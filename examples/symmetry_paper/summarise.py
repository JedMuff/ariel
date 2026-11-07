"""
Tables of Experiment 1 and 2 results, in the three spaces of the paper.

Experiment 1 (--e1 ROOT ...), per run (e1_runs.csv):
  score            main score: mean of the final population (runs.run_row);
                   champion: its best
  learning         over the final generation's trainings: curve_jitter (mean
                   |change| of the CMA-ES generation best between generations,
                   m), curve_median_jitter (same for the generation median),
                   evals_to_90 (evaluations until the best so far first reaches
                   90% of its final value); over the whole run: retrain_sd (the
                   SD of a body's scores across its retrainings, mean over
                   bodies trained at least 3 times)
  morphology       final population means of bodies.descriptors
  search           offspring fallbacks, mean draws per child
Per condition (e1_conditions.csv): mean, SD and n of every run column. Paired
gaps (e1_gaps.csv): the paper's contrasts per brain and budget, paired by seed.

Experiment 2 (--e2 ROOT ... [--bodies FILE]), per body and brain
(e2_bodies.csv): mean, SD, best and worst score over seeds, evals_to_90, body
descriptors; per genome and brain (e2_conditions.csv).

Usage:
    python summarise.py --e1 runs/e1 --e2 runs/e2 --bodies runs/e2/bodies.jsonl -o runs/analysis
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import runs
from bodies import descriptors

CONTRASTS = [  # (a, b): a - b
    ("tree_symmetric", "tree"),             # total effect of symmetry
    ("tree_symmetric", "tree_tied"),        # geometry vs duplication at the same share
    ("tree_tied_full", "tree_tied"),        # how much tying helps
    ("tree_tied", "tree"),                  # duplication alone
    ("grn", "tree"),                        # indirect encoding
]


def evals_to_frac(best: list[float], frac: float = 0.9, popsize: int = 20) -> float:
    """Evaluations until the best so far first reaches frac of its final value
    (NaN if the final best is not positive)."""
    so_far = np.maximum.accumulate(np.asarray(best, dtype=float))
    if not len(so_far) or so_far[-1] <= 0:
        return np.nan
    return float((np.argmax(so_far >= frac * so_far[-1]) + 1) * popsize)


def learning_metrics(records: list[dict]) -> dict:
    """See the module docstring; records are one generation's trainings."""
    jit = [np.mean(np.abs(np.diff(r["curve"]["best"]))) for r in records if len(r["curve"]["best"]) > 1]
    mjit = [np.mean(np.abs(np.diff(r["curve"]["median"]))) for r in records if len(r["curve"]["median"]) > 1]
    e90 = [evals_to_frac(r["curve"]["best"]) for r in records]
    return {"curve_jitter": float(np.mean(jit)) if jit else np.nan,
            "curve_median_jitter": float(np.mean(mjit)) if mjit else np.nan,
            "evals_to_90": float(np.nanmean(e90)) if np.isfinite(e90).any() else np.nan}


def e1_run(run_dir: Path) -> Optional[dict]:
    row = runs.run_row(run_dir)
    if row is None:
        return None
    evals = runs.evaluations(run_dir)
    last = runs.generations(run_dir)[-1]
    final_ids = set(last["ids"])
    final = [e for e in evals if e["gen"] == last["gen"] and e["id"] in final_ids]
    by_body = defaultdict(list)
    for e in evals:
        by_body[e["id"]].append(e["score"])
    sds = [np.std(v, ddof=1) for v in by_body.values() if len(v) >= 3]
    genome_of = runs.genomes_by_id(run_dir)
    morph = pd.DataFrame([descriptors(genome_of[i]) for i in last["ids"]]).mean(numeric_only=True)
    new = [e for e in evals if e["born"] == e["gen"] and e["gen"] > 0]
    return {**row, **learning_metrics(final),
            "retrain_sd": float(np.mean(sds)) if sds else np.nan,
            **{f"final_{k}": v for k, v in morph.items()},
            "child_draws": float(np.mean([e["attempts"] for e in new])) if new else np.nan}


def conditions(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    num = df.select_dtypes("number").drop(columns=[c for c in ("seed",) if c in df], errors="ignore")
    g = pd.concat([df[keys], num.drop(columns=[k for k in keys if k in num])], axis=1).groupby(keys)
    out = g.agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n"] = g.size()
    return out.reset_index()


def e1_tables(roots: list[Path], out: Path) -> None:
    rows = [r for d in runs.find(roots, "e1") if (r := e1_run(d))]
    if not rows:
        print("no E1 runs")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "e1_runs.csv", index=False)
    conditions(df, ["genome", "brain", "evals"]).to_csv(out / "e1_conditions.csv", index=False)
    gaps = pd.concat([runs.gap_table(df, ["brain", "evals"], a, b).assign(contrast=f"{a} - {b}")
                      for a, b in CONTRASTS if {a, b} <= set(df["genome"])], ignore_index=True)
    gaps.to_csv(out / "e1_gaps.csv", index=False)
    done = df["done"].sum()
    print(f"E1: {len(df)} runs ({done} done) in {df.groupby(['genome', 'brain', 'evals']).ngroups} conditions")


def e2_tables(roots: list[Path], bodies: Optional[Path], out: Path) -> None:
    rows = []
    for d in runs.find(roots, "e2"):
        cfg = runs.config(d)
        res = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]
        scores = np.array([r["score"] for r in res])
        rows.append({"body_id": cfg["body_id"], "genome": cfg["genome_type"], "brain": cfg["brain"],
                     "evals": cfg["evals"], "n_seeds": len(res), "score_mean": scores.mean(),
                     "score_sd": scores.std(ddof=1) if len(res) > 1 else np.nan,
                     "score_max": scores.max(), "score_min": scores.min(),
                     "evals_to_90": float(np.nanmean([evals_to_frac(r["curve"]["best"]) for r in res])),
                     "stopped_early": float(np.mean([r["stopped_early"] for r in res]))})
    if not rows:
        print("no E2 trainings")
        return
    df = pd.DataFrame(rows)
    if bodies is not None:
        desc = pd.DataFrame([{"body_id": b["body_id"], "rejected": b["rejected"], **descriptors(b["genome"])}
                             for b in map(json.loads, bodies.read_text().splitlines())])
        df = df.merge(desc, on="body_id", how="left")
    df.to_csv(out / "e2_bodies.csv", index=False)
    conditions(df, ["genome", "brain"]).to_csv(out / "e2_conditions.csv", index=False)
    print(f"E2: {len(df)} body x brain trainings, {int(df['n_seeds'].sum())} seeds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1", type=Path, nargs="*", default=[])
    ap.add_argument("--e2", type=Path, nargs="*", default=[])
    ap.add_argument("--bodies", type=Path, help="sample_bodies.py output, for E2 body descriptors")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.e1:
        e1_tables(args.e1, args.out)
    if args.e2:
        e2_tables(args.e2, args.bodies, args.out)


if __name__ == "__main__":
    main()
