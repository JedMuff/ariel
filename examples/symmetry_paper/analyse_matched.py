"""
E2 controls for the symmetric advantage on random bodies (R2, R6):
  shape-matched tree bodies   does body shape, not mirroring, carry the gap?
  random tied bodies          do paired limbs without mirroring get it too?

Bodies come from sample_bodies.py (random, tied, shape_matched; all with the
hinge counts of the random symmetric bodies), trained with
train_fixed_body.py like the random set. Per body, scores are the seed mean.

Readout, per brain and metric (final score, score after 200 evaluations):
  mean score of each body set: tree, symmetric, tied (random) and tree
  (shape-matched); sym - X unpaired gap with bootstrap CI and Mann-Whitney p;
  sym - shape-matched also paired by the body it was matched to (Wilcoxon);
  share of the random-body gap closed by shape matching; and the gap adjusted
  for the residual shape differences (OLS of score on symmetric + the
  SHAPE_MEASURES over the symmetric and shape-matched bodies, bootstrap CI over
  bodies), since tree bodies cannot match every measure (they always reach the
  module cap, so they stay taller).

Writes to --out:
  matched_bodies.csv  one row per body and brain: set, scores, shape measures
  matched_gaps.csv    the readout above
  fig9_matched.png    mean final score per body set (± SEM), one panel per brain

Usage:
    python analyse_matched.py --roots $FINAL_ROOT/e2/random $FINAL_ROOT/e2/tied $FINAL_ROOT/e2/shape_matched \\
        --bodies $FINAL_ROOT/e2/bodies_{random,tied,shape_matched}.jsonl --out $FINAL_ROOT/e2/analysis_matched
"""

from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import runs  # noqa: E402
from analyse_bodies import body_measures  # noqa: E402
from analyse_e2 import load_trainings, unpaired_gap  # noqa: E402
from sample_bodies import SHAPE_MEASURES  # noqa: E402

SETS = ["tree", "tree_symmetric", "tree_tied", "tree (shape-matched)"]
SET_STYLE = {"tree": (runs.GENOME_COLOR["tree"], 1.0), "tree_symmetric": (runs.GENOME_COLOR["tree_symmetric"], 1.0),
             "tree_tied": (runs.GENOME_COLOR["tree_tied"], 1.0), "tree (shape-matched)": (runs.GENOME_COLOR["tree"], 0.45)}
METRICS = ["final", "score_at_200"]


def body_set(row: pd.Series) -> str:
    return "tree (shape-matched)" if row["body_source"] == "shape_matched" else row["genome_type"]


def pair_key(body_id: str) -> str:
    """match_tree_007 and rand_tree_symmetric_007 share the key 007."""
    return body_id.rsplit("_", 1)[1]


def adjusted_gap(d: pd.DataFrame, metric: str, rng: np.random.Generator) -> dict:
    """Coefficient of symmetric in OLS metric ~ symmetric + SHAPE_MEASURES."""
    d = d.dropna(subset=[metric, *SHAPE_MEASURES])
    if d["set"].nunique() < 2 or len(d) <= len(SHAPE_MEASURES) + 2:
        return {"adj_gap": math.nan, "adj_ci_lo": math.nan, "adj_ci_hi": math.nan}
    X = np.column_stack([np.ones(len(d)), (d["set"] == "tree_symmetric").to_numpy(float), d[list(SHAPE_MEASURES)].to_numpy(float)])
    y = d[metric].to_numpy(float)

    def coef(i: np.ndarray) -> float:
        return float(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])

    boot = [coef(rng.integers(0, len(y), len(y))) for _ in range(2000)]
    return {"adj_gap": coef(np.arange(len(y))), "adj_ci_lo": float(np.percentile(boot, 2.5)),
            "adj_ci_hi": float(np.percentile(boot, 97.5))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=Path, nargs="+", required=True, help="train_fixed_body.py output roots")
    parser.add_argument("--bodies", type=Path, nargs="+", required=True, help="the sample_bodies.py files they trained")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tr, _ = load_trainings(args.roots)
    tr["set"] = tr.apply(body_set, axis=1)
    bodies = tr.groupby(["body_id", "set", "brain", "n_hinges"], as_index=False)[METRICS].mean()

    genomes = {b["body_id"]: b["genome"] for f in args.bodies for b in map(json.loads, f.read_text().splitlines())}
    ids = sorted(set(bodies["body_id"]) & set(genomes))
    with Pool(args.workers) as p:
        measures = dict(zip(ids, p.map(body_measures, [genomes[i] for i in ids])))
    shape = pd.DataFrame([{"body_id": i, **{k: measures[i][k] for k in SHAPE_MEASURES}} for i in ids])
    bodies = bodies.merge(shape, on="body_id", how="left")
    bodies.to_csv(args.out / "matched_bodies.csv", index=False)

    rng = np.random.default_rng(0)
    rows = []
    for (brain, metric), (_, g) in [((b, m), (b, bodies[bodies["brain"] == b])) for b in sorted(bodies["brain"].unique()) for m in METRICS]:
        sym = g.loc[g["set"] == "tree_symmetric", metric].to_numpy(float)
        row = {"brain": brain, "metric": metric, **{f"mean_{s}": g.loc[g["set"] == s, metric].mean() for s in SETS}}
        for s in SETS[:1] + SETS[2:]:
            other = g.loc[g["set"] == s, metric].to_numpy(float)
            gap = unpaired_gap(sym, other, rng) if len(other) else {}
            row |= {f"sym-{s}_{k}": gap.get(k, math.nan) for k in ("gap", "ci_lo", "ci_hi", "p")}
        m = g[g["set"] == "tree (shape-matched)"]
        if len(m):
            s = g[g["set"] == "tree_symmetric"].assign(key=lambda x: x["body_id"].map(pair_key)).set_index("key")[metric]
            mm = m.assign(key=lambda x: x["body_id"].map(pair_key)).set_index("key")[metric]
            d = (s - mm).dropna().to_numpy()
            row |= {"paired_gap": float(d.mean()), "paired_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else math.nan,
                    "n_pairs": len(d)}
            raw = row["sym-tree_gap"]
            row["gap_closed"] = (raw - row["sym-tree (shape-matched)_gap"]) / raw if raw else math.nan
            row |= adjusted_gap(g[g["set"].isin(["tree_symmetric", "tree (shape-matched)"])], metric, rng)
        rows.append(row)
    gaps = pd.DataFrame(rows)
    gaps.to_csv(args.out / "matched_gaps.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("mean score per body set:")
        print(gaps[["brain", "metric", *[f"mean_{s}" for s in SETS]]].round(3).to_string(index=False))
        print("\nsymmetric - X (unpaired, bootstrap 95% CI, Mann-Whitney p):")
        for s in SETS[:1] + SETS[2:]:
            c = [f"sym-{s}_{k}" for k in ("gap", "ci_lo", "ci_hi", "p")]
            print(f"  vs {s}:")
            print(gaps[["brain", "metric", *c]].rename(columns=dict(zip(c, ["gap", "ci_lo", "ci_hi", "p"]))).round(3).to_string(index=False))
        if "paired_gap" in gaps:
            print("\nshape matching: paired gap, share of the random-body gap closed, gap adjusted for residual shape:")
            print(gaps[["brain", "metric", "paired_gap", "paired_p", "n_pairs", "gap_closed", "adj_gap", "adj_ci_lo", "adj_ci_hi"]]
                  .round(3).to_string(index=False))

    brains = [b for b in runs.BRAIN_LABEL if b in set(bodies["brain"])]
    sets = [s for s in SETS if s in set(bodies["set"])]
    fig, axes = plt.subplots(1, len(brains), figsize=(3.6 * len(brains), 3.4), squeeze=False, sharey=True)
    for ax, brain in zip(axes[0], brains):
        g = bodies[bodies["brain"] == brain]
        for i, s in enumerate(sets):
            v = g.loc[g["set"] == s, "final"]
            color, alpha = SET_STYLE[s]
            ax.bar(i, v.mean(), 0.7, yerr=v.sem(), color=color, alpha=alpha, error_kw={"ecolor": runs.INK_2})
        ax.set_xticks(range(len(sets)), [s.replace("tree_symmetric", "symmetric").replace("tree_tied", "tied")
                                         .replace(" (", "\n(") for s in sets], fontsize=8)
        ax.set_title(runs.BRAIN_LABEL[brain], fontsize=10, color=runs.INK)
        runs.style(ax)
    axes[0][0].set_ylabel("final score per body (± SEM)", fontsize=9, color=runs.INK_2)
    fig.tight_layout()
    fig.savefig(args.out / "fig9_matched.png", dpi=160)
    plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
