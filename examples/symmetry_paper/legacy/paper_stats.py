"""
Statistics quoted in the paper that no analyse_*.py script prints.

  fixed bodies    per body set and evaluation count: mean; versus tree and
                  symmetric, the difference of body means with a bootstrap 95%
                  CI, a permutation test on the mean and Mann-Whitney p; the
                  gain from 20 evaluations to the end of training
  evolution       modules and hinges of each run's champion, and the paired gaps
                  when champions are limited to the module cap
  phase shift     locked share of locked symmetric trainings under each condition

Usage:
    python paper_stats.py --data $V2 --analysis $V2/paper --out $V2/paper/stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

import runs
from analyse_e2 import load_trainings

NAME = {"shape_matched": "shape-matched", "leg_matched": "leg-matched"}
GENOME = {"tree": "tree", "tree_symmetric": "symmetric", "tree_tied": "tied"}
COMPARE = [("symmetric", "tree"), ("tied", "tree"), ("shape-matched", "tree"), ("leg-matched", "tree"),
           ("symmetric", "tied"), ("symmetric", "shape-matched"), ("tied", "leg-matched")]
N = 10_000
MODULE_CAP = 26  # 25 modules plus the core


def compare(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    gap = a.mean() - b.mean()
    boot = rng.choice(a, (N, len(a))).mean(1) - rng.choice(b, (N, len(b))).mean(1)
    pool = np.concatenate([a, b])
    perm = np.array([rng.permuted(pool) for _ in range(N)])
    null = perm[:, :len(a)].mean(1) - perm[:, len(a):].mean(1)
    return {"gap": gap, "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
            "perm_p": (1 + (np.abs(null) >= abs(gap)).sum()) / (N + 1), "mw_p": mannwhitneyu(a, b).pvalue}


def end_of_training(d: pd.DataFrame) -> pd.DataFrame:
    """The rows of each controller's largest evaluation count."""
    return d[d["evals"] == d.groupby("brain")["evals"].transform("max")]


def fixed_bodies(data: Path, e2_name: str, out: Path, rng: np.random.Generator) -> None:
    roots = [data / e2_name / s for s in ("random", "tied", "shape_matched", "leg_matched")]
    tr, cur = load_trainings(roots)
    for d in (tr, cur):
        d["set"] = [NAME.get(s, GENOME[g]) for s, g in zip(d["body_source"], d["genome_type"])]
    per = cur.groupby(["brain", "set", "body_id", "evals"])["score"].mean().reset_index()
    per.to_csv(out / "fixed_body_curves.csv", index=False)
    print("mean score per set and evaluations:")
    at = per[per["evals"].isin([20, 60, 200, 800, 1600, 4800, 5000, 15000])]
    print(at.pivot_table(index=["brain", "set"], columns="evals", values="score").round(3).to_string())

    rows = []
    for (brain, evals), g in at.groupby(["brain", "evals"]):
        get = {s: x["score"].to_numpy() for s, x in g.groupby("set")}
        for a, b in COMPARE:
            if a in get and b in get:
                rows.append({"brain": brain, "evals": evals, "a": a, "b": b, "ratio": get[a].mean() / get[b].mean(),
                             **compare(get[a], get[b], rng)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "fixed_body_gaps.csv", index=False)
    print("\ngaps at the end of training:")
    last = end_of_training(t)
    print(last.round(3).to_string(index=False))
    final = end_of_training(per)
    print("\nmedian final score per set:")
    print(final.pivot_table(index="brain", columns="set", values="score", aggfunc="median").round(3).to_string())
    print("\nshare of the symmetric - tree gap closed, bootstrap 95% CI over bodies:")
    for brain in ("bang_bang", "dann"):
        get = {k: x["score"].to_numpy() for k, x in final[final["brain"] == brain].groupby("set")}
        boot = {k: rng.choice(v, (N, len(v))).mean(1) for k, v in get.items()}
        for k in ("shape-matched", "tied", "leg-matched"):
            share = (boot[k] - boot["tree"]) / (boot["symmetric"] - boot["tree"])
            point = (get[k].mean() - get["tree"].mean()) / (get["symmetric"].mean() - get["tree"].mean())
            print(f"  {brain} {k}: {point:.0%} [{np.percentile(share, 2.5):.0%}, {np.percentile(share, 97.5):.0%}]")
    print("\nsymmetric - tree by evaluations:")
    print(t[(t["a"] == "symmetric") & (t["b"] == "tree")].round(3).to_string(index=False))

    print("\ngain from 20 evaluations to the end of training, symmetric vs tree:")
    for brain, g in per.groupby("brain"):
        w = g[g["set"].isin(["symmetric", "tree"])].pivot_table(index=["set", "body_id"], columns="evals", values="score")
        gain = w[w.columns.max()] - w[20]
        c = compare(gain.loc["symmetric"].to_numpy(), gain.loc["tree"].to_numpy(), rng)
        print(f"  {brain}: symmetric {gain.loc['symmetric'].mean():.3f}, tree {gain.loc['tree'].mean():.3f}, "
              f"difference {c['gap']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}], perm p={c['perm_p']:.3f}")


def evolved_sizes(data: Path, out: Path) -> None:
    rows = []
    roots = [data / "e1"]
    if (data / "e1_dann" / "sb10").exists():
        roots.append(data / "e1_dann" / "sb10")
    for r in runs.load_runs(*roots).itertuples():
        final = [json.loads((c / "meta.json").read_text()) for c in runs.final_checkpoints(Path(r.path))]
        best = min(final, key=lambda m: m["fitness"])
        rows.append({"brain": r.brain, "genome": r.genome, "seed": r.seed, "score": r.score,
                     "modules": best["num_modules"],
                     "pop_modules": np.mean([m["num_modules"] for m in final]),
                     "pop_over_cap": np.mean([m["num_modules"] > MODULE_CAP for m in final])})
    d = pd.DataFrame(rows)
    d.to_csv(out / "evolved_sizes.csv", index=False)
    print("\nchampion and final-population size in evolution:")
    print(d.groupby(["brain", "genome"]).agg(score=("score", "mean"), modules=("modules", "mean"), max_modules=("modules", "max"),
                                             over_cap=("modules", lambda x: (x > MODULE_CAP).sum()),
                                             pop_modules=("pop_modules", "mean"), pop_over_cap=("pop_over_cap", "mean")).round(2).to_string())
    print("\ntied - tree, seeds whose tied champion is within the module cap:")
    for brain, g in d.groupby("brain"):
        score = g.pivot(index="seed", columns="genome", values="score")
        within = g.pivot(index="seed", columns="genome", values="modules")["tree_tied"] <= MODULE_CAP
        x = (score["tree_tied"] - score["tree"])[within].to_numpy()
        boot = np.random.default_rng(0).choice(x, (N, len(x))).mean(1)
        print(f"  {brain}: {x.mean():+.3f} [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}], "
              f"Wilcoxon p={wilcoxon(x).pvalue:.3f}, n={len(x)}")
    print("\nSpearman of champion modules with score, per brain and genome:")
    for (b, g), x in d.groupby(["brain", "genome"]):
        rho = spearmanr(x["modules"], x["score"])
        print(f"  {b} {g}: rho={rho.statistic:+.2f} p={rho.pvalue:.3f}")


def phase_shift(data: Path, ps_name: str, analysis: Path, out: Path) -> None:
    eps = pd.concat([pd.read_csv(p, usecols=["run", "brain", "genome", "condition", "ckpt", "skipped", "score", "phase_locked_frac"])
                     for p in sorted((data / "e3" / ps_name).glob("*/phase_shift.csv"))], ignore_index=True)
    eps = eps[~eps["skipped"].astype(bool)]
    base = eps[eps["condition"] == "baseline"].set_index("run")["phase_locked_frac"]
    locked = eps[(eps["genome"] == "tree_symmetric") & (eps["run"].map(base) >= 0.5)]
    tab = locked.groupby(["brain", "condition"])["phase_locked_frac"].mean().unstack(0).round(2)
    tab.to_csv(out / "locked_share_by_condition.csv")
    print("\nlocked share of locked symmetric trainings, by condition:")
    print(tab.to_string())
    bl = pd.read_csv(analysis / "phase_shift" / "phase_shift_bodies.csv")
    bl = bl[bl["group"] == "symmetric locked"]
    print("\nside - random per body, locked symmetric: range per brain")
    print(bl.groupby("brain").agg(lo=("mean", "min"), hi=("mean", "max"), min_p=("p", "min"), n=("n", "max")).round(3).to_string())

    ck = pd.read_csv(analysis / "phase_shift" / "phase_shift_ckpts.csv")
    print("\nunperturbed score per training, by group:")
    print(ck.groupby(["brain", "group"])["baseline"].agg(["mean", "count"]).round(3).to_string())
    print("\nlocked minus unlocked symmetric, seeds within mixed bodies, and between bodies:")
    for brain, g in ck[ck["genome"] == "tree_symmetric"].groupby("brain"):
        w = g.pivot_table(index="body", columns="group", values="baseline")
        both = w.dropna()
        d = (both["symmetric locked"] - both["symmetric unlocked"]).to_numpy()
        p = wilcoxon(d).pvalue if len(d) > 1 else float("nan")
        n_lock = g.groupby("body")["group"].apply(lambda s: (s == "symmetric locked").sum())
        print(f"  {brain}: within-body {d.mean():+.3f} over {len(d)} bodies, Wilcoxon p={p:.3f}; "
              f"seeds locked per body counts {n_lock.value_counts().sort_index().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--e2-name", default="e2", help="folder of the fixed-body trainings under --data")
    parser.add_argument("--phase-shift-name", default="phase_shift_e2_random", help="folder under --data/e3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)
    fixed_bodies(args.data, args.e2_name, args.out, rng)
    evolved_sizes(args.data, args.out)
    phase_shift(args.data, args.phase_shift_name, args.analysis, args.out)


if __name__ == "__main__":
    main()
