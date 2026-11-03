"""
Does retraining every survivor every generation change the genome gaps, and does
that depend on the learning budget?

Compares evolution runs without retraining (--plain) and with it (--retrained),
paired by seed within each controller and budget: the mean score per arm, each
paired gap (bootstrap 95% CI, Wilcoxon p), and the difference between the two gaps.

Absolute scores are not comparable across the two protocols. Without retraining
a run's score is the best of every training in the run, so it keeps lucky
results; with retraining it comes from the last generation only. Compare gaps,
not scores.

Usage:
    python compare_retraining.py --plain $FINAL_ROOT/e1_dann/sb* --retrained $FINAL_ROOT/e1_dann_retrain/sb*
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import runs

COMPARISONS = [("tree_symmetric", "tree"), ("tree_tied", "tree"), ("tree_symmetric", "tree_tied")]
KEYS = ["brain", "evals_per_body"]


def paired(df: pd.DataFrame, a: str, b: str) -> pd.Series | None:
    wide = df.pivot(index="seed", columns="genome", values="score")
    return (wide[a] - wide[b]).dropna() if a in wide and b in wide else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plain", type=Path, nargs="+", required=True)
    parser.add_argument("--retrained", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    plain, retrained = runs.load_runs(*args.plain), runs.load_runs(*args.retrained)
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 200)

    for label, df in (("without retraining", plain), ("with retraining", retrained)):
        print(f"\nscores {label}:")
        print(df.groupby([*KEYS, "genome"])["score"].agg(["mean", "median", "count"]).round(2).to_string())
        gaps = [runs.gap_table(df, KEYS, a=a, b=b).assign(comparison=f"{a} - {b}") for a, b in COMPARISONS
                if {a, b} <= set(df["genome"])]
        print(f"\npaired gaps {label}:")
        print(pd.concat(gaps)[[*KEYS, "comparison", "gap", "ci_lo", "ci_hi", "p", "wins", "n"]]
              .sort_values(["comparison", *KEYS]).round(3).to_string(index=False))

    print("\nchange in each gap from retraining (retrained - plain, paired by seed):")
    rows = []
    for (brain, evals), p in plain.groupby(KEYS):
        r = retrained[(retrained["brain"] == brain) & (retrained["evals_per_body"] == evals)]
        if r.empty:
            continue
        for a, b in COMPARISONS:
            dp, dr = paired(p, a, b), paired(r, a, b)
            if dp is None or dr is None:
                continue
            seeds = dp.index.intersection(dr.index)
            change = (dr[seeds] - dp[seeds]).to_numpy()
            boot = rng.choice(change, (runs.N_BOOT, len(change))).mean(1)
            rows.append({"brain": brain, "evals_per_body": evals, "comparison": f"{a} - {b}", "plain": dp[seeds].mean(),
                         "retrained": dr[seeds].mean(), "change": change.mean(), "ci_lo": np.percentile(boot, 2.5),
                         "ci_hi": np.percentile(boot, 97.5), "n": len(seeds)})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
