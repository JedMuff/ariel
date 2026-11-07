"""
Head-mass experiment: what brings about rowing under evolution?

Reads head_mass_overnight.sh's runs and gait labels. A run's rowing share is the
share of its episodes in the last GENS_LAST generations labelled rowing; the
seed is the unit of analysis. Factor effects compare symmetric, legacy-protocol
factorial cells paired by seed and matched on the other factors, with a
bootstrap CI over seeds. With 3 seeds these are effect sizes, not tests.

Usage: python analyse_head_mass.py --root __data__/ariel_head_mass/evolution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import runs

GENS_LAST = 5
FACTORS = ["head", "hinge", "modules"]


def load(root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted((root / "runs").glob("*")):
        cfg = runs.run_config(run_dir)
        phys = cfg["body_physics"]
        csv = root / "labels" / run_dir.name / "episodes.csv"
        if not csv.exists():
            continue
        ep = pd.read_csv(csv)
        if "gen" not in ep or ep.empty:
            continue
        last = ep[ep.gen > ep.gen.max() - GENS_LAST]
        fits = runs.fitness_by_gen(run_dir)
        best = min((min(v) for v in fits.values() if v), default=np.nan)
        rows.append({
            "run": run_dir.name, "genome": cfg["genome_type"], "seed": cfg["seed"],
            "head": phys["core_mass"], "modules": phys["module_set"],
            "hinge": "old" if cfg["hinge_actuator"]["kv"] == 1 else "zoo", "protocol": cfg["eval_profile"],
            "complete": (run_dir / "DONE").exists(), "last_gen": int(ep.gen.max()),
            "rowing": last.rowing_label.eq("rowing").mean(), "alternating": last.rowing_label.eq("alternating").mean(),
            "rowing_gen0": ep[ep.gen == 0].rowing_label.eq("rowing").mean(), "displacement_median": -last.replayed_fitness.median(),
            "raw_median": last.raw_score.median(), "best_fitness": best, "n_episodes": len(last),
        })
    return pd.DataFrame(rows)


def _ci(diffs: np.ndarray, n: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = rng.choice(diffs, (n, len(diffs))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def effect(df: pd.DataFrame, factor: str, a: object, b: object, value: str = "rowing") -> str:
    """b minus a, matched on the other factors and seed."""
    others = [f for f in FACTORS if f != factor] + ["seed"]
    wide = df[df[factor].isin([a, b])].pivot_table(index=others, columns=factor, values=value)
    if a not in wide.columns or b not in wide.columns:
        return f"{factor} {a} -> {b}: no matched runs"
    wide = wide.dropna()
    if wide.empty:
        return f"{factor} {a} -> {b}: no matched runs"
    per_seed = (wide[b] - wide[a]).groupby("seed").mean().to_numpy()
    lo, hi = _ci((wide[b] - wide[a]).to_numpy())
    return (f"{factor} {a} -> {b}: {wide[a].mean():.2f} -> {wide[b].mean():.2f}, diff {wide[b].sub(wide[a]).mean():+.2f} "
            f"(CI over {len(wide)} matched pairs {lo:+.2f} to {hi:+.2f}; per seed {np.round(per_seed, 2).tolist()})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    df = load(root)
    if df.empty:
        raise SystemExit("no labelled runs")
    df.to_csv(root / "runs.csv", index=False)
    print(f"{len(df)} labelled runs, {int(df.complete.sum())} complete; last generation {df.last_gen.min()}-{df.last_gen.max()}")
    fact = df[(df.genome == "tree_symmetric") & (df.protocol == "legacy")]

    print("\nRowing share, symmetric, legacy protocol (mean over seeds, last generations)")
    print(fact.pivot_table(index=["hinge", "modules"], columns="head", values="rowing").round(2).to_string())
    print("\nSeeds per cell")
    print(fact.pivot_table(index=["hinge", "modules"], columns="head", values="rowing", aggfunc="count").to_string())
    print("\nMedian forward displacement (m)")
    print(fact.pivot_table(index=["hinge", "modules"], columns="head", values="raw_median").round(2).to_string())

    print("\nEffects on rowing share")
    print(effect(fact, "head", 1.0, 1.363))
    print(effect(fact, "head", 0.25, 1.0))
    print(effect(fact, "head", 1.0, 2.0))
    print(effect(fact, "hinge", "old", "zoo"))
    print(effect(fact, "modules", "local", "zoo"))
    print("\nEffects on median displacement")
    print(effect(fact, "head", 1.0, 1.363, "raw_median"))
    print(effect(fact, "hinge", "old", "zoo", "raw_median"))
    print(effect(fact, "modules", "local", "zoo", "raw_median"))

    print("\nProtocol and genome at the two ends (old = head 1.0, old hinge, old modules; current = 1.363, current, current)")
    ends = df[((df["head"] == 1.0) & (df.hinge == "old") & (df.modules == "local"))
              | ((df["head"] == 1.363) & (df.hinge == "zoo") & (df.modules == "zoo"))]
    print(ends.groupby(["modules", "genome", "protocol"])[["rowing", "alternating", "raw_median"]].agg(["mean", "count"])
          .round(2).to_string())


if __name__ == "__main__":
    main()
