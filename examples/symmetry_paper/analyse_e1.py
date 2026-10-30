"""
E1: the coordination ablation (R6).

{bang_bang, bang_bang_free} x {tree, tree_symmetric, tree_tied} x 10 seeds
(slurm e1_coordination_ablation.sh). Paired by seed:
  symmetric - tree   the symmetric advantage, with and without a shared frequency
  tied - tree        paired limbs without mirroring
  symmetric - tied   mirroring itself
Checks for the ablation: the spread of evolved hinge frequencies within each
bang_bang_free champion (if they converge, the ablation did not remove
coordination), and the absolute scores per arm (extra parameters can hurt
CMA-ES on every genome). With --behaviour (replay.py output for these runs,
--gens last), the gait class and phase locking of every run's champion.

Writes to --out (default <root>/analysis):
  e1_runs.csv, e1_gaps.csv, e1_champions.csv
  fig7_ablation.png   paired gaps per brain with 95% CI, share of phase-locked
                      champions per genome beside each bar

Usage:
    python analyse_e1.py --root /scratch/$USER/ariel_paper_e1 [--behaviour /scratch/$USER/ariel_paper_e3/e1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runs  # noqa: E402
from brains import BangBangFreeBrain  # noqa: E402

COMPARISONS = [("tree_symmetric", "tree"), ("tree_tied", "tree"), ("tree_symmetric", "tree_tied")]
LOCKED_MIN = 0.5  # a champion is phase-locked if locked pairs carry at least this share of its stroke


def frequency_spread(run_dir: Path) -> float:
    """Coefficient of variation of the hinges' mean evolved frequency in a
    bang_bang_free champion (0 = all hinges share one frequency)."""
    ckpt = runs.best_checkpoint(run_dir)
    w = np.load(ckpt / "fwd_weights.npy")
    brain = BangBangFreeBrain(len(w) // 9)
    brain.set_params(w)
    f = (np.asarray(brain.freq.start) + np.asarray(brain.freq.end)) / 2
    return float(f.std() / f.mean())


def champions(behaviour: Path) -> pd.DataFrame:
    """The best (lowest stored fitness) last-generation episode of every replayed run."""
    eps = pd.concat([pd.read_csv(p) for p in sorted(behaviour.glob("*/episodes.csv"))], ignore_index=True)
    best = eps.loc[eps.groupby("run")["stored_fitness"].idxmin()].copy()
    best["phase_locked"] = best["phase_locked_frac"] >= LOCKED_MIN
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--behaviour", type=Path, help="replay.py output root for the E1 runs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = runs.load_runs(args.root)
    df["freq_cv"] = [frequency_spread(Path(p)) if b == "bang_bang_free" else 0.0 for p, b in zip(df["path"], df["brain"])]
    df.to_csv(out / "e1_runs.csv", index=False)
    print("mean score per arm (and frequency spread of bang_bang_free champions):")
    print(df.groupby(["brain", "genome"])[["score", "freq_cv"]].agg(["mean", "count"]).round(3).to_string())

    gaps = pd.concat([runs.gap_table(df, ["brain"], a=a, b=b).assign(comparison=f"{a} - {b}")
                      .rename(columns={f"mean_{a}": "mean_a", f"mean_{b}": "mean_b"}) for a, b in COMPARISONS])
    gaps.to_csv(out / "e1_gaps.csv", index=False)
    print("\npaired gaps:")
    print(gaps[["brain", "comparison", "gap", "ci_lo", "ci_hi", "p", "wins", "n"]].round(3).to_string(index=False))

    locked = {}
    if args.behaviour:
        champ = champions(args.behaviour)
        champ.to_csv(out / "e1_champions.csv", index=False)
        tab = champ.groupby(["brain", "genome"]).agg(
            n=("run", "size"), phase_locked=("phase_locked", "mean"), phase_quality=("phase_quality", "mean"),
            alternating=("rowing_label", lambda s: (s == "alternating").mean()), score=("score", "mean"))
        print(f"\nchampions (phase-locked: locked pairs >= {LOCKED_MIN:.0%} of the stroke):")
        print(tab.round(2).to_string())
        locked = tab["phase_locked"].to_dict()

    brains = [b for b in ("bang_bang", "bang_bang_free") if b in set(df["brain"])]
    fig, axes = plt.subplots(1, len(brains), figsize=(4.4 * len(brains), 3.4), sharey=True, squeeze=False,
                             constrained_layout=True)
    for ax, brain in zip(axes[0], brains):
        runs.style(ax)
        ax.axhline(0, color=runs.INK_2, linewidth=0.8)
        g = gaps[gaps["brain"] == brain]
        x = np.arange(len(g))
        colors = [runs.GENOME_COLOR[c.split(" - ")[0]] for c in g["comparison"]]
        ax.bar(x, g["gap"], width=0.6, color=colors, edgecolor="white", linewidth=2)
        ax.errorbar(x, g["gap"], yerr=[g["gap"] - g["ci_lo"], g["ci_hi"] - g["gap"]], fmt="none",
                    ecolor=runs.INK, capsize=3, linewidth=1)
        labels = [c.replace("tree_symmetric", "sym").replace("tree_tied", "tied") for c in g["comparison"]]
        ax.set_xticks(x, labels, fontsize=8)
        for xi, (comp, p) in enumerate(zip(g["comparison"], g["p"])):
            a, b = comp.split(" - ")
            note = f"p={p:.2g}"
            if locked:
                note += f"\nlocked {locked.get((brain, a), np.nan):.0%} vs {locked.get((brain, b), np.nan):.0%}"
            ax.annotate(note, (xi, g["ci_hi"].iloc[xi]), textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=7, color=runs.INK_2)
        ax.set_title(runs.BRAIN_LABEL[brain], fontsize=10, color=runs.INK)
    axes[0, 0].set_ylabel("paired score gap (95% CI)", fontsize=9, color=runs.INK_2)
    fig.savefig(out / "fig7_ablation.png", dpi=200)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
