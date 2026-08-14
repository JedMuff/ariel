"""Fitness + diversity curves for a single rep (no cross-rep aggregation).

Each panel shows the per-generation population mean ±1 std (shaded band,
population std within that generation's cohort — not cross-rep std, since
there's only one rep), the best-of-generation (thin dashed — the top
individual newly born that generation, which is noisy since it misses
elites that survived from earlier generations), and the running-best
(solid — cumulative max across generations). For metric="fitness" under this
framework's (mu+lambda) elitism, running-best coincides with the true
best-alive-individual's fitness at every generation, so unlike
best-of-generation it can only go up — see curve_utils.single_run_curve.

Usage:
    uv run examples/d_social_learning/analysis/plot_single_run.py \
        --rep-dir __data__/social/ariel/lamarckian/x10/rep_0 \
        [--out-dir __data__/social/ariel/lamarckian/x10/rep_0/plots]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from curve_utils import single_run_curve

C_MEAN = "tab:blue"
C_GEN_BEST = "tab:orange"
C_RUNNING_BEST = "tab:green"

METRICS = [("fitness", "fitness"), ("novelty", "novelty (diversity)")]


def plot_single_run(rep_dir: Path, out_dir: Path) -> None:
    present = [(m, lbl) for m, lbl in METRICS if single_run_curve(rep_dir, m) is not None]
    if not present:
        print(f"  [skip] no data in {rep_dir}")
        return

    fig, axes = plt.subplots(1, len(present), figsize=(6.0 * len(present), 4.2), squeeze=False)
    axes = axes[0]

    label = "/".join(rep_dir.parts[-3:])
    fig.suptitle(f"{label} — per-generation curves (single run)", fontsize=13)

    for ax, (metric, ylabel) in zip(axes, present):
        gens, mean, std, best_of_gen, running_best = single_run_curve(rep_dir, metric)
        ax.fill_between(gens, mean - std, mean + std, alpha=0.2, color=C_MEAN, label="mean ±1 std")
        ax.plot(gens, mean, lw=1.8, color=C_MEAN, label="mean")
        ax.plot(gens, best_of_gen, lw=1.0, ls="--", color=C_GEN_BEST, alpha=0.7, label="best of generation")
        ax.plot(gens, running_best, lw=1.8, color=C_RUNNING_BEST, label="running best")
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("generation", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[0].legend(fontsize=8, loc="best")
    plt.tight_layout()
    out_path = out_dir / "single_run_curves.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-dir", required=True, help="e.g. __data__/social/ariel/lamarckian/x10/rep_0")
    parser.add_argument("--out-dir", default=None, help="Defaults to <rep-dir>/plots")
    args = parser.parse_args()

    rep_dir = Path(args.rep_dir)
    out_dir = Path(args.out_dir) if args.out_dir else rep_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_single_run(rep_dir, out_dir)


if __name__ == "__main__":
    main()
