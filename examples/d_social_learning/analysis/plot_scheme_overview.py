"""All-schemes overview: every subscheme overlaid on one plot, faceted by x
(crowding), with separate "mean" and "best" panels (each including a ±std
shaded band).

One figure per (experiment, metric) — e.g. ariel_fitness_overview.png,
ariel_novelty_overview.png, ariel_muplus_fitness_overview.png, ... — each with
rows=x values and 2 columns (mean, best), all 8 schemes drawn as colored
lines+bands (aggregated across reps). Non-finite values (inf/nan) are ignored.

The fitness y-scale is shared per x-row (mean/max panels each get their own
scale, shared across schemes and experiments within that row); the diversity
y-scale is shared across the whole grid.

Usage:
    uv run examples/d_social_learning/analysis/plot_scheme_overview.py \
        --data-dir __data__/snellius_test_data/social \
        --domains ariel,ariel_muplus \
        --out-dir __data__/snellius_test_data/social/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from curve_utils import (
    SCHEME_COLORS,
    SCHEMES,
    X_VALUES,
    aggregate_across_reps,
    compute_global_ylim_split,
    compute_row_ylim_splits,
)


def _split_ylim_for_row(ylims, xv: float) -> dict:
    if ylims is None:
        return {}
    if xv in ylims:  # row-wise: {x_value: {"mean": .., "max": ..}}
        return ylims[xv] or {}
    return ylims  # whole-grid: {"mean": .., "max": ..}


def plot_overview(
    exp_dir: Path,
    label: str,
    metric: str,
    ylabel: str,
    out_dir: Path,
    ylims=None,
) -> None:
    data = {}
    for scheme in SCHEMES:
        for xv in X_VALUES:
            result = aggregate_across_reps(exp_dir, scheme, xv, metric)
            if result is not None:
                data[(scheme, xv)] = result

    present_x = [xv for xv in X_VALUES if any((s, xv) in data for s in SCHEMES)]
    present_schemes = [s for s in SCHEMES if any((s, xv) in data for xv in X_VALUES)]
    if not present_x:
        print(f"  [skip] no data for {label}/{metric}")
        return

    fig, axes = plt.subplots(
        len(present_x), 2,
        figsize=(11, 3.2 * len(present_x)),
        squeeze=False, sharex=True,
    )
    fig.suptitle(f"{label} — {ylabel} by scheme (aggregated across reps)", fontsize=14)

    for xi, xv in enumerate(present_x):
        ax_mean, ax_max = axes[xi][0], axes[xi][1]
        for scheme in present_schemes:
            result = data.get((scheme, xv))
            if result is None:
                continue
            gens, mean_of_mean, std_of_mean, mean_of_max, std_of_max = result
            c = SCHEME_COLORS[scheme]
            ax_mean.fill_between(
                gens, mean_of_mean - std_of_mean, mean_of_mean + std_of_mean,
                alpha=0.10, color=c,
            )
            ax_mean.plot(gens, mean_of_mean, lw=1.6, color=c, label=scheme)
            ax_max.fill_between(
                gens, mean_of_max - std_of_max, mean_of_max + std_of_max,
                alpha=0.10, color=c,
            )
            ax_max.plot(gens, mean_of_max, lw=1.6, color=c, label=scheme)

        ax_mean.set_ylabel(f"x={xv}\n{ylabel}", fontsize=8)
        for ax in (ax_mean, ax_max):
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if xi == len(present_x) - 1:
                ax.set_xlabel("generation", fontsize=8)
        if xi == 0:
            ax_mean.set_title("mean", fontsize=10)
            ax_max.set_title("best", fontsize=10)

        row_ylims = _split_ylim_for_row(ylims, xv)
        if row_ylims.get("mean") is not None:
            ax_mean.set_ylim(row_ylims["mean"])
        if row_ylims.get("max") is not None:
            ax_max.set_ylim(row_ylims["max"])

    axes[0][1].legend(fontsize=6.5, loc="best")
    plt.tight_layout()
    out_path = out_dir / f"{label}_{metric}_overview.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="__data__/social")
    parser.add_argument("--out-dir", default="__data__/social/plots")
    parser.add_argument(
        "--domains", default="ariel,evogym",
        help="Comma-separated experiment subdirectory names under --data-dir",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    labels = args.domains.split(",")
    exp_dirs = [data_dir / label for label in labels]

    # Fitness: per-x-row scale (mean/max panels each get their own range).
    fitness_ylims = compute_row_ylim_splits(exp_dirs, "fitness")
    for label, exp_dir in zip(labels, exp_dirs):
        plot_overview(exp_dir, label, "fitness", "fitness", out_dir, fitness_ylims)

    # Diversity: single scale shared across the whole grid.
    novelty_ylims = compute_global_ylim_split(exp_dirs, "novelty")
    for label, exp_dir in zip(labels, exp_dirs):
        plot_overview(exp_dir, label, "novelty", "novelty (diversity)", out_dir, novelty_ylims)


if __name__ == "__main__":
    main()
