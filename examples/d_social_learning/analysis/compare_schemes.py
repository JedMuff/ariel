"""Compare (mu,lambda) vs (mu+lambda) selection for the social-learning
subschemes.

Produces two kinds of figures:

1. Per-scheme comparison (plot_scheme): for each subscheme (darwinian,
   lamarckian, random, ...), one figure with rows=x values and 2 columns
   (fitness, diversity/novelty), overlaying the two selection strategies'
   mean±std and best±std curves (aggregated across reps).

2. Combined scheme-comparison grid (plot_combined_grid): one figure per
   metric (fitness, diversity) — {metric}_curve_scheme_comparison.png — with
   rows=x values x cols=schemes (like plot_run_curves.py's grid), both
   selection strategies overlaid (mean±std + best±std) in every cell.

Non-finite fitness/novelty values (inf/nan) are ignored. The fitness y-scale
is shared per x-row (across schemes/strategies) but not across rows, since
fitness ranges vary a lot with crowding (x); the diversity y-scale is shared
across the whole grid.

Usage:
    uv run examples/d_social_learning/analysis/compare_schemes.py \
        --data-dir __data__/snellius_test_data/social \
        --comma-dir ariel --plus-dir ariel_muplus \
        --out-dir __data__/snellius_test_data/social/plots/compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from curve_utils import (
    SCHEMES,
    X_VALUES,
    aggregate_across_reps,
    compute_global_ylim,
    compute_row_ylims,
)

METRICS = [("fitness", "fitness"), ("novelty", "novelty (diversity)")]
LABELS = ["(mu,lambda)", "(mu+lambda)"]
COLORS = {"(mu,lambda)": "tab:blue", "(mu+lambda)": "tab:orange"}


def _ylim_for_row(ylim, xv: float):
    if ylim is None:
        return None
    if isinstance(ylim, dict):
        return ylim.get(xv)
    return ylim


def _draw_overlay(ax, exp_dirs: dict[str, Path], scheme: str, xv: float, metric: str) -> None:
    for label in LABELS:
        result = aggregate_across_reps(exp_dirs[label], scheme, xv, metric)
        if result is None:
            continue
        gens, mean_of_mean, std_of_mean, mean_of_max, std_of_max = result
        c = COLORS[label]
        ax.fill_between(
            gens, mean_of_mean - std_of_mean, mean_of_mean + std_of_mean,
            alpha=0.18, color=c,
        )
        ax.plot(gens, mean_of_mean, lw=1.8, color=c, label=f"{label} mean")
        ax.fill_between(
            gens, mean_of_max - std_of_max, mean_of_max + std_of_max,
            alpha=0.10, color=c,
        )
        ax.plot(gens, mean_of_max, lw=1.1, ls="--", color=c, label=f"{label} best")


def plot_scheme(
    scheme: str,
    comma_dir: Path,
    plus_dir: Path,
    out_dir: Path,
    ylims: dict[str, object],
) -> None:
    exp_dirs = {"(mu,lambda)": comma_dir, "(mu+lambda)": plus_dir}
    present_x = [
        xv for xv in X_VALUES
        if any(
            aggregate_across_reps(exp_dirs[label], scheme, xv, metric) is not None
            for metric, _ in METRICS for label in LABELS
        )
    ]

    if not present_x:
        print(f"  [skip] no data for scheme={scheme}")
        return

    fig, axes = plt.subplots(
        len(present_x), len(METRICS),
        figsize=(5.5 * len(METRICS), 3.2 * len(present_x)),
        squeeze=False, sharex=True,
    )
    fig.suptitle(f"{scheme} — (mu,lambda) vs (mu+lambda)", fontsize=14)

    for xi, xv in enumerate(present_x):
        for mi, (metric, metric_label) in enumerate(METRICS):
            ax = axes[xi][mi]
            _draw_overlay(ax, exp_dirs, scheme, xv, metric)
            if xi == 0:
                ax.set_title(metric_label, fontsize=10)
            if mi == 0:
                ax.set_ylabel(f"x={xv}", fontsize=9)
            if xi == len(present_x) - 1:
                ax.set_xlabel("generation", fontsize=8)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            row_ylim = _ylim_for_row(ylims.get(metric), xv)
            if row_ylim is not None:
                ax.set_ylim(row_ylim)

    axes[0][0].legend(fontsize=7, loc="best")
    plt.tight_layout()
    out_path = out_dir / f"{scheme}_compare.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined_grid(
    metric: str,
    ylabel: str,
    comma_dir: Path,
    plus_dir: Path,
    out_dir: Path,
    ylim,
) -> None:
    exp_dirs = {"(mu,lambda)": comma_dir, "(mu+lambda)": plus_dir}
    present_schemes = [
        s for s in SCHEMES
        if any(
            aggregate_across_reps(exp_dirs[label], s, xv, metric) is not None
            for xv in X_VALUES for label in LABELS
        )
    ]
    present_x = [
        xv for xv in X_VALUES
        if any(
            aggregate_across_reps(exp_dirs[label], s, xv, metric) is not None
            for s in SCHEMES for label in LABELS
        )
    ]

    if not present_schemes or not present_x:
        print(f"  [skip] no data for combined grid metric={metric}")
        return

    fig, axes = plt.subplots(
        len(present_x), len(present_schemes),
        figsize=(4.2 * len(present_schemes), 3.0 * len(present_x)),
        squeeze=False, sharex=True,
    )
    fig.suptitle(
        f"(mu,lambda) vs (mu+lambda) — {ylabel} per generation (aggregated across reps)",
        fontsize=14,
    )

    for xi, xv in enumerate(present_x):
        row_ylim = _ylim_for_row(ylim, xv)
        for si, scheme in enumerate(present_schemes):
            ax = axes[xi][si]
            _draw_overlay(ax, exp_dirs, scheme, xv, metric)
            if xi == 0:
                ax.set_title(scheme, fontsize=9)
            if si == 0:
                ax.set_ylabel(f"x={xv}\n{ylabel}", fontsize=8)
            if xi == len(present_x) - 1:
                ax.set_xlabel("generation", fontsize=8)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if row_ylim is not None:
                ax.set_ylim(row_ylim)

    axes[0][0].legend(fontsize=7, loc="best")
    plt.tight_layout()
    out_path = out_dir / f"{metric}_curve_scheme_comparison.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="__data__/social")
    parser.add_argument("--comma-dir", default="ariel", help="Subdir name for (mu,lambda) runs")
    parser.add_argument("--plus-dir", default="ariel_muplus", help="Subdir name for (mu+lambda) runs")
    parser.add_argument("--out-dir", default="__data__/social/plots/compare")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comma_dir = data_dir / args.comma_dir
    plus_dir = data_dir / args.plus_dir
    exp_dirs = [comma_dir, plus_dir]

    # Fitness: shared scale per x-row only. Diversity: shared across the
    # whole grid (crowding affects fitness range far more than novelty range).
    ylims = {
        "fitness": compute_row_ylims(exp_dirs, "fitness"),
        "novelty": compute_global_ylim(exp_dirs, "novelty"),
    }

    for scheme in SCHEMES:
        plot_scheme(scheme, comma_dir, plus_dir, out_dir, ylims)

    for metric, metric_label in METRICS:
        plot_combined_grid(metric, metric_label, comma_dir, plus_dir, out_dir, ylims[metric])


if __name__ == "__main__":
    main()
