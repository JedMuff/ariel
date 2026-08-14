"""Fitness curves aggregated across reps: mean±std (shaded) + best±std
(dashed, shaded) per generation.

Produces one grid PNG per experiment directory (rows=x, cols=scheme), where
each cell aggregates all replicate runs found for that (scheme, x) pair.
Non-finite fitness values (inf/nan, e.g. from simulator failures) are ignored.

Usage:
    uv run examples/d_social_learning/analysis/plot_run_curves.py \
        --data-dir __data__/snellius_test_data/social \
        --domains ariel,ariel_muplus \
        --out-dir __data__/snellius_test_data/social/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from curve_utils import SCHEMES, X_VALUES, aggregate_across_reps, compute_row_ylims

C_MEAN = "tab:blue"
C_BEST = "tab:orange"


def _ylim_for_row(ylim, xv: float):
    if ylim is None:
        return None
    if isinstance(ylim, dict):
        return ylim.get(xv)
    return ylim


def plot_experiment(
    exp_dir: Path,
    label: str,
    out_dir: Path,
    metric: str = "fitness",
    ylabel: str = "fitness",
    ylim=None,
) -> None:
    """ylim may be a single (lo, hi) tuple applied to every subplot, a
    {x_value: (lo, hi)} dict applied per row, or None for autoscaling.
    """
    data = {}
    for scheme in SCHEMES:
        for xv in X_VALUES:
            result = aggregate_across_reps(exp_dir, scheme, xv, metric)
            if result is not None:
                data[(scheme, xv)] = result

    if not data:
        print(f"  [skip] no data for {label} ({exp_dir})")
        return

    present_schemes = [s for s in SCHEMES if any((s, x) in data for x in X_VALUES)]
    present_x = [x for x in X_VALUES if any((s, x) in data for s in SCHEMES)]

    fig, axes = plt.subplots(
        len(present_x), len(present_schemes),
        figsize=(4.2 * len(present_schemes), 3.0 * len(present_x)),
        squeeze=False, sharex=True,
    )
    fig.suptitle(
        f"{label} — {ylabel} per generation (aggregated across reps)",
        fontsize=14,
    )

    for xi, xv in enumerate(present_x):
        row_ylim = _ylim_for_row(ylim, xv)
        for si, scheme in enumerate(present_schemes):
            ax = axes[xi][si]
            result = data.get((scheme, xv))
            if result is not None:
                gens, mean_of_mean, std_of_mean, mean_of_max, std_of_max = result
                ax.fill_between(
                    gens, mean_of_mean - std_of_mean, mean_of_mean + std_of_mean,
                    alpha=0.25, color=C_MEAN, label="mean ±1 std",
                )
                ax.plot(gens, mean_of_mean, lw=1.8, color=C_MEAN, label="mean")
                ax.fill_between(
                    gens, mean_of_max - std_of_max, mean_of_max + std_of_max,
                    alpha=0.15, color=C_BEST, label="best ±1 std",
                )
                ax.plot(gens, mean_of_max, lw=1.3, ls="--", color=C_BEST, label="best")
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
    out_path = out_dir / f"{label}_{metric}_curves.png"
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

    # Fitness: shared y-scale per x-row (across schemes and experiments), not
    # across rows — fitness ranges vary a lot with crowding (x).
    row_ylims = compute_row_ylims(exp_dirs, metric="fitness")
    for label, exp_dir in zip(labels, exp_dirs):
        plot_experiment(exp_dir, label, out_dir, metric="fitness", ylabel="fitness", ylim=row_ylims)


if __name__ == "__main__":
    main()
