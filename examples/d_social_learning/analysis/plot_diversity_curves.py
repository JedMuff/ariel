"""Diversity (novelty) curves aggregated across reps: mean±std (shaded) +
best±std (dashed, shaded) per generation.

Reuses plot_run_curves.plot_experiment against tags_['novelty'] instead of
fitness_. Produces one grid PNG per experiment directory (rows=x, cols=scheme).
Unlike the fitness curves, the y-scale is shared across the whole grid (all x
rows), not per row.

Usage:
    uv run examples/d_social_learning/analysis/plot_diversity_curves.py \
        --data-dir __data__/snellius_test_data/social \
        --domains ariel,ariel_muplus \
        --out-dir __data__/snellius_test_data/social/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

from curve_utils import compute_global_ylim
from plot_run_curves import plot_experiment


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

    ylim = compute_global_ylim(exp_dirs, metric="novelty")
    for label, exp_dir in zip(labels, exp_dirs):
        plot_experiment(exp_dir, label, out_dir, metric="novelty", ylabel="novelty", ylim=ylim)


if __name__ == "__main__":
    main()
