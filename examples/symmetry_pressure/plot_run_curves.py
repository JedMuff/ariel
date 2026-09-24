"""
Per-run fitness and diversity curves for the symmetry-pressure sweep.

For every run directory found under __data__/sympress_*, plots land in a
single collated tree at <sweep_root>/analysis/<task>/<genome>/, where
<sweep_root> is the run directory's parent:
  .../analysis/<task>/<genome>/<run_name>_fitness.png   : per-gen mean±std/best fitness
  .../analysis/<task>/<genome>/<run_name>_diversity.png : mean±std pairwise tree-edit-distance
and one combined small-multiples overview across all runs, saved to
<sweep_root>/analysis/overview.png.

Handles both on-disk schemas transparently: the flat gecko_food_skills
records and the nested gecko_skill_tasks/<task> records both carry a
top-level "fitness" field, so fitness plotting needs no branching; the
tree-edit-distance diversity metric only reads best_genome.json/meta.json,
which both schemas' checkpoints also share.

Usage:
    python plot_run_curves.py
    python plot_run_curves.py --run-dirs ../../__data__/sympress_food_tree_rep0_37568_3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diversity_common import load_diversity_by_gen
from sweep_common import (
    DATA_ROOT,
    analysis_dir,
    analysis_overview_path,
    discover_run_dirs,
    load_fitness_by_gen,
    parse_run_tag,
    run_config,
)

parser = argparse.ArgumentParser()
parser.add_argument("--run-dirs", nargs="*", type=Path, default=None)
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()


def plot_fitness(fitness_by_gen: dict[int, list[float]], out_path: Path, title: str, dpi: int) -> None:
    gens = sorted(fitness_by_gen)
    mins = [min(fitness_by_gen[g]) for g in gens]
    means = np.array([np.mean(fitness_by_gen[g]) for g in gens])
    stds = np.array([np.std(fitness_by_gen[g]) for g in gens])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(gens, means - stds, means + stds, color="#4C72B0", alpha=0.2, linewidth=0, label="mean ± std")
    ax.plot(gens, means, color="#4C72B0", linewidth=2, label="mean fitness")
    ax.plot(gens, mins, color="#DD8452", linewidth=2, label="best fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (lower is better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_diversity_single(div_by_gen: dict[int, tuple[float, float]], out_path: Path, title: str, dpi: int) -> None:
    gens = sorted(div_by_gen)
    means = np.array([div_by_gen[g][0] for g in gens])
    stds = np.array([div_by_gen[g][1] for g in gens])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(
        gens, np.maximum(means - stds, 0), means + stds,
        color="#55A868", alpha=0.2, linewidth=0, label="mean ± std",
    )
    ax.plot(gens, means, color="#55A868", linewidth=2, label="mean pairwise TED")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean pairwise tree edit distance")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(rows: list[tuple], out_path: Path, dpi: int) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n), squeeze=False)
    for i, (name, fitness_by_gen, diversity_by_gen) in enumerate(rows):
        gens = sorted(fitness_by_gen)
        mins = [min(fitness_by_gen[g]) for g in gens]
        means = np.array([np.mean(fitness_by_gen[g]) for g in gens])
        stds = np.array([np.std(fitness_by_gen[g]) for g in gens])

        ax_f = axes[i][0]
        ax_f.fill_between(gens, means - stds, means + stds, color="#4C72B0", alpha=0.2, linewidth=0)
        ax_f.plot(gens, means, color="#4C72B0", linewidth=1.5, label="mean")
        ax_f.plot(gens, mins, color="#DD8452", linewidth=1.5, label="best")
        ax_f.set_title(f"{name}\nfitness", fontsize=9)
        ax_f.legend(fontsize=7)
        ax_f.grid(alpha=0.3)

        ax_d = axes[i][1]
        if diversity_by_gen:
            dg = sorted(diversity_by_gen)
            d_means = np.array([diversity_by_gen[g][0] for g in dg])
            d_stds = np.array([diversity_by_gen[g][1] for g in dg])
            ax_d.fill_between(dg, np.maximum(d_means - d_stds, 0), d_means + d_stds,
                               color="#55A868", alpha=0.2, linewidth=0)
            ax_d.plot(dg, d_means, color="#55A868", linewidth=1.5)
        ax_d.set_title(f"{name}\ndiversity (TED)", fontsize=9)
        ax_d.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nOverview saved -> {out_path}")


def main() -> None:
    run_dirs = args.run_dirs if args.run_dirs else discover_run_dirs()
    if not run_dirs:
        print(f"No run dirs found under {DATA_ROOT}")
        return

    overview_rows = []
    for d in run_dirs:
        info = parse_run_tag(d.name)
        if info is None:
            print(f"  Skipping {d.name} (unrecognized run-dir naming)")
            continue
        task, genome = info["task"], info["genome"]
        print(f"[{d.name}] task={task} genome={genome}")

        fitness_by_gen = load_fitness_by_gen(d, task)
        if not fitness_by_gen:
            print("  no run_data.jsonl records found, skipping")
            continue
        max_modules = run_config(d, task).get("max_modules", 25)
        diversity_by_gen = load_diversity_by_gen(d, task, genome, max_modules)

        out_dir = analysis_dir(d, task, genome)
        out_dir.mkdir(parents=True, exist_ok=True)
        title = f"{d.name}\n(task={task}, genome={genome})"

        fitness_path = out_dir / f"{d.name}_fitness.png"
        plot_fitness(fitness_by_gen, fitness_path, title, args.dpi)
        print(f"  Saved -> {fitness_path}")

        if diversity_by_gen:
            diversity_path = out_dir / f"{d.name}_diversity.png"
            plot_diversity_single(diversity_by_gen, diversity_path, title, args.dpi)
            print(f"  Saved -> {diversity_path}")
        else:
            print("  no checkpoints found, skipping diversity plot")

        overview_rows.append((d.name, fitness_by_gen, diversity_by_gen))

    if overview_rows:
        overview_path = analysis_overview_path(run_dirs[0])
        overview_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_overview(overview_rows, overview_path, args.dpi)


if __name__ == "__main__":
    main()
