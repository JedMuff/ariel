"""
Per-task fitness and diversity summaries for the symmetry-pressure sweep,
amalgamating every repeat run of each genome type onto one axis.

For each task (forward, multidirection, turn_avg, food), produces three
figures, each showing all 3 genome types (tree, tree_symmetric, cppn)
overlaid on the same axis:
  analysis/<task>/combined_fitness_mean.png
    line + shaded band = group mean of each rep's per-generation *mean*
    fitness, averaged across reps
  analysis/<task>/combined_fitness_best.png
    line + shaded band = group mean of each rep's per-generation *best*
    (min) fitness, averaged across reps
  analysis/<task>/combined_diversity.png
    line + shaded band = group mean pairwise tree-edit-distance across reps

"Group" here means: for each rep, first reduce to one per-generation value
(mean fitness, min fitness, or mean pairwise TED), then average those
per-rep curves across reps with a ±1 SEM band — the same aggregation
convention already used by plot_diversity.py / plot_morphological_descriptors.py
for their sym-vs-nosym comparisons, generalized from 2 groups to 3 (one per
genome type). The shaded band on every line (mean and best alike) is this
same across-rep SEM, not a within-generation std — see group_stats().

Usage:
    python plot_task_genome_summary.py
    python plot_task_genome_summary.py --sweep-root ../../__data__/ariel_symmetry_pressure_sweep
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from diversity_common import load_diversity_by_gen
from sweep_common import (
    GENOME_TYPES,
    TASKS,
    load_fitness_by_gen,
    parse_run_tag,
    run_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--sweep-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_symmetry_pressure_sweep")
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()

GENOME_COLOR = {
    "tree": "#2a78d6",
    "tree_symmetric": "#eb6834",
    "cppn": "#1baf7a",
}
GENOME_LABEL = {
    "tree": "tree",
    "tree_symmetric": "tree (symmetric)",
    "cppn": "cppn",
}
CI_ALPHA = 0.15


# ── Discovery ─────────────────────────────────────────────────────────────

def discover_runs_by_task_genome(sweep_root: Path) -> dict[tuple[str, str], list[Path]]:
    by_task_genome: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for d in sorted(sweep_root.glob("sympress_*")):
        if not d.is_dir():
            continue
        info = parse_run_tag(d.name)
        if info is None:
            continue
        by_task_genome[(info["task"], info["genome"])].append(d)
    return by_task_genome


# ── Aggregation across reps ──────────────────────────────────────────────

def group_stats(per_rep_series: list[dict[int, float]]) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± 1 SEM across reps, generation-aligned. Gens present in only
    some reps are still included (aggregated over however many reps have
    that gen), so a rep that's still mid-run doesn't truncate the others.

    Uses ±1 standard error rather than a t-scaled 95% CI: with only
    N_REPS=5 reps, the 95% CI's t-critical multiplier (~2.78) makes the
    band wide enough to swamp the trend line; SEM is ~2.78x narrower.
    """
    by_gen: dict[int, list[float]] = defaultdict(list)
    for series in per_rep_series:
        for g, v in series.items():
            by_gen[g].append(v)
    gens = sorted(by_gen)
    means, lo, hi = [], [], []
    for g in gens:
        vs = np.array(by_gen[g])
        m = float(np.mean(vs))
        sem = float(stats.sem(vs)) if len(vs) > 1 else 0.0
        means.append(m)
        lo.append(m - sem)
        hi.append(m + sem)
    return gens, np.array(means), np.array(lo), np.array(hi)


def per_rep_mean_and_best(run_dirs: list[Path], task: str) -> tuple[list[dict[int, float]], list[dict[int, float]]]:
    mean_series, best_series = [], []
    for run_dir in run_dirs:
        fitness_by_gen = load_fitness_by_gen(run_dir, task)
        if not fitness_by_gen:
            continue
        mean_series.append({g: float(np.mean(vs)) for g, vs in fitness_by_gen.items()})
        best_series.append({g: float(np.min(vs)) for g, vs in fitness_by_gen.items()})
    return mean_series, best_series


def per_rep_diversity(run_dirs: list[Path], task: str, genome: str) -> list[dict[int, float]]:
    series = []
    for run_dir in run_dirs:
        max_modules = run_config(run_dir, task).get("max_modules", 25)
        div_by_gen = load_diversity_by_gen(run_dir, task, genome, max_modules)
        if div_by_gen:
            series.append({g: mean for g, (mean, _std) in div_by_gen.items()})
    return series


# ── Plotting ──────────────────────────────────────────────────────────────

def fitness_ylim(
    data: dict[str, tuple[list, np.ndarray, np.ndarray, np.ndarray, list]],
    pad_frac: float = 0.05,
) -> Optional[tuple[float, float]]:
    """Shared y-axis range for a task's mean and best plots (computed from
    both lines' CI bands), so the two are visually comparable rather than
    each auto-scaling to its own data."""
    los, his = [], []
    for genome in GENOME_TYPES:
        if genome not in data:
            continue
        _gens, _mean_m, mean_lo, mean_hi, best_series = data[genome]
        los.append(np.min(mean_lo))
        his.append(np.max(mean_hi))
        _best_gens, _best_m, best_lo, best_hi = group_stats(best_series)
        if len(best_lo):
            los.append(np.min(best_lo))
            his.append(np.max(best_hi))
    if not los:
        return None
    lo, hi = min(los), max(his)
    pad = (hi - lo) * pad_frac or 1.0
    return lo - pad, hi + pad


def plot_fitness_mean(
    task: str, data: dict[str, tuple[list, np.ndarray, np.ndarray, np.ndarray, list]],
    out_path: Path, dpi: int, ylim: Optional[tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for genome in GENOME_TYPES:
        if genome not in data:
            continue
        gens, mean_m, mean_lo, mean_hi, _best_series = data[genome]
        color = GENOME_COLOR[genome]
        label = GENOME_LABEL[genome]
        ax.fill_between(gens, mean_lo, mean_hi, color=color, alpha=CI_ALPHA, linewidth=0)
        ax.plot(gens, mean_m, color=color, linewidth=2.2, label=label)

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (lower is better)")
    ax.set_title(f"Mean fitness across genome types — task={task}\n"
                 "(mean of each rep's per-generation mean fitness, ± 1 SEM across reps)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_fitness_best(
    task: str, data: dict[str, tuple[list, np.ndarray, np.ndarray, np.ndarray, list]],
    out_path: Path, dpi: int, ylim: Optional[tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for genome in GENOME_TYPES:
        if genome not in data:
            continue
        _gens, _mean_m, _mean_lo, _mean_hi, best_series = data[genome]
        color = GENOME_COLOR[genome]
        label = GENOME_LABEL[genome]

        best_gens, best_m, best_lo, best_hi = group_stats(best_series)
        ax.fill_between(best_gens, best_lo, best_hi, color=color, alpha=CI_ALPHA, linewidth=0)
        ax.plot(best_gens, best_m, color=color, linewidth=2.2, label=label)

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (lower is better)")
    ax.set_title(f"Best fitness across genome types — task={task}\n"
                 "(mean of each rep's per-generation best fitness, ± 1 SEM across reps)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_diversity(task: str, data: dict[str, tuple[list, np.ndarray, np.ndarray, np.ndarray]], out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for genome in GENOME_TYPES:
        if genome not in data:
            continue
        gens, means, lo, hi = data[genome]
        color = GENOME_COLOR[genome]
        ax.fill_between(gens, np.maximum(lo, 0), hi, color=color, alpha=CI_ALPHA, linewidth=0)
        ax.plot(gens, means, color=color, linewidth=2.2, label=GENOME_LABEL[genome])

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean pairwise tree edit distance")
    ax.set_title(f"Diversity across genome types — task={task}\n(mean ± 1 SEM across reps)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    by_task_genome = discover_runs_by_task_genome(args.sweep_root)
    if not by_task_genome:
        print(f"No run dirs found under {args.sweep_root}")
        return

    for task in TASKS:
        fitness_data = {}
        diversity_data = {}
        for genome in GENOME_TYPES:
            run_dirs = by_task_genome.get((task, genome), [])
            if not run_dirs:
                continue

            mean_series, best_series = per_rep_mean_and_best(run_dirs, task)
            if mean_series:
                gens, m, lo, hi = group_stats(mean_series)
                fitness_data[genome] = (gens, m, lo, hi, best_series)

            div_series = per_rep_diversity(run_dirs, task, genome)
            if div_series:
                diversity_data[genome] = group_stats(div_series)

        out_dir = args.sweep_root / "analysis" / task
        out_dir.mkdir(parents=True, exist_ok=True)

        if fitness_data:
            ylim = fitness_ylim(fitness_data)

            mean_path = out_dir / "combined_fitness_mean.png"
            plot_fitness_mean(task, fitness_data, mean_path, args.dpi, ylim=ylim)
            print(f"[{task}] Saved -> {mean_path}")

            best_path = out_dir / "combined_fitness_best.png"
            plot_fitness_best(task, fitness_data, best_path, args.dpi, ylim=ylim)
            print(f"[{task}] Saved -> {best_path}")
        else:
            print(f"[{task}] no fitness data found, skipping")

        if diversity_data:
            diversity_path = out_dir / "combined_diversity.png"
            plot_diversity(task, diversity_data, diversity_path, args.dpi)
            print(f"[{task}] Saved -> {diversity_path}")
        else:
            print(f"[{task}] no diversity data found, skipping")


if __name__ == "__main__":
    main()
