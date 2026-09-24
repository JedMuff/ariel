"""
Morphological descriptors vs. generation for the symmetry-pressure sweep.

For each task, produces one 2x4 grid (analysis/<task>/descriptors_over_generations.png)
with all 8 MorphologicalMeasures descriptors, each subplot overlaying the 3
genome types (tree, tree_symmetric, cppn) as separate colored series: group
mean ± 1 SEM across reps (faint individual-rep traces behind), following
the same aggregation convention as plot_task_genome_summary.py and the
older plot_morphological_descriptors.py (generalized from 2 groups to 3).

Usage:
    python plot_descriptors_over_generations.py
    python plot_descriptors_over_generations.py --sweep-root ../../__data__/ariel_symmetry_pressure_sweep
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from descriptor_common import DESCRIPTORS, load_checkpoint_records
from sweep_common import GENOME_TYPES, TASKS, parse_run_tag, run_config

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
CI_ALPHA = 0.18


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


def per_rep_gen_means_all_keys(run_dirs: list[Path], task: str, genome: str) -> list[dict[str, dict[int, float]]]:
    """One {descriptor_key: {gen: mean}} dict per rep. Loads each rep's
    checkpoints (and decodes each genome) exactly once, rather than once per
    descriptor key — cppn decoding is the expensive step here."""
    per_rep = []
    for run_dir in run_dirs:
        max_modules = run_config(run_dir, task).get("max_modules", 25)
        records = load_checkpoint_records(run_dir, task, genome, max_modules)
        if not records:
            continue
        by_gen: dict[int, list[dict]] = defaultdict(list)
        for r in records:
            by_gen[r["gen"]].append(r)
        rep_data = {
            key: {g: float(np.mean([r[key] for r in rs])) for g, rs in by_gen.items()}
            for key, _label in DESCRIPTORS
        }
        per_rep.append(rep_data)
    return per_rep


def group_stats(per_rep_series: list[dict[int, float]]) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± 1 SEM across reps (see plot_task_genome_summary.group_stats
    for why SEM rather than a t-scaled 95% CI)."""
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


def plot_task(task: str, by_genome_records: dict[str, list[dict[int, float]]], out_path: Path, dpi: int) -> None:
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows))

    for idx, (key, label) in enumerate(DESCRIPTORS):
        ax = axes[idx // n_cols, idx % n_cols]
        for genome in GENOME_TYPES:
            per_rep = by_genome_records.get(genome)
            if not per_rep:
                continue
            series = [rep[key] for rep in per_rep]
            gens, means, lo, hi = group_stats(series)
            color = GENOME_COLOR[genome]
            ax.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA, linewidth=0)
            ax.plot(gens, means, color=color, linewidth=2.0, label=GENOME_LABEL[genome])
            for s in series:
                g_list = sorted(s)
                ax.plot(g_list, [s[g] for g in g_list], color=color, linewidth=0.5, alpha=0.2)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Generation", fontsize=8)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Morphological descriptors vs. generation — task={task}\n"
        "(thick = group mean ± 1 SEM across reps; faint = individual reps)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    by_task_genome = discover_runs_by_task_genome(args.sweep_root)
    if not by_task_genome:
        print(f"No run dirs found under {args.sweep_root}")
        return

    for task in TASKS:
        by_genome_records: dict[str, list[dict[str, dict[int, float]]]] = {}
        for genome in GENOME_TYPES:
            run_dirs = by_task_genome.get((task, genome), [])
            if not run_dirs:
                continue
            per_rep = per_rep_gen_means_all_keys(run_dirs, task, genome)
            if per_rep:
                by_genome_records[genome] = per_rep

        if not by_genome_records:
            print(f"[{task}] no data found, skipping")
            continue

        out_dir = args.sweep_root / "analysis" / task
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "descriptors_over_generations.png"
        plot_task(task, by_genome_records, out_path, args.dpi)
        print(f"[{task}] Saved -> {out_path}")


if __name__ == "__main__":
    main()
