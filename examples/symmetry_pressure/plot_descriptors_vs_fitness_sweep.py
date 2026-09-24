"""
Morphological descriptors vs. fitness for the symmetry-pressure sweep.

For each task, produces one 2x4 grid (analysis/<task>/descriptors_vs_fitness.png)
with all 8 MorphologicalMeasures descriptors, each subplot overlaying the 3
genome types (tree, tree_symmetric, cppn) as separate colored series:
scatter of every evaluated individual (pooled across all reps of that
genome type) plus a per-genome OLS regression line ± 1 SE band and
Pearson r/p annotation — directly generalizes the older
plot_descriptors_vs_fitness.py (2-group sym/nosym) to 3 genome-type groups.

Usage:
    python plot_descriptors_vs_fitness_sweep.py
    python plot_descriptors_vs_fitness_sweep.py --sweep-root ../../__data__/ariel_symmetry_pressure_sweep
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
ALPHA_PT = 0.3


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


def pooled_records(run_dirs: list[Path], task: str, genome: str) -> list[dict]:
    records = []
    for run_dir in run_dirs:
        max_modules = run_config(run_dir, task).get("max_modules", 25)
        records.extend(load_checkpoint_records(run_dir, task, genome, max_modules))
    return records


def ols_line_and_band(
    x: np.ndarray, y: np.ndarray, x_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fitted line ± 1 standard error of the fit (not a t-scaled 95% CI —
    with the hundreds of pooled individuals here the t-critical multiplier
    is close to 2, so a plain SE band is about half as wide and leaves the
    trend line more visible; see plot_task_genome_summary.group_stats for
    the same reasoning applied to the across-rep bands elsewhere)."""
    n = len(x)
    if n < 3:
        nan = np.full_like(x_grid, np.nan, dtype=float)
        return nan, nan, nan
    slope, intercept, *_ = stats.linregress(x, y)
    y_hat = intercept + slope * x_grid

    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean) ** 2)
    if ss_x == 0:
        nan = np.full_like(x_grid, np.nan, dtype=float)
        return y_hat, nan, nan
    residuals = y - (intercept + slope * x)
    s2 = np.sum(residuals ** 2) / max(n - 2, 1)

    se_hat = np.sqrt(s2 * (1 / n + (x_grid - x_mean) ** 2 / ss_x))
    return y_hat, y_hat - se_hat, y_hat + se_hat


def plot_task(task: str, by_genome: dict[str, list[dict]], out_path: Path, dpi: int) -> None:
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))

    for idx, (key, label) in enumerate(DESCRIPTORS):
        ax = axes[idx // n_cols, idx % n_cols]

        for i, genome in enumerate(GENOME_TYPES):
            records = by_genome.get(genome)
            if not records:
                continue
            color = GENOME_COLOR[genome]
            fit_arr = np.array([r["fitness"] for r in records])
            vals = np.array([r[key] for r in records])

            ax.scatter(fit_arr, vals, color=color, alpha=ALPHA_PT, s=12, linewidths=0, label=GENOME_LABEL[genome])

            x_grid = np.linspace(fit_arr.min(), fit_arr.max(), 200)
            y_hat, lo, hi = ols_line_and_band(fit_arr, vals, x_grid)
            ax.plot(x_grid, y_hat, color=color, linewidth=1.8)
            ax.fill_between(x_grid, lo, hi, color=color, alpha=0.15)

            if len(fit_arr) >= 3 and np.std(fit_arr) > 0 and np.std(vals) > 0:
                r, p = stats.pearsonr(fit_arr, vals)
                p_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
                ax.annotate(
                    f"r={r:+.2f}  {p_str}",
                    xy=(0.03, 0.94 - 0.08 * i),
                    xycoords="axes fraction",
                    fontsize=7, color=color,
                )

        ax.set_xlabel("Fitness (lower is better)", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right", markerscale=1.5)

    fig.suptitle(
        f"Morphological descriptors vs. fitness — task={task}\n"
        "(scatter = all evaluated individuals across reps; line = OLS fit ± 1 SE; r = Pearson)",
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
        by_genome: dict[str, list[dict]] = {}
        for genome in GENOME_TYPES:
            run_dirs = by_task_genome.get((task, genome), [])
            if not run_dirs:
                continue
            records = pooled_records(run_dirs, task, genome)
            if records:
                by_genome[genome] = records
                print(f"[{task}/{genome}] {len(records)} individuals")

        if not by_genome:
            print(f"[{task}] no data found, skipping")
            continue

        out_dir = args.sweep_root / "analysis" / task
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "descriptors_vs_fitness.png"
        plot_task(task, by_genome, out_path, args.dpi)
        print(f"[{task}] Saved -> {out_path}")


if __name__ == "__main__":
    main()
