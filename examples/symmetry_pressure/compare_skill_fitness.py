"""
Compare per-skill fitness (forward loco, turn-left, turn-right) between
symmetry-enforced and no-symmetry food_skills runs.

Each skill's fitness is the best (min) value of its CMA-ES learning curve,
logged as {loco,left,right}_learning_curve in run_data.jsonl.

Outputs (under --out-dir):
  skill_fitness_comparison.png   3-row × 2-col grid:
    row 0: forward-loco  |  row 1: turn-left  |  row 2: turn-right
    col 0: running best (elite)  |  col 1: per-generation offspring mean

Usage:
    python compare_skill_fitness.py \\
        --sym-dirs   ../../__data__/food_skills/food_skills_..._35117 ... \\
        --nosym-dirs ../../__data__/food_skills/food_skills_..._35301 ... \\
        --out-dir    ../../__data__/food_skills_comparison
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--sym-dirs",   nargs="+", type=Path, required=True)
parser.add_argument("--nosym-dirs", nargs="+", type=Path, required=True)
parser.add_argument("--out-dir",    type=Path, default=Path("__data__/food_skills_comparison"))
parser.add_argument("--dpi",        type=int, default=150)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

SYM_COLOR   = "#4C72B0"
NOSYM_COLOR = "#DD8452"
CI_ALPHA    = 0.20
GLITCH_FIT  = 1.0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> dict[str, Any]:
    data_dir = run_dir / "__data__" / "gecko_food_skills"
    config = json.loads((data_dir / "run_config.json").read_text())
    records: list[dict] = []
    with (data_dir / "run_data.jsonl").open() as f:
        for line in f:
            records.append(json.loads(line))
    return {"name": run_dir.name, "config": config, "records": records}


def per_gen_best_of_curve(records: list[dict], curve_key: str) -> dict[int, list[float]]:
    """Per-generation list of each individual's best CMA-ES value."""
    by_gen: dict[int, list[float]] = {}
    for r in records:
        curve = r.get(curve_key)
        if not curve:
            continue
        best = min(curve)
        if not np.isfinite(best):
            continue
        by_gen.setdefault(r["gen"], []).append(best)
    return by_gen


def running_best(by_gen: dict[int, list[float]]) -> tuple[list[int], list[float]]:
    gens = sorted(by_gen)
    rb = list(np.minimum.accumulate([min(by_gen[g]) for g in gens]))
    return gens, rb


# ── Statistics ────────────────────────────────────────────────────────────────

def group_running_best_stats(
    runs: list[dict], curve_key: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    all_rb: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        by_gen = per_gen_best_of_curve(run["records"], curve_key)
        gens, rb = running_best(by_gen)
        for g, v in zip(gens, rb):
            all_rb[g].append(v)
    gens = sorted(all_rb)
    means, lo, hi = [], [], []
    for g in gens:
        vs = np.array(all_rb[g])
        m = float(np.mean(vs))
        ci = (stats.sem(vs) * stats.t.ppf(0.975, len(vs) - 1)
              if len(vs) > 1 else 0.0)
        means.append(m)
        lo.append(m - ci)
        hi.append(m + ci)
    return gens, means, lo, hi


def group_per_gen_mean_stats(
    runs: list[dict], curve_key: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    all_means: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        by_gen = per_gen_best_of_curve(run["records"], curve_key)
        for g, vals in by_gen.items():
            all_means[g].append(float(np.mean(vals)))
    gens = sorted(all_means)
    means, lo, hi = [], [], []
    for g in gens:
        vs = np.array(all_means[g])
        m = float(np.mean(vs))
        ci = (stats.sem(vs) * stats.t.ppf(0.975, len(vs) - 1)
              if len(vs) > 1 else 0.0)
        means.append(m)
        lo.append(m - ci)
        hi.append(m + ci)
    return gens, means, lo, hi


# ── Plotting ──────────────────────────────────────────────────────────────────

SKILLS = [
    ("loco_learning_curve",  "Forward-locomotion skill fitness",  "Forward displacement (higher displacement → more negative)"),
    ("left_learning_curve",  "Turn-left skill fitness",           "Accumulated yaw (higher yaw → more negative)"),
    ("right_learning_curve", "Turn-right skill fitness",          "Accumulated yaw (higher yaw → more negative)"),
]


def plot_skill_fitness(
    sym_runs: list[dict], nosym_runs: list[dict], out_path: Path, dpi: int,
) -> None:
    n_skills = len(SKILLS)
    fig, axes = plt.subplots(n_skills, 2, figsize=(14, 5.5 * n_skills))

    for row, (curve_key, skill_title, ylabel) in enumerate(SKILLS):
        ax_elite = axes[row, 0]
        ax_mean  = axes[row, 1]

        # ── Left col: running-best elite progress ─────────────────────────────
        for runs, color, label in [
            (sym_runs,   SYM_COLOR,   "Symmetry enforced (y_zero)"),
            (nosym_runs, NOSYM_COLOR, "No symmetry (none)"),
        ]:
            gens, means, lo, hi = group_running_best_stats(runs, curve_key)
            ax_elite.plot(gens, means, color=color, linewidth=2.5, label=label)
            ax_elite.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA)
            for run in runs:
                by_gen = per_gen_best_of_curve(run["records"], curve_key)
                if not by_gen:
                    continue
                g_list, rb = running_best(by_gen)
                ax_elite.plot(g_list, rb, color=color, linewidth=0.6, alpha=0.25)

        ax_elite.set_xlabel("Generation")
        ax_elite.set_ylabel(f"{ylabel}\n(lower = better)")
        ax_elite.set_title(f"{skill_title}\nElite progress — running best\n"
                           "(thick = group mean ± 95% CI, thin = individual runs)")
        ax_elite.legend(fontsize=9)
        ax_elite.grid(alpha=0.3)

        # ── Right col: per-generation offspring mean ───────────────────────────
        for runs, color, label in [
            (sym_runs,   SYM_COLOR,   "Symmetry enforced (y_zero)"),
            (nosym_runs, NOSYM_COLOR, "No symmetry (none)"),
        ]:
            gens, means, lo, hi = group_per_gen_mean_stats(runs, curve_key)
            ax_mean.plot(gens, means, color=color, linewidth=2.5, label=label)
            ax_mean.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA)

        ax_mean.set_xlabel("Generation")
        ax_mean.set_ylabel(f"{ylabel}\n(lower = better)")
        ax_mean.set_title(f"{skill_title}\nPer-generation offspring mean\n"
                          "(thick = group mean ± 95% CI)")
        ax_mean.legend(fontsize=9)
        ax_mean.grid(alpha=0.3)

    fig.suptitle(
        "Skill fitness breakdown: symmetry enforcement vs no symmetry\n"
        "(best of CMA-ES learning curve per individual)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading runs...")
    sym_runs   = [load_run(d) for d in args.sym_dirs]
    nosym_runs = [load_run(d) for d in args.nosym_dirs]
    print(f"  {len(sym_runs)} symmetric  |  {len(nosym_runs)} no-symmetry")

    print("Plotting skill fitness comparison...")
    plot_skill_fitness(
        sym_runs, nosym_runs,
        args.out_dir / "skill_fitness_comparison.png",
        args.dpi,
    )
    print(f"\nDone. Plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
