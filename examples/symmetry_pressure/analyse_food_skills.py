"""
Summarise + plot gecko_food_skills.py replicate runs (same config, different
seeds) from one or more run directories, e.g. data/food_skills4/.

Breaks out the two independently trained skill fitnesses (forward-locomotion
and turning) as their own figures, since they are separate CMA-ES objectives
(loco_learning_curve / left_learning_curve / right_learning_curve in
run_data.jsonl), not sub-terms of the food-task fitness that body-evolution
actually optimises.

Outputs (under --out-dir):
  summary.txt               text summary of the runs
  fitness_curves.png        food-task fitness (waypoints + distance), per generation
  loco_fitness_curves.png   forward-locomotion skill fitness (best of loco_learning_curve)
  turn_fitness_curves.png   turning skill fitness, left vs right (best of {left,right}_learning_curve)
  diversity.png             skill-usage mix + yz_symmetry spread per generation

Usage:
    python analyse_food_skills.py \\
        --run-dirs ../../data/food_skills4/food_skills_20260802_144224_35117 \\
                   ../../data/food_skills4/food_skills_20260802_144224_35118
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Summarise gecko_food_skills.py replicate runs")
parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, default=Path("__data__/food_skills_analysis"))
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def load_run(run_dir: Path) -> dict[str, Any]:
    data_dir = run_dir / "__data__" / "gecko_food_skills"
    config = json.loads((data_dir / "run_config.json").read_text())

    records: list[dict] = []
    with (data_dir / "run_data.jsonl").open() as f:
        for line in f:
            records.append(json.loads(line))

    timing: list[dict] = []
    timing_path = data_dir / "timing.jsonl"
    if timing_path.exists():
        with timing_path.open() as f:
            for line in f:
                timing.append(json.loads(line))

    return {"name": run_dir.name, "config": config, "records": records, "timing": timing}


def per_gen_fitness(records: list[dict]) -> dict[int, list[float]]:
    by_gen: dict[int, list[float]] = {}
    for r in records:
        fit = r.get("fitness")
        if fit is None or not np.isfinite(fit):
            continue
        by_gen.setdefault(r["gen"], []).append(fit)
    return by_gen


def per_gen_best_of_curve(records: list[dict], curve_key: str) -> dict[int, list[float]]:
    """Like per_gen_fitness, but sources each individual's value from the best
    (min) point of its own CMA-ES learning curve (e.g. loco_learning_curve)
    rather than the food-task `fitness` field. Individuals with an empty curve
    (e.g. --loco-only runs skip turn training) are skipped.
    """
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


def per_gen_symmetry(records: list[dict]) -> dict[int, list[float]]:
    by_gen: dict[int, list[float]] = {}
    for r in records:
        sym = r.get("yz_symmetry")
        if sym is None:
            continue
        by_gen.setdefault(r["gen"], []).append(sym)
    return by_gen


def per_gen_skill_mix(records: list[dict]) -> dict[int, dict[str, float]]:
    by_gen: dict[int, list[dict]] = {}
    for r in records:
        pct = r.get("skill_pct")
        if not pct:
            continue
        by_gen.setdefault(r["gen"], []).append(pct)
    out: dict[int, dict[str, float]] = {}
    for g, pcts in by_gen.items():
        out[g] = {
            k: float(np.mean([p.get(k, 0.0) for p in pcts]))
            for k in ("loco", "left", "right")
        }
    return out


def summarise(run: dict[str, Any]) -> str:
    records = run["records"]
    cfg = run["config"]
    fits_by_gen = per_gen_fitness(records)
    gens = sorted(fits_by_gen)
    all_fits = [r["fitness"] for r in records if r.get("fitness") is not None and np.isfinite(r["fitness"])]
    best_rec = min((r for r in records if r.get("fitness") is not None and np.isfinite(r["fitness"])),
                   key=lambda r: r["fitness"], default=None)

    n_glitch = sum(1 for r in records if r.get("fitness") == 1.0)
    n_total = len(records)

    total_wall = sum(t.get("wall_time_s", 0.0) for t in run["timing"])

    skill_totals = {"loco": 0.0, "left": 0.0, "right": 0.0}
    n_skill = 0
    for r in records:
        pct = r.get("skill_pct") or {}
        if pct:
            for k in skill_totals:
                skill_totals[k] += pct.get(k, 0.0)
            n_skill += 1
    if n_skill:
        for k in skill_totals:
            skill_totals[k] /= n_skill

    loco_bests = [min(r["loco_learning_curve"]) for r in records if r.get("loco_learning_curve")]
    left_bests = [min(r["left_learning_curve"]) for r in records if r.get("left_learning_curve")]
    right_bests = [min(r["right_learning_curve"]) for r in records if r.get("right_learning_curve")]

    lines = []
    lines.append(f"Run: {run['name']}  (job seed={cfg['seed']}, strategy={cfg['strategy_type']})")
    lines.append(f"  Generations completed: {len(gens)} / budget={cfg['budget']}  "
                 f"(pop={cfg['pop']}, lam={cfg['lam']})")
    lines.append(f"  Individuals evaluated: {n_total}  "
                 f"({n_glitch} hinge-glitch penalised = {100*n_glitch/max(n_total,1):.1f}%)")
    if all_fits:
        lines.append(f"  Food-task fitness (lower=better): min={min(all_fits):.4f}  "
                     f"mean={np.mean(all_fits):.4f}  max={max(all_fits):.4f}")
    if gens:
        first_best = min(fits_by_gen[gens[0]])
        last_best  = min(fits_by_gen[gens[-1]])
        lines.append(f"  Best food-task fitness: gen0={first_best:.4f}  ->  gen{gens[-1]}={last_best:.4f}")
    if best_rec is not None:
        lines.append(f"  Best individual: gen={best_rec['gen']} ind_id={best_rec['ind_id']} "
                     f"fitness={best_rec['fitness']:.4f} genome_hash={best_rec['genome_hash'][:10]}")
    if loco_bests:
        lines.append(f"  Loco skill fitness (best-of-CMA-ES, lower=better): min={min(loco_bests):.4f}  "
                     f"mean={np.mean(loco_bests):.4f}")
    if left_bests:
        lines.append(f"  Left-turn skill fitness: min={min(left_bests):.4f}  mean={np.mean(left_bests):.4f}")
    if right_bests:
        lines.append(f"  Right-turn skill fitness: min={min(right_bests):.4f}  mean={np.mean(right_bests):.4f}")
    lines.append(f"  Mean skill usage (%% of controlled steps): "
                 f"loco={skill_totals['loco']:.1f}  left={skill_totals['left']:.1f}  "
                 f"right={skill_totals['right']:.1f}")
    lines.append(f"  Total wall time: {total_wall/3600:.2f} h")
    return "\n".join(lines)


def _plot_running_best_and_spread(
    axes: tuple[plt.Axes, plt.Axes],
    runs: list[dict[str, Any]],
    by_gen_fn,
    ylabel: str,
    title_prefix: str,
) -> None:
    ax_best, ax_spread = axes

    for i, run in enumerate(runs):
        by_gen = by_gen_fn(run["records"])
        gens = sorted(by_gen)
        if not gens:
            continue
        running_best = np.minimum.accumulate([min(by_gen[g]) for g in gens])
        color = COLORS[i % len(COLORS)]
        label = f"{run['name']} (seed={run['config']['seed']})"
        ax_best.plot(gens, running_best, color=color, linewidth=2, label=label)
    ax_best.set_xlabel("Generation")
    ax_best.set_ylabel(ylabel)
    ax_best.set_title(f"{title_prefix} — elite progress (cumulative best)")
    ax_best.legend(fontsize=8, loc="best")
    ax_best.grid(alpha=0.3)

    for i, run in enumerate(runs):
        by_gen = by_gen_fn(run["records"])
        gens = sorted(by_gen)
        if not gens:
            continue
        mean = [float(np.mean(by_gen[g])) for g in gens]
        worst = [max(by_gen[g]) for g in gens]
        best = [min(by_gen[g]) for g in gens]
        color = COLORS[i % len(COLORS)]
        label = f"{run['name']} (seed={run['config']['seed']})"
        ax_spread.plot(gens, mean, color=color, linewidth=1.5, label=label)
        ax_spread.fill_between(gens, best, worst, color=color, alpha=0.1)
    ax_spread.set_xlabel("Generation")
    ax_spread.set_ylabel(ylabel)
    ax_spread.set_title(f"{title_prefix} — per-generation offspring batch (line=mean, band=best..worst)")
    ax_spread.legend(fontsize=8, loc="best")
    ax_spread.grid(alpha=0.3)


def plot_fitness_curves(runs: list[dict[str, Any]], out_path: Path, dpi: int) -> None:
    # Only newly-evaluated offspring are recorded each generation (parents that
    # survive under the `plus` strategy are not re-evaluated), so per-generation
    # min/mean/max of *this file* is the offspring batch, not the live population.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _plot_running_best_and_spread(
        (axes[0], axes[1]), runs, per_gen_fitness,
        ylabel="Food-task fitness (lower = better)",
        title_prefix="Food-task fitness",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_loco_fitness_curves(runs: list[dict[str, Any]], out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _plot_running_best_and_spread(
        (axes[0], axes[1]), runs,
        lambda records: per_gen_best_of_curve(records, "loco_learning_curve"),
        ylabel="Loco skill fitness (lower = better, i.e. more forward displacement)",
        title_prefix="Forward-locomotion skill fitness",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_turn_fitness_curves(runs: list[dict[str, Any]], out_path: Path, dpi: int) -> None:
    # Left and right turn skills are trained independently (not mirror images
    # by construction), so give each its own row rather than overlaying them.
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    _plot_running_best_and_spread(
        (axes[0, 0], axes[0, 1]), runs,
        lambda records: per_gen_best_of_curve(records, "left_learning_curve"),
        ylabel="Left-turn skill fitness (lower = better, i.e. more accumulated yaw)",
        title_prefix="Turn-left skill fitness",
    )
    _plot_running_best_and_spread(
        (axes[1, 0], axes[1, 1]), runs,
        lambda records: per_gen_best_of_curve(records, "right_learning_curve"),
        ylabel="Right-turn skill fitness (lower = better, i.e. more accumulated yaw)",
        title_prefix="Turn-right skill fitness",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_diversity(runs: list[dict[str, Any]], out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    skill_linestyle = {"loco": "-", "left": "--", "right": ":"}
    for i, run in enumerate(runs):
        mix = per_gen_skill_mix(run["records"])
        gens = sorted(mix)
        color = COLORS[i % len(COLORS)]
        for skill, ls in skill_linestyle.items():
            vals = [mix[g][skill] for g in gens]
            ax.plot(gens, vals, color=color, linewidth=1.3, linestyle=ls, alpha=0.85)
        ax.plot([], [], color=color, linewidth=1.5,
                label=f"{run['name']} (seed={run['config']['seed']})")
    for skill, ls in skill_linestyle.items():
        ax.plot([], [], color="black", linestyle=ls, linewidth=1.3, label=f"({skill})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean %% of controlled steps")
    ax.set_title("Skill-controller usage mix (loco=solid, left=dashed, right=dotted)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, run in enumerate(runs):
        sym_by_gen = per_gen_symmetry(run["records"])
        gens = sorted(sym_by_gen)
        mean_sym = [float(np.mean(sym_by_gen[g])) for g in gens]
        std_sym  = [float(np.std(sym_by_gen[g])) for g in gens]
        color = COLORS[i % len(COLORS)]
        mean_sym_arr = np.array(mean_sym)
        std_sym_arr = np.array(std_sym)
        ax.plot(gens, mean_sym, color=color, linewidth=1.5,
                label=f"{run['name']} (seed={run['config']['seed']})")
        ax.fill_between(gens, mean_sym_arr - std_sym_arr, mean_sym_arr + std_sym_arr,
                        color=color, alpha=0.15)
    ax.set_xlabel("Generation")
    ax.set_ylabel("yz_symmetry (mean ± std)")
    ax.set_title("Bilateral symmetry spread")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def main() -> None:
    runs = [load_run(d) for d in args.run_dirs]

    summary_lines = ["gecko_food_skills.py — replicate run summary", "=" * 60, ""]
    for run in runs:
        summary_lines.append(summarise(run))
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (args.out_dir / "summary.txt").write_text(summary_text)
    print(f"\nSaved -> {args.out_dir / 'summary.txt'}")

    plot_fitness_curves(runs, args.out_dir / "fitness_curves.png", args.dpi)
    plot_loco_fitness_curves(runs, args.out_dir / "loco_fitness_curves.png", args.dpi)
    plot_turn_fitness_curves(runs, args.out_dir / "turn_fitness_curves.png", args.dpi)
    plot_diversity(runs, args.out_dir / "diversity.png", args.dpi)


if __name__ == "__main__":
    main()
