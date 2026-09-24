"""
Compare symmetry-enforced vs no-symmetry food_skills runs.

Generates:
  fitness_comparison.png     – food-task fitness (group mean ± CI bands)
  diversity_comparison.png   – yz_symmetry + unique-genome diversity per group
  best_morphologies.png      – top-3 body plans per group (top-down 2D layout)

Usage:
    python compare_food_skills.py \\
        --sym-dirs  ../../__data__/food_skills/food_skills_..._35117 ... \\
        --nosym-dirs ../../__data__/food_skills/food_skills_..._35301 ... \\
        --out-dir   ../../__data__/food_skills_comparison
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

from genome_diagram import MODULE_COLORS, draw_morphology, genome_layout

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--sym-dirs",   nargs="+", type=Path, required=True)
parser.add_argument("--nosym-dirs", nargs="+", type=Path, required=True)
parser.add_argument("--out-dir",    type=Path, default=Path("__data__/food_skills_comparison"))
parser.add_argument("--dpi",        type=int, default=150)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

SYM_COLOR   = "#4C72B0"   # blue
NOSYM_COLOR = "#DD8452"   # orange
CI_ALPHA    = 0.20
GLITCH_FIT  = 1.0         # penalty value to exclude


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> dict[str, Any]:
    data_dir = run_dir / "__data__" / "gecko_food_skills"
    config = json.loads((data_dir / "run_config.json").read_text())
    records: list[dict] = []
    with (data_dir / "run_data.jsonl").open() as f:
        for line in f:
            records.append(json.loads(line))
    return {"name": run_dir.name, "config": config, "records": records,
            "ckpt_base": data_dir / "checkpoints"}


def per_gen_fitness(records: list[dict]) -> dict[int, list[float]]:
    by_gen: dict[int, list[float]] = {}
    for r in records:
        fit = r.get("fitness")
        if fit is None or not np.isfinite(fit) or fit == GLITCH_FIT:
            continue
        by_gen.setdefault(r["gen"], []).append(fit)
    return by_gen


def per_gen_symmetry(records: list[dict]) -> dict[int, list[float]]:
    by_gen: dict[int, list[float]] = {}
    for r in records:
        sym = r.get("yz_symmetry")
        if sym is None:
            continue
        by_gen.setdefault(r["gen"], []).append(sym)
    return by_gen


def per_gen_unique_ratio(records: list[dict]) -> dict[int, float]:
    by_gen: dict[int, list[str]] = {}
    for r in records:
        gh = r.get("genome_hash")
        if gh:
            by_gen.setdefault(r["gen"], []).append(gh)
    return {g: len(set(hs)) / len(hs) for g, hs in by_gen.items() if hs}


def running_best(by_gen: dict[int, list[float]]) -> tuple[list[int], list[float]]:
    gens = sorted(by_gen)
    bests = [min(by_gen[g]) for g in gens]
    rb = list(np.minimum.accumulate(bests))
    return gens, rb


def group_stats(
    runs: list[dict], fn
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Compute per-generation mean ± 95% CI across runs using `fn` to extract values."""
    all_series: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        vals = fn(run["records"])
        for g, v in vals.items():
            all_series[g].append(v if isinstance(v, float) else np.mean(v))
    gens = sorted(all_series)
    means, lo, hi = [], [], []
    for g in gens:
        vs = np.array(all_series[g])
        m = float(np.mean(vs))
        if len(vs) > 1:
            se = stats.sem(vs)
            ci = se * stats.t.ppf(0.975, len(vs) - 1)
        else:
            ci = 0.0
        means.append(m)
        lo.append(m - ci)
        hi.append(m + ci)
    return gens, means, lo, hi


def group_running_best_stats(
    runs: list[dict],
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Mean ± 95% CI of the running-best curve across replicate runs."""
    all_rb: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        by_gen = per_gen_fitness(run["records"])
        gens, rb = running_best(by_gen)
        for g, v in zip(gens, rb):
            all_rb[g].append(v)
    gens = sorted(all_rb)
    means, lo, hi = [], [], []
    for g in gens:
        vs = np.array(all_rb[g])
        m = float(np.mean(vs))
        if len(vs) > 1:
            se = stats.sem(vs)
            ci = se * stats.t.ppf(0.975, len(vs) - 1)
        else:
            ci = 0.0
        means.append(m)
        lo.append(m - ci)
        hi.append(m + ci)
    return gens, means, lo, hi


# ── Fitness comparison plot ───────────────────────────────────────────────────

def plot_fitness_comparison(
    sym_runs: list[dict], nosym_runs: list[dict], out_path: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: running best (elite progress)
    ax = axes[0]
    for runs, color, label in [
        (sym_runs,   SYM_COLOR,   "Symmetry enforced (y_zero)"),
        (nosym_runs, NOSYM_COLOR, "No symmetry (none)"),
    ]:
        gens, means, lo, hi = group_running_best_stats(runs)
        ax.plot(gens, means, color=color, linewidth=2.5, label=label)
        ax.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA)
        # Individual run traces (faint)
        for run in runs:
            bg = per_gen_fitness(run["records"])
            g_list, rb = running_best(bg)
            ax.plot(g_list, rb, color=color, linewidth=0.6, alpha=0.25)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Food-task fitness (lower = better)")
    ax.set_title("Elite progress — running best\n(thick = group mean ± 95% CI, thin = individual runs)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: per-generation offspring mean (population spread)
    ax = axes[1]
    for runs, color, label in [
        (sym_runs,   SYM_COLOR,   "Symmetry enforced (y_zero)"),
        (nosym_runs, NOSYM_COLOR, "No symmetry (none)"),
    ]:
        gens, means, lo, hi = group_stats(
            runs, lambda recs: {g: np.mean(v) for g, v in per_gen_fitness(recs).items()}
        )
        ax.plot(gens, means, color=color, linewidth=2.5, label=label)
        ax.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Food-task fitness (lower = better)")
    ax.set_title("Per-generation offspring mean\n(thick = group mean ± 95% CI)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Food-task fitness: symmetry enforcement vs no symmetry", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Diversity comparison plot ─────────────────────────────────────────────────

def plot_diversity_comparison(
    sym_runs: list[dict], nosym_runs: list[dict], out_path: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: yz_symmetry mean over generations
    ax = axes[0]
    for runs, color, label in [
        (sym_runs,   SYM_COLOR,   "Symmetry enforced (y_zero)"),
        (nosym_runs, NOSYM_COLOR, "No symmetry (none)"),
    ]:
        gens, means, lo, hi = group_stats(
            runs, lambda recs: {g: np.mean(v) for g, v in per_gen_symmetry(recs).items()}
        )
        ax.plot(gens, means, color=color, linewidth=2.5, label=label)
        ax.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA)
        for run in runs:
            sym = per_gen_symmetry(run["records"])
            g_list = sorted(sym)
            ax.plot(g_list, [np.mean(sym[g]) for g in g_list],
                    color=color, linewidth=0.6, alpha=0.25)

    ax.set_xlabel("Generation")
    ax.set_ylabel("yz_symmetry score (1 = perfectly symmetric)")
    ax.set_title("Bilateral symmetry over evolution")
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: unique genome ratio per generation (genome diversity)
    ax = axes[1]
    for runs, color, label in [
        (sym_runs,   SYM_COLOR,   "Symmetry enforced (y_zero)"),
        (nosym_runs, NOSYM_COLOR, "No symmetry (none)"),
    ]:
        all_series: dict[int, list[float]] = defaultdict(list)
        for run in runs:
            ur = per_gen_unique_ratio(run["records"])
            for g, v in ur.items():
                all_series[g].append(v)
        gens = sorted(all_series)
        means = [np.mean(all_series[g]) for g in gens]
        lo_ci, hi_ci = [], []
        for g in gens:
            vs = np.array(all_series[g])
            if len(vs) > 1:
                se = stats.sem(vs)
                ci = se * stats.t.ppf(0.975, len(vs) - 1)
            else:
                ci = 0.0
            lo_ci.append(np.mean(vs) - ci)
            hi_ci.append(np.mean(vs) + ci)

        ax.plot(gens, means, color=color, linewidth=2.5, label=label)
        ax.fill_between(gens, lo_ci, hi_ci, color=color, alpha=CI_ALPHA)
        for run in runs:
            ur = per_gen_unique_ratio(run["records"])
            g_list = sorted(ur)
            ax.plot(g_list, [ur[g] for g in g_list],
                    color=color, linewidth=0.6, alpha=0.25)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Unique genome ratio (per generation batch)")
    ax.set_title("Genome diversity within offspring batch")
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Diversity: symmetry enforcement vs no symmetry", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Morphology layout + plotting ──────────────────────────────────────────────
# genome_layout / MODULE_COLORS / draw_morphology now live in genome_diagram.py
# (shared with the ancestry-tree viewer's genome-thumbnail renderer).


def find_best_checkpoint(run: dict, target_fitness: float) -> Path | None:
    """Scan checkpoint meta.json files and return the closest match."""
    ckpt_base: Path = run["ckpt_base"]
    best_ckpt, best_diff = None, float("inf")
    for ckpt in sorted(ckpt_base.iterdir()):
        meta_path = ckpt / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            diff = abs(meta.get("fitness", float("nan")) - target_fitness)
            if diff < best_diff:
                best_diff = diff
                best_ckpt = ckpt
    return best_ckpt


def best_individuals(runs: list[dict], top_n: int = 3) -> list[dict]:
    """
    Return info for the top-n best individuals, one per run (different runs).
    Each entry: {fitness, run_name, ckpt_dir, genome}.
    """
    per_run_best = []
    for run in runs:
        valid = [
            r for r in run["records"]
            if r.get("fitness") is not None
            and np.isfinite(r["fitness"])
            and r["fitness"] != GLITCH_FIT
        ]
        if not valid:
            continue
        best_rec = min(valid, key=lambda r: r["fitness"])
        ckpt_dir = find_best_checkpoint(run, best_rec["fitness"])
        if ckpt_dir is None:
            continue
        genome = json.loads((ckpt_dir / "best_genome.json").read_text())
        per_run_best.append({
            "fitness": best_rec["fitness"],
            "gen": best_rec["gen"],
            "run_name": run["name"],
            "ckpt_dir": ckpt_dir,
            "genome": genome,
        })
    per_run_best.sort(key=lambda x: x["fitness"])
    return per_run_best[:top_n]


def plot_best_morphologies(
    sym_runs: list[dict], nosym_runs: list[dict], out_path: Path, dpi: int
) -> None:
    sym_best   = best_individuals(sym_runs,   top_n=3)
    nosym_best = best_individuals(nosym_runs, top_n=3)

    n_sym   = len(sym_best)
    n_nosym = len(nosym_best)
    n_cols  = max(n_sym, n_nosym)

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, ind in enumerate(sym_best):
        run_short = ind["run_name"].split("_")[-1]
        draw_morphology(
            axes[0, col], ind["genome"],
            title=f"Sym #{col+1}  run={run_short}  gen={ind['gen']}",
            fitness=ind["fitness"],
        )
    for col in range(n_sym, n_cols):
        axes[0, col].axis("off")

    for col, ind in enumerate(nosym_best):
        run_short = ind["run_name"].split("_")[-1]
        draw_morphology(
            axes[1, col], ind["genome"],
            title=f"NoSym #{col+1}  run={run_short}  gen={ind['gen']}",
            fitness=ind["fitness"],
        )
    for col in range(n_nosym, n_cols):
        axes[1, col].axis("off")

    # Row labels
    for row, label, color in [(0, "Symmetry enforced (y_zero)", SYM_COLOR),
                               (1, "No symmetry (none)",         NOSYM_COLOR)]:
        fig.text(
            0.01, 0.75 - row * 0.5, label,
            va="center", ha="left", fontsize=11, fontweight="bold", color=color,
            rotation=90,
        )

    fig.suptitle(
        "Top-3 best body plans per group (2D top-down module layout)\n"
        "Square=CORE  Circle=BRICK  Diamond=HINGE  |  lower fitness = better",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0.04, 0, 1, 0.96])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading runs...")
    sym_runs   = [load_run(d) for d in args.sym_dirs]
    nosym_runs = [load_run(d) for d in args.nosym_dirs]
    print(f"  {len(sym_runs)} symmetric  |  {len(nosym_runs)} no-symmetry")

    print("\nPlotting fitness comparison...")
    plot_fitness_comparison(sym_runs, nosym_runs,
                            args.out_dir / "fitness_comparison.png", args.dpi)

    print("Plotting diversity comparison...")
    plot_diversity_comparison(sym_runs, nosym_runs,
                              args.out_dir / "diversity_comparison.png", args.dpi)

    print("Plotting best morphologies...")
    plot_best_morphologies(sym_runs, nosym_runs,
                           args.out_dir / "best_morphologies.png", args.dpi)

    print(f"\nDone. All plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
