"""
Scatter each morphological descriptor against food-task fitness, with a
per-group OLS linear regression line to reveal correlations.

Data source: all checkpoint directories (one entry per individual per run).
Each checkpoint has meta.json (gen, fitness) and best_genome.json.

Outputs (under --out-dir):
  descriptors_vs_fitness.png   2×4 grid, one panel per descriptor.
    Points are coloured by group (sym=blue, nosym=orange).
    A fitted OLS line + 95% confidence band is overlaid per group.
    Pearson r and p-value annotated in each panel.

Usage:
    python plot_descriptors_vs_fitness.py \\
        --sym-dirs   ../../__data__/food_skills/food_skills_..._35117 ... \\
        --nosym-dirs ../../__data__/food_skills/food_skills_..._35301 ... \\
        --out-dir    ../../__data__/food_skills_comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from ariel.utils.morphological_descriptor import MorphologicalMeasures
from shared import bilateral_symmetry_score

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
ALPHA_PT    = 0.35

DESCRIPTORS = [
    ("num_modules",      "Num modules"),
    ("branching",        "Branching (B)"),
    ("limbs",            "Limbs (L)"),
    ("length_of_limbs",  "Length of limbs (E)"),
    ("coverage",         "Coverage (C)"),
    ("joints",           "Joints (J)"),
    ("symmetry",         "Symmetry (S)"),
    ("module_diversity", "Module diversity (D)"),
]


# ── Genome helpers ────────────────────────────────────────────────────────────

def genome_to_nx(genome: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for nid, attrs in genome["nodes"].items():
        G.add_node(nid, type=attrs["type"], rotation=attrs.get("rotation", "DEG_0"))
    for e in genome["edges"]:
        G.add_edge(str(e["parent"]), str(e["child"]), face=e["face"])
    return G


def compute_descriptors(genome: dict) -> dict[str, float] | None:
    try:
        m = MorphologicalMeasures(genome_to_nx(genome))
    except Exception:
        return None
    return {
        "num_modules":      float(m.num_modules),
        "branching":        m.branching,
        "limbs":            m.limbs,
        "length_of_limbs":  m.length_of_limbs,
        "coverage":         m.coverage,
        "joints":           m.joints,
        "symmetry":         bilateral_symmetry_score(genome),
        "module_diversity": m.module_diversity,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_group_data(run_dirs: list[Path]) -> dict[str, list[float]]:
    """
    Iterate all checkpoints across all run_dirs.
    Returns {descriptor_key: [values...], "fitness": [values...]}.
    """
    accum: dict[str, list[float]] = defaultdict(list)

    for run_dir in run_dirs:
        ckpt_base = run_dir / "__data__" / "gecko_food_skills" / "checkpoints"
        if not ckpt_base.exists():
            continue
        for ckpt in sorted(ckpt_base.iterdir()):
            meta_path   = ckpt / "meta.json"
            genome_path = ckpt / "best_genome.json"
            if not meta_path.exists() or not genome_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            fit  = meta.get("fitness")
            if fit is None or not np.isfinite(fit):
                continue
            genome = json.loads(genome_path.read_text())
            desc   = compute_descriptors(genome)
            if desc is None:
                continue
            accum["fitness"].append(float(fit))
            for key, val in desc.items():
                accum[key].append(float(val))

    return dict(accum)


# ── OLS helpers ───────────────────────────────────────────────────────────────

def ols_line_and_band(
    x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_hat, lower, upper) on x_grid for OLS fit of y ~ x."""
    n = len(x)
    if n < 3:
        nan = np.full_like(x_grid, np.nan, dtype=float)
        return nan, nan, nan
    slope, intercept, *_ = stats.linregress(x, y)
    y_hat = intercept + slope * x_grid

    x_mean = np.mean(x)
    ss_x   = np.sum((x - x_mean) ** 2)
    residuals = y - (intercept + slope * x)
    s2 = np.sum(residuals ** 2) / (n - 2)

    se_hat = np.sqrt(s2 * (1 / n + (x_grid - x_mean) ** 2 / ss_x))
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    return y_hat, y_hat - t_crit * se_hat, y_hat + t_crit * se_hat


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_descriptors_vs_fitness(
    sym_data: dict[str, list[float]],
    nosym_data: dict[str, list[float]],
    out_path: Path,
    dpi: int,
) -> None:
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))

    sym_fit   = np.array(sym_data.get("fitness", []))
    nosym_fit = np.array(nosym_data.get("fitness", []))

    for idx, (key, label) in enumerate(DESCRIPTORS):
        ax = axes[idx // n_cols, idx % n_cols]

        for data, fit_arr, color, grp_label in [
            (sym_data,   sym_fit,   SYM_COLOR,   "Symmetry enforced (y_zero)"),
            (nosym_data, nosym_fit, NOSYM_COLOR, "No symmetry (none)"),
        ]:
            vals = np.array(data.get(key, []))
            if len(vals) == 0 or len(vals) != len(fit_arr):
                continue

            # Scatter
            ax.scatter(
                fit_arr, vals,
                color=color, alpha=ALPHA_PT, s=14, linewidths=0, label=grp_label,
            )

            # OLS line + band
            x_grid = np.linspace(fit_arr.min(), fit_arr.max(), 200)
            y_hat, lo, hi = ols_line_and_band(fit_arr, vals, x_grid)
            ax.plot(x_grid, y_hat, color=color, linewidth=1.8)
            ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18)

            # Pearson r annotation
            r, p = stats.pearsonr(fit_arr, vals)
            p_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
            ax.annotate(
                f"r={r:+.2f}  {p_str}",
                xy=(0.04, 0.90 if color == SYM_COLOR else 0.82),
                xycoords="axes fraction",
                fontsize=7.5, color=color,
            )

        ax.set_xlabel("Food-task fitness (lower = better)", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right", markerscale=1.5)

    fig.suptitle(
        "Morphological descriptors vs food-task fitness\n"
        "(scatter = all evaluated individuals;  line = OLS fit ± 95% CI;  r = Pearson)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading checkpoint data for symmetric runs...")
    sym_data = load_group_data(list(args.sym_dirs))
    print(f"  {len(sym_data.get('fitness', []))} individuals")

    print("Loading checkpoint data for no-symmetry runs...")
    nosym_data = load_group_data(list(args.nosym_dirs))
    print(f"  {len(nosym_data.get('fitness', []))} individuals")

    print("Plotting descriptors vs fitness...")
    plot_descriptors_vs_fitness(
        sym_data, nosym_data,
        args.out_dir / "descriptors_vs_fitness.png",
        args.dpi,
    )
    print(f"Done. Output -> {args.out_dir}/descriptors_vs_fitness.png")


if __name__ == "__main__":
    main()
