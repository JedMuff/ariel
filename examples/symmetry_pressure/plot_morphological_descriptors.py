"""
Compare morphological descriptors across symmetry-enforced vs no-symmetry runs.

Plots all 8 descriptors from MorphologicalMeasures in a 2×4 grid:
  num_modules, branching (B), limbs (L), length_of_limbs (E),
  coverage (C), joints (J), symmetry (S), module_diversity (D)

Each subplot shows group mean ± 95% CI (shaded) over generations,
with faint individual-run traces behind.

Usage:
    python plot_morphological_descriptors.py \\
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
CI_ALPHA    = 0.20

DESCRIPTORS = [
    ("num_modules",     "Num modules",             False),
    ("branching",       "Branching (B)",            False),
    ("limbs",           "Limbs (L)",                False),
    ("length_of_limbs", "Length of limbs (E)",      False),
    ("coverage",        "Coverage (C)",             False),
    ("joints",          "Joints (J)",               False),
    ("symmetry",        "Symmetry (S)",             False),
    ("module_diversity","Module diversity (D)",     False),
]


# ── Genome → descriptors ──────────────────────────────────────────────────────

def genome_to_nx(genome: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for nid, attrs in genome["nodes"].items():
        G.add_node(nid, type=attrs["type"], rotation=attrs.get("rotation", "DEG_0"))
    for e in genome["edges"]:
        G.add_edge(str(e["parent"]), str(e["child"]), face=e["face"])
    return G


def compute_descriptors(genome: dict) -> dict[str, float]:
    try:
        m = MorphologicalMeasures(genome_to_nx(genome))
    except Exception:
        return {}
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

def load_run_descriptors(run_dir: Path) -> dict[int, list[dict[str, float]]]:
    """
    Returns gen -> [descriptor_dict, ...] for every checkpoint in the run.
    """
    ckpt_base = run_dir / "__data__" / "gecko_food_skills" / "checkpoints"
    by_gen: dict[int, list[dict[str, float]]] = defaultdict(list)
    for ckpt in sorted(ckpt_base.iterdir()):
        meta_path = ckpt / "meta.json"
        genome_path = ckpt / "best_genome.json"
        if not meta_path.exists() or not genome_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        genome = json.loads(genome_path.read_text())
        desc = compute_descriptors(genome)
        if desc:
            by_gen[meta["gen"]].append(desc)
    return dict(by_gen)


# ── Statistics ────────────────────────────────────────────────────────────────

def per_gen_mean(by_gen: dict[int, list[dict]], key: str) -> dict[int, float]:
    return {g: float(np.mean([d[key] for d in ds if key in d]))
            for g, ds in by_gen.items() if any(key in d for d in ds)}


def group_stats(
    runs_data: list[dict[int, list[dict]]], key: str
) -> tuple[list[int], list[float], list[float], list[float]]:
    all_means: dict[int, list[float]] = defaultdict(list)
    for run in runs_data:
        gm = per_gen_mean(run, key)
        for g, v in gm.items():
            all_means[g].append(v)
    gens = sorted(all_means)
    means, lo, hi = [], [], []
    for g in gens:
        vs = np.array(all_means[g])
        m = float(np.mean(vs))
        ci = (stats.sem(vs) * stats.t.ppf(0.975, max(len(vs)-1, 1))
              if len(vs) > 1 else 0.0)
        means.append(m)
        lo.append(m - ci)
        hi.append(m + ci)
    return gens, means, lo, hi


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_descriptors(
    sym_data: list[dict], nosym_data: list[dict],
    out_path: Path, dpi: int,
) -> None:
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))

    for idx, (key, label, _) in enumerate(DESCRIPTORS):
        ax = axes[idx // n_cols, idx % n_cols]

        for runs_data, color, grp_label in [
            (sym_data,   SYM_COLOR,   "Symmetry enforced (y_zero)"),
            (nosym_data, NOSYM_COLOR, "No symmetry (none)"),
        ]:
            gens, means, lo, hi = group_stats(runs_data, key)
            ax.plot(gens, means, color=color, linewidth=2.2, label=grp_label)
            ax.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA)

            for run in runs_data:
                gm = per_gen_mean(run, key)
                g_list = sorted(gm)
                ax.plot(g_list, [gm[g] for g in g_list],
                        color=color, linewidth=0.6, alpha=0.2)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Generation", fontsize=8)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Morphological descriptors: symmetry enforcement vs no symmetry\n"
        "(thick line = group mean ± 95% CI,  faint = individual runs)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading checkpoint descriptors for symmetric runs...")
    sym_data = []
    for i, d in enumerate(args.sym_dirs, 1):
        print(f"  [{i}/{len(args.sym_dirs)}] {d.name}")
        sym_data.append(load_run_descriptors(d))

    print("Loading checkpoint descriptors for no-symmetry runs...")
    nosym_data = []
    for i, d in enumerate(args.nosym_dirs, 1):
        print(f"  [{i}/{len(args.nosym_dirs)}] {d.name}")
        nosym_data.append(load_run_descriptors(d))

    print("Plotting...")
    plot_descriptors(sym_data, nosym_data,
                     args.out_dir / "morphological_descriptors.png", args.dpi)
    print(f"Done. Output -> {args.out_dir}/morphological_descriptors.png")


if __name__ == "__main__":
    main()
