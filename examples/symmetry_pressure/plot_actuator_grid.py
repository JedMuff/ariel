"""
Hinge-actuator grid (slurm/run_actuator_grid.sh): where does tree_symmetric
beat tree on forward locomotion?

Each run is one (brain, genome, hinge cell, seed). Its score is the best
forward score of its last generation (-fitness: the normalised translate
score in [0, 2] minus penalties, so higher is better). Per brain and cell,
delta = mean(tree_symmetric) - mean(tree) over reps, with a percentile
bootstrap CI (reps resampled within each genome).

Outputs (in <root>/analysis/):
  actuator_grid_delta.png     per brain: kp x kv heatmaps, one panel per
                              armature x joint limit, coloured by delta and
                              annotated with both genomes' means and n
  actuator_grid_ranked.csv    cells sorted by delta per brain, with CI, means,
                              n, and how many runs hit the score cap of 2
  actuator_grid_runs.csv      one row per run

Usage:
    python plot_actuator_grid.py --root /scratch/jed/ariel_actuator_grid
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from shared import TRANSLATE_MAX, HingeActuator
from sweep_common import load_fitness_by_gen, run_config

TASK = "forward"
GENOMES = ("tree", "tree_symmetric")
BRAINS = ("dann", "bang_bang")
N_BOOT = 5000
CAP_TOL = 1e-3

# Diverging blue (symmetric better) <-> red (tree better) through a neutral gray.
DIVERGING = LinearSegmentedColormap.from_list(
    "tree_sym", ["#b3261e", "#e34948", "#f2a8a4", "#f0efec", "#86b6ef", "#2a78d6", "#104281"],
)
TEXT = "#1f1f1d"
MUTED = "#6b6a63"


def load_runs(root: Path) -> list[dict]:
    """One row per finished actgrid_* run: brain, genome, actuator, seed, last gen, score."""
    rows = []
    for run_dir in sorted(root.glob("actgrid_*")):
        cfg = run_config(run_dir, TASK)
        by_gen = load_fitness_by_gen(run_dir, TASK)
        if not cfg or not by_gen or "hinge_actuator" not in cfg:
            continue
        last = max(by_gen)
        act = HingeActuator(**cfg["hinge_actuator"])
        rows.append({
            "run": run_dir.name, "brain": cfg["brain"], "genome": cfg["genome_type"],
            "seed": cfg["seed"], "label": act.label, "kp": act.kp, "kv": act.kv,
            "armature": act.armature, "limited": act.limited,
            "last_gen": last, "score": -min(by_gen[last]),
        })
    return rows


def bootstrap_delta(sym: np.ndarray, tree: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(sym) < 2 or len(tree) < 2:
        return math.nan, math.nan
    d = (rng.choice(sym, (N_BOOT, len(sym))).mean(1) - rng.choice(tree, (N_BOOT, len(tree))).mean(1))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cell_stats(rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(0)
    groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["brain"], r["kp"], r["kv"], r["armature"], r["limited"])][r["genome"]].append(r["score"])
    out = []
    for (brain, kp, kv, arm, lim), by_genome in groups.items():
        sym, tree = (np.array(by_genome.get(g, [])) for g in ("tree_symmetric", "tree"))
        lo, hi = bootstrap_delta(sym, tree, rng)
        out.append({
            "brain": brain, "label": HingeActuator(kp, kv, arm, lim).label,
            "kp": kp, "kv": kv, "armature": arm, "limited": lim,
            "delta": float(sym.mean() - tree.mean()) if len(sym) and len(tree) else math.nan,
            "ci_lo": lo, "ci_hi": hi,
            "mean_sym": float(sym.mean()) if len(sym) else math.nan,
            "mean_tree": float(tree.mean()) if len(tree) else math.nan,
            "n_sym": len(sym), "n_tree": len(tree),
            "n_capped": int(np.sum(np.concatenate([sym, tree]) >= TRANSLATE_MAX - CAP_TOL)),
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_delta(stats: list[dict], out_path: Path) -> None:
    kps = sorted({s["kp"] for s in stats})
    kvs = sorted({s["kv"] for s in stats})
    arms = sorted({s["armature"] for s in stats})
    lims = sorted({s["limited"] for s in stats}, reverse=True)
    brains = [b for b in BRAINS if any(s["brain"] == b for s in stats)]
    panels = list(itertools.product(arms, lims))
    by_key = {(s["brain"], s["kp"], s["kv"], s["armature"], s["limited"]): s for s in stats}

    finite = [abs(s["delta"]) for s in stats if math.isfinite(s["delta"])]
    vmax = max(finite) if finite else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(len(brains), len(panels), figsize=(3.4 * len(panels), 3.3 * len(brains) + 0.6),
                             squeeze=False, constrained_layout=True)
    for (bi, brain), (pi, (arm, lim)) in itertools.product(enumerate(brains), enumerate(panels)):
        ax = axes[bi, pi]
        grid = np.full((len(kps), len(kvs)), np.nan)
        for (i, kp), (j, kv) in itertools.product(enumerate(kps), enumerate(kvs)):
            s = by_key.get((brain, kp, kv, arm, lim))
            if s is None:
                continue
            grid[i, j] = s["delta"]
            ink = "white" if math.isfinite(s["delta"]) and abs(s["delta"]) > 0.6 * vmax else TEXT
            ax.text(j, i - 0.12, f"{s['delta']:+.2f}", ha="center", va="center", fontsize=10, color=ink,
                    fontweight="bold")
            ax.text(j, i + 0.2, f"S {s['mean_sym']:.2f} / T {s['mean_tree']:.2f}\nn={s['n_sym']}/{s['n_tree']}",
                    ha="center", va="center", fontsize=6.5, color=ink, linespacing=1.1)
        ax.imshow(grid, cmap=DIVERGING, norm=norm, origin="lower", aspect="auto")
        ax.set_xticks(range(len(kvs)), [f"{v:g}" for v in kvs], fontsize=8, color=MUTED)
        ax.set_yticks(range(len(kps)), [f"{v:g}" for v in kps], fontsize=8, color=MUTED)
        ax.set_xticks(np.arange(len(kvs) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(kps) + 1) - 0.5, minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if bi == len(brains) - 1:
            ax.set_xlabel("kv (damping)", fontsize=9, color=MUTED)
        if pi == 0:
            ax.set_ylabel(f"{brain}\nkp (gain)", fontsize=9, color=TEXT)
        if bi == 0:
            ax.set_title(f"armature {arm:g}, limit {'on' if lim else 'off'}", fontsize=9, color=TEXT)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), ax=axes, shrink=0.8, pad=0.01)
    cbar.set_label("mean score: symmetric − tree\n(blue = symmetric better)", fontsize=9, color=TEXT)
    cbar.outline.set_visible(False)
    fig.suptitle("Symmetric advantage on forward, by hinge actuator (S / T = mean score per genome)",
                 fontsize=11, color=TEXT)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/scratch/jed/ariel_actuator_grid"))
    args = parser.parse_args()

    rows = load_runs(args.root)
    if not rows:
        raise SystemExit(f"no finished actgrid_* runs under {args.root}")
    out_dir = args.root / "analysis"
    out_dir.mkdir(exist_ok=True)

    write_csv(out_dir / "actuator_grid_runs.csv", rows)
    stats = cell_stats(rows)
    ranked = sorted(stats, key=lambda s: (s["brain"], -s["delta"] if math.isfinite(s["delta"]) else math.inf))
    write_csv(out_dir / "actuator_grid_ranked.csv", ranked)
    plot_delta(stats, out_dir / "actuator_grid_delta.png")

    print(f"{len(rows)} runs, {len(stats)} brain x cell groups -> {out_dir}")
    for brain in BRAINS:
        top = [s for s in ranked if s["brain"] == brain][:5]
        if not top:
            continue
        print(f"\n{brain}: top cells by symmetric - tree")
        for s in top:
            print(f"  {s['label']:40s} delta {s['delta']:+.3f}  CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]  "
                  f"sym {s['mean_sym']:.3f}  tree {s['mean_tree']:.3f}  n {s['n_sym']}/{s['n_tree']}  "
                  f"capped {s['n_capped']}")
    capped = sum(r["score"] >= TRANSLATE_MAX - CAP_TOL for r in rows)
    print(f"\nruns at the score cap ({TRANSLATE_MAX}): {capped} / {len(rows)}")


if __name__ == "__main__":
    main()
