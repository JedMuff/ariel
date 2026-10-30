"""
E0: the symmetric advantage and its conditions (R1).

  hinge grid      evolve.py runs over kp x kv x armature (slurm e0_actuator_grid.sh):
                  paired symmetric - tree score gap per brain and cell, pooled
                  per brain over every (cell, seed) pair
  budget series   DANN runs at 200 / 800 / 1600 evaluations per body
                  (e0_dann_budget.sh): paired gap per budget

Score = the best forward score of a run's last generation (runs.run_row).

Writes to --out (default <grid-root>/analysis):
  e0_grid_cells.csv, e0_grid_pooled.csv, e0_budget.csv, e0_runs.csv
  fig1a_grid_gap.png    kp x kv heatmaps of the gap, one row per brain, one column per armature
  fig1b_budget_gap.png  DANN gap vs evaluations per body, 95% CI

Usage:
    python analyse_e0.py --grid-root /scratch/$USER/ariel_paper_e0_grid \\
        --budget-roots /scratch/$USER/ariel_paper_e0_budget/sb10 /scratch/$USER/ariel_paper_e0_budget/sb40 ...
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import runs  # noqa: E402
from sim import TRANSLATE_MAX, ZOO_HINGE  # noqa: E402

CELL = ["brain", "kp", "kv", "armature", "limited"]
CAP_TOL = 1e-3
# Diverging: orange (tree better) <-> blue (symmetric better) through a neutral gray.
DIVERGING = LinearSegmentedColormap.from_list("tree_sym", ["#c4501f", "#eb6834", "#f4b89c", "#f0efec",
                                                           "#86b6ef", "#2a78d6", "#104281"])


def grid(df: pd.DataFrame, out: Path) -> None:
    cells = runs.gap_table(df, CELL)
    cells["n_capped"] = [int((df.loc[(df[CELL] == c[CELL]).all(axis=1), "score"] >= TRANSLATE_MAX - CAP_TOL).sum())
                         for _, c in cells.iterrows()]
    cells.to_csv(out / "e0_grid_cells.csv", index=False)
    pooled = df.assign(pair=df["hinge"] + "_s" + df["seed"].astype(str))
    pooled = runs.gap_table(pooled, ["brain"], pair_on="pair")
    pooled["cells_sym_ahead"] = [int((cells.loc[cells["brain"] == b, "gap"] > 0).sum()) for b in pooled["brain"]]
    pooled["n_cells"] = [int((cells["brain"] == b).sum()) for b in pooled["brain"]]
    pooled.to_csv(out / "e0_grid_pooled.csv", index=False)
    print("hinge grid, pooled per brain (symmetric - tree):")
    print(pooled.round(3).to_string(index=False))

    kps, kvs, arms = (sorted(cells[c].unique()) for c in ("kp", "kv", "armature"))
    brains = [b for b in runs.BRAIN_LABEL if b in set(cells["brain"])]
    vmax = float(np.nanmax(np.abs(cells["gap"]))) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(len(brains), len(arms), figsize=(3.5 * len(arms), 3.2 * len(brains) + 0.6),
                             squeeze=False, constrained_layout=True)
    for (bi, brain), (ai, arm) in itertools.product(enumerate(brains), enumerate(arms)):
        ax = axes[bi, ai]
        sub = cells[(cells["brain"] == brain) & (cells["armature"] == arm)]
        img = np.full((len(kps), len(kvs)), np.nan)
        for s in sub.itertuples():
            i, j = kps.index(s.kp), kvs.index(s.kv)
            img[i, j] = s.gap
            ink = "white" if abs(s.gap) > 0.6 * vmax else runs.INK
            zoo = math.isclose(s.kp, ZOO_HINGE.kp) and math.isclose(s.kv, ZOO_HINGE.kv) and math.isclose(arm, ZOO_HINGE.armature)
            ax.text(j, i - 0.1, f"{s.gap:+.2f}{' (zoo)' if zoo else ''}", ha="center", va="center", fontsize=9, color=ink)
            ax.text(j, i + 0.22, f"S {s.mean_tree_symmetric:.2f} / T {s.mean_tree:.2f}", ha="center", va="center",
                    fontsize=6.5, color=ink)
        ax.imshow(img, cmap=DIVERGING, norm=norm, origin="lower", aspect="auto")
        ax.set_xticks(range(len(kvs)), [f"{v:g}" for v in kvs])
        ax.set_yticks(range(len(kps)), [f"{v:g}" for v in kps])
        ax.tick_params(length=0, labelsize=8, colors=runs.INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if bi == len(brains) - 1:
            ax.set_xlabel("kv (damping)", fontsize=9, color=runs.INK_2)
        if ai == 0:
            ax.set_ylabel(f"{runs.BRAIN_LABEL[brain]}\nkp (gain)", fontsize=9, color=runs.INK)
        if bi == 0:
            ax.set_title(f"armature {arm:g}", fontsize=9, color=runs.INK)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), ax=axes, shrink=0.8)
    cbar.set_label("score gap, symmetric − tree\n(blue = symmetric better)", fontsize=9, color=runs.INK)
    cbar.outline.set_visible(False)
    fig.savefig(out / "fig1a_grid_gap.png", dpi=200)
    plt.close(fig)


def budget(df: pd.DataFrame, out: Path) -> None:
    df = df[(df["brain"] == "dann") & (df["hinge"] == ZOO_HINGE.label)]
    tab = runs.gap_table(df, ["evals_per_body"])
    tab.to_csv(out / "e0_budget.csv", index=False)
    print("\nDANN budget series (symmetric - tree):")
    print(tab.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    runs.style(ax)
    ax.axhline(0, color=runs.INK_2, linewidth=0.8)
    ax.errorbar(tab["evals_per_body"], tab["gap"], yerr=[tab["gap"] - tab["ci_lo"], tab["ci_hi"] - tab["gap"]],
                color=runs.GENOME_COLOR["tree_symmetric"], linewidth=2, marker="o", markersize=6, capsize=3)
    for r in tab.itertuples():
        ax.annotate(f"n={r.n}", (r.evals_per_body, r.ci_hi), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=7, color=runs.INK_2)
    ax.set_xscale("log")
    ax.set_xticks(tab["evals_per_body"], [str(v) for v in tab["evals_per_body"]])
    ax.minorticks_off()
    ax.set_xlabel("evaluations per body (inner CMA-ES)", fontsize=9, color=runs.INK_2)
    ax.set_ylabel("score gap, symmetric − tree", fontsize=9, color=runs.INK_2)
    ax.set_title("DANN: symmetric − tree gap vs training budget", fontsize=10, color=runs.INK)
    fig.savefig(out / "fig1b_budget_gap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-root", type=Path)
    parser.add_argument("--budget-roots", type=Path, nargs="+", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or (args.grid_root or args.budget_roots[0]) / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.grid_root:
        g = runs.load_runs(args.grid_root)
        grid(g, out)
        frames.append(g.assign(experiment="grid"))
    if args.budget_roots:
        b = runs.load_runs(*args.budget_roots)
        budget(b, out)
        frames.append(b.assign(experiment="budget"))
    if frames:
        pd.concat(frames).to_csv(out / "e0_runs.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
