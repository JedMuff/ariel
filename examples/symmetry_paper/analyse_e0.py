"""
E0: the symmetric advantage and its conditions (R1), paper v2: DANN only.

  hinge grid      evolve.py runs over kp x kv (x armature) (slurm e0_dann_grid.sh,
                  plus the default cell from the budget series' largest arm):
                  paired symmetric - tree score gap per cell at the largest
                  inner budget in the data, pooled over every (cell, seed) pair
  budget series   DANN runs at 200 / 800 / 1600 evaluations per body at the
                  default hinge (e0_dann_budget.sh): paired gap per budget

Score = the best forward score of a run's last generation (runs.run_row; v2:
the retest mean, in metres). v1 data (normalised score, both brains, 18 cells)
still loads; --brains picks the brains to show.

Writes to --out (default <first grid root>/analysis):
  e0_grid_cells.csv, e0_grid_pooled.csv, e0_budget.csv, e0_budget_pairs.csv, e0_runs.csv
  fig1a_grid_gap.png    kp x kv heatmap of the gap per brain (one column per armature)
  fig1b_budget_gap.png  DANN gap vs evaluations per body: paired seeds and mean with 95% CI
  fig1_grid_budget.png  both panels side by side (the paper's Figure 1)

Usage:
    python analyse_e0.py --grid-roots $FINAL_ROOT/e0_grid $FINAL_ROOT/e0_budget/sb80 \\
        --budget-roots $FINAL_ROOT/e0_budget/sb10 $FINAL_ROOT/e0_budget/sb40 $FINAL_ROOT/e0_budget/sb80
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
from matplotlib.patches import Rectangle  # noqa: E402

import runs  # noqa: E402
from sim import ZOO_HINGE  # noqa: E402

CELL = ["brain", "kp", "kv", "armature", "limited"]
# Diverging: orange (tree better) <-> blue (symmetric better) through a neutral gray.
DIVERGING = LinearSegmentedColormap.from_list("tree_sym", ["#c4501f", "#eb6834", "#f4b89c", "#f0efec",
                                                           "#86b6ef", "#2a78d6", "#104281"])
DEFAULT = ZOO_HINGE  # the default hinge (fitted to the physical servo)


def _is_default(kp: float, kv: float, arm: float) -> bool:
    return math.isclose(kp, DEFAULT.kp) and math.isclose(kv, DEFAULT.kv) and math.isclose(arm, DEFAULT.armature)


def grid_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-cell gaps, pooled gap per brain) at each brain's largest budget."""
    top = df.groupby("brain")["evals_per_body"].transform("max")
    df = df[df["evals_per_body"] == top]
    cells = runs.gap_table(df, CELL + ["evals_per_body"])
    pooled = df.assign(pair=df["hinge"] + "_s" + df["seed"].astype(str))
    pooled = runs.gap_table(pooled, ["brain", "evals_per_body"], pair_on="pair")
    pooled["cells_sym_ahead"] = [int((cells.loc[cells["brain"] == b, "gap"] > 0).sum()) for b in pooled["brain"]]
    pooled["n_cells"] = [int((cells["brain"] == b).sum()) for b in pooled["brain"]]
    return cells, pooled


def draw_grid(ax, cells: pd.DataFrame, brain: str, arm: float, norm, unit: str) -> None:  # noqa: ANN001
    kps, kvs = sorted(cells["kp"].unique()), sorted(cells["kv"].unique())
    sub = cells[(cells["brain"] == brain) & np.isclose(cells["armature"], arm)]
    img = np.full((len(kps), len(kvs)), np.nan)
    vmax = norm.vmax
    for s in sub.itertuples():
        i, j = kps.index(s.kp), kvs.index(s.kv)
        img[i, j] = s.gap
        ink = "white" if abs(s.gap) > 0.6 * vmax else runs.INK
        ax.text(j, i + 0.08, f"{s.gap:+.2f}", ha="center", va="center", fontsize=12, color=ink, weight="semibold")
        ax.text(j, i - 0.22, f"sym {s.mean_tree_symmetric:.2f} · tree {s.mean_tree:.2f}\n{s.wins}/{s.n} seeds",
                ha="center", va="center", fontsize=6.5, color=ink, linespacing=1.2)
        if _is_default(s.kp, s.kv, arm):
            ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=runs.INK, linewidth=2))
    ax.imshow(img, cmap=DIVERGING, norm=norm, origin="lower", aspect="auto")
    ax.set_xticks(range(len(kvs)), [f"{v:.3g}" + (" (default)" if math.isclose(v, DEFAULT.kv) else "") for v in kvs])
    ax.set_yticks(range(len(kps)), [f"{v:.3g}" + ("\n(default)" if math.isclose(v, DEFAULT.kp) else "") for v in kps])
    ax.tick_params(length=0, labelsize=8, colors=runs.INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("damping kv", fontsize=9, color=runs.INK_2)
    ax.set_ylabel("gain kp", fontsize=9, color=runs.INK_2)


def grid(df: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, TwoSlopeNorm]:
    cells, pooled = grid_cells(df)
    cells.to_csv(out / "e0_grid_cells.csv", index=False)
    pooled.to_csv(out / "e0_grid_pooled.csv", index=False)
    print("hinge grid, pooled per brain (symmetric - tree, largest budget):")
    print(pooled.round(3).to_string(index=False))

    unit = runs.unit_suffix(df)
    arms = sorted(cells["armature"].unique())
    brains = [b for b in runs.BRAIN_LABEL if b in set(cells["brain"])]
    vmax = float(np.nanmax(np.abs(cells["gap"]))) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(len(brains), len(arms), figsize=(4.0 * len(arms) + 1, 3.6 * len(brains)),
                             squeeze=False, constrained_layout=True)
    for (bi, brain), (ai, arm) in itertools.product(enumerate(brains), enumerate(arms)):
        draw_grid(axes[bi, ai], cells, brain, arm, norm, unit)
        evals = int(cells.loc[cells["brain"] == brain, "evals_per_body"].iloc[0])
        axes[bi, ai].set_title(f"{runs.BRAIN_LABEL[brain]}, {evals} evaluations per body, armature {arm:.2g}",
                               fontsize=9, color=runs.INK)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), ax=axes, shrink=0.8)
    cbar.set_label(f"symmetric − tree{unit}\n(blue: symmetric better)", fontsize=9, color=runs.INK)
    cbar.outline.set_visible(False)
    fig.savefig(out / "fig1a_grid_gap.png", dpi=200)
    plt.close(fig)
    return cells, norm


def budget_pairs(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index=["evals_per_body", "seed"], columns="genome", values="score").dropna()
    return (wide["tree_symmetric"] - wide["tree"]).rename("gap").reset_index()


def draw_budget(ax, df: pd.DataFrame, tab: pd.DataFrame) -> None:  # noqa: ANN001
    runs.style(ax)
    ax.axhline(0, color=runs.INK_2, linewidth=0.8)
    pairs = budget_pairs(df)
    for _, g in pairs.groupby("seed"):
        g = g.sort_values("evals_per_body")
        ax.plot(g["evals_per_body"], g["gap"], color=runs.GRID, linewidth=1, marker="o", markersize=3,
                markerfacecolor=runs.INK_2, markeredgewidth=0, zorder=1)
    ax.errorbar(tab["evals_per_body"], tab["gap"], yerr=[tab["gap"] - tab["ci_lo"], tab["ci_hi"] - tab["gap"]],
                color=runs.GENOME_COLOR["tree_symmetric"], linewidth=2.2, marker="o", markersize=7, capsize=4, zorder=3,
                label="mean, 95% CI")
    for r in tab.itertuples():
        ax.annotate(f"{r.wins}/{r.n}", (r.evals_per_body, r.ci_hi), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=7, color=runs.INK_2)
    ax.set_xscale("log")
    ax.set_xticks(tab["evals_per_body"], [f"{v:,}" for v in tab["evals_per_body"]])
    ax.minorticks_off()
    ax.set_xlabel("evaluations per body (inner CMA-ES)", fontsize=9, color=runs.INK_2)
    ax.set_ylabel(f"symmetric − tree{runs.unit_suffix(df)}", fontsize=9, color=runs.INK_2)
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def budget(df: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[(df["brain"] == "dann") & (df["hinge"] == DEFAULT.label)]
    tab = runs.gap_table(df, ["evals_per_body"])
    tab.to_csv(out / "e0_budget.csv", index=False)
    budget_pairs(df).to_csv(out / "e0_budget_pairs.csv", index=False)
    print("\nDANN budget series (symmetric - tree; wins = seeds where symmetric is ahead):")
    print(tab.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(4.4, 3.4), constrained_layout=True)
    draw_budget(ax, df, tab)
    ax.set_title("DANN, default hinge: gap vs training budget", fontsize=10, color=runs.INK)
    fig.savefig(out / "fig1b_budget_gap.png", dpi=200)
    plt.close(fig)
    return df, tab


def combined(cells: pd.DataFrame, norm, bdf: pd.DataFrame, tab: pd.DataFrame, out: Path) -> None:  # noqa: ANN001
    """Figure 1: (a) DANN grid at the largest budget, default armature; (b) budget series."""
    arm = DEFAULT.armature if np.isclose(cells["armature"], DEFAULT.armature).any() else float(cells["armature"].iloc[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.15, 1]}, constrained_layout=True)
    draw_grid(a, cells, "dann", arm, norm, runs.unit_suffix(bdf))
    evals = int(cells.loc[cells["brain"] == "dann", "evals_per_body"].iloc[0])
    a.set_title(f"(a) symmetric − tree by hinge setting, {evals:,} evaluations per body", fontsize=9, color=runs.INK, loc="left")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), ax=a, shrink=0.85, pad=0.02)
    cbar.set_label(f"symmetric − tree{runs.unit_suffix(bdf)}", fontsize=8, color=runs.INK)
    cbar.outline.set_visible(False)
    draw_budget(b, bdf, tab)
    b.set_title("(b) default hinge: gap vs training budget", fontsize=9, color=runs.INK, loc="left")
    fig.savefig(out / "fig1_grid_budget.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-roots", "--grid-root", type=Path, nargs="+", default=[])
    parser.add_argument("--budget-roots", type=Path, nargs="+", default=[])
    parser.add_argument("--brains", nargs="+", default=["dann"], help="grid brains to show (v1 data also has bang_bang)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or (args.grid_roots or args.budget_roots)[0] / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    frames, cells, b = [], None, None
    if args.grid_roots:
        g = runs.load_runs(*args.grid_roots)
        g = g[g["brain"].isin(args.brains)].drop_duplicates("path")
        cells, norm = grid(g, out)
        frames.append(g.assign(experiment="grid"))
    if args.budget_roots:
        b = runs.load_runs(*args.budget_roots)
        bdf, tab = budget(b, out)
        frames.append(b.assign(experiment="budget"))
    if cells is not None and b is not None and "dann" in set(cells["brain"]):
        combined(cells, norm, bdf, tab, out)
    if frames:
        pd.concat(frames).to_csv(out / "e0_runs.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
