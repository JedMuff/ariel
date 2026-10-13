"""
Plots of rowing_detect.py output: does the symmetric genome converge on a
rowing gait over the generations?

Input: <stride-root>/rowing/individuals.csv (one row per evaluated
individual, with its rowing_label). The detector was validated against 170
hand labels (agreement 0.94, see ROWING.md).

Writes to <stride-root>/rowing/plots/:
  rowing_share_over_gens.png
    cols = stride; rows = share of the population per generation that is
      rowing (of all individuals),
      rowing among judgeable individuals (leaving out unpaired and static,
        whose gait the detector can't call),
      alternating (of all individuals).
    Per genome, mean ± 1 SEM across reps, with each rep as a thin line.
  label_composition.png
    rows = genome, cols = stride: stacked share of each gait class per
    generation (3 reps pooled): rowing, alternating, other gait (mixed /
    aperiodic), can't judge (unpaired / static).
  rowing_share_by_gen.csv
    per run x gen: n, the counts and shares behind the plots.
  rowing_early_late.csv
    per genome x stride x rep: rowing share over the first and last
    EARLY_LATE_GENS generations, and the change. Printed too.

Usage:
    python plot_rowing_stride_sweep.py [--stride-root ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GENOMES = ("tree", "tree_symmetric")
GENOME_LABEL = {"tree": "tree", "tree_symmetric": "tree (symmetric)"}
GENOME_COLOR = {"tree": "#2a78d6", "tree_symmetric": "#eb6834"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CI_ALPHA = 0.18
EARLY_LATE_GENS = 5
# Gait classes for the composition plot, bottom to top: detector labels -> class.
CLASSES = {"rowing": ("rowing",), "alternating": ("alternating",), "other gait": ("mixed", "aperiodic"),
           "can't judge": ("unpaired", "static")}
CLASS_COLOR = {"rowing": "#1baf7a", "alternating": "#4a3aa7", "other gait": "#eda100", "can't judge": "#d6d4cd"}
JUDGEABLE = ("rowing", "alternating", "mixed", "aperiodic")


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def stride_title(stride: int) -> str:
    return f"stride {stride} (~{500 / stride:.0f} Hz control)"


def shares_by_gen(ind: pd.DataFrame) -> pd.DataFrame:
    """Per run x gen: counts of each detector label, and the shares plotted."""
    counts = (ind.groupby(["run", "genome", "stride", "rep", "gen"])["rowing_label"].value_counts()
              .unstack(fill_value=0).reset_index())
    labels = [c for c in ("rowing", "alternating", "mixed", "aperiodic", "unpaired", "static") if c in counts]
    for c in ("rowing", "alternating", "mixed", "aperiodic", "unpaired", "static"):
        if c not in counts:
            counts[c] = 0
    counts["n"] = counts[labels].sum(axis=1)
    counts["n_judgeable"] = counts[list(JUDGEABLE)].sum(axis=1)
    counts["rowing_share"] = counts["rowing"] / counts["n"]
    counts["alternating_share"] = counts["alternating"] / counts["n"]
    counts["rowing_share_judgeable"] = counts["rowing"] / counts["n_judgeable"].where(counts["n_judgeable"] > 0)
    for cls, labs in CLASSES.items():
        counts[f"class_{cls}"] = counts[list(labs)].sum(axis=1) / counts["n"]
    return counts.sort_values(["genome", "stride", "rep", "gen"]).reset_index(drop=True)


def gen_curve(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Per-rep share per gen -> gens, mean and SEM across reps, and the per-rep table."""
    per_rep = df.pivot_table(index="gen", columns="rep", values=col)
    sem = per_rep.apply(lambda r: stats.sem(r.dropna()) if r.notna().sum() > 1 else 0.0, axis=1)
    return per_rep.index.to_numpy(), per_rep.mean(axis=1).to_numpy(), sem.to_numpy(), per_rep


def plot_shares(sh: pd.DataFrame, out: Path, dpi: int) -> None:
    strides = sorted(sh["stride"].unique())
    rows = [("rowing_share", "rowing\n(share of all individuals)"),
            ("rowing_share_judgeable", "rowing\n(share of judgeable individuals)"),
            ("alternating_share", "alternating\n(share of all individuals)")]
    fig, axes = plt.subplots(len(rows), len(strides), figsize=(4.2 * len(strides), 3.0 * len(rows)),
                             sharex=True, sharey=True, squeeze=False)
    for r, (col, ylabel) in enumerate(rows):
        for c, stride in enumerate(strides):
            ax = axes[r, c]
            style(ax)
            for genome in GENOMES:
                df = sh[(sh["stride"] == stride) & (sh["genome"] == genome)]
                if df.empty:
                    continue
                gens, m, sem, per_rep = gen_curve(df, col)
                color = GENOME_COLOR[genome]
                for rep in per_rep.columns:
                    ax.plot(per_rep.index, per_rep[rep], color=color, linewidth=0.8, alpha=0.35)
                ax.fill_between(gens, m - sem, m + sem, color=color, alpha=CI_ALPHA, linewidth=0)
                ax.plot(gens, m, color=color, linewidth=2, label=f"{GENOME_LABEL[genome]} ({per_rep.shape[1]} reps)")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            if r == 0:
                ax.set_title(stride_title(stride), fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
            if r == len(rows) - 1:
                ax.set_xlabel("generation", fontsize=9, color=INK_2)
    axes[0, 0].legend(fontsize=8.5, frameon=False, labelcolor=INK_2, loc="upper left")
    fig.suptitle("Rowing gait over evolution: share of each generation's population "
                 "(thick = mean ± 1 SEM across reps, thin = each rep)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_composition(sh: pd.DataFrame, out: Path, dpi: int) -> None:
    strides = sorted(sh["stride"].unique())
    fig, axes = plt.subplots(len(GENOMES), len(strides), figsize=(4.2 * len(strides), 3.2 * len(GENOMES)),
                             sharex=True, sharey=True, squeeze=False)
    cols = [f"class_{c}" for c in CLASSES]
    for r, genome in enumerate(GENOMES):
        for c, stride in enumerate(strides):
            ax = axes[r, c]
            style(ax)
            ax.grid(False)
            df = sh[(sh["stride"] == stride) & (sh["genome"] == genome)]
            if df.empty:
                continue
            # Pool the reps: weight each rep's shares by its population size.
            w = df[cols].mul(df["n"], axis=0).assign(gen=df["gen"], n=df["n"]).groupby("gen").sum()
            frac = w[cols].div(w["n"], axis=0)
            ax.stackplot(frac.index, frac.T.to_numpy(), colors=[CLASS_COLOR[k] for k in CLASSES],
                         edgecolor="white", linewidth=1.0)
            # Direct labels at the right edge, mid-band, where the band is tall enough to read.
            last = frac.iloc[-1]
            bottom = 0.0
            for cls, col in zip(CLASSES, cols):
                h = last[col]
                if h >= 0.10:
                    ax.text(frac.index[-1] - 0.5, bottom + h / 2, f"{cls} {h:.0%}", ha="right", va="center",
                            fontsize=7.5, color="white" if cls in ("rowing", "alternating") else INK)
                bottom += h
            ax.set_xlim(frac.index[0], frac.index[-1])
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            if r == 0:
                ax.set_title(stride_title(stride), fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"{GENOME_LABEL[genome]}\nshare of population", fontsize=9, color=INK_2)
            if r == len(GENOMES) - 1:
                ax.set_xlabel("generation", fontsize=9, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[k]) for k in CLASSES]
    labels = ["rowing", "alternating", "other gait (mixed / aperiodic)", "can't judge (unpaired / static)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Gait classes over evolution (3 reps pooled; labels: share in the final generation)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def early_late(sh: pd.DataFrame) -> pd.DataFrame:
    """Rowing share over the first and last EARLY_LATE_GENS gens, per rep (counts pooled within the window)."""
    g0, g1 = sh["gen"].min(), sh["gen"].max()
    rows = []
    for (genome, stride, rep), df in sh.groupby(["genome", "stride", "rep"]):
        early = df[df["gen"] < g0 + EARLY_LATE_GENS]
        late = df[df["gen"] > g1 - EARLY_LATE_GENS]
        e, l = early["rowing"].sum() / early["n"].sum(), late["rowing"].sum() / late["n"].sum()
        ej = early["rowing"].sum() / max(early["n_judgeable"].sum(), 1)
        lj = late["rowing"].sum() / max(late["n_judgeable"].sum(), 1)
        rows.append({"genome": genome, "stride": stride, "rep": rep, "early": e, "late": l, "change": l - e,
                     "early_judgeable": ej, "late_judgeable": lj, "n_early": int(early["n"].sum()),
                     "n_late": int(late["n"].sum())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
    parser.add_argument("--rowing-dir", type=Path, default=None, help="default: <stride-root>/rowing")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    rdir = args.rowing_dir or args.stride_root / "rowing"
    out = rdir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    ind = pd.read_csv(rdir / "individuals.csv")
    sh = shares_by_gen(ind)
    sh.to_csv(out / "rowing_share_by_gen.csv", index=False)
    plot_shares(sh, out / "rowing_share_over_gens.png", args.dpi)
    plot_composition(sh, out / "label_composition.png", args.dpi)

    el = early_late(sh)
    el.to_csv(out / "rowing_early_late.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nRowing share, first vs last {EARLY_LATE_GENS} generations, per rep:")
    print(el.to_string(index=False, float_format="%.2f"))
    print("\nMean over reps:")
    print(el.groupby(["genome", "stride"])[["early", "late", "change", "early_judgeable", "late_judgeable"]]
          .mean().to_string(float_format="%.2f"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
