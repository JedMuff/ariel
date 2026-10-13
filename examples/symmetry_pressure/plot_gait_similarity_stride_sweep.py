"""
Plots of stride_gait_similarity.py output: does the symmetric genome
converge on a shared gait?

Writes to <stride-root>/gait_similarity/:
  gait_diversity_over_gens.png
    rows = gait distance (cycle_shape, cycle), cols = stride; per genome the
    mean pairwise gait distance within each generation's population (the
    behavioural counterpart of the tree-edit-distance diversity curves),
    mean ± 1 SEM across reps.
  medoid_spread.png
    rows = gait distance, cols = stride; per genome the distances from the
    pooled final generation's medoid (smallest median distance to the rest)
    to every other individual in the pool: box = IQR, line = median, dots =
    individuals. Mann-Whitney U p-value tree vs tree_symmetric in each panel.
  medoid_spread.csv
    the medoid-spread numbers behind the box plot, per genome x stride x distance.
Static versions of the stride_gait_page.py views (same MDS layouts and groups):
  medoid_overview.png
    the page's overview table: per experiment the medoid's IQR (bar), median
    (tick) and the pool's all-pairs median (diamond), for both distances.
  pool_maps_<variant>.png
    rows = genome, cols = stride: 2D MDS map of the pooled final generation,
    medoid and the nearest / in-between / farthest of its run highlighted.
  medoid_strips_<variant>.png
    one row per experiment on a shared axis: distance from the medoid to every
    pooled individual, IQR band and median.

Usage:
    python plot_gait_similarity_stride_sweep.py [--stride-root ...]
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
VARIANT_LABEL = {"cycle_shape": "gait-cycle shape distance\n(unit-RMS cycles, dimensionless)",
                 "cycle": "gait-cycle distance (mm)"}
VARIANT_SCALE = {"cycle_shape": 1.0, "cycle": 1000.0}  # cycle shown in mm
CI_ALPHA = 0.18


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


def gen_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-rep mean pairwise distance per gen -> mean and SEM across reps."""
    per_rep = df.pivot_table(index="gen", columns="rep", values="mean")
    sem = per_rep.apply(lambda r: stats.sem(r.dropna()) if r.notna().sum() > 1 else 0.0, axis=1)
    return per_rep.index.to_numpy(), per_rep.mean(axis=1).to_numpy(), sem.to_numpy()


def plot_diversity(summary: pd.DataFrame, out: Path, dpi: int) -> None:
    strides = sorted(summary["stride"].unique())
    variants = [v for v in VARIANT_LABEL if v in set(summary["variant"])]
    fig, axes = plt.subplots(len(variants), len(strides), figsize=(4.2 * len(strides), 3.4 * len(variants)),
                             sharex=True, sharey="row", squeeze=False)
    for r, variant in enumerate(variants):
        for c, stride in enumerate(strides):
            ax = axes[r, c]
            style(ax)
            for genome in GENOMES:
                df = summary[(summary["variant"] == variant) & (summary["stride"] == stride)
                             & (summary["genome"] == genome)]
                if df.empty:
                    continue
                gens, m, sem = gen_curve(df)
                k = VARIANT_SCALE[variant]
                n_reps = df["rep"].nunique()
                ax.fill_between(gens, (m - sem) * k, (m + sem) * k, color=GENOME_COLOR[genome],
                                alpha=CI_ALPHA, linewidth=0)
                ax.plot(gens, m * k, color=GENOME_COLOR[genome], linewidth=2,
                        label=f"{GENOME_LABEL[genome]} ({n_reps} reps)")
            if r == 0:
                ax.set_title(stride_title(stride), fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"mean pairwise {VARIANT_LABEL[variant]}", fontsize=9, color=INK_2)
            if r == len(variants) - 1:
                ax.set_xlabel("generation", fontsize=9, color=INK_2)
    axes[0, 0].legend(fontsize=8.5, frameon=False, labelcolor=INK_2)
    fig.suptitle("Behavioural diversity: mean pairwise gait distance within each generation "
                 "(lower = more similar gaits; mean ± 1 SEM across reps)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def medoid_distances(data_dir: Path, medoids: pd.DataFrame, genome: str, stride: int, variant: str) -> np.ndarray | None:
    row = medoids[(medoids["scope"] == "pool") & (medoids["genome"] == genome)
                  & (medoids["stride"] == stride) & (medoids["variant"] == variant)]
    path = data_dir / f"pool_{genome}_s{stride}_{variant}.csv"
    if row.empty or not path.exists():
        return None
    label = row["medoid_label"].iloc[0]
    return pd.read_csv(path, index_col=0).loc[label].drop(label).to_numpy()


def plot_medoid_spread(data_dir: Path, medoids: pd.DataFrame, out: Path, dpi: int) -> pd.DataFrame:
    pool = medoids[medoids["scope"] == "pool"]
    strides = sorted(pool["stride"].unique())
    variants = [v for v in VARIANT_LABEL if v in set(pool["variant"])]
    rng = np.random.default_rng(0)
    rows = []
    fig, axes = plt.subplots(len(variants), len(strides), figsize=(3.4 * len(strides), 3.8 * len(variants)),
                             sharey="row", squeeze=False)
    for r, variant in enumerate(variants):
        k = VARIANT_SCALE[variant]
        for c, stride in enumerate(strides):
            ax = axes[r, c]
            style(ax)
            dists, ticks = {}, [GENOME_LABEL[g] for g in GENOMES]
            for x, genome in enumerate(GENOMES):
                d = medoid_distances(data_dir, medoids, genome, stride, variant)
                if d is None:
                    continue
                d = d * k
                dists[genome] = d
                color = GENOME_COLOR[genome]
                q25, q50, q75 = np.percentile(d, [25, 50, 75])
                ax.add_patch(plt.Rectangle((x - 0.28, q25), 0.56, q75 - q25, facecolor=color, alpha=0.18,
                                           edgecolor=color, linewidth=1.2))
                ax.plot([x - 0.28, x + 0.28], [q50, q50], color=color, linewidth=2.2, solid_capstyle="round")
                ax.scatter(x + rng.uniform(-0.2, 0.2, len(d)), d, s=12, color=color, alpha=0.75,
                           edgecolors="white", linewidths=0.5, zorder=3)
                ticks[x] += f"\nmedian {q50:.2f}\nIQR {q75 - q25:.2f}"
                rows.append({"variant": variant, "stride": stride, "genome": genome, "n_others": len(d),
                             "median": q50, "q25": q25, "q75": q75, "iqr": q75 - q25})
            title = stride_title(stride) if r == 0 else ""
            if len(dists) == 2:
                p = stats.mannwhitneyu(dists["tree"], dists["tree_symmetric"]).pvalue
                title += ("\n" if title else "") + f"Mann-Whitney p = {p:.2g}"
            ax.set_title(title, fontsize=9.5, color=INK)
            ax.set_xticks(range(len(GENOMES)), ticks, fontsize=8.5, color=INK_2)
            ax.set_xlim(-0.6, len(GENOMES) - 0.4)
            ax.tick_params(axis="x", length=0)
            if c == 0:
                ax.set_ylabel(f"medoid's {VARIANT_LABEL[variant]}\nto every other individual", fontsize=9, color=INK_2)
    fig.suptitle("Spread around the most central gait: final generation, 3 reps pooled "
                 "(box = IQR, line = median)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return pd.DataFrame(rows)


# ── Static versions of the stride_gait_page.py views ─────────────────────────

GROUP_COLOR = {"medoid": INK, "near": "#2a78d6", "mid": "#1baf7a", "far": "#eb6834", "run": "#b5b3ac", "pool": "#b5b3ac"}
GROUP_ORDER = ["medoid", "near", "mid", "far", "run", "pool"]
HIGHLIGHT = {"medoid", "near", "mid", "far"}


def group_label(group: str, k: int) -> str:
    return {"medoid": "medoid (most central gait)", "near": f"{k} nearest in its run", "mid": "in between",
            "far": f"{k} farthest in its run", "run": "rest of its run", "pool": "other reps"}[group]


def short_label(meta: pd.DataFrame, label: str, medoid: str) -> str:
    return f"b{int(meta.loc[label, 'body_idx']):02d}" if meta.loc[label, "run"] == meta.loc[medoid, "run"] else label


def experiment_title(genome: str, stride: int) -> str:
    return f"{GENOME_LABEL[genome]}, stride {stride}"


def plot_overview(medoids: pd.DataFrame, out: Path, dpi: int) -> None:
    """The page's overview table: per experiment the medoid's IQR (bar) and
    median (tick) distance to the pool, and the all-pairs median (diamond)."""
    pool = medoids[medoids["scope"] == "pool"]
    strides = sorted(pool["stride"].unique())
    variants = [v for v in VARIANT_LABEL if v in set(pool["variant"])]
    rows = [(s, g) for s in strides for g in GENOMES]
    fig, axes = plt.subplots(1, len(variants), figsize=(6.4 * len(variants), 0.42 * len(rows) + 1.6), squeeze=False)
    for ax, variant in zip(axes[0], variants):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        k = VARIANT_SCALE[variant]
        hi = pool.loc[pool["variant"] == variant, ["medoid_q75", "pairs_median"]].to_numpy().max() * k
        for y, (stride, genome) in enumerate(rows):
            r = pool[(pool["variant"] == variant) & (pool["stride"] == stride) & (pool["genome"] == genome)]
            if r.empty:
                continue
            r = r.iloc[0]
            color = GENOME_COLOR[genome]
            ax.plot([r["medoid_q25"] * k, r["medoid_q75"] * k], [y, y], color=color, linewidth=7,
                    solid_capstyle="round", alpha=0.45)
            ax.plot([r["medoid_median"] * k] * 2, [y - 0.28, y + 0.28], color=INK, linewidth=2)
            ax.scatter([r["pairs_median"] * k], [y], marker="D", s=26, facecolor="white", edgecolor=MUTED,
                       linewidth=1.2, zorder=3)
            ax.text(hi * 1.04, y, f"median {r['medoid_median'] * k:.2f} · IQR {r['medoid_iqr'] * k:.2f}",
                    va="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(range(len(rows)), [f"s{s}  {GENOME_LABEL[g]}" for s, g in rows], fontsize=9, color=INK_2)
        for y in range(2, len(rows), 2):
            ax.axhline(y - 0.5, color=GRID, linewidth=0.8)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(0, hi * 1.42)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"medoid's {VARIANT_LABEL[variant].replace(chr(10), ' ')} to the rest of the pool",
                      fontsize=9, color=INK_2)
    handles = [plt.Line2D([], [], color=GENOME_COLOR[g], linewidth=7, alpha=0.45, label=f"IQR, {GENOME_LABEL[g]}")
               for g in GENOMES]
    handles += [plt.Line2D([], [], color=INK, linewidth=2, label="median"),
                plt.Line2D([], [], marker="D", linestyle="", markerfacecolor="white", markeredgecolor=MUTED,
                           label="all-pairs median (scale)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Spread around the most central gait (final generation, 3 reps pooled)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _place_labels(ax: plt.Axes, points: list[tuple[float, float, float, str, bool]]) -> None:
    """Greedy label placement (as on the page): right, left, above, below,
    keeping clear of markers and earlier labels. points: (x, y, radius_px, text, bold)."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    placed = []
    for x, y, r, *_ in points:
        px, py = to_px((x, y))
        placed.append((px - r, px + r, py - r, py + r))
    char_w, h = 5.4 * fig.dpi / 100, 9 * fig.dpi / 100
    for x, y, r, text, bold in points:
        px, py = to_px((x, y))
        w, r = len(text) * char_w, r + 2
        options = [((r, 0), "left", "center", (px + r, px + r + w, py - h / 2, py + h / 2)),
                   ((-r, 0), "right", "center", (px - r - w, px - r, py - h / 2, py + h / 2)),
                   ((0, r), "center", "bottom", (px - w / 2, px + w / 2, py + r, py + r + h)),
                   ((0, -r), "center", "top", (px - w / 2, px + w / 2, py - r - h, py - r))]
        free = [o for o in options if not any(o[3][0] < b[1] and o[3][1] > b[0] and o[3][2] < b[3] and o[3][3] > b[2]
                                               for b in placed)]
        (dx, dy), ha, va, box = (free or options)[0]
        placed.append(box)
        ax.annotate(text, (x, y), xytext=(dx * 72 / fig.dpi, dy * 72 / fig.dpi), textcoords="offset points",
                    ha=ha, va=va, fontsize=7.5, color=INK if bold else INK_2, fontweight="bold" if bold else "normal")


def plot_pool_maps(layouts: dict, variant: str, k: int, out: Path, dpi: int) -> None:
    """The page's MDS maps, one per genome x stride."""
    strides = sorted({s for _, s in layouts})
    fig, axes = plt.subplots(len(GENOMES), len(strides), figsize=(5.2 * len(strides), 5.0 * len(GENOMES)), squeeze=False)
    scale = VARIANT_SCALE[variant]
    for r, genome in enumerate(GENOMES):
        for c, stride in enumerate(strides):
            ax = axes[r, c]
            style(ax)
            ax.grid(axis="both", color=GRID, linewidth=0.8)
            lay = layouts.get((genome, stride))
            if lay is None:
                ax.set_visible(False)
                continue
            meta, groups, medoid = lay["meta"], lay["groups"], lay["medoid"]
            xy = pd.DataFrame(lay["xy"][variant] * scale, index=lay["labels"], columns=["x", "y"])
            size = lambda lab: 18 + 7 * meta.loc[lab, "n_joints"]  # noqa: E731  -- area grows with hinge count
            for lab in groups:
                if groups[lab] == "near":
                    ax.plot(*xy.loc[[medoid, lab]].to_numpy().T, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            for group in reversed(GROUP_ORDER):
                labs = [lab for lab, g in groups.items() if g == group]
                if not labs:
                    continue
                s = [size(lab) * (1.8 if group == "medoid" else 1) for lab in labs]
                if group == "pool":
                    ax.scatter(xy.loc[labs, "x"], xy.loc[labs, "y"], s=s, facecolor="none",
                               edgecolor=GROUP_COLOR[group], linewidth=1.1, zorder=2)
                else:
                    ax.scatter(xy.loc[labs, "x"], xy.loc[labs, "y"], s=s, color=GROUP_COLOR[group],
                               edgecolor="white", linewidth=1.2, zorder=3 if group != "medoid" else 4)
            ax.set_aspect("equal", adjustable="datalim")
            radius_px = lambda lab, g: np.sqrt(size(lab) * (1.8 if g == "medoid" else 1)) / 2 * dpi / 72  # noqa: E731
            order = sorted((lab for lab in groups if groups[lab] in HIGHLIGHT), key=lambda l: GROUP_ORDER.index(groups[l]))
            _place_labels(ax, [(xy.loc[lab, "x"], xy.loc[lab, "y"], radius_px(lab, groups[lab]),
                                ("medoid " if groups[lab] == "medoid" else "") + short_label(meta, lab, medoid),
                                groups[lab] == "medoid") for lab in order])
            unit = " (mm)" if variant == "cycle" else ""
            ax.set_title(f"{experiment_title(genome, stride)}\nmedoid {medoid}, MDS stress-1 {lay['stress'][variant]:.2f}",
                         fontsize=9.5, color=INK)
            ax.set_xlabel("MDS 1" + unit, fontsize=8.5, color=INK_2)
            ax.set_ylabel("MDS 2" + unit, fontsize=8.5, color=INK_2)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                          markerfacecolor="none" if g == "pool" else GROUP_COLOR[g],
                          markeredgecolor=GROUP_COLOR[g] if g == "pool" else "white", label=group_label(g, k))
               for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"2D MDS maps of the pooled final generations: {VARIANT_LABEL[variant].replace(chr(10), ' ')}\n"
                 "groups picked in the medoid's run by gait-shape distance; dashed = medoid to its nearest; "
                 "dot area = hinge count", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_medoid_strips(layouts: dict, variant: str, k: int, out: Path, dpi: int) -> None:
    """The page's distance-from-the-medoid strips, one row per experiment on a shared axis."""
    keys = [(g, s) for s in sorted({s for _, s in layouts}) for g in GENOMES if (g, s) in layouts]
    scale = VARIANT_SCALE[variant]
    hi = max(lay["dists"][variant].loc[lay["medoid"]].max() for lay in layouts.values()) * scale * 1.03
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 1.25 * len(keys) + 1.2), sharex=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, key in zip(axes[:, 0], keys):
        lay = layouts[key]
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        st, medoid, groups = lay["stats"][variant], lay["medoid"], lay["groups"]
        d = lay["dists"][variant].loc[medoid] * scale
        ax.axvspan(st["medoid_q25"] * scale, st["medoid_q75"] * scale, color="#2a78d6", alpha=0.10, linewidth=0)
        ax.axvline(st["medoid_median"] * scale, color=INK_2, linewidth=1.8)
        for group in reversed(GROUP_ORDER):
            labs = [lab for lab, g in groups.items() if g == group]
            if not labs:
                continue
            lane = 0.68 if group in HIGHLIGHT else 0.3
            y = lane + rng.uniform(-0.1, 0.1, len(labs))
            if group == "pool":
                ax.scatter(d[labs], y, s=20, facecolor="none", edgecolor=GROUP_COLOR[group], linewidth=1)
            else:
                ax.scatter(d[labs], y if group != "medoid" else [lane], s=55 if group == "medoid" else 30,
                           color=GROUP_COLOR[group], edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        genome, stride = key
        ax.text(0, 1.02, f"{experiment_title(genome, stride)}   ", transform=ax.transAxes, fontsize=9,
                color=GENOME_COLOR[genome], fontweight="bold", va="bottom")
        ax.text(1, 1.02, f"median {st['medoid_median'] * scale:.2f} · IQR {st['medoid_iqr'] * scale:.2f} · "
                f"n = {st['n']}", transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom", ha="right")
    axes[-1, 0].set_xlim(-0.015 * hi, hi)
    axes[-1, 0].set_xlabel(f"distance to the medoid: {VARIANT_LABEL[variant].replace(chr(10), ' ')}", fontsize=9, color=INK_2)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                          markerfacecolor="none" if g == "pool" else GROUP_COLOR[g],
                          markeredgecolor=GROUP_COLOR[g] if g == "pool" else "white", label=group_label(g, k))
               for g in GROUP_ORDER]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#2a78d6", alpha=0.10, label="IQR (line = median)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Distance from the medoid to every pooled individual (final generation, 3 reps pooled)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
    parser.add_argument("--data-dir", type=Path, default=None, help="default: <stride-root>/gait_similarity")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--k", type=int, default=5, help="as stride_gait_page.py: nearest/farthest highlighted")
    parser.add_argument("--n-mid", type=int, default=3, help="as stride_gait_page.py: highlighted in between")
    args = parser.parse_args()
    data_dir = args.data_dir or args.stride_root / "gait_similarity"

    summary = pd.read_csv(data_dir / "summary.csv")
    plot_diversity(summary, data_dir / "gait_diversity_over_gens.png", args.dpi)
    aperiodic = summary[summary["variant"] == "cycle_shape"].groupby(["genome", "stride"])[["n_aperiodic", "n"]].sum()
    print("Aperiodic episodes (fallback cycle over the whole episode):")
    print((aperiodic["n_aperiodic"].astype(str) + " / " + aperiodic["n"].astype(str)).to_string())

    medoids = pd.read_csv(data_dir / "medoid_summary.csv")
    spread = plot_medoid_spread(data_dir, medoids, data_dir / "medoid_spread.png", args.dpi)
    spread.to_csv(data_dir / "medoid_spread.csv", index=False)
    print(spread.to_string(index=False, float_format="%.3f"))

    # Static versions of the page views (same MDS layouts and groups as stride_gait_page.py).
    from stride_gait_page import pool_layout  # heavy imports (zoo benchmark), only needed here

    plot_overview(medoids, data_dir / "medoid_overview.png", args.dpi)
    strides = sorted(medoids.loc[medoids["scope"] == "pool", "stride"].unique())
    layouts = {}
    for stride in strides:
        for genome in GENOMES:
            lay = pool_layout(data_dir, f"{genome}_s{stride}", args.k, args.n_mid)
            if lay is not None:
                layouts[(genome, int(stride))] = lay
    for variant in ("cycle_shape", "cycle"):
        plot_pool_maps(layouts, variant, args.k, data_dir / f"pool_maps_{variant}.png", args.dpi)
        plot_medoid_strips(layouts, variant, args.k, data_dir / f"medoid_strips_{variant}.png", args.dpi)


if __name__ == "__main__":
    main()
