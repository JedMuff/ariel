"""Publication figures for part A: one standalone image per graph, legends as
separate images, large fonts, major + minor gridlines, no titles, saved as
vector PDF and high-resolution PNG.

Reads only the CSVs that parta_report.py writes (so it re-renders in seconds):
    <report-dir>/parta_curves_per_run.csv
    <report-dir>/parta_included_runs.csv

Every graph is made twice, summarising runs by their mean or their median
(<stat> below):
  - mean:   curves = mean with a 95% CI band (normal approximation);
            end-of-run = box plots of mean (line), 95% CI (box), +-1 SD (whiskers)
  - median: curves = median with a bootstrap 95% CI band;
            end-of-run = box plots (median, IQR, 1.5 IQR whiskers)

Output tree (pdf/ and png/ mirror each other, so a LaTeX project can use
\\graphicspath{{figures/pdf/}} and \\includegraphics{mean/displacement/best_so_far_x05}):
    <out-dir>/{pdf,png}/legend/schemes_{horizontal,vertical}
    <out-dir>/{pdf,png}/<stat>/displacement/{survivor_mean,best_so_far}_x{00,05,10}
    <out-dir>/{pdf,png}/<stat>/inheritance/{pre_learning,learning_gain}_x{00,05,10}
    <out-dir>/{pdf,png}/<stat>/final/{best_so_far,survivor_mean}_x{00,05,10}
    <out-dir>/manifest.csv

Usage:
    uv run examples/d_social_learning/analysis/paper_figures.py \
        --report-dir __data__/social/figures/parta/report \
        --out-dir __data__/social/figures/parta/paper
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from parta_report import SCHEME_COLORS, X_VALUES, ci95, schemes_for  # noqa: E402

# Display names; *_many schemes average the k=5 donors (core/inheritance.py).
SCHEME_LABELS = {
    "darwinian": "Darwinian",
    "lamarckian": "Lamarckian",
    "random": "Random",
    "random_many": "Random (k=5)",
    "best": "Best",
    "best_many": "Best (k=5)",
    "similar": "Similar",
    "similar_many": "Similar (k=5)",
}

PAPER_RC = {
    "figure.figsize": (6.4, 4.8),
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "lines.linewidth": 2.5,
    "pdf.fonttype": 42,  # embed TrueType so text stays text in the PDF
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# (column, output name, y-label)
CURVES = {
    "displacement": [
        ("surv_disp", "survivor_mean", "Survivor mean\ndisplacement (m)"),
        ("best_so_far_disp", "best_so_far", "Best displacement\nso far (m)"),
    ],
    "inheritance": [
        ("born_init_disp", "pre_learning", "Offspring displacement\nbefore learning (m)"),
        ("born_learn_gain", "learning_gain", "Lifetime-learning\ngain (m)"),
    ],
}
FINAL = [
    ("best_so_far_disp", "best_so_far", "Best displacement\nfound (m)"),
    ("surv_disp", "survivor_mean", "Survivor mean\ndisplacement (m)"),
]
STATS = ("mean", "median")
N_BOOT = 2000

X_DESC = {0.0: "x=0 (novelty only)", 0.5: "x=0.5 (blend)", 1.0: "x=1 (displacement only)"}


def label(scheme: str) -> str:
    base = scheme.removesuffix("_DESCR").removesuffix("_STRUCT").removesuffix("_TOPDOWN")
    return SCHEME_LABELS.get(base, scheme)


def x_tag(xv: float) -> str:
    return f"x{int(round(xv * 10)):02d}"


def style_grid(ax: plt.Axes) -> None:
    ax.minorticks_on()
    ax.grid(which="major", lw=0.9, alpha=0.5)
    ax.grid(which="minor", lw=0.6, ls=":", alpha=0.35)
    ax.set_axisbelow(True)


def center_band(v: np.ndarray, stat: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """(center, lo, hi): mean with a 95% CI, or median with a bootstrap 95% CI."""
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    if stat == "mean":
        m, c = v.mean(), ci95(v)
        return m, m - c, m + c
    boot = np.median(rng.choice(v, size=(N_BOOT, len(v))), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.median(v)), lo, hi


class Saver:
    def __init__(self, out_dir: Path, dpi: int) -> None:
        self.out_dir, self.dpi, self.rows = out_dir, dpi, []

    def __call__(self, fig: plt.Figure, rel: str, **meta) -> None:
        for fmt in ("pdf", "png"):
            path = self.out_dir / fmt / f"{rel}.{fmt}"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        self.rows.append(dict(file=rel, **meta))
        print(f"Saved {rel}")

    def write_manifest(self) -> None:
        keys = ["file", "kind", "stat", "metric", "x", "ylabel", "n_runs", "caption"]
        with open(self.out_dir / "manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: r.get(k, "") for k in keys} for r in self.rows])
        print(f"Saved {self.out_dir / 'manifest.csv'}")


def n_runs_str(n_reps: pd.Series, schemes: list[str], xv: float) -> str:
    return "; ".join(f"{label(s)}={n_reps.get((s, xv), 0)}" for s in schemes)


def plot_curves(curves: pd.DataFrame, schemes: list[str], n_reps: pd.Series, save: Saver,
                stat: str) -> None:
    gens = int(curves.gen.max())
    rng = np.random.default_rng(0)
    for group, panels in CURVES.items():
        for col, name, ylabel in panels:
            for xv in X_VALUES:
                fig, ax = plt.subplots()
                for scheme in schemes:
                    s = curves[(curves.scheme == scheme) & (curves.x == xv)]
                    if s.empty:
                        continue
                    band = pd.DataFrame(
                        [center_band(v.to_numpy(dtype=float), stat, rng) for _, v in s.groupby("gen")[col]],
                        index=sorted(s.gen.unique()), columns=["c", "lo", "hi"],
                    ).dropna()
                    color = SCHEME_COLORS[scheme]
                    ax.fill_between(band.index, band.lo, band.hi, color=color, alpha=0.12, lw=0)
                    ax.plot(band.index, band.c, color=color)
                ax.set_xlim(0 if col.startswith(("surv", "best")) else 1, gens)
                ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
                ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
                ax.set_xlabel("Generation")
                ax.set_ylabel(ylabel)
                style_grid(ax)
                save(fig, f"{stat}/{group}/{name}_{x_tag(xv)}", kind="curve", stat=stat, metric=col, x=xv,
                     ylabel=ylabel.replace("\n", " "), n_runs=n_runs_str(n_reps, schemes, xv),
                     caption=f"{ylabel.replace(chr(10), ' ')} per generation, {X_DESC[xv]}; "
                             + ("mean over runs, shaded 95% CI." if stat == "mean"
                                else "median over runs, shaded bootstrap 95% CI."))


def plot_final(curves: pd.DataFrame, schemes: list[str], n_reps: pd.Series,
               save: Saver, stat: str) -> None:
    gens = int(curves.gen.max())
    final = curves[curves.gen == gens]
    rng = np.random.default_rng(0)
    for col, name, ylabel in FINAL:
        for xv in X_VALUES:
            fx = final[final.x == xv]
            data = [fx[fx.scheme == s][col].dropna().to_numpy() for s in schemes]
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            box_style = dict(widths=0.6, showfliers=False, patch_artist=True,
                             medianprops=dict(color="black", lw=2), whiskerprops=dict(lw=1.4),
                             capprops=dict(lw=1.4), boxprops=dict(lw=1.4))
            if stat == "median":
                bp = ax.boxplot(data, **box_style)
            else:
                # Same glyph, mean-based: line = mean, box = 95% CI, whiskers = mean +- 1 SD.
                boxes = []
                for d in data:
                    m, lo, hi = center_band(d, "mean", rng)
                    sd = d.std(ddof=1) if len(d) > 1 else 0.0
                    boxes.append(dict(med=m, q1=lo, q3=hi, whislo=m - sd, whishi=m + sd, fliers=[]))
                bp = ax.bxp(boxes, **box_style)
            for patch, s in zip(bp["boxes"], schemes):
                patch.set_facecolor(SCHEME_COLORS[s])
                patch.set_alpha(0.4)
            for i, (d, s) in enumerate(zip(data, schemes)):
                ax.scatter(i + 1 + rng.uniform(-0.17, 0.17, len(d)), d, s=22,
                           color=SCHEME_COLORS[s], zorder=3, lw=0)
            ax.set_xticks(range(1, len(schemes) + 1))
            ax.set_xticklabels([label(s) for s in schemes], rotation=45, ha="right",
                               rotation_mode="anchor")
            ax.set_ylabel(ylabel)
            style_grid(ax)
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.grid(axis="x", which="both", visible=False)
            save(fig, f"{stat}/final/{name}_{x_tag(xv)}", kind="boxplot",
                 stat=stat, metric=col, x=xv,
                 ylabel=ylabel.replace("\n", " "),
                 n_runs=n_runs_str(n_reps, schemes, xv),
                 caption=f"{ylabel.replace(chr(10), ' ')} at generation {gens}, {X_DESC[xv]}; one point per run; "
                         + ("line = mean, box = 95% CI of the mean, whiskers = mean ± 1 SD." if stat == "mean"
                            else "box = median and IQR, whiskers 1.5 IQR."))


def plot_legends(schemes: list[str], save: Saver) -> None:
    handles = [
        (Line2D([], [], color=SCHEME_COLORS[s], lw=3),
         Patch(facecolor=SCHEME_COLORS[s], alpha=0.12, lw=0))
        for s in schemes
    ]
    labels = [label(s) for s in schemes]
    for name, ncol in (("horizontal", 4), ("vertical", 1)):
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.legend(
            [(patch, line) for line, patch in handles], labels, ncol=ncol, frameon=False,
            handlelength=2.2, columnspacing=1.5, loc="center",
            handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=1)},
        )
        save(fig, f"legend/schemes_{name}", kind="legend")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="__data__/social/figures/parta/report")
    parser.add_argument("--out-dir", default="__data__/social/figures/parta/paper")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--run-novelty", default="STRUCT", choices=["DESCR", "STRUCT", "TOPDOWN"])
    args = parser.parse_args()

    report = Path(args.report_dir)
    curves = pd.read_csv(report / "parta_curves_per_run.csv")
    runs = pd.read_csv(report / "parta_included_runs.csv")
    n_reps = runs.groupby(["scheme", "x"]).size()
    schemes = schemes_for(args.run_novelty)

    save = Saver(Path(args.out_dir), args.dpi)
    with plt.rc_context(PAPER_RC):
        plot_legends(schemes, save)
        for stat in STATS:
            plot_curves(curves, schemes, n_reps, save, stat)
            plot_final(curves, schemes, n_reps, save, stat)
    save.write_manifest()


if __name__ == "__main__":
    main()
