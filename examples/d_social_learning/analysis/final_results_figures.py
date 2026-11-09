r"""Figures, LaTeX tables and quoted numbers for the proposed results section
(overleaf_final_results/final_results_section.tex), from the caches final_results_data.py
writes. Re-renders in seconds.

x in {0, 0.5, 1}. A condition without data (e.g. ARIEL ig20 at x = 0)
is skipped in the figures (the .tex shows a placeholder panel) and printed
as \tbd{} (a highlighted blank, defined in the .tex) in the tables, so
rerunning once the data exists fills both in.

Headline metric: survivor mean displacement at the last generation, per run.
Head start: offspring displacement with the inherited controller, before
learning (born_init_disp); learning gain: what lifetime learning adds to it.

Outputs:
    <out-dir>/{pdf,png}/
        legend/schemes_horizontal
        displacement_<dom>_<x>          Fig 1, survivor mean over generations
        best_so_far_<dom>_<x>           appendix, best displacement so far
        headstart_share_<dom>           Fig 2, head start vs learning share per scheme and x
        headstart_vs_distance_<dom>     Fig 3, one panel per x: scheme means and runs
        diversity_<dom>_<x>             Fig 4, survivor diversity (mean pairwise distance)
        donor_distance_<dom>_<x>        Fig 5, donor-offspring distance
        budget_advantage_ariel_<x>      Fig 6, ig10 (open) vs ig20 (filled)
    <tables-dir>/tab_{advantage,kruskal,novelty_cost}.tex   Tables 1-3
    <out-dir>/stats.csv                 every number the text quotes

Usage (from the repo root):
    uv run examples/d_social_learning/analysis/final_results_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_figures import PAPER_RC, Saver, style_grid, x_tag
from parta_report import a12, ci95
from zoo_budget_compare import boot_mean, holm

FIGURES = Path("__data__/social/figures")
SCHEMES = ["darwinian", "lamarckian", "random", "random_many", "best", "best_many",
           "similar_TOPDOWN", "similar_many_TOPDOWN"]
SOCIAL = SCHEMES[1:]
# k = 5 donors in ARIEL, 8 in EvoGym, so the "-Many" label leaves k open.
LABELS = {"darwinian": "Darwinian", "lamarckian": "Lamarckian", "random": "Random",
          "random_many": "Random-Many", "best": "Best", "best_many": "Best-Many",
          "similar_TOPDOWN": "Similar", "similar_many_TOPDOWN": "Similar-Many"}
# The paper's palette: one hue per scheme, so the k-donor variants need no
# dotted line or hollow marker.
SCHEME_COLORS = {"darwinian": "#808080", "lamarckian": "#9c5a33", "best": "#d1352b", "best_many": "#ef8632",
                 "similar_TOPDOWN": "#4a7cb3", "similar_many_TOPDOWN": "#68ad57", "random": "#8e529f",
                 "random_many": "#e887bd"}
DOMAINS = {"ariel": "ARIEL", "evogym": "EvoGym"}
DIST_LABEL = {"ariel": "Donor–offspring\ntree edit distance", "evogym": "Donor–offspring\naligned Hamming dist."}
X_VALUES = [0.0, 0.5, 1.0]
TBD = r"\tbd{}"


def label(s: str) -> str:
    return LABELS[s]


def tex(s: str) -> str:
    return label(s)


def final(curves: pd.DataFrame, col: str = "surv_disp") -> pd.DataFrame:
    """One value per run at the last generation."""
    return curves[curves.gen == curves.gen.max()][["scheme", "x", "rep", col]]


def values(df: pd.DataFrame, scheme: str, xv: float, col: str) -> np.ndarray:
    return df[(df.scheme == scheme) & (df.x == xv)][col].dropna().to_numpy(dtype=float)


def ci95_series(v: pd.Series) -> float:
    return ci95(v.to_numpy(dtype=float))


def has(df: pd.DataFrame, xv: float) -> bool:
    return bool((df.x == xv).any())


def run_means(curves: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-run mean of a per-generation column over the whole run."""
    return curves.groupby(["scheme", "x", "rep"])[col].mean().reset_index()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_legend(save: Saver) -> None:
    handles = [(Patch(facecolor=SCHEME_COLORS[s], alpha=0.12, lw=0),
                Line2D([], [], color=SCHEME_COLORS[s], lw=3)) for s in SCHEMES]
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.legend(handles, [label(s) for s in SCHEMES], ncol=4, frameon=False, handlelength=2.2,
               columnspacing=1.5, loc="center", handler_map={tuple: HandlerTuple(ndivide=1)})
    save(fig, "legend/schemes_horizontal", kind="legend")


def plot_curve(curves: pd.DataFrame, col: str, ylabel: str, name: str, schemes: list[str],
               save: Saver, ylim: tuple[float, float] | None = None, own_scale: bool = False) -> None:
    """own_scale: the panel has its own y-range instead of the domain's shared
    one, marked by a label so it is not misread.
    """
    fig, ax = plt.subplots()
    if own_scale:
        ax.text(0.03, 0.96, "different y-scale", transform=ax.transAxes, ha="left", va="top", fontsize=14,
                fontstyle="italic", color="0.25",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.5", "lw": 1})
    for s in schemes:
        g = curves[curves.scheme == s].groupby("gen")[col]
        m = g.mean().dropna()
        c = g.agg(lambda v: ci95(v.to_numpy(dtype=float))).reindex(m.index)
        ax.fill_between(m.index, m - c, m + c, color=SCHEME_COLORS[s], alpha=0.12, lw=0)
        ax.plot(m.index, m, color=SCHEME_COLORS[s])
    ax.set_xlim(0, curves.gen.max())
    if ylim:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    style_grid(ax)
    save(fig, name, kind="curve", metric=col)


def shared_ylim(curves: pd.DataFrame, col: str) -> tuple[float, float]:
    """y-range of the mean +- CI over all schemes and x of one domain, so the
    x = 0.5 and x = 1 panels share an axis."""
    g = curves.groupby(["scheme", "x", "gen"])[col]
    m, c = g.mean(), g.agg(lambda v: ci95(v.to_numpy(dtype=float)))
    lo, hi = (m - c).min(), (m + c).max()
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def headstart_share(curves: pd.DataFrame) -> pd.DataFrame:
    """Per run: the share of its offspring's final displacement (generations
    1-60) that the head start gives, i.e. total head start over total final
    displacement, clipped to 0-100% (a run whose inherited controllers do worse
    than standing still on the whole counts as 0%, all from learning);
    learning gives the rest. A share per offspring would blow up for offspring
    that end near zero.
    """
    born = curves[curves.gen > 0]
    g = born.assign(final=born.born_init_disp + born.born_learn_gain).groupby(["scheme", "x", "rep"])
    raw = g.born_init_disp.sum() / g.final.sum()
    return pd.DataFrame({"head_share": raw.clip(0, 1), "head_share_raw": raw}).reset_index()


def plot_headstart_share(share: pd.DataFrame, dom: str, save: Saver) -> None:
    """Stacked bars per scheme and x: head start share (dark) and learning
    share (light), adding up to 100%, with the 95% CI of the head start share
    over runs at the boundary. A condition without data is a dashed outline.
    """
    width, gap = 0.8, 0.9
    fig, ax = plt.subplots(figsize=(15, 2.4))
    centers = []
    for i, s in enumerate(SOCIAL):
        base = i * (len(X_VALUES) * width + gap)
        centers.append(base + width * (len(X_VALUES) - 1) / 2)
        color = SCHEME_COLORS[s]
        for j, xv in enumerate(X_VALUES):
            pos = base + j * width
            v = values(share, s, xv, "head_share")
            ax.text(pos, -0.08, f"{xv:g}", ha="center", va="top", fontsize=12,
                    transform=ax.get_xaxis_transform())
            if not len(v):
                ax.bar(pos, 1, width * 0.92, fill=False, ls="--", lw=1.2, edgecolor="0.6")
                ax.text(pos, 0.5, "?", ha="center", va="center", fontsize=16, color="0.5")
                continue
            m, c = v.mean(), ci95(v)
            ax.bar(pos, m, width * 0.92, color=color, lw=0)
            ax.bar(pos, 1 - m, width * 0.92, bottom=m, color=color, alpha=0.3, lw=0)
            # The shares are clipped to 0-100%, so is the CI
            ax.errorbar(pos, m, yerr=[[min(c, m)], [min(c, 1 - m)]], color="black", capsize=3, lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels([label(s) for s in SOCIAL])
    ax.text(-width * 0.9, -0.08, "$x$:", ha="right", va="top", fontsize=12, transform=ax.get_xaxis_transform())
    ax.tick_params(axis="x", which="both", length=0, pad=26)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of final\ndisplacement")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-width, centers[-1] + width * (len(X_VALUES) + 1) / 2)
    style_grid(ax)
    ax.grid(axis="x", which="both", visible=False)
    ax.tick_params(axis="x", which="minor", bottom=False)
    handles = [Patch(facecolor="0.3", lw=0), Patch(facecolor="0.3", alpha=0.3, lw=0)]
    ax.legend(handles, ["head start (before learning)", "learning gain"], ncol=2, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, f"headstart_share_{dom}", kind="bars")


def plot_headstart_vs_distance(per_run: pd.DataFrame, dom: str, save: Saver) -> dict:
    """Mean donor-offspring distance vs mean head start, one panel per x on
    shared axes: each scheme's mean over runs with its 95% CI on both axes, and
    the runs themselves as faint dots, and a dashed least-squares line. rho in
    the title and the line are over all runs at that x.
    """
    xs = [xv for xv in X_VALUES if has(per_run, xv)]
    fig, axes = plt.subplots(1, len(xs), figsize=(15, 4.4), sharex=True, sharey=True, squeeze=False)
    rho = {}
    for ax, xv in zip(axes[0], xs, strict=False):
        p = per_run[per_run.x == xv]
        for s in SOCIAL:
            q = p[p.scheme == s]
            d, h = q.donor_dist.to_numpy(dtype=float), q.init_fitness.to_numpy(dtype=float)
            ax.scatter(d, h, s=22, color=SCHEME_COLORS[s], alpha=0.22, edgecolors="none", zorder=2)
            ax.errorbar(d.mean(), h.mean(), xerr=ci95(d), yerr=ci95(h), color=SCHEME_COLORS[s], lw=2, zorder=3)
            ax.scatter(d.mean(), h.mean(), s=110, color=SCHEME_COLORS[s], edgecolors="white", linewidths=1.5, zorder=4)
        fit = stats.linregress(p.donor_dist, p.init_fitness)
        ends = np.array([p.donor_dist.min(), p.donor_dist.max()])
        ax.plot(ends, fit.slope * ends + fit.intercept, color="0.2", ls="--", lw=1.8, zorder=2.5)
        rho[xv] = stats.spearmanr(p.donor_dist, p.init_fitness)
        ax.set_title(f"$x = {xv:g}$   ($\\rho = {rho[xv].statistic:.2f}$)", fontsize=17)
        ax.set_xlabel(DIST_LABEL[dom])
        style_grid(ax)
    axes[0, 0].set_ylabel("Head start\n(before learning)")
    fig.tight_layout()
    save(fig, f"headstart_vs_distance_{dom}", kind="scatter")
    return {xv: (r.statistic, r.pvalue) for xv, r in rho.items()}


def distance_headstart_split(per_run: pd.DataFrame, xv: float) -> dict[str, tuple[float, float]]:
    """Spearman (rho, p) of distance vs head start at one x, split into between
    schemes (over the scheme means) and within a scheme (over the runs, each
    measured from its scheme's mean).
    """
    cols = ["donor_dist", "init_fitness"]
    p = per_run[per_run.x == xv]
    by_scheme = p.groupby("scheme")[cols]
    between, within = by_scheme.mean(), p[cols] - by_scheme.transform("mean")
    return {name: tuple(stats.spearmanr(df.donor_dist, df.init_fitness))
            for name, df in (("between", between), ("within", within))}


def distance_headstart_fits(per_run: pd.DataFrame, xv: float) -> dict[str, tuple[float, float]]:
    """Least-squares (slope, R^2) of head start on distance at one x: over all
    runs (the figure's line) and within a scheme (each run measured from its
    scheme's mean).
    """
    cols = ["donor_dist", "init_fitness"]
    p = per_run[per_run.x == xv]
    within = p[cols] - p.groupby("scheme")[cols].transform("mean")
    fits = {"all_runs": stats.linregress(p.donor_dist, p.init_fitness),
            "within": stats.linregress(within.donor_dist, within.init_fitness)}
    return {name: (fit.slope, fit.rvalue ** 2) for name, fit in fits.items()}


def plot_budget_advantage(adv: pd.DataFrame, save: Saver) -> None:
    # Only x values both budgets have (ig20 has no x = 0 runs)
    for xv in filter(lambda xv: all(has(adv[adv.budget == b], xv) for b in ("ig10", "ig20")), X_VALUES):
        a = adv[adv.x == xv].set_index(["budget", "scheme"])
        fig, ax = plt.subplots()
        for i, s in enumerate(SOCIAL):
            for j, (bud, mk) in enumerate((("ig10", "o"), ("ig20", "s"))):
                r = a.loc[bud, s]
                ax.errorbar(i + (j - 0.5) * 0.3, r.rel, yerr=[[r.rel - r.lo], [r.hi - r.rel]], fmt=mk,
                                 color=SCHEME_COLORS[s], mfc="white" if j == 0 else SCHEME_COLORS[s],
                                 ms=10, mew=2, capsize=4, lw=2)
        ax.axhline(0, color=SCHEME_COLORS["darwinian"], lw=2.5)
        ax.set_xticks(range(len(SOCIAL)))
        ax.set_xticklabels([label(s) for s in SOCIAL], rotation=45, ha="right", rotation_mode="anchor")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("Relative advantage\nover Darwinian")
        style_grid(ax)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.grid(axis="x", which="both", visible=False)
        save(fig, f"budget_advantage_ariel_{x_tag(xv)}", kind="advantage", x=xv)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def advantage(fin: pd.DataFrame, col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Per scheme and x: relative advantage of the mean over Darwinian with a
    bootstrap 95% CI, A12 vs Darwinian and a Holm-corrected Mann-Whitney p."""
    rows = []
    for xv in filter(lambda xv: has(fin, xv), X_VALUES):
        d = values(fin, "darwinian", xv, col)
        bd = boot_mean(d, rng)
        block = []
        for s in SOCIAL:
            v = values(fin, s, xv, col)
            lo, hi = np.percentile(boot_mean(v, rng) / bd - 1, [2.5, 97.5])
            block.append(dict(x=xv, scheme=s, rel=v.mean() / d.mean() - 1, lo=lo, hi=hi, a12=a12(v, d),
                              p=stats.mannwhitneyu(v, d).pvalue))
        for r, ph in zip(block, holm([r["p"] for r in block])):
            r["p_holm"] = ph
        rows += block
    return pd.DataFrame(rows)


def social_tests(fin: pd.DataFrame, head: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis over the seven inheritance schemes, and Best vs Similar."""
    rows = []
    for xv in filter(lambda xv: has(fin, xv), X_VALUES):
        for metric, df, col in (("final", fin, "surv_disp"), ("head", head, "born_init_disp")):
            groups = [values(df, s, xv, col) for s in SOCIAL]
            kw = stats.kruskal(*groups)
            b, sim = values(df, "best", xv, col), values(df, "similar_TOPDOWN", xv, col)
            rows.append(dict(x=xv, metric=metric, kw_H=kw.statistic, kw_p=kw.pvalue,
                             a12_best_vs_similar=a12(b, sim), p_best_vs_similar=stats.mannwhitneyu(b, sim).pvalue))
    return pd.DataFrame(rows)


def novelty_cost(fin: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Share of the x = 1 survivor mean displacement each scheme keeps at
    x = 0.5 and at x = 0.
    """
    rows = []
    for xv in filter(lambda xv: xv < 1 and has(fin, xv), X_VALUES):
        for s in SCHEMES:
            lo_v, hi_v = values(fin, s, xv, "surv_disp"), values(fin, s, 1.0, "surv_disp")
            boot = boot_mean(lo_v, rng) / boot_mean(hi_v, rng)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({"x": xv, "scheme": s, "share": lo_v.mean() / hi_v.mean(), "lo": lo, "hi": hi,
                             "a12_vs_x10": a12(lo_v, hi_v)})
    return pd.DataFrame(rows)


def stars(p: float) -> str:
    return "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


def fmt_p(p: float) -> str:
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"


def table_advantage(adv: dict[str, pd.DataFrame]) -> str:
    cols = [(dom, xv) for dom in DOMAINS for xv in X_VALUES]
    lines = [r"\begin{tabular}{l" + "rc" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{2 * len(X_VALUES)}}}{{c}}{{{DOMAINS[d]}}}" for d in DOMAINS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * len(X_VALUES) * i}-{1 + 2 * len(X_VALUES) * (i + 1)}}}"
                      for i in range(len(DOMAINS))),
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$x={xv:g}$}}" for _, xv in cols) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
             "Scheme & " + " & ".join([r"$\Delta$ & $\hat A_{12}$"] * len(cols)) + r" \\", r"\midrule"]
    for s in SOCIAL:
        cells = []
        for dom, xv in cols:
            r = adv[dom][(adv[dom].scheme == s) & (adv[dom].x == xv)]
            if r.empty:
                cells += [TBD, TBD]
                continue
            r = r.iloc[0]
            cells += [rf"${100 * r.rel:+.0f}\%$", rf"${r.a12:.2f}{stars(r.p_holm)}$"]
        lines.append(f"{tex(s)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_kruskal(tests: dict[str, pd.DataFrame], rho: dict[str, dict]) -> str:
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Kruskal--Wallis $p$} & \multicolumn{2}{c}{$\hat A_{12}$ Best vs.\ Similar}"
             r" & Spearman $\rho$ \\",
             r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7}",
             r"Domain & $x$ & head start & final & head start & final & distance, head start \\", r"\midrule"]
    for dom in DOMAINS:
        t = tests[dom].set_index(["x", "metric"])
        for xv in X_VALUES:
            if xv not in rho[dom]:
                lines.append(f"{DOMAINS[dom]} & {xv:g} & " + " & ".join([TBD] * 5) + r" \\")
                continue
            h, f = t.loc[xv, "head"], t.loc[xv, "final"]
            r, p = rho[dom][xv]
            lines.append(f"{DOMAINS[dom]} & {xv:g} & {fmt_p(h.kw_p)} & {fmt_p(f.kw_p)} & "
                         f"${h.a12_best_vs_similar:.2f}{stars(h.p_best_vs_similar)}$ & "
                         f"${f.a12_best_vs_similar:.2f}{stars(f.p_best_vs_similar)}$ & ${r:.2f}{stars(p)}$" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_novelty_cost(cost: dict[str, pd.DataFrame]) -> str:
    xs = [xv for xv in X_VALUES if xv < 1]
    cols = [(dom, xv) for dom in DOMAINS for xv in xs]
    lines = [r"\begin{tabular}{l" + "rr" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{2 * len(xs)}}}{{c}}{{{DOMAINS[d]}}}" for d in DOMAINS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * len(xs) * i}-{1 + 2 * len(xs) * (i + 1)}}}" for i in range(len(DOMAINS))),
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$x={xv:g}$}}" for _, xv in cols) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
             "Scheme & " + " & ".join([r"kept & 95\% CI"] * len(cols)) + r" \\", r"\midrule"]
    for s in SCHEMES:
        cells = []
        for dom, xv in cols:
            r = cost[dom][(cost[dom].scheme == s) & (cost[dom].x == xv)]
            if r.empty:
                cells += [TBD, TBD]
                continue
            r = r.iloc[0]
            cells += [rf"{100 * r.share:.0f}\%", rf"[{100 * r.lo:.0f}, {100 * r.hi:.0f}]"]
        lines.append(f"{tex(s)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(FIGURES / "final_results/cache"))
    parser.add_argument("--out-dir", default=str(FIGURES / "final_results"))
    parser.add_argument("--tables-dir", default=str(FIGURES / "overleaf_final_results/final_results_tables"))
    args = parser.parse_args()
    cache, out, tdir = Path(args.cache_dir), Path(args.out_dir), Path(args.tables_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PAPER_RC)
    save = Saver(out, dpi=200)
    rng = np.random.default_rng(0)

    curves = {"ariel": pd.read_csv(cache / "curves_ariel_ig10.csv"), "evogym": pd.read_csv(cache / "curves_evogym.csv")}
    donor = {"ariel": pd.read_csv(cache / "donor_dist_ariel_ig10.csv"),
             "evogym": pd.read_csv(cache / "donor_dist_evogym.csv")}
    curves_ig20 = pd.read_csv(cache / "curves_ariel_ig20.csv")

    plot_legend(save)
    adv, tests, cost, rho, quoted = {}, {}, {}, {}, []
    for dom, c in curves.items():
        dd = donor[dom].groupby(["scheme", "x", "rep", "gen"]).donor_dist.mean().reset_index()
        for col, name, ylabel, schemes, data in (
                ("surv_disp", "displacement", "Survivor mean\ndisplacement", SCHEMES, c),
                ("best_so_far_disp", "best_so_far", "Best displacement\nso far", SCHEMES, c),
                ("surv_diversity", "diversity", "Diversity", SCHEMES, c),
                ("donor_dist", "donor_distance", DIST_LABEL[dom], SOCIAL, dd)):
            ylim = shared_ylim(data[data.gen > 0] if col == "donor_dist" else data, col)
            for xv in filter(lambda xv: has(data, xv), X_VALUES):
                # At x = 0 survivor displacement stays near 0 on the shared axis
                own = col == "surv_disp" and xv == 0.0
                plot_curve(data[data.x == xv], col, ylabel, f"{name}_{dom}_{x_tag(xv)}", schemes, save,
                           None if own else ylim, own_scale=own)
        share = headstart_share(c)
        plot_headstart_share(share, dom, save)

        per_run = donor[dom].groupby(["scheme", "x", "rep"])[["donor_dist", "init_fitness"]].mean().reset_index()
        rho[dom] = plot_headstart_vs_distance(per_run, dom, save)

        fin = final(c)
        head = run_means(c[c.gen > 0], "born_init_disp")
        adv[dom] = advantage(fin, "surv_disp", rng)
        tests[dom] = social_tests(fin, head)
        cost[dom] = novelty_cost(fin, rng)

        # Numbers for the text.
        share_means = (share.groupby(["scheme", "x"])
                       .agg(mean=("head_share", "mean"), ci95_series=("head_share", ci95_series),
                            n_clipped=("head_share_raw", lambda v: int(((v < 0) | (v > 1)).sum())))
                       .reset_index())
        for df, what in ((adv[dom], "advantage"), (tests[dom], "tests"), (cost[dom], "novelty_cost"),
                         (share_means, "headstart_share")):
            quoted += [dict(domain=dom, table=what, **r) for r in df.to_dict("records")]
        for xv, (r, p) in rho[dom].items():
            quoted.append(dict(domain=dom, table="spearman_distance_headstart", x=xv, rho=r, p=p))
            for part, (r_part, p_part) in distance_headstart_split(per_run, xv).items():
                quoted.append(dict(domain=dom, table=f"spearman_distance_headstart_{part}", x=xv, rho=r_part, p=p_part))
            for part, (slope, r2) in distance_headstart_fits(per_run, xv).items():
                quoted.append(dict(domain=dom, table=f"linfit_distance_headstart_{part}", x=xv, slope=slope, r2=r2))
        last = c[c.gen == c.gen.max()]
        early = c[c.gen.between(1, 10)]
        for (s, xv), g in last.groupby(["scheme", "x"]):
            e = early[(early.scheme == s) & (early.x == xv)]
            ddx = dd[(dd.scheme == s) & (dd.x == xv)]
            quoted.append({"domain": dom, "table": "final_means", "scheme": s, "x": xv, "surv_disp": g.surv_disp.mean(),
                               "best_so_far": g.best_so_far_disp.mean(), "diversity_last": g.surv_diversity.mean(),
                               "diversity_gen0": c[(c.scheme == s) & (c.x == xv) & (c.gen == 0)].surv_diversity.mean(),
                               "head_start_mean": head[(head.scheme == s) & (head.x == xv)].born_init_disp.mean(),
                               "head_start_last": g.born_init_disp.mean(), "learn_gain_last": g.born_learn_gain.mean(),
                               "head_start_gen1_10": e.born_init_disp.mean(), "learn_gain_gen1_10": e.born_learn_gain.mean(),
                               "donor_dist_gen1_10": ddx[ddx.gen <= 10].donor_dist.mean(),
                               "donor_dist_last10": ddx[ddx.gen > ddx.gen.max() - 10].donor_dist.mean()})

    budget_adv = pd.concat([advantage(final(cv), "surv_disp", rng).assign(budget=b)
                            for b, cv in (("ig10", curves["ariel"]), ("ig20", curves_ig20))])
    plot_budget_advantage(budget_adv, save)
    quoted += [dict(domain="ariel", table="budget_advantage", **r) for r in budget_adv.to_dict("records")]

    (tdir / "tab_advantage.tex").write_text(table_advantage(adv))
    (tdir / "tab_kruskal.tex").write_text(table_kruskal(tests, rho))
    (tdir / "tab_novelty_cost.tex").write_text(table_novelty_cost(cost))
    pd.DataFrame(quoted).to_csv(out / "stats.csv", index=False)
    print(f"Saved tables to {tdir} and {out / 'stats.csv'}")


if __name__ == "__main__":
    main()
