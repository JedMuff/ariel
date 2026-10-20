"""Rowing gait over evolution in the zoo fe10 runs: the plots of
ariel_symmetry's plot_rowing_stride_sweep.py, with scheme in place of genome
and budget x selection weight in place of control stride.

Reads behavior_replay.py --select all output (<behavior-dir>/runs/**/
{episodes,history}.csv). A robot's label is its most common label over its
episodes, per weight set. A generation's population is its 20 survivors
(--population survivors, generation 0 = the initial population) or the 20
robots born in it (offspring).

Writes to <behavior-dir>/rowing/<population>_<weights>/:
  rowing_share_over_gens.png
    cols = budget x selection weight; rows = share of each generation's
    population that is rowing (of all), rowing among judgeable robots
    (leaving out unpaired, midline and static, which have no left/right pairs
    to judge), and alternating (of all). Per scheme, mean +- 1 SEM over reps.
  label_composition.png
    rows = scheme, cols = budget x selection weight: stacked share of each gait
    class per generation, reps pooled.
  rowing_learned_vs_init.png
    per scheme, rowing share with the learned and with the starting weights
    (budgets and selection weights pooled): does learning make the rowing, or
    is it inherited?
  rowing_share_by_gen.csv   per run x gen: the counts and shares plotted
  rowing_early_late.csv     per run: rowing share over the first and last
                            EARLY_LATE_GENS generations, and the change
  (a per-condition summary of the last is printed)

Usage:
    uv run examples/d_social_learning/analysis/plot_rowing_over_gens.py \\
        --behavior-dir __data__/social/behavior_zoo_all [--population offspring] [--weights init]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from gait_analysis import LABEL_COLORS, SCHEME_ORDER  # noqa: E402
from paper_figures import SCHEME_COLORS, SCHEME_LS, label  # noqa: E402

RUN = ["budget", "x", "scheme", "k", "rep"]
CONDITIONS = [("ig10", 0.5), ("ig10", 1.0), ("ig20", 0.5), ("ig20", 1.0)]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CI_ALPHA = 0.15
EARLY_LATE_GENS = 5
LABELS = ("rowing", "alternating", "mixed", "aperiodic", "midline", "unpaired", "static")
# Gait classes for the composition plot, bottom to top: detector labels -> class.
CLASSES = {"rowing": ("rowing",), "alternating": ("alternating",), "other gait": ("mixed", "aperiodic"),
           "midline": ("midline",), "can't judge": ("unpaired", "static")}
CLASS_COLOR = {"rowing": LABEL_COLORS["rowing"], "alternating": LABEL_COLORS["alternating"],
               "other gait": LABEL_COLORS["mixed"], "midline": LABEL_COLORS["midline"],
               "can't judge": LABEL_COLORS["static"]}
CLASS_LEGEND = {"other gait": "other gait (mixed / aperiodic)", "midline": "midline (chain flexing on the split plane)",
                "can't judge": "can't judge (unpaired / static)"}
JUDGEABLE = ("rowing", "alternating", "mixed", "aperiodic")


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))


def cond_title(budget: str, x: float) -> str:
    return f"{budget.removeprefix('ig')} CMA-ES gens, x={x:g}"


def conditions_in(df: pd.DataFrame) -> list[tuple[str, float]]:
    return [c for c in CONDITIONS if ((df.budget == c[0]) & (df.x == c[1])).any()]


def schemes_in(df: pd.DataFrame) -> list[str]:
    return [s for s in SCHEME_ORDER if s in set(df.scheme)]


def load(behavior_dir: Path, schemes: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robot labels (one row per run x id x weights) and the runs' histories."""
    eps, hists = [], []
    for p in sorted((behavior_dir / "runs").glob("ig*/*/x*/k*/rep_*/episodes.csv")):
        if schemes and p.parts[-5] not in schemes:
            continue
        ep = pd.read_csv(p, usecols=RUN + ["id", "birth_gen", "weights", "episode", "rowing_label"])
        eps.append(ep)
        hists.append(pd.read_csv(p.parent / "history.csv").assign(**ep.iloc[0][RUN].to_dict()))
    if not eps:
        raise SystemExit(f"no runs/**/episodes.csv under {behavior_dir}")
    ep = pd.concat(eps, ignore_index=True)
    rob = (ep.groupby(RUN + ["id", "birth_gen", "weights"]).rowing_label
           .agg(lambda s: s.mode().iloc[0]).rename("label").reset_index())
    print(f"{len(eps)} runs, {rob.id.size} robot x weight labels")
    return rob, pd.concat(hists, ignore_index=True)


def population(rob: pd.DataFrame, hist: pd.DataFrame, which: str) -> pd.DataFrame:
    """One row per run x gen x member of that generation's population."""
    if which == "offspring":
        return rob.rename(columns={"birth_gen": "gen"})
    surv = hist[hist.survivor == 1].rename(columns={"generation": "gen"})[RUN + ["gen", "id"]]
    return surv.merge(rob.drop(columns="birth_gen"), on=RUN + ["id"])


def shares_by_gen(pop: pd.DataFrame) -> pd.DataFrame:
    """Per run x gen: counts of each detector label, and the shares plotted."""
    counts = pd.crosstab([pop[c] for c in RUN + ["gen"]], pop.label).reset_index()
    for c in LABELS:
        if c not in counts:
            counts[c] = 0
    counts["n"] = counts[list(LABELS)].sum(axis=1)
    counts["n_judgeable"] = counts[list(JUDGEABLE)].sum(axis=1)
    counts["rowing_share"] = counts["rowing"] / counts["n"]
    counts["alternating_share"] = counts["alternating"] / counts["n"]
    counts["rowing_share_judgeable"] = counts["rowing"] / counts["n_judgeable"].where(counts["n_judgeable"] > 0)
    for cls, labs in CLASSES.items():
        counts[f"class_{cls}"] = counts[list(labs)].sum(axis=1) / counts["n"]
    return counts.sort_values(RUN + ["gen"]).reset_index(drop=True)


def gen_curve(df: pd.DataFrame, col: str) -> tuple[pd.Index, pd.Series, pd.Series]:
    """Per-rep share per gen -> gens, mean and SEM across reps."""
    per_rep = df.pivot_table(index="gen", columns="rep", values=col)
    sem = per_rep.apply(lambda r: stats.sem(r.dropna()) if r.notna().sum() > 1 else 0.0, axis=1)
    return per_rep.index, per_rep.mean(axis=1), sem


def scheme_legend(fig: plt.Figure, schemes: list[str]) -> None:
    handles = [plt.Line2D([], [], color=SCHEME_COLORS[s], ls=SCHEME_LS[s], lw=2) for s in schemes]
    fig.legend(handles, [label(s) for s in schemes], loc="lower center", ncol=len(schemes), frameon=False,
               fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.01))


def plot_shares(sh: pd.DataFrame, title: str, out: Path, dpi: int) -> None:
    rows = [("rowing_share", "rowing\n(share of all robots)"),
            ("rowing_share_judgeable", "rowing\n(share of judgeable robots)"),
            ("alternating_share", "alternating\n(share of all robots)")]
    schemes, conditions = schemes_in(sh), conditions_in(sh)
    fig, axes = plt.subplots(len(rows), len(conditions), figsize=(4.2 * len(conditions), 2.9 * len(rows)),
                             sharex=True, sharey=True, squeeze=False)
    for r, (col, ylabel) in enumerate(rows):
        for c, (budget, x) in enumerate(conditions):
            ax = axes[r, c]
            style(ax)
            for scheme in schemes:
                df = sh[(sh.budget == budget) & (sh.x == x) & (sh.scheme == scheme)]
                if df.empty:
                    continue
                gens, m, sem = gen_curve(df, col)
                color = SCHEME_COLORS[scheme]
                ax.fill_between(gens, m - sem, m + sem, color=color, alpha=CI_ALPHA, linewidth=0)
                ax.plot(gens, m, color=color, ls=SCHEME_LS[scheme], linewidth=1.8)
            ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(cond_title(budget, x), fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
            if r == len(rows) - 1:
                ax.set_xlabel("generation", fontsize=9, color=INK_2)
    scheme_legend(fig, schemes)
    n_reps = sh.groupby(["budget", "x", "scheme"]).rep.nunique().median()
    fig.suptitle(f"Rowing gait over evolution: {title} (mean ± 1 SEM over {n_reps:.0f} reps)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_composition(sh: pd.DataFrame, title: str, out: Path, dpi: int) -> None:
    schemes, conditions = schemes_in(sh), conditions_in(sh)
    cols = [f"class_{c}" for c in CLASSES]
    fig, axes = plt.subplots(len(schemes), len(conditions), figsize=(4.2 * len(conditions), 2.0 * len(schemes) + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    for r, scheme in enumerate(schemes):
        for c, (budget, x) in enumerate(conditions):
            ax = axes[r, c]
            style(ax)
            ax.grid(False)
            df = sh[(sh.budget == budget) & (sh.x == x) & (sh.scheme == scheme)]
            if df.empty:
                continue
            # Pool the reps: weight each rep's shares by its population size.
            w = df[cols].mul(df["n"], axis=0).assign(gen=df["gen"], n=df["n"]).groupby("gen").sum()
            frac = w[cols].div(w["n"], axis=0)
            ax.stackplot(frac.index, frac.T.to_numpy(), colors=[CLASS_COLOR[k] for k in CLASSES],
                         edgecolor="white", linewidth=0.8)
            # Direct labels at the right edge, mid-band, where the band is tall enough to read.
            last, bottom = frac.iloc[-1], 0.0
            for cls, col in zip(CLASSES, cols):
                h = last[col]
                if h >= 0.12:
                    ax.text(frac.index[-1] - 0.5, bottom + h / 2, f"{h:.0%}", ha="right", va="center",
                            fontsize=7.5, color=INK if cls == "can't judge" else "white")
                bottom += h
            ax.set_xlim(frac.index[0], frac.index[-1])
            ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(cond_title(budget, x), fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(label(scheme), fontsize=9, color=INK_2)
            if r == len(schemes) - 1:
                ax.set_xlabel("generation", fontsize=9, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[k]) for k in CLASSES]
    fig.legend(handles, [CLASS_LEGEND.get(k, k) for k in CLASSES], loc="lower center", ncol=len(CLASSES),
               frameon=False, fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Gait classes over evolution: {title} (reps pooled; numbers: share in the last generation)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.5 / fig.get_figheight(), 1, 1))  # room for the legend
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_learned_vs_init(learned: pd.DataFrame, init: pd.DataFrame, title: str, out: Path, dpi: int) -> None:
    """Rowing share per scheme, learned vs starting weights; each rep is one run (budgets and x pooled).
    With a single scheme, one panel per budget x selection weight instead."""
    schemes = schemes_in(learned)
    if len(schemes) == 1:
        panels = [(schemes[0], c) for c in conditions_in(learned)]
        pooled = f"{label(schemes[0])}, mean ± 1 SEM over reps"
    else:
        panels = [(s, None) for s in schemes]
        pooled = "mean ± 1 SEM over runs, budgets and x pooled"
    ncols = min(4, len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (scheme, cond) in zip(axes.flat, panels):
        style(ax)
        color = SCHEME_COLORS[scheme]
        for sh, alpha, name in ((init, 0.45, "starting weights"), (learned, 1.0, "learned weights")):
            df = sh[sh.scheme == scheme]
            if cond is not None:
                df = df[(df.budget == cond[0]) & (df.x == cond[1])]
            df = df.assign(rep=lambda d: d[RUN].astype(str).agg("/".join, axis=1))
            gens, m, sem = gen_curve(df, "rowing_share")
            ax.fill_between(gens, m - sem, m + sem, color=color, alpha=CI_ALPHA * alpha, linewidth=0)
            ax.plot(gens, m, color=color, alpha=alpha, linewidth=1.8, ls="-" if alpha == 1.0 else "--", label=name)
        ax.set_title(cond_title(*cond) if cond is not None else label(scheme), fontsize=10, color=INK)
    axes[0, 0].set_ylim(bottom=0)  # after plotting, so the shared top fits every panel
    for ax in axes[:, 0]:
        ax.set_ylabel("rowing (share of all robots)", fontsize=9, color=INK_2)
    for ax in axes[-1]:
        ax.set_xlabel("generation", fontsize=9, color=INK_2)
    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper left")
    fig.suptitle(f"Rowing with learned vs starting weights: {title} ({pooled})", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def early_late(sh: pd.DataFrame) -> pd.DataFrame:
    """Rowing share over the first and last EARLY_LATE_GENS gens, per run (counts pooled within the window)."""
    g0, g1 = sh["gen"].min(), sh["gen"].max()
    rows = []
    for keys, df in sh.groupby(RUN):
        early = df[df["gen"] < g0 + EARLY_LATE_GENS]
        late = df[df["gen"] > g1 - EARLY_LATE_GENS]
        e, l = early["rowing"].sum() / early["n"].sum(), late["rowing"].sum() / late["n"].sum()
        ej = early["rowing"].sum() / max(early["n_judgeable"].sum(), 1)
        lj = late["rowing"].sum() / max(late["n_judgeable"].sum(), 1)
        rows.append({**dict(zip(RUN, keys)), "early": e, "late": l, "change": l - e,
                     "early_judgeable": ej, "late_judgeable": lj,
                     "n_early": int(early["n"].sum()), "n_late": int(late["n"].sum())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--behavior-dir", type=Path, default=Path("__data__/social/behavior_zoo_all"))
    parser.add_argument("--population", choices=["survivors", "offspring"], default="survivors")
    parser.add_argument("--weights", choices=["learned", "init"], default="learned")
    parser.add_argument("--schemes", nargs="+", default=None,
                        help="only these schemes (run-directory names, e.g. darwinian similar_TOPDOWN)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    rob, hist = load(args.behavior_dir, args.schemes)
    tag = f"{args.population}_{args.weights}" + ("_" + "+".join(args.schemes) if args.schemes else "")
    out = args.behavior_dir / "rowing" / tag
    out.mkdir(parents=True, exist_ok=True)
    pop = population(rob, hist, args.population)
    by_weights = {w: shares_by_gen(pop[pop.weights == w]) for w in ("learned", "init")}
    sh = by_weights[args.weights]
    weights_desc = "learned weights" if args.weights == "learned" else "starting weights"
    title = f"{args.population}, {weights_desc}"

    sh.to_csv(out / "rowing_share_by_gen.csv", index=False)
    el = early_late(sh)
    el.to_csv(out / "rowing_early_late.csv", index=False)
    summary = (el.groupby(["budget", "x", "scheme"])[["early", "late", "change", "late_judgeable"]].mean()
               .join(el.groupby(["budget", "x", "scheme"]).change.apply(
                   lambda d: stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0).rename("p_change")))
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(f"Rowing share, first vs last {EARLY_LATE_GENS} generations ({title}; mean over reps, "
              "p: Wilcoxon signed-rank of the per-rep change):")
        print(summary)

    plot_shares(sh, title, out / "rowing_share_over_gens.png", args.dpi)
    plot_composition(sh, title, out / "label_composition.png", args.dpi)
    plot_learned_vs_init(by_weights["learned"], by_weights["init"], args.population,
                         out / "rowing_learned_vs_init.png", args.dpi)


if __name__ == "__main__":
    main()
