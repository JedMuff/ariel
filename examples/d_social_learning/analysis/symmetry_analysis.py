"""Step 1 of SYMMETRY_BEHAVIOUR_PLAN.md: body symmetry in the zoo fe10 runs.

Reads what symmetry_scores.py wrote to --data-dir (individuals.csv,
survivors.csv, random.csv) and writes, under --data-dir:
    agreement.csv     Spearman between the symmetry scores over all distinct
                      bodies, and how often tree_symmetry calls a body fully
                      symmetric that geometric_symmetry (strict) does not
    random_by_size.csv  mean scores of the random genomes per module count
    final.csv         per budget/x/scheme: survivors' mean paired and excess
                      over size-matched random genomes, initial population vs
                      the last generation, mean over reps, Wilcoxon (signed-rank
                      over reps) of the change and of the final excess
    fitness.csv       per budget/x/scheme: Spearman of excess paired vs
                      distance among a run's offspring (median over reps,
                      share of reps with rho > 0), the same with module and
                      hinge counts partialled out, and across reps of the
                      final survivors' mean excess vs mean distance
    figures/*.png     survivors' paired, excess and module count over
                      generations, one panel per budget and x

"excess" is a body's paired score minus the mean paired score of random
genomes with the same number of modules, since small bodies are more often
symmetric by chance.

Usage:
    uv run examples/d_social_learning/analysis/symmetry_analysis.py --data-dir __data__/social/symmetry
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

from paper_figures import label, style_grid  # noqa: E402
from parta_report import SCHEME_COLORS, SCHEME_LS  # noqa: E402

RUN = ["budget", "x", "scheme", "k", "rep"]
SCORES = ["paired", "strict", "sym", "tree_y", "tree_d"]
SCHEME_ORDER = ["darwinian", "lamarckian", "random", "random_many", "best", "best_many",
                "similar_TOPDOWN", "similar_many_TOPDOWN"]


def add_excess(df: pd.DataFrame, by_size: pd.DataFrame) -> pd.DataFrame:
    return df.assign(excess=df.paired - df.n_modules.map(by_size.paired))


def agreement(ind: pd.DataFrame, rnd: pd.DataFrame) -> pd.DataFrame:
    bodies = pd.concat([ind.drop_duplicates(["budget", "scheme", "x", "k", "rep", "id"]), rnd])
    rho = bodies[SCORES].corr(method="spearman").stack().rename("spearman").reset_index()
    rho = rho[rho.level_0 < rho.level_1]
    full = bodies[bodies.tree_y == 1.0]
    extra = pd.DataFrame([{"level_0": "tree_y==1", "level_1": "share with strict<1",
                           "spearman": float((full.strict < 1).mean())}])
    return pd.concat([rho, extra]).rename(columns={"level_0": "a", "level_1": "b"})


def survivor_curves(ind: pd.DataFrame, surv: pd.DataFrame) -> pd.DataFrame:
    """Per run and generation (0 = initial population): survivors' mean scores."""
    init = ind[ind.time_of_birth == 0].assign(generation=0)
    later = surv.merge(ind, left_on=RUN + ["individual_id"], right_on=RUN + ["id"])
    both = pd.concat([init, later])
    return both.groupby(RUN + ["generation"])[["paired", "excess", "strict", "n_modules", "distance"]].mean().reset_index()


def final_table(curves: pd.DataFrame) -> pd.DataFrame:
    last = curves.generation.max()
    rows = []
    for (b, x, s), g in curves.groupby(["budget", "x", "scheme"]):
        g0 = g[g.generation == 0].set_index("rep")
        g1 = g[g.generation == last].set_index("rep")
        reps = g0.index.intersection(g1.index)
        d = g1.loc[reps, "paired"] - g0.loc[reps, "paired"]
        rows.append({
            "budget": b, "x": x, "scheme": s, "n_reps": len(reps),
            "paired_gen0": g0.loc[reps, "paired"].mean(), f"paired_gen{last}": g1.loc[reps, "paired"].mean(),
            "p_change": stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0,
            "excess_gen0": g0.loc[reps, "excess"].mean(), f"excess_gen{last}": g1.loc[reps, "excess"].mean(),
            "p_excess": stats.wilcoxon(g1.loc[reps, "excess"], nan_policy="omit").pvalue,
            "modules_gen0": g0.loc[reps, "n_modules"].mean(), f"modules_gen{last}": g1.loc[reps, "n_modules"].mean(),
        })
    return pd.DataFrame(rows)


def partial_rho(g: pd.DataFrame) -> float:
    """Spearman of paired vs distance with module and hinge counts partialled
    out (Pearson of the rank residuals): hinges drive distance and are hard
    to pair, so the raw correlation is mostly the hinge count."""
    X = np.column_stack([np.ones(len(g)), stats.rankdata(g.n_modules), stats.rankdata(g.n_hinges)])

    def resid(v):
        v = stats.rankdata(v)
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    return stats.pearsonr(resid(g.paired), resid(g.distance)).statistic


def fitness_table(ind: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    kids = ind[(ind.time_of_birth > 0) & np.isfinite(ind.distance)]
    per_run = kids.groupby(RUN).apply(
        lambda g: pd.Series({"rho": stats.spearmanr(g.excess, g.distance, nan_policy="omit").statistic,
                             "partial": partial_rho(g)}), include_groups=False)
    per_run = per_run.reset_index()
    last = curves[curves.generation == curves.generation.max()]
    rows = []
    for (b, x, s), g in per_run.groupby(["budget", "x", "scheme"]):
        f = last[(last.budget == b) & (last.x == x) & (last.scheme == s)]
        across = stats.spearmanr(f.excess, f.distance, nan_policy="omit")
        rows.append({"budget": b, "x": x, "scheme": s, "median_rho_within": g.rho.median(),
                     "share_rho_pos": (g.rho > 0).mean(),
                     "median_partial_within": g.partial.median(), "share_partial_pos": (g.partial > 0).mean(),
                     "rho_across_reps": across.statistic,
                     "p_across_reps": across.pvalue})
    return pd.DataFrame(rows)


def plot_curves(curves: pd.DataFrame, col: str, ylabel: str, out: Path, hline: float | None = None,
                hline_label: str = "") -> None:
    budgets, xs = sorted(curves.budget.unique()), sorted(curves.x.unique())
    fig, axes = plt.subplots(len(budgets), len(xs), figsize=(5.5 * len(xs), 3.8 * len(budgets)),
                             sharex=True, sharey=True, squeeze=False)
    for i, b in enumerate(budgets):
        for j, x in enumerate(xs):
            ax = axes[i, j]
            for s in [s for s in SCHEME_ORDER if s in set(curves.scheme)]:
                g = curves[(curves.budget == b) & (curves.x == x) & (curves.scheme == s)]
                m = g.groupby("generation")[col].agg(["mean", "sem"])
                ax.plot(m.index, m["mean"], color=SCHEME_COLORS[s], ls=SCHEME_LS[s], lw=2, label=label(s))
                ax.fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"], color=SCHEME_COLORS[s], alpha=0.12, lw=0)
            if hline is not None:
                ax.axhline(hline, color="0.4", ls="--", lw=1.2, label=hline_label)
            ax.set_title(f"{b}, x={x}")
            style_grid(ax)
            if i == len(budgets) - 1:
                ax.set_xlabel("generation")
            if j == 0:
                ax.set_ylabel(ylabel)
    axes[0, -1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="__data__/social/symmetry")
    args = parser.parse_args()
    d = Path(args.data_dir)
    ind = pd.read_csv(d / "individuals.csv")
    surv = pd.read_csv(d / "survivors.csv")
    rnd = pd.read_csv(d / "random.csv")

    by_size = rnd.groupby("n_modules")[SCORES].mean().assign(n=rnd.groupby("n_modules").size())
    by_size = by_size[by_size.n >= 100]  # sizes random_tree rarely reaches stay unmatched (NaN excess)
    by_size.to_csv(d / "random_by_size.csv")
    ind = add_excess(ind, by_size)
    agreement(ind, rnd).to_csv(d / "agreement.csv", index=False)

    curves = survivor_curves(ind, surv)
    curves.to_csv(d / "survivor_curves.csv", index=False)
    final_table(curves).to_csv(d / "final.csv", index=False)
    fitness_table(ind, curves).to_csv(d / "fitness.csv", index=False)

    fig_dir = d / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_curves(curves, "paired", "survivors' paired symmetry", fig_dir / "paired.png", hline=None)
    plot_curves(curves, "excess", "paired minus size-matched random", fig_dir / "excess.png", hline=0.0,
                hline_label="size-matched random")
    plot_curves(curves, "strict", "survivors' strict symmetry", fig_dir / "strict.png", hline=None)
    plot_curves(curves, "n_modules", "survivors' modules", fig_dir / "modules.png", hline=13.0,
                hline_label="initial bodies")
    print(f"-> {d}")


if __name__ == "__main__":
    main()
