"""Zoo learning-budget pilot: mechanism extras per budget and the comparison
between two real inner budgets (default ig10 vs ig20).

Reads what parta_report.py and learning_budget.py wrote for each budget:
    <report-root>/<budget>/parta_curves_per_run.csv, parta_included_runs.csv
    <budget-root>/<budget>/budget_curves.npz
and the run databases (for donor ids).

Per budget (CSV, one row per budget/x/scheme):
    mechanism.csv        offspring fitness before learning and learning gain,
                         generations 21-30 per run, mean/SD over runs, and
                         Mann-Whitney vs Similar
    donor_is_parent.csv  share of births whose donor(s) include the parent
    first_gen_share.csv  share of the full learning gain reached after one
                         CMA-ES generation, (B(1)-B(0)) / (B(full)-B(0))
    kruskal_social.csv   Kruskal-Wallis over the inheritance schemes only, gen 30
    pairwise_social.csv  every pair of inheritance schemes at gen 30: Mann-Whitney,
                         A12, Holm-corrected over the pairs of one budget/x/metric

Between budgets (CSV + figures under <paper-dir>/{pdf,png}/compare/):
    budget_final.csv     final survivor / best-found displacement per scheme,
                         short vs long budget (Mann-Whitney, A12 of long vs short)
    budget_advantage.csv advantage over Darwinian per budget, absolute and
                         relative, with a bootstrap 95% CI of the change
    budget_headstart.csv Similar's head start before learning per budget
    budget_groups.csv    head-start schemes (Lamarckian, Similar) minus
                         averaged-donor schemes, per budget, for each set in
                         LEARNER_SETS (all k=5, or the pilot's Best/Random k=5)
    truncation.csv       long-budget runs truncated to the short budget's
                         inner generations, B(short), vs the real short runs

Usage:
    uv run examples/d_social_learning/analysis/zoo_budget_compare.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

from learning_budget import per_run_means  # noqa: E402
from paper_figures import PAPER_RC, Saver, label, style_grid, x_tag  # noqa: E402
from parta_report import SCHEME_COLORS, a12, ci95  # noqa: E402

# All 8 part A schemes in part A's order; darwinian must come first.
SCHEMES = ["darwinian", "lamarckian", "random", "random_many", "best", "best_many",
           "similar_TOPDOWN", "similar_many_TOPDOWN"]
SIMILAR = "similar_TOPDOWN"
# Part A's two routes: close single donors (head start) vs averaged donors
# (learning gain). Single-donor random/best fit neither and are left out.
HEADSTART = ["lamarckian", "similar_TOPDOWN"]
# Similar (k=5) mostly averages the parent with its near neighbours, so the
# group difference is also given without it ("pilot": the first batch's set).
LEARNER_SETS = {"k5": ["random_many", "best_many", "similar_many_TOPDOWN"],
                "pilot": ["random_many", "best_many"]}
FINAL = [("surv_disp", "survivor_mean", "Survivor mean\ndisplacement (m)"),
         ("best_so_far_disp", "best_so_far", "Best displacement\nfound (m)")]
LATE = (21, 30)
N_BOOT = 10000


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, p[i] * (len(p) - rank)))
        adj[i] = running
    return adj.tolist()


def boot_mean(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(v, size=(N_BOOT, len(v))).mean(axis=1)


# ---------------------------------------------------------------------------
# Per-budget mechanism extras
# ---------------------------------------------------------------------------

def mechanism(curves: pd.DataFrame, budget: str) -> list[dict]:
    late = curves[(curves.gen >= LATE[0]) & (curves.gen <= LATE[1])]
    per_run = late.groupby(["scheme", "x", "rep"])[["born_init_disp", "born_learn_gain"]].mean()
    rows = []
    for xv in sorted(curves.x.unique()):
        px = per_run.xs(xv, level="x")
        ref = px.xs(SIMILAR, level="scheme")
        for s in SCHEMES:
            v = px.xs(s, level="scheme")
            row = dict(budget=budget, x=xv, scheme=s, n=len(v))
            for col, short in (("born_init_disp", "init"), ("born_learn_gain", "gain")):
                row[f"{short}_mean"] = v[col].mean()
                row[f"{short}_sd"] = v[col].std(ddof=1)
                if s != SIMILAR:
                    row[f"{short}_a12_vs_similar"] = a12(v[col].to_numpy(), ref[col].to_numpy())
                    row[f"{short}_p_vs_similar"] = stats.mannwhitneyu(v[col], ref[col]).pvalue
            rows.append(row)
    return rows


def donor_is_parent(runs: pd.DataFrame, budget: str) -> list[dict]:
    rows = []
    for r in runs.itertuples():
        if r.scheme == "darwinian":
            continue
        con = sqlite3.connect(f"file:{r.db}?mode=ro", uri=True)
        tags = con.execute("SELECT json_extract(tags_, '$.parent_id'), json_extract(tags_, '$.donor_ids') "
                           "FROM individual WHERE time_of_birth > 0").fetchall()
        con.close()
        hits = [p in json.loads(d) for p, d in tags if d and p is not None]
        rows.append(dict(budget=budget, scheme=r.scheme, x=r.x, rep=r.rep, births=len(hits),
                         parent_share=np.mean(hits)))
    return rows


def first_gen_share(meta: pd.DataFrame, B: np.ndarray, budget: str) -> list[dict]:
    runs_b = per_run_means(meta, B, (1, 30))
    share = (runs_b[1] - runs_b[0]) / (runs_b[runs_b.columns[-1]] - runs_b[0])
    out = share.groupby(level=["scheme", "x"]).agg(["mean", "std", "count"]).reset_index()
    out.insert(0, "budget", budget)
    return out.to_dict("records")


def kruskal_social(curves: pd.DataFrame, budget: str) -> list[dict]:
    final = curves[curves.gen == curves.gen.max()]
    rows = []
    for xv in sorted(final.x.unique()):
        fx = final[final.x == xv]
        for col, _, _ in FINAL:
            groups = [fx[fx.scheme == s][col].to_numpy() for s in SCHEMES[1:]]
            rows.append(dict(budget=budget, x=xv, metric=col, kruskal_p_social=stats.kruskal(*groups).pvalue))
    return rows


def pairwise_social(curves: pd.DataFrame, budget: str) -> list[dict]:
    final = curves[curves.gen == curves.gen.max()]
    rows = []
    for xv in sorted(final.x.unique()):
        fx = final[final.x == xv]
        for col, _, _ in FINAL:
            block = []
            for i, a in enumerate(SCHEMES[1:], 1):
                for b in SCHEMES[i + 1:]:
                    va, vb = fx[fx.scheme == a][col].to_numpy(), fx[fx.scheme == b][col].to_numpy()
                    block.append(dict(budget=budget, x=xv, metric=col, scheme_a=a, scheme_b=b,
                                      diff=va.mean() - vb.mean(), a12=a12(va, vb),
                                      p=stats.mannwhitneyu(va, vb).pvalue))
            for row, pa in zip(block, holm([r["p"] for r in block])):
                row["p_holm"] = pa
            rows += block
    return rows


# ---------------------------------------------------------------------------
# Between budgets
# ---------------------------------------------------------------------------

def final_values(curves: pd.DataFrame) -> pd.DataFrame:
    return curves[curves.gen == curves.gen.max()].set_index(["scheme", "x", "rep"])


def budget_final(fin: dict[str, pd.DataFrame], short: str, long: str) -> pd.DataFrame:
    rows = []
    for xv in sorted(fin[short].index.unique("x")):
        for col, _, _ in FINAL:
            block = []
            for s in SCHEMES:
                a = fin[short].xs((s, xv), level=("scheme", "x"))[col].to_numpy()
                b = fin[long].xs((s, xv), level=("scheme", "x"))[col].to_numpy()
                block.append(dict(x=xv, metric=col, scheme=s, n_short=len(a), n_long=len(b),
                                  mean_short=a.mean(), mean_long=b.mean(), diff=b.mean() - a.mean(),
                                  a12_long_vs_short=a12(b, a), p=stats.mannwhitneyu(b, a).pvalue))
            for row, pa in zip(block, holm([r["p"] for r in block])):
                row["p_holm"] = pa
            rows += block
    return pd.DataFrame(rows)


def budget_advantage(fin: dict[str, pd.DataFrame], short: str, long: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for xv in sorted(fin[short].index.unique("x")):
        for col, _, _ in FINAL:
            boot, point = {}, {}
            for bud in (short, long):
                d = fin[bud].xs(("darwinian", xv), level=("scheme", "x"))[col].to_numpy()
                bd = boot_mean(d, rng)
                for s in SCHEMES[1:]:
                    v = fin[bud].xs((s, xv), level=("scheme", "x"))[col].to_numpy()
                    bv = boot_mean(v, rng)
                    point[bud, s] = (v.mean() - d.mean(), v.mean() / d.mean() - 1)
                    boot[bud, s] = (bv - bd, bv / bd - 1)
            for s in SCHEMES[1:]:
                row = dict(x=xv, metric=col, scheme=s)
                for i, kind in enumerate(("abs", "rel")):
                    for bud, tag in ((short, "short"), (long, "long")):
                        row[f"{kind}_{tag}"] = point[bud, s][i]
                        row[f"{kind}_{tag}_lo"], row[f"{kind}_{tag}_hi"] = np.percentile(boot[bud, s][i], [2.5, 97.5])
                    change = boot[long, s][i] - boot[short, s][i]
                    row[f"{kind}_change"] = point[long, s][i] - point[short, s][i]
                    row[f"{kind}_change_lo"], row[f"{kind}_change_hi"] = np.percentile(change, [2.5, 97.5])
                rows.append(row)
    return pd.DataFrame(rows)


def budget_groups(fin: dict[str, pd.DataFrame], short: str, long: str) -> pd.DataFrame:
    """Mean of the head-start schemes minus mean of the learning schemes."""
    rng = np.random.default_rng(1)
    rows = []
    for (lname, learners), xv in ((ls, xv) for ls in LEARNER_SETS.items()
                                  for xv in sorted(fin[short].index.unique("x"))):
        for col, _, _ in FINAL:
            row, boots = dict(learners=lname, x=xv, metric=col), {}
            for bud, tag in ((short, "short"), (long, "long")):
                f = fin[bud].xs(xv, level="x")[col]
                grp = lambda ss: [f.xs(s, level="scheme").to_numpy() for s in ss]  # noqa: E731
                hs, ln = grp(HEADSTART), grp(learners)
                row[f"diff_{tag}"] = np.mean([v.mean() for v in hs]) - np.mean([v.mean() for v in ln])
                boots[tag] = (np.mean([boot_mean(v, rng) for v in hs], axis=0)
                              - np.mean([boot_mean(v, rng) for v in ln], axis=0))
                row[f"diff_{tag}_lo"], row[f"diff_{tag}_hi"] = np.percentile(boots[tag], [2.5, 97.5])
                pooled_h, pooled_l = np.concatenate(hs), np.concatenate(ln)
                row[f"a12_{tag}"] = a12(pooled_h, pooled_l)
                row[f"p_{tag}"] = stats.mannwhitneyu(pooled_h, pooled_l).pvalue
            row["change"] = row["diff_long"] - row["diff_short"]
            row["change_lo"], row["change_hi"] = np.percentile(boots["long"] - boots["short"], [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)


def budget_headstart(curves: dict[str, pd.DataFrame], short: str, long: str) -> pd.DataFrame:
    rows = []
    for bud in (short, long):
        c = curves[bud]
        for wname, (lo, hi) in (("all", (1, 30)), ("late", LATE)):
            w = c[(c.gen >= lo) & (c.gen <= hi)].groupby(["scheme", "x", "rep"]).born_init_disp.mean()
            for xv in sorted(c.x.unique()):
                wx = w.xs(xv, level="x")
                sim = wx.xs(SIMILAR, level="scheme").to_numpy()
                for s in SCHEMES:
                    if s == SIMILAR:
                        continue
                    v = wx.xs(s, level="scheme").to_numpy()
                    rows.append(dict(budget=bud, window=wname, x=xv, scheme=s, similar_init=sim.mean(),
                                     other_init=v.mean(), headstart=sim.mean() - v.mean(),
                                     a12_similar=a12(sim, v), p=stats.mannwhitneyu(sim, v).pvalue))
    return pd.DataFrame(rows)


def truncation(npz: dict[str, tuple], short: str, long: str) -> tuple[pd.DataFrame, dict]:
    """B(b_short) of long-budget runs vs the real short-budget runs."""
    b_short = npz[short][1].shape[1] - 1
    rows, means = [], {}
    for wname, window in (("all", (1, 30)), ("early", (1, 10)), ("late", LATE)):
        rb = {bud: per_run_means(*npz[bud], window) for bud in (short, long)}
        means[wname] = rb
        for xv in sorted(rb[short].index.unique("x")):
            for b in (0, 1, b_short):
                real = {s: rb[short].xs((s, xv), level=("scheme", "x"))[b].to_numpy() for s in SCHEMES}
                trunc = {s: rb[long].xs((s, xv), level=("scheme", "x"))[b].to_numpy() for s in SCHEMES}
                for s in SCHEMES:
                    rows.append(dict(
                        window=wname, x=xv, budget_b=b, scheme=s,
                        real_short=real[s].mean(), truncated_long=trunc[s].mean(),
                        diff=trunc[s].mean() - real[s].mean(),
                        a12_trunc_vs_real=a12(trunc[s], real[s]),
                        p=stats.mannwhitneyu(trunc[s], real[s]).pvalue,
                        adv_real=real[s].mean() - real["darwinian"].mean(),
                        adv_trunc=trunc[s].mean() - trunc["darwinian"].mean(),
                        gap_sim_real=real[s].mean() - real[SIMILAR].mean(),
                        gap_sim_trunc=trunc[s].mean() - trunc[SIMILAR].mean(),
                    ))
    return pd.DataFrame(rows), means


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def budget_name(bud: str) -> str:
    return f"{int(bud.removeprefix('ig'))} inner gens"


def plot_final_pairs(fin: dict[str, pd.DataFrame], short: str, long: str, save: Saver) -> None:
    rng = np.random.default_rng(0)
    for col, name, ylabel in FINAL:
        for xv in sorted(fin[short].index.unique("x")):
            fig, ax = plt.subplots(figsize=(max(7.2, 1.3 * len(SCHEMES)), 4.8))
            for i, s in enumerate(SCHEMES):
                for j, (bud, alpha) in enumerate(((short, 0.35), (long, 0.9))):
                    v = fin[bud].xs((s, xv), level=("scheme", "x"))[col].to_numpy()
                    pos = i + 1 + (j - 0.5) * 0.38
                    m, c = v.mean(), ci95(v)
                    ax.scatter(pos + rng.uniform(-0.07, 0.07, len(v)), v, s=18,
                               color=SCHEME_COLORS[s], alpha=alpha, lw=0, zorder=3)
                    ax.errorbar(pos, m, yerr=c, fmt="_", color="black", ms=18,
                                mew=2.5, capsize=5, lw=1.6, zorder=4)
            ax.set_xticks(range(1, len(SCHEMES) + 1))
            ax.set_xticklabels([label(s) for s in SCHEMES], rotation=45, ha="right", rotation_mode="anchor")
            ax.set_ylabel(ylabel)
            style_grid(ax)
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.grid(axis="x", which="both", visible=False)
            save(fig, f"final_{name}_{x_tag(xv)}", kind="boxplot", metric=col, x=xv,
                 caption=f"{ylabel.replace(chr(10), ' ')} at generation 30, x={xv}; per scheme, left = "
                         f"{budget_name(short)} (faint), right = {budget_name(long)}; one point per run, "
                         "bar = mean with 95% CI.")


def plot_advantage(adv: pd.DataFrame, short: str, long: str, save: Saver) -> None:
    for col, name, _ in FINAL:
        for kind, ylabel in (("abs", "Advantage over\nDarwinian (m)"), ("rel", "Relative advantage\nover Darwinian")):
            for xv in sorted(adv.x.unique()):
                a = adv[(adv.x == xv) & (adv.metric == col)].set_index("scheme")
                fig, ax = plt.subplots()
                for i, s in enumerate(SCHEMES[1:]):
                    for j, (tag, mk) in enumerate((("short", "o"), ("long", "s"))):
                        m = a.loc[s, f"{kind}_{tag}"]
                        err = [[m - a.loc[s, f"{kind}_{tag}_lo"]], [a.loc[s, f"{kind}_{tag}_hi"] - m]]
                        ax.errorbar(i + (j - 0.5) * 0.3, m, yerr=err, fmt=mk, color=SCHEME_COLORS[s],
                                    mfc="white" if j == 0 else SCHEME_COLORS[s], ms=10, mew=2, capsize=4, lw=2)
                ax.axhline(0, color=SCHEME_COLORS["darwinian"], lw=2.5)
                ax.set_xticks(range(len(SCHEMES) - 1))
                ax.set_xticklabels([label(s) for s in SCHEMES[1:]], rotation=45, ha="right",
                                   rotation_mode="anchor")
                if kind == "rel":
                    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
                ax.set_ylabel(ylabel)
                style_grid(ax)
                ax.tick_params(axis="x", which="minor", bottom=False)
                ax.grid(axis="x", which="both", visible=False)
                save(fig, f"advantage_{kind}_{name}_{x_tag(xv)}", kind="advantage", metric=col, x=xv,
                     caption=f"{'Absolute' if kind == 'abs' else 'Relative'} advantage over Darwinian in "
                             f"{name.replace('_', ' ')} displacement, x={xv}; open = {budget_name(short)}, "
                             f"filled = {budget_name(long)}; bootstrap 95% CI.")


def plot_truncation(means: dict, short: str, long: str, save: Saver) -> None:
    for wname, rb in means.items():
        for xv in sorted(rb[short].index.unique("x")):
            for kind, ylabel in (("learning_curve", "Mean offspring\nfitness (m)"),
                                 ("advantage", "Advantage over\nDarwinian (m)")):
                fig, ax = plt.subplots()
                for bud, ls in ((long, "-"), (short, "--")):
                    rx = rb[bud].xs(xv, level="x")
                    budgets = np.array(rx.columns, dtype=int)
                    darw = rx.xs("darwinian", level="scheme").to_numpy().mean(axis=0)
                    for s in SCHEMES:
                        v = rx.xs(s, level="scheme").to_numpy()
                        m = v.mean(axis=0) - (darw if kind == "advantage" else 0)
                        if kind == "advantage" and s == "darwinian":
                            continue
                        if ls == "-":
                            c = np.array([ci95(v[:, j]) for j in range(v.shape[1])])
                            ax.fill_between(budgets, m - c, m + c, color=SCHEME_COLORS[s], alpha=0.10, lw=0)
                        ax.plot(budgets, m, color=SCHEME_COLORS[s], ls=ls)
                if kind == "advantage":
                    ax.axhline(0, color=SCHEME_COLORS["darwinian"], lw=2.5)
                b_short = int(rb[short].columns[-1])
                ax.axvline(b_short, color="0.4", lw=1.2, ls=":")
                ax.set_xscale("symlog", linthresh=1, linscale=0.4)
                ax.set_xlim(0, int(rb[long].columns[-1]))
                ax.xaxis.set_major_locator(mticker.FixedLocator([0, 1, 2, 5, 10, 20]))
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
                ax.set_xlabel("Learning budget (CMA-ES generations)")
                ax.set_ylabel(ylabel)
                style_grid(ax)
                ax.xaxis.set_minor_locator(mticker.FixedLocator([3, 4, 6, 7, 8, 9]))
                save(fig, f"truncation_{kind}_{wname}_{x_tag(xv)}", kind=kind, x=xv,
                     caption=f"{ylabel.replace(chr(10), ' ')} vs. learning budget, x={xv}, offspring of "
                             f"window {wname}; solid = {budget_name(long)} runs (shaded 95% CI), "
                             f"dashed = {budget_name(short)} runs.")


def plot_pre_learning(curves: dict[str, pd.DataFrame], short: str, long: str, save: Saver) -> None:
    for col, name, ylabel in (("born_init_disp", "pre_learning", "Offspring fitness\nbefore learning (m)"),
                              ("born_learn_gain", "learning_gain", "Lifetime-learning\ngain (m)")):
        for xv in sorted(curves[short].x.unique()):
            fig, ax = plt.subplots()
            for bud, ls in ((long, "-"), (short, "--")):
                c = curves[bud]
                for s in SCHEMES:
                    m = c[(c.scheme == s) & (c.x == xv) & (c.gen > 0)].groupby("gen")[col].mean()
                    ax.plot(m.index, m, color=SCHEME_COLORS[s], ls=ls)
            ax.set_xlim(1, 30)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
            ax.set_xlabel("Generation")
            ax.set_ylabel(ylabel)
            style_grid(ax)
            save(fig, f"{name}_{x_tag(xv)}", kind="curve", metric=col, x=xv,
                 caption=f"{ylabel.replace(chr(10), ' ')} per generation, x={xv}; mean over runs; solid = "
                         f"{budget_name(long)}, dashed = {budget_name(short)}.")


def plot_budget_legend(short: str, long: str, save: Saver) -> None:
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.legend([Line2D([], [], color="0.2", lw=3, ls="--"), Line2D([], [], color="0.2", lw=3)],
               [budget_name(short), budget_name(long)], ncol=2, frameon=False, handlelength=2.6,
               columnspacing=1.5, loc="center")
    save(fig, "legend_budgets", kind="legend")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = "__data__/social/figures/zoo_budget"
    parser.add_argument("--report-root", default=f"{root}/report")
    parser.add_argument("--budget-root", default=f"{root}/learning_budget")
    parser.add_argument("--paper-dir", default=f"{root}/paper")
    parser.add_argument("--out-dir", default=f"{root}/compare")
    parser.add_argument("--budgets", nargs=2, default=["ig10", "ig20"], metavar=("SHORT", "LONG"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    short, long = args.budgets
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    curves, runs, npz = {}, {}, {}
    for bud in (short, long):
        rep = Path(args.report_root) / bud
        curves[bud] = pd.read_csv(rep / "parta_curves_per_run.csv")
        runs[bud] = pd.read_csv(rep / "parta_included_runs.csv")
        z = np.load(Path(args.budget_root) / bud / "budget_curves.npz", allow_pickle=False)
        npz[bud] = (pd.DataFrame({k: z[k] for k in ("scheme", "x", "rep", "gen")}), z["B"])

    mech, donors, share, kw, pairs = [], [], [], [], []
    for bud in (short, long):
        mech += mechanism(curves[bud], bud)
        donors += donor_is_parent(runs[bud], bud)
        share += first_gen_share(*npz[bud], bud)
        kw += kruskal_social(curves[bud], bud)
        pairs += pairwise_social(curves[bud], bud)
    pd.DataFrame(mech).to_csv(out / "mechanism.csv", index=False)
    d = pd.DataFrame(donors)
    d.to_csv(out / "donor_is_parent_per_run.csv", index=False)
    (d.groupby(["budget", "scheme", "x"]).agg(parent_share=("parent_share", "mean"), births=("births", "sum"))
     .reset_index().to_csv(out / "donor_is_parent.csv", index=False))
    pd.DataFrame(share).to_csv(out / "first_gen_share.csv", index=False)
    pd.DataFrame(kw).to_csv(out / "kruskal_social.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "pairwise_social.csv", index=False)

    fin = {bud: final_values(curves[bud]) for bud in (short, long)}
    budget_final(fin, short, long).to_csv(out / "budget_final.csv", index=False)
    adv = budget_advantage(fin, short, long)
    adv.to_csv(out / "budget_advantage.csv", index=False)
    budget_groups(fin, short, long).to_csv(out / "budget_groups.csv", index=False)
    budget_headstart(curves, short, long).to_csv(out / "budget_headstart.csv", index=False)
    trunc, means = truncation(npz, short, long)
    trunc.to_csv(out / "truncation.csv", index=False)
    print(f"Saved CSVs to {out}")

    save = Saver(Path(args.paper_dir), args.dpi, "compare")
    with plt.rc_context(PAPER_RC):
        plot_budget_legend(short, long, save)
        plot_final_pairs(fin, short, long, save)
        plot_advantage(adv, short, long, save)
        plot_truncation(means, short, long, save)
        plot_pre_learning(curves, short, long, save)
    save.write_manifest()


if __name__ == "__main__":
    main()
