"""Part-A learning-budget analysis: how much does each inheritance scheme's
head start matter as the lifetime-learning budget grows?

Every offspring stores its full inner CMA-ES history (tags['learning_curve'],
inner_gens x inner_pop episode fitnesses) plus the fitness of its inherited
brain before learning (tags['init_fitness']). From these we rebuild, per
robot, the best fitness it would have had if learning had stopped after b
inner generations:

    B(0) = init_fitness                       (inherited brain, no learning)
    B(b) = max fitness over inner gens 1..b   (b = 1..inner_gens)

B(inner_gens) is the robot's recorded distance (fitness scale: the episode
fitness incl. the jerk penalty, same scale as the pre-learning plots) -- except
for runs made with experiment.py --final-evals, whose recorded distance is the
mean over perturbed re-evaluations while B(b) stays CMA-ES's best single
episode, so B(inner_gens) is then distance_cma, an optimistic score.

This is a *retrospective truncation* of the runs made with the full budget:
selection and the donor brains on offer were shaped by full-budget learning.
It shows how the within-lifetime advantage of each scheme evolves with the
budget, not what a full re-run with a smaller budget would produce.

Stage 1 (slow, cached): extract B(b) for every offspring of every included run
    -> <out-dir>/budget_curves.npz
Stage 2: figures and tables
    <out-dir>/{pdf,png}/<window>/learning_curve_x{05,10}       mean B(b) per scheme
    <out-dir>/{pdf,png}/<window>/advantage_x{05,10}            B(b) - Darwinian B(b)
    <out-dir>/{pdf,png}/<window>/gap_to_similar_x{05,10}       B(b) - Similar B(b)
    <out-dir>/budget_stats.csv      per budget: Kruskal-Wallis over schemes and
                                    each scheme vs Similar (A12, Mann-Whitney)
    <out-dir>/catch_up.csv          per scheme: the budget from which its mean
                                    curve stays at or above Similar's
                                    (overtakes; empty = never), and the budget
                                    from which it is no longer significantly
                                    below Similar (Mann-Whitney p >= 0.05 at
                                    every larger budget)

The inner budget is read from the data (learning_curve length), so runs with
any --inner-gens work; budgets beyond it are dropped from the tables. With
--prefix P, figures go to <out-dir>/{pdf,png}/P/<window>/... and the cache and
CSVs to <out-dir>/P/.

Usage:
    uv run examples/d_social_learning/analysis/learning_budget.py \
        --report-dir __data__/social/figures/parta/report \
        --out-dir __data__/social/figures/parta/learning_budget

    uv run examples/d_social_learning/analysis/learning_budget.py \
        --report-dir __data__/social/figures/zoo_budget/report/ig10 \
        --out-dir __data__/social/figures/zoo_budget/learning_budget --prefix ig10 \
        --run-novelty TOPDOWN \
        --schemes darwinian lamarckian similar_TOPDOWN best_many random_many
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from paper_figures import PAPER_RC, Saver, label, style_grid, x_tag  # noqa: E402
from curve_utils import read_individuals  # noqa: E402
from parta_report import SCHEME_COLORS, a12, ci95, schemes_for  # noqa: E402



def windows(gens: int) -> dict[str, tuple[int, int]]:
    """Generation windows (by offspring birth generation) the curves are
    averaged over: the whole run and its thirds."""
    t = gens // 3
    return {"all": (1, gens), "early": (1, t), "mid": (t + 1, 2 * t), "late": (2 * t + 1, gens)}


WINDOWS = windows(30)  # set from the data in main()
BUDGETS = [0, 1, 2, 5, 10, 20, 30, 50]  # inner generations reported in budget_stats.csv
REFERENCE = "similar_STRUCT"  # scheme the catch-up point is measured against (--reference)


# ---------------------------------------------------------------------------
# Stage 1: extraction
# ---------------------------------------------------------------------------

def _extract_run(job: tuple) -> tuple[list[tuple], np.ndarray]:
    scheme, x, rep, db = job
    rows = read_individuals(
        db, "time_of_birth, json_extract(tags_, '$.init_fitness'), json_extract(tags_, '$.learning_curve')",
        "fitness_ IS NOT NULL AND time_of_birth > 0",
    )
    meta, curves = [], []
    for _, gen, init, lc in rows.itertuples(index=False):
        lc = np.array(json.loads(lc), dtype=np.float64) if isinstance(lc, str) else np.empty((0, 0))
        if pd.isna(init) or lc.size == 0 or not np.isfinite(init):
            continue  # failed / timed-out evaluation
        best = np.maximum.accumulate(np.nanmax(lc, axis=1))
        meta.append((scheme, x, rep, gen))
        curves.append(np.concatenate([[init], best]))
    return meta, np.array(curves, dtype=np.float32)


def extract(runs: pd.DataFrame, cache: Path, workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        meta = pd.DataFrame({k: z[k] for k in ("scheme", "x", "rep", "gen")})
        return meta, z["B"]
    jobs = [(r.scheme, r.x, r.rep, r.db) for r in runs.itertuples()]
    with Pool(workers) as pool:
        parts = pool.map(_extract_run, jobs, chunksize=1)
    meta = pd.DataFrame([m for ms, _ in parts for m in ms], columns=["scheme", "x", "rep", "gen"])
    B = np.concatenate([c for _, c in parts if len(c)])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, B=B, **{k: (meta[k].to_numpy(dtype=str) if k == "scheme" else meta[k].to_numpy()) for k in meta})
    print(f"Saved {cache} ({len(meta)} offspring)")
    return meta, B


# ---------------------------------------------------------------------------
# Stage 2: per-run curves, figures, stats
# ---------------------------------------------------------------------------

def per_run_means(meta: pd.DataFrame, B: np.ndarray, window: tuple[int, int]) -> pd.DataFrame:
    """Mean B(b) over the offspring born in `window`, one row per run."""
    lo, hi = window
    sel = ((meta.gen >= lo) & (meta.gen <= hi)).to_numpy()
    df = pd.DataFrame(B[sel], columns=range(B.shape[1]))
    keys = meta[sel].reset_index(drop=True)
    return pd.concat([keys[["scheme", "x", "rep"]], df], axis=1).groupby(["scheme", "x", "rep"]).mean()


def plot_window(runs_b: pd.DataFrame, schemes: list[str], wname: str, save: Saver) -> None:
    budgets = np.array(runs_b.columns, dtype=int)
    for xv in sorted(runs_b.index.unique("x")):
        rx = runs_b.xs(xv, level="x")
        darw = rx.xs("darwinian", level="scheme").to_numpy()
        ref = rx.xs(REFERENCE, level="scheme").to_numpy()
        for kind, ylabel in (("learning_curve", "Mean offspring\nfitness (m)"),
                             ("advantage", "Advantage over\nDarwinian (m)"),
                             ("gap_to_similar", "Difference to\nSimilar (m)")):
            base = {"advantage": ("darwinian", darw), "gap_to_similar": (REFERENCE, ref)}.get(kind)
            fig, ax = plt.subplots()
            for scheme in schemes:
                if base and scheme == base[0]:
                    ax.axhline(0, color=SCHEME_COLORS[scheme], lw=2.5)
                    continue
                v = rx.xs(scheme, level="scheme").to_numpy()
                m = v.mean(axis=0)
                c = np.array([ci95(v[:, j]) for j in range(v.shape[1])])
                if kind == "gap_to_similar":
                    # CI of a difference of two independent means.
                    c = np.hypot(c, [ci95(ref[:, j]) for j in range(ref.shape[1])])
                if base:
                    m = m - base[1].mean(axis=0)
                color = SCHEME_COLORS[scheme]
                ax.fill_between(budgets, m - c, m + c, color=color, alpha=0.12, lw=0)
                ax.plot(budgets, m, color=color)
            ax.set_xscale("symlog", linthresh=1, linscale=0.4)
            ax.set_xlim(0, budgets[-1])
            ax.xaxis.set_major_locator(mticker.FixedLocator(
                [b for b in (0, 1, 2, 5, 10, 20, 50) if b < budgets[-1]] + [budgets[-1]]))
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_xlabel("Learning budget (CMA-ES generations)")
            ax.set_ylabel(ylabel)
            style_grid(ax)
            ax.xaxis.set_minor_locator(mticker.FixedLocator(
                [b for b in (3, 4, 6, 7, 8, 9, 30, 40) if b < budgets[-1]]))
            caption = (f"{ylabel.replace(chr(10), ' ')} vs. learning budget, x={xv}, offspring born in "
                       f"generations {WINDOWS[wname][0]}-{WINDOWS[wname][1]}; mean over runs, shaded 95% CI"
                       + {"advantage": " (of the scheme, not of the difference).",
                          "gap_to_similar": " of the difference."}.get(kind, "."))
            save(fig, f"{wname}/{kind}_{x_tag(xv)}", kind=kind, metric="B(b)", x=xv,
                 ylabel=ylabel.replace("\n", " "), caption=caption)


def budget_stats(runs_b: pd.DataFrame, schemes: list[str], wname: str) -> list[dict]:
    rows = []
    full = int(runs_b.columns[-1])
    for xv in sorted(runs_b.index.unique("x")):
        rx = runs_b.xs(xv, level="x")
        for b in sorted({b for b in BUDGETS if b < full} | {full}):
            groups = {s: rx.xs(s, level="scheme")[b].to_numpy() for s in schemes}
            kw = stats.kruskal(*groups.values()).pvalue
            kw_social = stats.kruskal(*[g for s, g in groups.items() if s != "darwinian"]).pvalue
            ref = groups[REFERENCE]
            for s, g in groups.items():
                rows.append(dict(
                    window=wname, x=xv, budget=b, scheme=s, n=len(g), mean=g.mean(),
                    a12_vs_similar=a12(g, ref) if s != REFERENCE else np.nan,
                    p_vs_similar=stats.mannwhitneyu(g, ref).pvalue if s != REFERENCE else np.nan,
                    kruskal_p=kw, kruskal_p_social=kw_social,
                ))
    return rows


def catch_up(runs_b: pd.DataFrame, schemes: list[str], wname: str) -> list[dict]:
    """Per scheme vs Similar: the budget from which its mean curve stays at or
    above Similar's, and the budget from which it is no longer significantly
    below it (one-sided Mann-Whitney p >= 0.05 at that and every larger budget)."""

    def first_from(ok: np.ndarray) -> int | None:
        if not ok[-1]:
            return None  # not true even at the full budget
        return 0 if ok.all() else int(np.where(~ok)[0][-1] + 1)

    rows = []
    for xv in sorted(runs_b.index.unique("x")):
        rx = runs_b.xs(xv, level="x")
        mean = rx.groupby(level="scheme").mean()
        ref_runs = rx.xs(REFERENCE, level="scheme").to_numpy()
        ref = ref_runs.mean(axis=0)
        for s in schemes:
            if s == REFERENCE:
                continue
            v = rx.xs(s, level="scheme").to_numpy()
            p_below = np.array([stats.mannwhitneyu(v[:, j], ref_runs[:, j], alternative="less").pvalue
                                for j in range(v.shape[1])])
            rows.append(dict(window=wname, x=xv, scheme=s,
                             overtakes_from=first_from(mean.loc[s].to_numpy() >= ref),
                             not_sig_below_from=first_from(p_below >= 0.05),
                             gap_b0=mean.loc[s, 0] - ref[0],
                             gap_b1=mean.loc[s, 1] - ref[1],
                             gap_b5=mean.loc[s, 5] - ref[5],
                             gap_full=mean.loc[s].iloc[-1] - ref[-1]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="__data__/social/figures/parta/report")
    parser.add_argument("--out-dir", default="__data__/social/figures/parta/learning_budget")
    parser.add_argument("--run-novelty", default="STRUCT", choices=["DESCR", "STRUCT", "TOPDOWN"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--schemes", nargs="+", default=None,
                        help="Schemes, darwinian first (default: part A's eight).")
    parser.add_argument("--reference", default=None,
                        help="Scheme for the catch-up analysis (default: similar_<run-novelty>).")
    parser.add_argument("--prefix", default="", help="Extra path level for figures and CSVs.")
    args = parser.parse_args()
    global REFERENCE, WINDOWS
    REFERENCE = args.reference or f"similar_{args.run_novelty}"

    out = Path(args.out_dir)
    data_out = out / args.prefix if args.prefix else out
    runs = pd.read_csv(Path(args.report_dir) / "parta_included_runs.csv")
    meta, B = extract(runs, data_out / "budget_curves.npz", args.workers)
    schemes = args.schemes or schemes_for(args.run_novelty)
    WINDOWS = windows(int(meta.gen.max()))

    save = Saver(out, args.dpi, args.prefix)
    stat_rows, catch_rows = [], []
    with plt.rc_context(PAPER_RC):
        for wname, window in WINDOWS.items():
            runs_b = per_run_means(meta, B, window)
            plot_window(runs_b, schemes, wname, save)
            stat_rows += budget_stats(runs_b, schemes, wname)
            catch_rows += catch_up(runs_b, schemes, wname)
    save.write_manifest()
    pd.DataFrame(stat_rows).to_csv(data_out / "budget_stats.csv", index=False)
    pd.DataFrame(catch_rows).to_csv(data_out / "catch_up.csv", index=False)
    print(f"Saved {data_out / 'budget_stats.csv'} and {data_out / 'catch_up.csv'}")


if __name__ == "__main__":
    main()
