"""Part-A report: fitness, displacement and novelty curves for every scheme,
overlaid per x value, plus end-of-run statistics.

Only complete runs (every generation up to --gens evaluated) of one run
novelty metric (--run-novelty, read from the run dir's ``_nov{M}_`` label)
are used. When a (scheme, x, rep) was run more than once, the newest
timestamped run dir wins, so stale reruns never double-count.

Curves are per generation, mean across reps with a 95% CI band:
  - survivors: the mu individuals alive after selection at generation g
  - best-so-far displacement: cumulative max over every individual born <= g
  - novelty / fitness: per-generation selection-pool values from the
    database's novelty_history table (core/novelty_history.py) for
    --novelty-metric (default TOPDOWN); runs made before that table existed
    get it from analysis/backfill_novelty.py
  - inherited displacement: tags['init_fitness'], the displacement an
    offspring achieves with its inherited brain *before* lifetime learning

Fitness is the pool-normalised blend x * distance + (1 - x) * novelty, so it is
only comparable within one x. For backfilled history whose metric differs from
the run's, it is the fitness selection *would* have seen, not what it used.

Usage:
    uv run examples/d_social_learning/analysis/parta_report.py \
        --data-dir __data__/social/ariel --run-novelty STRUCT \
        --out-dir __data__/social/figures/parta/report
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.novelty_history import read_run_history  # noqa: E402

BASE_SCHEMES = ["darwinian", "lamarckian", "random", "random_many", "best", "best_many"]


def schemes_for(run_novelty: str) -> list[str]:
    """Scheme list for runs of one novelty metric: the similar_* schemes use
    the run's structural metric (DESCR runs use the descriptor variants)."""
    return BASE_SCHEMES + [f"similar_{run_novelty}", f"similar_many_{run_novelty}"]


SCHEMES = schemes_for("TOPDOWN")  # set from --run-novelty in main()
X_VALUES = [0.0, 0.5, 1.0]
# Fixed categorical order (validated palette), one hue per scheme slot.
_SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SCHEME_COLORS = {
    scheme: _SLOT_COLORS[i]
    for metric in ("DESCR", "STRUCT", "TOPDOWN")
    for i, scheme in enumerate(schemes_for(metric))
}
X_LABEL = {0.0: "x=0 (novelty only)", 0.5: "x=0.5 (blend)", 1.0: "x=1 (displacement only)"}

# Must match sim_ariel/evaluator.py when the data was generated (see curve_utils).
JERK_PENALTY_WEIGHT = 3.0
JERK_THRESHOLD = 0.15

RUN_RE = re.compile(r"rep_(\d+)_.*nov([A-Z]+)_(\d{8}_\d{6})$")


def discover_runs(data_dir: Path, gens: int, run_novelty: str) -> pd.DataFrame:
    rows = []
    for db in data_dir.glob("*/x*/k*/rep_*/database.db"):
        m = RUN_RE.search(db.parent.name)
        if not m or m[2] != run_novelty:
            continue
        scheme = db.parents[3].name
        x = int(db.parents[2].name[1:]) / 10
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        n_done = con.execute(
            "SELECT count(*) FROM individual WHERE time_of_birth = ? AND fitness_ IS NOT NULL",
            (gens,),
        ).fetchone()[0]
        con.close()
        rows.append(dict(scheme=scheme, x=x, rep=int(m[1]), stamp=m[3],
                         db=str(db), complete=n_done > 0))
    runs = pd.DataFrame(rows, columns=["scheme", "x", "rep", "stamp", "db", "complete"])
    runs = runs[runs.complete & runs.scheme.isin(schemes_for(run_novelty))]
    return runs.sort_values("stamp").drop_duplicates(["scheme", "x", "rep"], keep="last")


def load_run(job: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(individuals, novelty_history rows for metric) of one run."""
    scheme, x, rep, db, metric = job
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    df = pd.read_sql(
        """SELECT time_of_birth AS gen, time_of_death AS tod, alive,
                  json_extract(tags_, '$.distance') AS distance,
                  json_extract(tags_, '$.mean_jerk') AS jerk,
                  json_extract(tags_, '$.init_fitness') AS init_fitness
           FROM individual WHERE fitness_ IS NOT NULL""",
        con,
    )
    con.close()
    df["disp"] = df.distance + (JERK_PENALTY_WEIGHT * df.jerk).where(df.jerk >= JERK_THRESHOLD, 0.0)
    hist = read_run_history(Path(db).parent, metric)
    for d in (df, hist):
        d[["scheme", "x", "rep"]] = scheme, x, rep
    return df, hist


def per_run_curves(ind: pd.DataFrame, hist: pd.DataFrame, gens: int) -> pd.DataFrame:
    out = []
    for (scheme, x, rep), r in ind.groupby(["scheme", "x", "rep"]):
        disp_by_gen = r.groupby("gen").disp.max().reindex(range(gens + 1))
        best_so_far = disp_by_gen.cummax().to_numpy()
        for g in range(gens + 1):
            surv = r[(r.gen <= g) & ((r.tod > g) | ((r.alive == 1) & (g == gens)))]
            born = r[r.gen == g]
            out.append(dict(
                scheme=scheme, x=x, rep=rep, gen=g,
                surv_disp=surv.disp.mean(),
                surv_best_disp=surv.disp.max(),
                best_so_far_disp=best_so_far[g],
                born_init_disp=born.init_fitness.mean() if g > 0 else np.nan,
                born_learn_gain=(born.distance - born.init_fitness).mean() if g > 0 else np.nan,
            ))
    curves = pd.DataFrame(out)
    if hist.empty:
        return curves
    surv = hist[hist.survivor == 1]
    keys = ["scheme", "x", "rep", "generation"]
    # Offspring = pool members in their first generation in the pool.
    first = hist.groupby(["scheme", "x", "rep", "individual_id"]).generation.transform("min")
    offspring = hist[hist.generation == first]
    nov = pd.DataFrame({
        "pool_novelty": hist.groupby(keys).novelty.mean(),
        "surv_novelty": surv.groupby(keys).novelty.mean(),
        "surv_fitness": surv.groupby(keys).fitness.mean(),
        "offspring_fitness": offspring.groupby(keys).fitness.mean(),
    }).reset_index().rename(columns={"generation": "gen"})
    return curves.merge(nov, on=["scheme", "x", "rep", "gen"], how="left")


def ci95(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def plot_overlay(curves: pd.DataFrame, panels: list[tuple[str, str]], title: str, out: Path,
                 n_reps: pd.Series, x_values: list[float] = X_VALUES) -> None:
    fig, axes = plt.subplots(len(x_values), len(panels), figsize=(5.6 * len(panels), 3.3 * len(x_values)),
                             squeeze=False, sharex=True)
    fig.suptitle(title, fontsize=13)
    for xi, xv in enumerate(x_values):
        for pi, (col, label) in enumerate(panels):
            ax = axes[xi][pi]
            for scheme in SCHEMES:
                s = curves[(curves.scheme == scheme) & (curves.x == xv)]
                if s.empty:
                    continue
                g = s.groupby("gen")[col]
                m, c = g.mean(), g.agg(lambda v: ci95(v.to_numpy()))
                color = SCHEME_COLORS[scheme]
                ax.fill_between(m.index, m - c, m + c, color=color, alpha=0.12, lw=0)
                ax.plot(m.index, m, color=color, lw=2,
                        label=f"{scheme} (n={n_reps.get((scheme, xv), 0)})")
            ax.grid(alpha=0.25, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if xi == 0:
                ax.set_title(label, fontsize=10)
            if pi == 0:
                ax.set_ylabel(X_LABEL[xv], fontsize=9)
            if xi == len(x_values) - 1:
                ax.set_xlabel("generation", fontsize=9)
            if pi == len(panels) - 1:
                ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def a12(a: np.ndarray, b: np.ndarray) -> float:
    """Vargha-Delaney A12: P(a > b) + 0.5 P(a == b)."""
    gt = (a[:, None] > b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return gt + 0.5 * eq


def final_stats(curves: pd.DataFrame, gens: int, out_dir: Path, metric: str) -> pd.DataFrame:
    final = curves[curves.gen == gens]
    # Diversity summarised over the last third of the run (less noisy than one gen).
    if "surv_novelty" in curves:
        late_nov = (curves[curves.gen > gens * 2 // 3]
                    .groupby(["scheme", "x", "rep"]).surv_novelty.mean().rename("late_novelty"))
    metrics = ["best_so_far_disp", "surv_disp"]
    if "surv_novelty" in curves:
        final = final.set_index(["scheme", "x", "rep"]).join(late_nov).reset_index()
        metrics.append("late_novelty")
    rows = []
    for xv in X_VALUES:
        fx = final[final.x == xv]
        for metric in metrics:
            groups = [fx[fx.scheme == s][metric].dropna().to_numpy() for s in SCHEMES]
            kw = stats.kruskal(*[g for g in groups if len(g)]).pvalue
            base = groups[0]
            pvals = []
            for s, g in zip(SCHEMES, groups):
                p = stats.mannwhitneyu(g, base).pvalue if s != "darwinian" else np.nan
                pvals.append(p)
            # Holm correction across the 7 comparisons vs darwinian.
            finite = [(p, i) for i, p in enumerate(pvals) if np.isfinite(p)]
            order = sorted(finite)
            adj = [np.nan] * len(pvals)
            running = 0.0
            for rank, (p, i) in enumerate(order):
                running = max(running, min(1.0, p * (len(order) - rank)))
                adj[i] = running
            for s, g, p, pa in zip(SCHEMES, groups, pvals, adj):
                rows.append(dict(x=xv, metric=metric, scheme=s, n=len(g),
                                 median=np.median(g), mean=g.mean(),
                                 a12_vs_darwinian=a12(g, base) if s != "darwinian" else np.nan,
                                 p_holm_vs_darwinian=pa, kruskal_p=kw))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "parta_final_stats.csv", index=False)
    print(f"Saved {out_dir / 'parta_final_stats.csv'}")

    fig, axes = plt.subplots(len(metrics), len(X_VALUES), figsize=(15, 3.6 * len(metrics)), squeeze=False)
    labels = {"best_so_far_disp": "best displacement found (m), gen %d" % gens,
              "surv_disp": "survivor mean displacement (m), gen %d" % gens,
              "late_novelty": f"survivor {metric} novelty, mean of last third"}
    rng = np.random.default_rng(0)
    for mi, metric in enumerate(metrics):
        for xi, xv in enumerate(X_VALUES):
            ax = axes[mi][xi]
            fx = final[final.x == xv]
            data = [fx[fx.scheme == s][metric].dropna().to_numpy() for s in SCHEMES]
            bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                            medianprops=dict(color="black", lw=1.5))
            for patch, s in zip(bp["boxes"], SCHEMES):
                patch.set_facecolor(SCHEME_COLORS[s])
                patch.set_alpha(0.35)
            for i, (d, s) in enumerate(zip(data, SCHEMES)):
                ax.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(d)), d, s=9,
                           color=SCHEME_COLORS[s], zorder=3, lw=0)
                sig = table[(table.x == xv) & (table.metric == metric) & (table.scheme == s)].p_holm_vs_darwinian
                if len(sig) and np.isfinite(sig.iloc[0]) and sig.iloc[0] < 0.05:
                    ax.text(i + 1, 1.0, "*", transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=14)
            ax.set_xticks(range(1, len(SCHEMES) + 1))
            ax.set_xticklabels([re.sub(r"_(DESCR|STRUCT|TOPDOWN)$", "", s) for s in SCHEMES],
                               rotation=35, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if mi == 0:
                ax.set_title(X_LABEL[xv], fontsize=10)
            if xi == 0:
                ax.set_ylabel(labels[metric], fontsize=9)
    fig.suptitle("End-of-run comparison (* = Holm-corrected Mann-Whitney p<0.05 vs darwinian)", fontsize=12)
    fig.tight_layout()
    out = out_dir / "parta_final_boxplots.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return table


def main() -> None:
    global SCHEMES
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="__data__/social/ariel")
    parser.add_argument("--out-dir", default="__data__/social/figures/parta/report")
    parser.add_argument("--gens", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--run-novelty", default="TOPDOWN", choices=["DESCR", "STRUCT", "TOPDOWN"],
                        help="Only use runs made with this --novelty-metric (part A: STRUCT).")
    parser.add_argument("--novelty-metric", default="TOPDOWN", choices=["DESCR", "STRUCT", "TOPDOWN"],
                        help="novelty_history metric for the novelty/fitness curves.")
    args = parser.parse_args()
    SCHEMES = schemes_for(args.run_novelty)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(Path(args.data_dir), args.gens, args.run_novelty)
    runs.to_csv(out_dir / "parta_included_runs.csv", index=False)
    n_reps = runs.groupby(["scheme", "x"]).size()
    print(f"{len(runs)} complete nov{args.run_novelty} runs\n{n_reps.unstack()}")
    if runs.empty:
        return

    jobs = [(*r, args.novelty_metric) for r in runs[["scheme", "x", "rep", "db"]].itertuples(index=False)]
    with Pool(args.workers) as pool:
        loaded = pool.map(load_run, jobs)
    ind = pd.concat([d for d, _ in loaded])
    hist = pd.concat([h for _, h in loaded])
    curves = per_run_curves(ind, hist, args.gens)
    curves.to_csv(out_dir / "parta_curves_per_run.csv", index=False)

    n = len(runs)
    plot_overlay(curves, [("surv_disp", "survivor mean displacement (m)"),
                          ("best_so_far_disp", "best displacement found so far (m)")],
                 f"Displacement per generation — mean ±95% CI over reps ({n} runs)",
                 out_dir / "parta_displacement.png", n_reps)
    plot_overlay(curves, [("born_init_disp", "offspring displacement before learning (m)"),
                          ("born_learn_gain", "lifetime-learning gain (m)")],
                 "What offspring inherit vs. what they learn",
                 out_dir / "parta_inheritance.png", n_reps)

    if hist.empty:
        print(f"No run has {args.novelty_metric} novelty_history: skipping novelty/fitness plots "
              "(run analysis/backfill_novelty.py to get them)")
        final_stats(curves, args.gens, out_dir, args.novelty_metric)
        return
    hist_reps = hist.drop_duplicates(["scheme", "x", "rep"]).groupby(["scheme", "x"]).size()
    missing = n - int(hist_reps.sum())
    if missing:
        print(f"{missing} runs have no {args.novelty_metric} novelty_history: "
              "run analysis/backfill_novelty.py; novelty/fitness use the rest")
    sources = sorted(hist.source.unique())
    note = " (backfilled)" if sources == ["backfill"] else ""
    plot_overlay(curves, [("surv_novelty", f"survivor novelty ({args.novelty_metric})"),
                          ("pool_novelty", "selection-pool novelty (survivors + offspring)")],
                 f"Morphological novelty per generation — {args.novelty_metric}{note}",
                 out_dir / "parta_novelty.png", hist_reps)
    fitness_note = (" — backfilled: what selection would have seen under this metric"
                    if note and args.novelty_metric != args.run_novelty else note)
    plot_overlay(curves, [("surv_fitness", "survivor mean fitness"),
                          ("offspring_fitness", "offspring mean fitness (vs. the pool they enter)")],
                 f"Selection fitness per generation ({args.novelty_metric} novelty){fitness_note}\n"
                 "only comparable within an x row",
                 out_dir / "parta_fitness.png", hist_reps)
    final_stats(curves, args.gens, out_dir, args.novelty_metric)


if __name__ == "__main__":
    main()
