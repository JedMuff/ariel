"""
E2: fixed-body learning curves (R2, R4).

One training = one body x brain x seed (train_fixed_body.py). Per training,
from curve.jsonl (score = -best fitness so far):
  score_at_<N>   score after N evaluations (N in CHECKPOINTS, if reached)
  final          score at the end of training
  auc            mean score over the curve's evaluation grid (learning speed)
  evals_to_half  evaluations to reach 50% of the final score
A training whose CMA-ES stopped early (flat fitness) keeps its final score to
the end of the budget (stopped_early). Seeds are averaged per body and brain.
Bodies differ between genomes, so the symmetric - tree gaps are unpaired:
difference of body means, bootstrap CI over bodies, Mann-Whitney p.
hinge_matched_gap averages the per-hinge-count gaps over the hinge counts both
genomes have (weighted by bodies), the gap at matched size for evolved bodies
(random bodies are matched by construction).

Writes to --out (default <first root>/analysis):
  e2_trainings.csv, e2_bodies.csv, e2_gaps.csv
  fig5_learning_curves.png  mean score vs evaluations (± 1 SEM over bodies), tree vs
                            symmetric, one panel per brain, one row per body source
  fig2_bodies.png           module, hinge and limb counts by genome and body source

Usage:
    python analyse_e2.py --roots /scratch/$USER/ariel_paper_e2/random /scratch/$USER/ariel_paper_e2/evolved
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

import runs  # noqa: E402

CHECKPOINTS = (200, 800, 1600, 5000)
POPSIZE = 20  # sim.PROFILE.cma_popsize: evaluations per CMA-ES generation
GENOMES = ("tree", "tree_symmetric")
METRICS = ["final", "auc", "evals_to_half", *(f"score_at_{n}" for n in CHECKPOINTS)]


def load_trainings(roots: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per training, long-form curves)."""
    rows, curves = [], []
    for root in roots:
        for d in sorted(root.glob("*/final.json")):
            d = d.parent
            cfg = json.loads((d / "run_config.json").read_text())
            c = pd.DataFrame([json.loads(line) for line in (d / "curve.jsonl").open()])
            # A training CMA-ES stopped early (flat fitness) keeps its final score to the end of the budget.
            grid = np.arange(1, cfg["skill_budget"] + 1) * POPSIZE
            stopped = int(c["evals"].iloc[-1]) < grid[-1]
            c = c.set_index("evals").reindex(grid).ffill().rename_axis("evals").reset_index()
            c["score"] = -c["best_so_far"]
            final = float(c["score"].iloc[-1])
            half = c.loc[c["score"] >= 0.5 * final, "evals"]
            row = {k: cfg[k] for k in ("body_id", "body_source", "genome_type", "brain", "seed",
                                       "n_modules", "n_hinges", "n_limbs")}
            fin = json.loads((d / "final.json").read_text())
            row.update({"fitness_mode": cfg.get("fitness", "normalised"),
                        "final_retest": fin.get("retest_mean", math.nan), "retest_sd": fin.get("retest_sd", math.nan)})
            row.update({"stopped_early": stopped, "final": final, "auc": float(c["score"].mean()),
                        "evals_to_half": int(half.iloc[0]) if final > 0 and len(half) else math.nan})
            for n in CHECKPOINTS:
                hit = c.loc[c["evals"] >= n, "score"]
                row[f"score_at_{n}"] = float(hit.iloc[0]) if len(hit) else math.nan
            rows.append(row)
            curves.append(c[["evals", "score"]].assign(**{k: row[k] for k in ("body_id", "body_source", "genome_type",
                                                                             "brain", "seed")}))
    return pd.DataFrame(rows), (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


def unpaired_gap(sym: np.ndarray, tree: np.ndarray, rng: np.random.Generator) -> dict:
    sym, tree = sym[np.isfinite(sym)], tree[np.isfinite(tree)]
    if len(sym) < 2 or len(tree) < 2:
        return {"gap": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "p": math.nan, "n_sym": len(sym), "n_tree": len(tree)}
    boot = rng.choice(sym, (runs.N_BOOT, len(sym))).mean(1) - rng.choice(tree, (runs.N_BOOT, len(tree))).mean(1)
    return {"gap": float(sym.mean() - tree.mean()), "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)), "p": float(mannwhitneyu(sym, tree).pvalue),
            "n_sym": len(sym), "n_tree": len(tree)}


def hinge_matched_gap(g: pd.DataFrame, metric: str) -> float:
    per_bin = g.groupby(["n_hinges", "genome_type"])[metric].agg(["mean", "count"]).unstack().dropna()
    if per_bin.empty or not {("count", "tree"), ("count", "tree_symmetric")} <= set(per_bin.columns):
        return math.nan
    w = per_bin[("count", "tree")] + per_bin[("count", "tree_symmetric")]
    d = per_bin[("mean", "tree_symmetric")] - per_bin[("mean", "tree")]
    return float((d * w).sum() / w.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.roots[0] / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    tr, curves = load_trainings(args.roots)
    if tr.empty:
        raise SystemExit("no finished trainings")
    tr.to_csv(out / "e2_trainings.csv", index=False)
    print(f"{len(tr)} trainings, {int(tr['stopped_early'].sum())} stopped early (flat fitness)")
    keys = ["body_id", "body_source", "genome_type", "brain", "n_modules", "n_hinges", "n_limbs"]
    bodies = tr.groupby(keys, as_index=False)[METRICS].mean()
    bodies.to_csv(out / "e2_bodies.csv", index=False)

    rng = np.random.default_rng(0)
    gaps = []
    for (source, brain), g in bodies.groupby(["body_source", "brain"]):
        for m in METRICS:
            sym = g.loc[g["genome_type"] == "tree_symmetric", m].to_numpy(float)
            tree = g.loc[g["genome_type"] == "tree", m].to_numpy(float)
            gaps.append({"body_source": source, "brain": brain, "metric": m, **unpaired_gap(sym, tree, rng),
                         "hinge_matched_gap": hinge_matched_gap(g, m)})
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(out / "e2_gaps.csv", index=False)
    print("symmetric - tree, per body source and brain:")
    print(gaps[gaps["metric"].isin(["final", "auc", "evals_to_half"])].round(3).to_string(index=False))

    # Learning curves: per body, the seed mean on the common evaluation grid.
    sources = sorted(curves["body_source"].unique())
    brains = [b for b in runs.BRAIN_LABEL if b in set(curves["brain"])]
    fig, axes = plt.subplots(len(sources), len(brains), figsize=(4.2 * len(brains), 3.2 * len(sources)),
                             squeeze=False, constrained_layout=True)
    for (si, source), (bi, brain) in [((si, s), (bi, b)) for si, s in enumerate(sources) for bi, b in enumerate(brains)]:
        ax = axes[si, bi]
        runs.style(ax)
        c = curves[(curves["body_source"] == source) & (curves["brain"] == brain)]
        for genome in GENOMES:
            body = c[c["genome_type"] == genome].groupby(["body_id", "evals"])["score"].mean().unstack()
            if body.empty:
                continue
            m, sem = body.mean(), body.std() / np.sqrt(body.count())
            color = runs.GENOME_COLOR[genome]
            ax.fill_between(m.index, m - sem, m + sem, color=color, alpha=0.2, linewidth=0)
            ax.plot(m.index, m, color=color, linewidth=2,
                    label=f"{runs.GENOME_LABEL[genome]} (n={len(body)})")
        ax.set_xscale("log")
        ax.set_title(f"{runs.BRAIN_LABEL[brain]} — {source} bodies", fontsize=10, color=runs.INK)
        if bi == 0:
            ax.set_ylabel("best score so far", fontsize=9, color=runs.INK_2)
        if si == len(sources) - 1:
            ax.set_xlabel("evaluations", fontsize=9, color=runs.INK_2)
        ax.legend(fontsize=8, frameon=False, labelcolor=runs.INK_2)
    fig.savefig(out / "fig5_learning_curves.png", dpi=200)
    plt.close(fig)

    # Body size by genome and source (one row per body).
    size = tr.drop_duplicates("body_id")
    cols = [("n_modules", "modules"), ("n_hinges", "hinges"), ("n_limbs", "limbs")]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.0), constrained_layout=True)
    groups = [(s, g) for s in sources for g in GENOMES]
    for ax, (col, label) in zip(axes, cols):
        runs.style(ax)
        data = [size.loc[(size["body_source"] == s) & (size["genome_type"] == g), col] for s, g in groups]
        parts = ax.boxplot(data, widths=0.6, patch_artist=True, medianprops={"color": runs.INK})
        for patch, (_, g) in zip(parts["boxes"], groups):
            patch.set_facecolor(runs.GENOME_COLOR[g])
            patch.set_edgecolor("white")
        ax.set_xticks(range(1, len(groups) + 1), [f"{runs.GENOME_LABEL[g]}\n{s}" for s, g in groups], fontsize=7)
        ax.set_title(label, fontsize=10, color=runs.INK)
    fig.savefig(out / "fig2_bodies.png", dpi=200)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
