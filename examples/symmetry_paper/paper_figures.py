"""
Paper figures with no per-experiment script, and the numbers PAPER_HANDOFF.md
lists as "computed inline".

Reads the raw E2 / E3 / grid data plus the CSVs the analyse_*.py scripts write
under --analysis (default __data__/ariel_paper/paper, the layout of
paper/make_figures.sh):
  matched/matched_bodies.csv   analyse_matched.py
  bodies/body_features.csv     analyse_bodies.py
  e0/e0_grid_cells.csv         analyse_e0.py
  e1/e1_champions.csv          analyse_e1.py --behaviour

Figures (to --out):
  fig_curves_sets.png     R2: best score so far vs evaluations for tree,
                          shape-matched tree, tied and symmetric bodies, per brain;
                          rows: score, gap to tree (95% bootstrap band for
                          symmetric), and the ratio mean / mean of tree bodies
                          at equal evaluations (from RATIO_MIN_EVALS on)
  fig_shape_measures.png  R3: rest-pose shape measures by body set
  fig_locked.png          R5: (a) final score of tree, unlocked and locked
                          symmetric E2 trainings; (b) phase-locked share of E1
                          champions per genome x controller
  fig_locked_seeds.png    appendix: locked seeds per symmetric body
  fig_gait_grid.png       with --gait-grid only (v1): bang_bang gait-class shares
                          per hinge-grid cell, tree vs symmetric

Numbers (to --out): inline_numbers.txt (readable) and inline_numbers.json.

Usage:
    python paper_figures.py [--data __data__/ariel_paper] [--analysis __data__/ariel_paper/paper] --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

import runs  # noqa: E402
from analyse_e2 import load_trainings  # noqa: E402
from analyse_e3 import CLASS_SHADE  # noqa: E402
from sample_bodies import SHAPE_MEASURES  # noqa: E402

LOCKED_MIN = 0.5
MAIN_BRAINS = ("bang_bang", "dann")
RATIO_MIN_EVALS = 50
SETS = ["tree", "tree (shape-matched)", "tree_tied", "tree_symmetric"]
SET_LABEL = {"tree": "tree", "tree (shape-matched)": "tree, shape-matched", "tree_tied": "tied",
             "tree_symmetric": "symmetric"}
SET_STYLE = {"tree": (runs.GENOME_COLOR["tree"], 1.0, "-"),
             "tree (shape-matched)": (runs.GENOME_COLOR["tree"], 0.5, "--"),
             "tree_tied": (runs.GENOME_COLOR["tree_tied"], 1.0, "-"),
             "tree_symmetric": (runs.GENOME_COLOR["tree_symmetric"], 1.0, "-")}
GAIT_ORDER = ["alternating", "rowing", "mixed", "aperiodic", "unpaired", "static"]

NUMBERS: dict[str, object] = {}
LINES: list[str] = []


def _jsonable(v: object) -> object:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def note(key: str, value: object, text: str) -> None:
    NUMBERS[key] = _jsonable(value)
    LINES.append(f"{key}: {text}")


def body_set(source: str, genome: str) -> str:
    return "tree (shape-matched)" if source == "shape_matched" else genome


def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def ols_coef(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, col: int = 1) -> tuple[float, float, float]:
    """Coefficient col of y ~ X (X includes the intercept), bootstrap 95% CI over rows."""
    def c(i: np.ndarray) -> float:
        return float(np.linalg.lstsq(X[i], y[i], rcond=None)[0][col])
    boot = [c(rng.integers(0, len(y), len(y))) for _ in range(2000)]
    return c(np.arange(len(y))), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ── R2: learning curves of the four body sets ────────────────────────────────


def fig_curves(e2_roots: list[Path], out: Path, rng: np.random.Generator) -> pd.DataFrame:
    tr, curves = load_trainings(e2_roots)
    tr["set"] = [body_set(s, g) for s, g in zip(tr["body_source"], tr["genome_type"])]
    curves["set"] = [body_set(s, g) for s, g in zip(curves["body_source"], curves["genome_type"])]
    curves = curves[curves["brain"].isin(MAIN_BRAINS)]
    # per body: seed mean on the common grid
    per_body = curves.groupby(["brain", "set", "body_id", "evals"])["score"].mean().reset_index()

    fig, axes = plt.subplots(3, len(MAIN_BRAINS), figsize=(4.6 * len(MAIN_BRAINS), 8.2), sharex="col",
                             squeeze=False, constrained_layout=True)
    for bi, brain in enumerate(MAIN_BRAINS):
        pb = per_body[per_body["brain"] == brain]
        wide = {s: pb[pb["set"] == s].pivot(index="body_id", columns="evals", values="score") for s in SETS}
        tree_mean = wide["tree"].mean()
        for s in SETS:
            w = wide[s]
            if w.empty:
                continue
            color, alpha, ls = SET_STYLE[s]
            m, sem = w.mean(), w.sem()
            ax = axes[0, bi]
            ax.fill_between(m.index, m - sem, m + sem, color=color, alpha=0.15 * alpha, linewidth=0)
            ax.plot(m.index, m, color=color, alpha=alpha, linestyle=ls, linewidth=2, label=f"{SET_LABEL[s]} (n={len(w)})")
            if s == "tree":
                continue
            axes[1, bi].plot(m.index, m - tree_mean, color=color, alpha=alpha, linestyle=ls, linewidth=2)
            # Ratio from RATIO_MIN_EVALS on: early tree means near 0 (or below,
            # with displacement fitness) make the ratio meaningless.
            r = (m / tree_mean)[m.index >= RATIO_MIN_EVALS]
            axes[2, bi].plot(r.index, r, color=color, alpha=alpha, linestyle=ls, linewidth=2)
            if s == "tree_symmetric":  # bootstrap band of the unpaired sym - tree gap per evaluation count
                a, b = w.to_numpy(), wide["tree"].to_numpy()
                ia = rng.integers(0, len(a), (1000, len(a)))
                ib = rng.integers(0, len(b), (1000, len(b)))
                boot = a[ia].mean(1) - b[ib].mean(1)
                axes[1, bi].fill_between(m.index, np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0),
                                         color=color, alpha=0.15, linewidth=0)
        axes[0, bi].set_title(runs.BRAIN_LABEL[brain], fontsize=10, color=runs.INK)
        axes[1, bi].axhline(0, color=runs.INK_2, linewidth=0.8)
        axes[2, bi].axhline(1, color=runs.INK_2, linewidth=0.8)
        for ax in axes[:, bi]:
            runs.style(ax)
            ax.set_xscale("log")
        axes[2, bi].set_xlabel("evaluations", fontsize=9, color=runs.INK_2)
    axes[0, 0].set_ylabel("best score so far\n(mean ± SEM over bodies)", fontsize=9, color=runs.INK_2)
    axes[1, 0].set_ylabel("gap to tree\n(band: sym 95% CI)", fontsize=9, color=runs.INK_2)
    axes[2, 0].set_ylabel("mean ÷ mean of tree\n(1 = no advantage)", fontsize=9, color=runs.INK_2)
    axes[0, 0].legend(fontsize=7, frameon=False, labelcolor=runs.INK_2)
    fig.savefig(out / "fig_curves_sets.png", dpi=200)
    plt.close(fig)

    # numbers: the gap after the first CMA-ES generation (20 evals, the fixed starting mean) and at checkpoints
    for brain in MAIN_BRAINS:
        pb = per_body[per_body["brain"] == brain]
        for n in (20, 200, 1600, 5000):
            at = pb[pb["evals"] == n].groupby("set")["score"].mean()
            if "tree" not in at:
                continue
            note(f"curve_{brain}_{n}", {s: round(float(at.get(s, math.nan)), 3) for s in SETS},
                 f"{brain} mean best-so-far at {n} evals: " +
                 ", ".join(f"{SET_LABEL[s]} {at.get(s, math.nan):.3f}" for s in SETS) +
                 f"; sym-tree {at['tree_symmetric'] - at['tree']:+.3f}, sym/tree {at['tree_symmetric'] / at['tree']:.2f}")
    stops = tr.groupby(["brain", "set"])["stopped_early"].mean().unstack().round(3)
    note("early_stops", stops.to_dict(), "share of trainings stopped early (flat fitness):\n" + stops.to_string())
    return tr


# ── R3: shape measures by set ────────────────────────────────────────────────


def fig_shapes(matched: pd.DataFrame, out: Path) -> None:
    b = matched.drop_duplicates("body_id")
    cols = [("height", "height (m)"), ("contact_len", "front–back contact spread (m)"), ("n_leg", "leg hinges"),
            ("n_leg_yaw", "yaw-axis leg hinges"), ("n_pitch", "lateral-axis hinges"), ("leg_reach", "leg reach (m)")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), constrained_layout=True)
    rng = np.random.default_rng(1)
    for ax, (col, label) in zip(axes.flat, cols):
        runs.style(ax)
        data = [b.loc[b["set"] == s, col].dropna().to_numpy() for s in SETS]
        parts = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": runs.INK})
        for i, (patch, s) in enumerate(zip(parts["boxes"], SETS)):
            color, alpha, _ = SET_STYLE[s]
            patch.set_facecolor(color)
            patch.set_alpha(0.35 * alpha + 0.15)
            patch.set_edgecolor(color)
            ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(data[i])), data[i], s=6, color=color, alpha=0.7 * alpha + 0.2,
                       linewidth=0)
        ax.set_xticks(range(1, len(SETS) + 1), [SET_LABEL[s].replace(", ", ",\n") for s in SETS], fontsize=7)
        ax.set_title(label, fontsize=9, color=runs.INK)
    fig.savefig(out / "fig_shape_measures.png", dpi=200)
    plt.close(fig)
    tab = b.groupby("set")[[c for c, _ in cols]].mean().reindex(SETS).round(3)
    note("shape_by_set", tab.to_dict(), "shape measures by set (mean):\n" + tab.T.to_string())


# ── R4: gait vs actuator ─────────────────────────────────────────────────────


def fig_gait_grid(individuals: Path, grid_cells: Path, out: Path) -> None:
    d = pd.read_csv(individuals)
    d = d[d["brain"] == "bang_bang"]
    d["arm"] = d["arm"].round(4)
    shares = pd.crosstab([d["arm"], d["kp"].round(3), d["kv"].round(3), d["genome"]], d["rowing_label"], normalize="index")
    shares = shares.reindex(columns=[c for c in GAIT_ORDER if c in shares]).fillna(0.0)
    shares.to_csv(out / "gait_grid_shares.csv")
    cells = pd.read_csv(grid_cells)
    cells = cells[cells["brain"] == "bang_bang"].assign(arm=lambda x: x["armature"].round(4), kp=lambda x: x["kp"].round(3),
                                                        kv=lambda x: x["kv"].round(3))
    arms = sorted(shares.index.get_level_values(0).unique())
    kps = sorted(shares.index.get_level_values(1).unique())
    kvs = sorted(shares.index.get_level_values(2).unique())
    fig, axes = plt.subplots(len(arms), 1, figsize=(10, 3.0 * len(arms)), squeeze=False, constrained_layout=True)
    for ai, arm in enumerate(arms):
        ax = axes[ai, 0]
        runs.style(ax)
        ax2 = ax.twinx()
        xt, xl = [], []
        for ci, (kp, kv) in enumerate([(kp, kv) for kp in kps for kv in kvs]):
            for gi, genome in enumerate(("tree", "tree_symmetric")):
                x = ci * 2.6 + gi
                key = (arm, kp, kv, genome)
                if key not in shares.index:
                    continue
                bottom = 0.0
                for c in shares.columns:
                    v = shares.loc[key, c]
                    ax.bar(x, v, bottom=bottom, width=0.9, color=CLASS_SHADE[c], edgecolor="white", linewidth=0.8,
                           label=c if (ci == 0 and gi == 0) else None)
                    bottom += v
                ax.text(x, 1.01, "T" if genome == "tree" else "S", ha="center", fontsize=6, color=runs.INK_2)
            g = cells[(cells["arm"] == arm) & (cells["kp"] == kp) & (cells["kv"] == kv)]
            if len(g):
                ax2.errorbar(ci * 2.6 + 0.5, g["gap"].iloc[0], yerr=[[g["gap"].iloc[0] - g["ci_lo"].iloc[0]],
                             [g["ci_hi"].iloc[0] - g["gap"].iloc[0]]], fmt="o", color=runs.INK, markersize=4,
                             capsize=2, linewidth=1)
            xt.append(ci * 2.6 + 0.5)
            xl.append(f"kp {kp:g}\nkv {kv:g}")
        ax.set_xticks(xt, xl, fontsize=7)
        ax.set_ylim(0, 1.06)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("share of episodes", fontsize=9, color=runs.INK_2)
        ax2.axhline(0, color=runs.INK_2, linewidth=0.6, linestyle=":")
        ax2.set_ylabel("sym − tree score (●, 95% CI)", fontsize=9, color=runs.INK_2)
        ax2.tick_params(colors=runs.INK_2, labelsize=8)
        for side in ("top",):
            ax2.spines[side].set_visible(False)
        ax.set_title(f"armature {arm:g} (T = tree, S = symmetric; bang_bang)", fontsize=10, color=runs.INK)
    axes[0, 0].legend(fontsize=7, frameon=False, labelcolor=runs.INK_2, ncol=6, loc="upper left", bbox_to_anchor=(0, 1.22))
    fig.savefig(out / "fig_gait_grid.png", dpi=200)
    plt.close(fig)

    pooled = pd.crosstab([d["genome"]], d["rowing_label"], normalize="index").round(3)
    by_kv = pd.crosstab([d["genome"], d["kv"].round(3)], d["rowing_label"], normalize="index").round(3)
    note("grid_gait_pooled", pooled.to_dict(), "grid bang_bang gait shares (all episodes):\n" + pooled.to_string())
    note("grid_gait_by_kv", by_kv.to_dict(), "grid bang_bang gait shares by kv:\n" + by_kv.to_string())


# ── R5: locked vs unlocked ───────────────────────────────────────────────────


def final_checkpoints(replays: Path) -> pd.DataFrame:
    eps = pd.concat([pd.read_csv(p) for p in sorted(replays.glob("*/episodes.csv"))], ignore_index=True)
    final = eps[eps["evals"] == eps.groupby("run")["evals"].transform("max")].copy()
    final["locked"] = final["phase_locked_frac"] >= LOCKED_MIN
    return final


def fig_locked(replays: Path, e1_champions: Path, out: Path) -> None:
    """(a) final score of tree, unlocked and locked symmetric trainings on E2
    bodies; (b) phase-locked share of E1 champions per genome x controller.
    The locked-seeds-per-body histogram goes to fig_locked_seeds.png (appendix)."""
    final = final_checkpoints(replays)
    brains = [b for b in ("bang_bang", "bang_bang_drift", "dann") if b in set(final["brain"])]
    groups = [("tree", None, "tree"), ("tree_symmetric", False, "symmetric, unlocked"),
              ("tree_symmetric", True, "symmetric, locked")]
    fig, (ax, cx) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [3, 2]})
    sfig, bx = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
    for a in (ax, bx, cx):
        runs.style(a)
    for bi, brain in enumerate(brains):
        f = final[final["brain"] == brain]
        for gi, (genome, locked, label) in enumerate(groups):
            g = f[f["genome"] == genome]
            if locked is not None:
                g = g[g["locked"] == locked]
            x = bi * 4 + gi
            color = runs.GENOME_COLOR[genome]
            alpha = 1.0 if locked in (None, True) else 0.45
            ax.bar(x, g["score"].mean(), 0.85, yerr=g["score"].sem(), color=color, alpha=alpha,
                   error_kw={"ecolor": runs.INK_2}, label=label if bi == 0 else None)
            ax.text(x, 0.01, f"n={len(g)}", ha="center", fontsize=6, color="white", rotation=90, va="bottom")
        key = brain
        tree = f[f["genome"] == "tree"]["score"]
        unl = f[(f["genome"] == "tree_symmetric") & ~f["locked"]]["score"]
        lk = f[(f["genome"] == "tree_symmetric") & f["locked"]]["score"]
        note(f"locked_{key}", {"tree": round(tree.mean(), 3), "unlocked": round(unl.mean(), 3), "locked": round(lk.mean(), 3),
                               "n_tree": len(tree), "n_unlocked": len(unl), "n_locked": len(lk),
                               "tree_locked_share": round(float(f[f["genome"] == "tree"]["locked"].mean()), 3)},
             f"{brain} final checkpoints (replay score): locked sym {lk.mean():.3f} (n={len(lk)}), unlocked sym "
             f"{unl.mean():.3f} (n={len(unl)}), tree {tree.mean():.3f} (n={len(tree)}); tree locked share "
             f"{f[f['genome'] == 'tree']['locked'].mean():.2f}")

        # locking as a body property: share of seeds locked per symmetric body
        sym = f[f["genome"] == "tree_symmetric"]
        per = sym.groupby("body_id").agg(share=("locked", "mean"), score=("score", "mean"), n=("locked", "size"))
        rho = spearmanr(per["share"], per["score"])
        counts = per["share"].round(2).value_counts().sort_index()
        # within-body: bodies with both locked and unlocked seeds, locked-seed mean - unlocked-seed mean
        mixed = sym.groupby("body_id").filter(lambda g: g["locked"].nunique() == 2)
        wb = mixed.groupby(["body_id", "locked"])["score"].mean().unstack()
        dwb = (wb[True] - wb[False]).to_numpy()
        p_wb = float(wilcoxon(dwb).pvalue) if len(dwb) >= 2 and np.any(dwb != 0) else math.nan
        note(f"lock_body_{brain}", {"all_or_none": int(((per["share"] == 0) | (per["share"] == 1)).sum()),
                                    "n_bodies": len(per), "rho": round(float(rho.statistic), 3), "rho_p": round(float(rho.pvalue), 3),
                                    "within_body_diff": round(float(dwb.mean()), 3) if len(dwb) else math.nan,
                                    "within_body_n": len(dwb), "within_body_p": round(p_wb, 3)},
             f"{brain}: {int(((per['share'] == 0) | (per['share'] == 1)).sum())}/{len(per)} sym bodies lock in all or no seeds "
             f"(counts by share: {dict(counts)}); Spearman share vs score rho={rho.statistic:+.2f} (p={rho.pvalue:.2f}); "
             f"within-body locked - unlocked seeds {dwb.mean() if len(dwb) else math.nan:+.3f} over {len(dwb)} mixed bodies "
             f"(Wilcoxon p={p_wb:.2f})")
        shares = [0.0, 1 / 3, 2 / 3, 1.0]
        cnt = [int((per["share"].round(3) == round(s, 3)).sum()) for s in shares]
        bx.bar(np.arange(4) + bi * 0.27 - 0.27, cnt, 0.25, color=runs.GENOME_COLOR["tree_symmetric"],
               alpha=[0.4, 0.7, 1.0][bi % 3], label=runs.BRAIN_LABEL[brain])
    ax.set_xticks([bi * 4 + 1 for bi in range(len(brains))], [runs.BRAIN_LABEL[b] for b in brains], fontsize=8)
    ax.set_ylabel(f"final score per training{runs.unit_suffix(final)} (± SEM)", fontsize=9, color=runs.INK_2)
    ax.legend(fontsize=7, frameon=False, labelcolor=runs.INK_2)
    ax.set_title("(a) fixed random bodies, final controller", fontsize=9, color=runs.INK, loc="left")
    bx.set_xticks(range(4), ["0/3", "1/3", "2/3", "3/3"], fontsize=8)
    bx.set_xlabel("training seeds that lock", fontsize=9, color=runs.INK_2)
    bx.set_ylabel("symmetric bodies", fontsize=9, color=runs.INK_2)
    bx.legend(fontsize=7, frameon=False, labelcolor=runs.INK_2)
    bx.set_title("Locking is mostly all-or-none per body", fontsize=10, color=runs.INK)
    sfig.savefig(out / "fig_locked_seeds.png", dpi=200)
    plt.close(sfig)

    if e1_champions.exists():
        ch = pd.read_csv(e1_champions)
        cbrains = [b for b in ("bang_bang", "bang_bang_drift", "bang_bang_free", "dann") if b in set(ch["brain"])]
        genomes = [g for g in ("tree", "tree_symmetric", "tree_tied") if g in set(ch["genome"])]
        for bi, brain in enumerate(cbrains):
            for gi, genome in enumerate(genomes):
                g = ch[(ch["brain"] == brain) & (ch["genome"] == genome)]
                if g.empty:
                    continue
                x = bi * (len(genomes) + 1) + gi
                cx.bar(x, g["phase_locked"].mean(), 0.85, color=runs.GENOME_COLOR[genome],
                       label=runs.GENOME_LABEL[genome] if bi == 0 else None)
                cx.text(x, g["phase_locked"].mean() + 0.02, f"{int(g['phase_locked'].sum())}/{len(g)}", ha="center",
                        fontsize=6, color=runs.INK_2)
        cx.set_xticks([bi * (len(genomes) + 1) + (len(genomes) - 1) / 2 for bi in range(len(cbrains))],
                      [runs.BRAIN_SHORT[b] for b in cbrains], fontsize=8)
        cx.set_ylim(0, 1.1)
        cx.set_ylabel("share of champions phase-locked", fontsize=9, color=runs.INK_2)
        cx.legend(fontsize=7, frameon=False, labelcolor=runs.INK_2)
        cx.set_title("(b) evolution (E1): last-generation champions", fontsize=9, color=runs.INK, loc="left")
    fig.savefig(out / "fig_locked.png", dpi=200)
    plt.close(fig)


# ── Numbers without a figure ─────────────────────────────────────────────────


def body_numbers(features: Path, matched: pd.DataFrame, bodies_random: Path, rng: np.random.Generator) -> None:
    f = pd.read_csv(features)
    sym = (f["genome"] == "tree_symmetric").to_numpy(float)
    for brain in MAIN_BRAINS:
        y = f[f"score_{brain}"].to_numpy(float)
        raw = ols_coef(np.column_stack([np.ones(len(y)), sym]), y, rng)
        adj = ols_coef(np.column_stack([np.ones(len(y)), sym, f[list(SHAPE_MEASURES)].to_numpy(float)]), y, rng)
        note(f"shape_adjusted_random_{brain}", {"raw": [round(v, 3) for v in raw], "adj": [round(v, 3) for v in adj]},
             f"{brain} random bodies (replay scores): sym-tree raw {raw[0]:+.3f} [{raw[1]:+.3f}, {raw[2]:+.3f}], "
             f"adjusted for {', '.join(SHAPE_MEASURES)} {adj[0]:+.3f} [{adj[1]:+.3f}, {adj[2]:+.3f}]")

    for brain in MAIN_BRAINS:
        m = matched[(matched["brain"] == brain) & matched["set"].isin(["tree", "tree_tied"])].dropna(subset=["final", "n_leg"])
        tied = (m["set"] == "tree_tied").to_numpy(float)
        y = m["final"].to_numpy(float)
        raw = ols_coef(np.column_stack([np.ones(len(y)), tied]), y, rng)
        adj = ols_coef(np.column_stack([np.ones(len(y)), tied, m["n_leg"].to_numpy(float)]), y, rng)
        legs = m.groupby("set")["n_leg"].mean()
        note(f"tied_leg_adjusted_{brain}", {"raw": [round(v, 3) for v in raw], "adj": [round(v, 3) for v in adj]},
             f"{brain}: tied-tree raw {raw[0]:+.3f} [{raw[1]:+.3f}, {raw[2]:+.3f}], adjusted for n_leg {adj[0]:+.3f} "
             f"[{adj[1]:+.3f}, {adj[2]:+.3f}]; n_leg tied {legs.get('tree_tied', math.nan):.2f} vs tree {legs.get('tree', math.nan):.2f}")

    f["diag_share"] = f["n_diag"] / f["n_hinges"]
    tree = f[f["genome"] == "tree"]
    r = spearmanr(tree["diag_share"], tree["score_bang_bang"])
    ds = f.groupby("genome")["diag_share"].mean()
    note("diag", {"share": ds.round(3).to_dict(), "rho_tree": round(float(r.statistic), 3)},
         f"diagonal hinge share: tree {ds['tree']:.2f}, sym {ds['tree_symmetric']:.2f}; within tree Spearman with "
         f"bang_bang score rho={r.statistic:+.2f} (p={r.pvalue:.2f})")

    rot: dict[str, Counter] = {}
    for line in bodies_random.read_text().splitlines():
        b = json.loads(line)
        c = rot.setdefault(b["genome_type"], Counter())
        c.update(n["rotation"] for n in b["genome"]["nodes"].values() if n["type"] == "HINGE")
    tab = {g: {k: round(v / sum(c.values()), 3) for k, v in sorted(c.items())} for g, c in rot.items()}
    note("hinge_rotations", tab, "hinge rotation shares: " + json.dumps(tab))


def e1_vs_grid(e1_root: Path, grid_root: Path) -> None:
    e1 = runs.load_runs(e1_root)
    e1 = e1[(e1["brain"] == "bang_bang") & e1["seed"].between(42, 44)]
    grid = runs.load_runs(grid_root)
    z = grid[(grid["brain"] == "bang_bang") & np.isclose(grid["kp"], 1.36051, atol=1e-3)
             & np.isclose(grid["kv"], 0.35555, atol=1e-3) & np.isclose(grid["armature"], 0.0811705, atol=1e-4)]
    m = e1.merge(z, on=["genome", "seed"], suffixes=("_e1", "_grid"))
    same = bool(len(m) and np.allclose(m["score_e1"], m["score_grid"]))
    note("e1_reproduces_grid", same, f"E1 bang_bang seeds 42-44 equal the grid zoo cell: {same} ({len(m)} runs compared)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("__data__/ariel_paper"))
    parser.add_argument("--analysis", type=Path, default=Path("__data__/ariel_paper/paper"))
    parser.add_argument("--grid-root", type=Path, default=Path("__data__/ariel_actuator_grid"),
                        help="v1 bang_bang hinge grid (only for --gait-grid and the E1 reproduction check)")
    parser.add_argument("--gait-grid", action="store_true", help="also draw the v1 gait-vs-hinge figure (appendix)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    e2 = args.data / "e2"
    matched = pd.read_csv(args.analysis / "matched" / "matched_bodies.csv")

    fig_curves([e2 / "random", e2 / "tied", e2 / "shape_matched"], args.out, rng)
    fig_shapes(matched, args.out)
    if args.gait_grid:
        fig_gait_grid(args.grid_root / "rowing_grid" / "individuals.csv", args.analysis / "e0" / "e0_grid_cells.csv",
                      args.out)
    fig_locked(args.data / "e3" / "e2_random", args.analysis / "e1" / "e1_champions.csv", args.out)
    body_numbers(args.analysis / "bodies" / "body_features.csv", matched, e2 / "bodies_random.jsonl", rng)
    if args.grid_root.exists():
        e1_vs_grid(args.data / "e1", args.grid_root)

    (args.out / "inline_numbers.txt").write_text("\n\n".join(LINES) + "\n")
    (args.out / "inline_numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=str))
    print("\n\n".join(LINES))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
