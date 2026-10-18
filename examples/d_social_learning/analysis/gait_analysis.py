"""Step 2 (preliminary) of SYMMETRY_BEHAVIOUR_PLAN.md: gaits of replayed zoo robots.

Reads behavior_replay.py's episodes.csv (--behavior-dir) and
symmetry_scores.py's individuals.csv (--symmetry-dir). A robot's metrics are
its mean over episodes and its label the most common one (separately for its
learned and its init weights). Writes, under --behavior-dir:
    reproduction.csv      replayed vs stored final-evaluation fitness, and how
                          often a robot's episodes agree on its label
    labels_by_scheme.csv  learned-gait label shares per budget/x/scheme
                          (and pooled over budget/x)
    labels_by_symmetry.csv  label shares per body symmetry (paired) bin
    symmetry_vs_gait.csv  Spearman of body paired vs the left/right gait metrics
    fitness_vs_gait.csv   Spearman of fitness vs each metric, and fitness by label
    learned_vs_init.csv   per scheme: share of robots whose learned gait keeps
                          the init gait's label, fitness and straightness of both
    figures/labels_by_scheme.png, figures/labels_by_symmetry.png

Usage:
    uv run examples/d_social_learning/analysis/gait_analysis.py \\
        --behavior-dir __data__/social/behavior_zoo --symmetry-dir __data__/social/symmetry
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

from paper_figures import label  # noqa: E402

RUN = ["budget", "x", "scheme", "k", "rep"]
ROBOT = RUN + ["id", "weights"]
METRICS = ["fitness", "straightness", "lateral_drift", "mean_abs_yaw_rate", "heading_consistency",
           "duty_factor_mean", "core_tilt_mean", "joint_amplitude_mean", "joint_corr_abs_mean",
           "mirror_pair_frac", "mirror_pair_corr_mean", "mirror_pair_phase_lag", "mirror_pair_amp_asym",
           "paired_frac", "rowing_frac", "alternating_frac", "rowing_score", "midline_amp_frac"]
LABEL_ORDER = ["rowing", "alternating", "mixed", "unpaired", "midline", "aperiodic", "static"]
# Label colours: one hue per gait family, greys for the non-gaits.
LABEL_COLORS = {"rowing": "#2a78d6", "alternating": "#e34948", "mixed": "#9b6cd1", "unpaired": "#eda100",
                "midline": "#1baf7a", "aperiodic": "#8c8c8c", "static": "#cfcfcf"}
SCHEME_ORDER = ["darwinian", "lamarckian", "random", "random_many", "best", "best_many",
                "similar_TOPDOWN", "similar_many_TOPDOWN"]
SYM_BINS = [-0.01, 0.1, 0.3, 0.5, 1.0]


def robots(ep: pd.DataFrame, sym: pd.DataFrame) -> pd.DataFrame:
    g = ep.groupby(ROBOT)
    r = g[METRICS].mean()
    r["label"] = g.rowing_label.agg(lambda s: s.mode().iloc[0])
    r["label_agree"] = g.rowing_label.agg(lambda s: s.value_counts().iloc[0] / len(s))
    return r.reset_index().merge(sym, on=RUN + ["id"], how="left")


def shares(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    t = pd.crosstab([df[b] for b in by], df.label, normalize="index")
    t = t[[c for c in LABEL_ORDER if c in t.columns]]
    return t.assign(n=df.groupby(by).size())


def spearman_rows(df: pd.DataFrame, x: str, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        ok = df[[x, c]].dropna()
        r = stats.spearmanr(ok[x], ok[c]) if len(ok) > 2 else None
        rows.append({"x": x, "y": c, "rho": r.statistic if r else np.nan, "p": r.pvalue if r else np.nan, "n": len(ok)})
    return pd.DataFrame(rows)


def plot_shares(t: pd.DataFrame, names: list[str], title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.6))
    left = np.zeros(len(t))
    for lab in [c for c in LABEL_ORDER if c in t.columns]:
        ax.barh(range(len(t)), t[lab], left=left, color=LABEL_COLORS[lab], edgecolor="white", lw=1.5, label=lab)
        left += t[lab].to_numpy()
    ax.set_yticks(range(len(t)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of robots (learned gait)")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18 - 0.6 / len(t)), frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--behavior-dir", default="__data__/social/behavior_zoo")
    parser.add_argument("--symmetry-dir", default="__data__/social/symmetry")
    args = parser.parse_args()
    out = Path(args.behavior_dir)
    ep = pd.read_csv(out / "episodes.csv")
    sym = pd.read_csv(Path(args.symmetry_dir) / "individuals.csv", usecols=RUN + ["id", "paired", "strict", "n_modules"])
    R = robots(ep, sym)
    L, I = R[R.weights == "learned"], R[R.weights == "init"]

    le = ep[ep.weights == "learned"].dropna(subset=["stored_fitness"])
    pd.DataFrame([{
        "n_episodes": len(le), "median_abs_diff": (le.fitness - le.stored_fitness).abs().median(),
        "pearson": np.corrcoef(le.fitness, le.stored_fitness)[0, 1],
        "label_all_episodes_agree": (L.label_agree == 1).mean(),
    }]).to_csv(out / "reproduction.csv", index=False)

    by_scheme = shares(L, ["scheme"]).reindex([s for s in SCHEME_ORDER if s in set(L.scheme)])
    pd.concat([shares(L, ["budget", "x", "scheme"]).reset_index(),
               by_scheme.reset_index().assign(budget="all", x="all")]).to_csv(out / "labels_by_scheme.csv", index=False)
    Lb = L.assign(paired_bin=pd.cut(L.paired, SYM_BINS))
    by_sym = shares(Lb, ["paired_bin"])
    by_sym.to_csv(out / "labels_by_symmetry.csv")

    spearman_rows(L, "paired", ["mirror_pair_frac", "mirror_pair_corr_mean", "mirror_pair_phase_lag", "mirror_pair_amp_asym",
                                "paired_frac", "rowing_frac", "alternating_frac", "rowing_score", "midline_amp_frac",
                                "straightness"]).to_csv(out / "symmetry_vs_gait.csv", index=False)
    fit = spearman_rows(L, "fitness", [m for m in METRICS if m != "fitness"])
    by_label = L.groupby("label").fitness.agg(["count", "mean", "median"]).reset_index().rename(
        columns={"label": "y", "count": "n", "mean": "fitness_mean", "median": "fitness_median"}).assign(x="label")
    pd.concat([fit, by_label]).to_csv(out / "fitness_vs_gait.csv", index=False)

    M = L.merge(I, on=RUN + ["id"], suffixes=("", "_init"))
    M.assign(same_label=M.label == M.label_init).groupby("scheme").agg(
        n=("id", "size"), same_label=("same_label", "mean"),
        fitness_init=("fitness_init", "mean"), fitness=("fitness", "mean"),
        straightness_init=("straightness_init", "mean"), straightness=("straightness", "mean"),
    ).reindex([s for s in SCHEME_ORDER if s in set(M.scheme)]).to_csv(out / "learned_vs_init.csv")

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_shares(by_scheme, [label(s) for s in by_scheme.index], "Gait of the best 5 gen-60 survivors per run",
                fig_dir / "labels_by_scheme.png")
    plot_shares(by_sym, [f"paired {iv.left:.1f}-{iv.right:.1f} (n={n})" for iv, n in zip(by_sym.index, by_sym.n)],
                "Gait by body symmetry", fig_dir / "labels_by_symmetry.png")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
