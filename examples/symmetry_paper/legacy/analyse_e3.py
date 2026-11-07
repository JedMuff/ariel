"""
E3: behaviour (R3, R5).

Input: replay.py outputs (<root>/<run>/episodes.csv). Evolution runs give one
episode per last-generation body; fixed-body trainings (E2) one per
checkpoint along training.

  R3   gait class shares and phase locking by genome x brain x budget, over
       all episodes and over run champions (each run's best body)
  R5   phase quality, phase-locked stroke share, alternation and score vs
       evaluations for the E2 checkpoints, symmetric vs tree, per brain

Writes to --out (default <first root>/analysis):
  e3_gait_classes.csv, e3_champions.csv, e3_along_training.csv
  fig3_gait_classes.png    gait class shares per genome x brain (x budget)
  fig6_phase_training_<body source>.png
                           phase-locked share, phase quality, alternation and score vs evaluations

Usage:
    python analyse_e3.py --roots /scratch/$USER/ariel_paper_e3/e1 /scratch/$USER/ariel_paper_e3/budget \\
        [--training-roots /scratch/$USER/ariel_paper_e3/e2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runs  # noqa: E402
from gait import LABELS  # noqa: E402

GENOMES = ("tree", "tree_symmetric", "tree_tied")
GROUPS = ["brain", "genome", "skill_budget"]
CLASS_SHADE = {"alternating": "#2a78d6", "rowing": "#eb6834", "mixed": "#86b6ef", "aperiodic": "#c3c2b7",
               "unpaired": "#9a9990", "static": "#e4e3de"}
TRACKS = [("phase_locked_frac", "phase-locked stroke share"), ("phase_quality", "phase quality"),
          ("alternating", "alternating episodes"), ("score", "score")]


def load(roots: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for root in roots for p in sorted(root.glob("*/episodes.csv"))]
    frames = [f for f in frames if not f.empty and "rowing_label" in f]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def class_shares(df: pd.DataFrame) -> pd.DataFrame:
    tab = pd.crosstab([df[g] for g in GROUPS], df["rowing_label"], normalize="index")
    tab = tab.reindex(columns=[c for c in LABELS if c in tab]).fillna(0.0)
    extra = df.groupby(GROUPS).agg(n=("run", "size"), score=("score", "mean"),
                                   phase_quality=("phase_quality", "mean"),
                                   phase_locked_frac=("phase_locked_frac", "mean"))
    return tab.join(extra).reset_index()


def evolution(df: pd.DataFrame, out: Path) -> None:
    shares = class_shares(df)
    shares.to_csv(out / "e3_gait_classes.csv", index=False)
    champ = df.loc[df.groupby("run")["stored_fitness"].idxmin()]
    champ_shares = class_shares(champ)
    champ_shares.to_csv(out / "e3_champions.csv", index=False)
    print("gait classes, all last-generation episodes:")
    print(shares.round(2).to_string(index=False))
    print("\ngait classes, run champions:")
    print(champ_shares.round(2).to_string(index=False))

    labels = [c for c in LABELS if c in shares]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.7 * len(shares) + 2), 3.6), constrained_layout=True)
    runs.style(ax)
    x = np.arange(len(shares))
    bottom = np.zeros(len(shares))
    for c in labels:
        ax.bar(x, shares[c], bottom=bottom, width=0.7, color=CLASS_SHADE[c], edgecolor="white", linewidth=2, label=c)
        bottom += shares[c].to_numpy()
    short = runs.BRAIN_SHORT
    ax.set_xticks(x, [f"{runs.GENOME_LABEL.get(r.genome, r.genome)}\n{short.get(r.brain, r.brain)}\n{r.skill_budget * 20} ev"
                      for r in shares.itertuples()], fontsize=7)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("share of episodes", fontsize=9, color=runs.INK_2)
    ax.legend(fontsize=8, frameon=False, labelcolor=runs.INK_2, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title("Gait class by genome, brain and budget", fontsize=10, color=runs.INK)
    fig.savefig(out / "fig3_gait_classes.png", dpi=200)
    plt.close(fig)


def along_training(df: pd.DataFrame, out: Path) -> None:
    df = df.assign(alternating=(df["rowing_label"] == "alternating").astype(float))
    cols = [c for c, _ in TRACKS]
    per_body = df.groupby(["body_source", "brain", "genome", "body_id", "evals"], as_index=False)[cols].mean()
    tab = per_body.groupby(["body_source", "brain", "genome", "evals"])[cols].agg(["mean", "sem", "count"])
    tab.to_csv(out / "e3_along_training.csv")
    print("\nalong training (mean over bodies):")
    print(per_body.groupby(["body_source", "brain", "genome", "evals"])[cols].mean().round(3).to_string())

    brains = [b for b in runs.BRAIN_LABEL if b in set(per_body["brain"])]
    for source, src in per_body.groupby("body_source"):
        fig, axes = plt.subplots(len(TRACKS), len(brains), figsize=(3.8 * len(brains), 2.3 * len(TRACKS)),
                                 sharex=True, sharey="row", squeeze=False, constrained_layout=True)
        for bi, brain in enumerate(brains):
            for ti, (col, label) in enumerate(TRACKS):
                ax = axes[ti, bi]
                runs.style(ax)
                for genome in GENOMES:
                    g = src[(src["brain"] == brain) & (src["genome"] == genome)].groupby("evals")[col]
                    if not len(g):
                        continue
                    m, sem = g.mean(), g.sem().fillna(0.0)
                    color = runs.GENOME_COLOR[genome]
                    ax.fill_between(m.index, m - sem, m + sem, color=color, alpha=0.2, linewidth=0)
                    ax.plot(m.index, m, color=color, linewidth=2, marker="o", markersize=4,
                            label=runs.GENOME_LABEL[genome])
                ax.set_xscale("log")
                if ti == 0:
                    ax.set_title(runs.BRAIN_LABEL[brain], fontsize=10, color=runs.INK)
                if bi == 0:
                    ax.set_ylabel(label, fontsize=9, color=runs.INK_2)
                if ti == len(TRACKS) - 1:
                    ax.set_xlabel("evaluations", fontsize=9, color=runs.INK_2)
        axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=runs.INK_2)
        fig.suptitle(f"Phase locking along training ({source} bodies; mean ± 1 SEM over bodies)",
                     fontsize=10, color=runs.INK)
        fig.savefig(out / f"fig6_phase_training_{source}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=Path, nargs="*", default=[], help="replay.py output of evolution runs")
    parser.add_argument("--training-roots", type=Path, nargs="*", default=[], help="replay.py output of E2 trainings")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.roots and not args.training_roots:
        parser.error("give --roots and/or --training-roots")
    out = args.out or (args.roots or args.training_roots)[0] / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    if args.roots:
        evolution(load(args.roots), out)
    if args.training_roots:
        along_training(load(args.training_roots), out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
