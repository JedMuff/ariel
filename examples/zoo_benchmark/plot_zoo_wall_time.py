"""Plot wall-clock cost of the zoo benchmark.

Reads every <root>/*/history.csv (+ config.json) from zoo_benchmark.py runs and
writes two figures to the output dir:
  zoo_wall_time.png             left: wall time per run by brain (one dot per
                                run, bar = median); right: per-body median wall
                                time against the body's hinge count
  zoo_fitness_vs_wall_time.png  mean best-so-far x-speed against wall-clock
                                time, averaged like zoo_fitness_aggregate.png
                                (mean of per-body means, 95% CI over bodies)

Wall time is the history.csv wall_s of the final generation (CMA-ES loop only,
all runs with the same --workers). A run that has finished holds its final
best-so-far for the rest of the time axis.

Usage:
  python plot_zoo_wall_time.py __data__/ariel_zoo_benchmark [-o OUTDIR]
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_zoo_fitness_curves import (  # noqa: E402
    BRAIN_COLORS, BRAINS, INK, MUTED, N_GRID, draw_brains, label_line_ends, load_curves, step_resample, style_axis,
)


def plot_wall_time(runs, out: Path, dpi: int) -> None:
    runs = runs.assign(wall_min=[w[-1] / 60 for w in runs["wall_s"]])
    workers = sorted(runs["workers"].unique())
    fig, (ax_brain, ax_hinge) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True,
                                             gridspec_kw={"width_ratios": [1, 1.3]})
    rng = np.random.default_rng(0)
    brains = [b for b in BRAINS if b in set(runs["brain"])]

    # Left: every run, jittered, with the median as a bar.
    for i, brain in enumerate(brains):
        wall = runs.loc[runs["brain"] == brain, "wall_min"].to_numpy()
        ax_brain.scatter(i + rng.uniform(-0.22, 0.22, len(wall)), wall, s=9, color=BRAIN_COLORS[brain],
                         alpha=0.45, lw=0)
        med = np.median(wall)
        ax_brain.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=2, solid_capstyle="round")
        ax_brain.annotate(f"{med:.1f} min", xy=(i + 0.32, med), va="center", fontsize=8, color=INK)
    ax_brain.set_xticks(range(len(brains)), brains)
    ax_brain.set_xlim(-0.5, len(brains) - 0.2)
    ax_brain.set_ylabel("Wall time per run (min)", fontsize=10, color=INK)
    ax_brain.set_title(f"Per run (n={len(runs) // len(brains)} per brain); bar = median",
                       fontsize=10, color=INK, loc="left")

    # Right: one dot per body (median over reps), with a line through the mean per hinge count.
    per_body = runs.groupby(["brain", "body", "n_hinges"], as_index=False)["wall_min"].median()
    offsets = dict(zip(brains, np.linspace(-0.18, 0.18, len(brains))))
    for brain in brains:
        sub = per_body[per_body["brain"] == brain]
        color = BRAIN_COLORS[brain]
        ax_hinge.scatter(sub["n_hinges"] + offsets[brain], sub["wall_min"], s=22, color=color,
                         edgecolor="white", lw=0.8, zorder=3, label=brain)
        trend = sub.groupby("n_hinges")["wall_min"].mean()
        ax_hinge.plot(trend.index + offsets[brain], trend.to_numpy(), color=color, lw=1.5, alpha=0.8)
    ax_hinge.set_xticks(sorted(per_body["n_hinges"].unique()))
    ax_hinge.set_xlabel("Hinges in body", fontsize=10, color=INK)
    ax_hinge.set_title("Per body (median over reps); line = mean per hinge count",
                       fontsize=10, color=INK, loc="left")
    ax_hinge.legend(loc="upper left", frameon=False, fontsize=9, title="Brain", title_fontsize=9)

    for ax in (ax_brain, ax_hinge):
        style_axis(ax)
        ax.tick_params(labelsize=9)
    ax_brain.set_ylim(bottom=0)
    fig.suptitle(f"CMA-ES wall-clock time per 10k-evaluation run ({', '.join(map(str, workers))} workers)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_fitness_vs_wall_time(runs, out: Path, dpi: int) -> None:
    t_end = max(w[-1] for w in runs["wall_s"]) / 60
    grid = np.linspace(0, t_end, N_GRID)
    curves = [step_resample(w / 60, b, grid) for w, b in zip(runs["wall_s"], runs["best"])]
    runs = runs.assign(curve=curves)

    groups = {}
    for brain, sub in runs.groupby("brain"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            body_means = [np.nanmean(np.vstack(g["curve"].to_list()), axis=0) for _, g in sub.groupby("body")]
        groups[brain] = np.vstack(body_means)

    fig, ax = plt.subplots(figsize=(8, 5))
    finals = draw_brains(ax, groups, grid, lw=2)
    style_axis(ax)
    ax.tick_params(labelsize=9)
    label_line_ends(ax, finals, grid[-1])

    # Median finish time per brain, so it's clear where each curve stops improving.
    for brain in groups:
        t_med = np.median([w[-1] / 60 for w in runs.loc[runs["brain"] == brain, "wall_s"]])
        ax.axvline(t_med, color=BRAIN_COLORS[brain], lw=1, ls=(0, (3, 3)), alpha=0.8)

    ax.set_xlim(0, grid[-1])
    ax.set_xlabel("Wall-clock time (min)", fontsize=10, color=INK)
    ax.set_ylabel("Best x-speed (m/s)", fontsize=10, color=INK)
    ax.set_title(f"CMA-ES best-so-far x-speed against wall-clock time across {runs['body'].nunique()} bodies\n"
                 "mean of per-body means, shaded 95% CI over bodies; dashed = median finish time",
                 fontsize=11, color=INK, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9, title="Brain", title_fontsize=9)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="default: root")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    out_dir = args.out_dir or args.root
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, _ = load_curves(args.root)
    print(f"{len(runs)} runs, {runs['body'].nunique()} bodies, {runs['brain'].nunique()} brains")

    plot_wall_time(runs, out_dir / "zoo_wall_time.png", args.dpi)
    plot_fitness_vs_wall_time(runs, out_dir / "zoo_fitness_vs_wall_time.png", args.dpi)


if __name__ == "__main__":
    main()
