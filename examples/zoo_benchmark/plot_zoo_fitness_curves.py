"""Plot mean CMA-ES fitness curves for the full zoo benchmark.

Reads every <root>/*/history.csv (+ config.json) from zoo_benchmark.py runs and
writes two figures to the output dir:
  zoo_fitness_per_body.png   one panel per body: mean best-so-far x-speed over
                             reps per brain, shaded 95% CI
  zoo_fitness_aggregate.png  mean best-so-far over all bodies per brain; the CI
                             is over body-level means (n = number of bodies)

Curves are resampled onto a common evaluation grid (popsize, and hence the eval
count per generation, differs between brains). Runs that stop early hold their
final best-so-far until the budget.

Usage:
  python plot_zoo_fitness_curves.py __data__/ariel_zoo_benchmark [-o OUTDIR]
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BRAINS = ["ann", "matsuoka", "revolve_cpg", "sine", "square", "square_sync"]
BRAIN_COLORS = {"ann": "#2a78d6", "matsuoka": "#eb6834", "revolve_cpg": "#1baf7a", "sine": "#eda100",
                "square": "#4a3aa7", "square_sync": "#e87ba4"}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
N_GRID = 201


def step_resample(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Step function: at each grid point, the last y with x <= grid; NaN before x[0]."""
    idx = np.searchsorted(x, grid, side="right") - 1
    return np.where(idx >= 0, y[np.clip(idx, 0, None)], np.nan)


def load_curves(root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (runs, grid): one row per run with its best-so-far resampled onto grid."""
    records = []
    for hist_path in sorted(root.glob("*/history.csv")):
        cfg = json.loads((hist_path.parent / "config.json").read_text())
        hist = pd.read_csv(hist_path)
        records.append({"body": cfg["body"], "brain": cfg["brain"], "seed": cfg["seed"],
                        "n_hinges": cfg["n_hinges"], "budget": cfg["budget"], "workers": cfg["workers"],
                        "evals": hist["evals"].to_numpy(), "wall_s": hist["wall_s"].to_numpy(),
                        "best": hist["best"].to_numpy()})
    if not records:
        raise SystemExit(f"No history.csv found under {root}")

    grid = np.linspace(0, max(r["budget"] for r in records), N_GRID)
    for r in records:
        r["curve"] = step_resample(r["evals"], r["best"], grid)
    return pd.DataFrame(records), grid


def mean_ci(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and 95% CI half-width (1.96 * SEM) across rows, ignoring NaN."""
    n = np.sum(~np.isnan(curves), axis=0)
    with warnings.catch_warnings():  # grid points before any run's first generation are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(curves, axis=0)
        sem = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, 1.96 * np.where(n > 1, sem, np.nan)


def style_axis(ax) -> None:
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def draw_brains(ax, groups: dict[str, np.ndarray], grid: np.ndarray, lw: float) -> dict[str, float]:
    """Draw mean +- CI per brain; return each brain's final mean."""
    finals = {}
    for brain in BRAINS:
        if brain not in groups:
            continue
        mean, ci = mean_ci(groups[brain])
        color = BRAIN_COLORS[brain]
        ax.fill_between(grid, mean - ci, mean + ci, color=color, alpha=0.18, lw=0)
        ax.plot(grid, mean, color=color, lw=lw, label=brain)
        finals[brain] = mean[-1]
    return finals


def label_line_ends(ax, finals: dict[str, float], x_end: float) -> None:
    """Direct labels at the line ends, nudged apart so they don't overlap."""
    ymin, ymax = ax.get_ylim()
    min_gap = 0.045 * (ymax - ymin)
    placed = []
    for brain, y in sorted(finals.items(), key=lambda kv: kv[1]):
        y_lab = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_lab)
        ax.annotate(f"{brain}  {y:.3f}", xy=(x_end, y), xytext=(x_end * 1.015, y_lab),
                    va="center", fontsize=9, color=INK, annotation_clip=False)
        ax.plot([x_end], [y], "o", ms=5, color=BRAIN_COLORS[brain], mec="white", mew=1.5, clip_on=False)


def plot_per_body(runs: pd.DataFrame, grid: np.ndarray, out: Path, dpi: int) -> None:
    bodies = sorted(runs["body"].unique())
    ncols = 6
    nrows = int(np.ceil((len(bodies) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.1 * nrows), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, body in zip(axes, bodies):
        sub = runs[runs["body"] == body]
        groups = {b: np.vstack(g["curve"].to_list()) for b, g in sub.groupby("brain")}
        draw_brains(ax, groups, grid, lw=1.6)
        n_reps = sub.groupby("brain").size().min()
        ax.set_title(f"{body}  (n={n_reps})", fontsize=9, color=INK, loc="left")
        style_axis(ax)

    for ax in axes[len(bodies):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[len(bodies)].legend(handles, labels, loc="center", frameon=False, fontsize=10,
                             title="Brain", title_fontsize=10)

    for ax in axes[(nrows - 1) * ncols:]:
        ax.set_xlabel("Evaluations", fontsize=9, color=INK)
    for ax in axes[::ncols]:
        ax.set_ylabel("Best x-speed (m/s)", fontsize=9, color=INK)
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    # sharex hides tick labels on rows above the last; the blank tail panels leave
    # the last populated row without an axis below it, so re-enable them there.
    for ax in axes[len(bodies) - ncols:len(bodies)]:
        if ax not in axes[(nrows - 1) * ncols:]:
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel("Evaluations", fontsize=9, color=INK)

    fig.suptitle("CMA-ES best-so-far x-speed per body: mean over reps, shaded 95% CI",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_aggregate(runs: pd.DataFrame, grid: np.ndarray, out: Path, dpi: int) -> None:
    # Average reps within each body first, so the CI reflects variation across bodies.
    groups = {}
    for brain, sub in runs.groupby("brain"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            body_means = [np.nanmean(np.vstack(g["curve"].to_list()), axis=0) for _, g in sub.groupby("body")]
        groups[brain] = np.vstack(body_means)
    n_bodies = runs["body"].nunique()

    fig, ax = plt.subplots(figsize=(8, 5))
    finals = draw_brains(ax, groups, grid, lw=2)
    style_axis(ax)
    ax.tick_params(labelsize=9)

    label_line_ends(ax, finals, grid[-1])

    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    ax.set_xlim(0, grid[-1])
    ax.set_xlabel("Evaluations", fontsize=10, color=INK)
    ax.set_ylabel("Best x-speed (m/s)", fontsize=10, color=INK)
    ax.set_title(f"CMA-ES best-so-far x-speed across {n_bodies} bodies\n"
                 f"mean of per-body means, shaded 95% CI over bodies",
                 fontsize=11, color=INK, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9, title="Brain", title_fontsize=9)
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
    runs, grid = load_curves(args.root)
    print(f"{len(runs)} runs, {runs['body'].nunique()} bodies, {runs['brain'].nunique()} brains")

    plot_per_body(runs, grid, out_dir / "zoo_fitness_per_body.png", args.dpi)
    plot_aggregate(runs, grid, out_dir / "zoo_fitness_aggregate.png", args.dpi)


if __name__ == "__main__":
    main()
