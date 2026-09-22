"""Compare the 4 survivor-selection variants from
run_darwinian_selection_parta_only.sh (darwinian scheme, x=1.0):

    0. (mu+lambda) + elitist
    1. (mu+lambda) + tournament(4)
    2. (mu,lambda) + elitist
    3. (mu,lambda) + tournament(4)

Unlike plot_run_curves.py/compare_schemes.py (which expect a
{scheme}/x{x}/rep_* directory tree), this operates on a flat directory of
run-dirs named rep_{rep}[_comma][_tourn4]_{timestamp} (experiment.py's naming
when --resume-dir isn't used), as produced by the darwinian-selection array
job and rsynced back from the cluster into one directory. Reps still missing
some variants (mid-run) are simply skipped for that variant.

Produces {out-dir}/fitness_comparison.png and {out-dir}/diversity_comparison.png,
each a single-panel plot with mean+-std (solid) and running-best+-std (dashed)
per variant, aggregated across whatever reps are currently present.

Usage:
    uv run examples/d_social_learning/analysis/compare_selection_variants.py \
        --data-dir __data__/cluster_data \
        --out-dir __data__/cluster_data/plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from curve_utils import load_run

RUN_DIR_RE = re.compile(
    r"^rep_(?P<rep>\d+)(?P<suffix>_comma_tourn4|_comma|_tourn4)?_\d{8}_\d{6}$"
)

VARIANT_ORDER = ["elitist", "tourn4", "comma", "comma_tourn4"]
VARIANT_LABELS = {
    "elitist": "(mu+lambda) + elitist",
    "tourn4": "(mu+lambda) + tournament(4)",
    "comma": "(mu,lambda) + elitist",
    "comma_tourn4": "(mu,lambda) + tournament(4)",
}
# Fixed categorical order, reusing curve_utils.SCHEME_COLORS' first 4 hues
# for visual consistency with the rest of this project's plots.
VARIANT_COLORS = {
    "elitist": "#4c72b0",
    "tourn4": "#dd8452",
    "comma": "#55a868",
    "comma_tourn4": "#c44e52",
}


def discover_variant_runs(data_dir: Path) -> dict[str, list[Path]]:
    """{variant: [rep_dir, ...]} for every rep_dir directly under data_dir
    that matches experiment.py's naming convention and has a database.db.
    """
    by_variant: dict[str, list[Path]] = {v: [] for v in VARIANT_ORDER}
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        m = RUN_DIR_RE.match(child.name)
        if not m or not (child / "database.db").exists():
            continue
        variant = m.group("suffix").lstrip("_") if m.group("suffix") else "elitist"
        by_variant[variant].append(child)
    return by_variant


def aggregate_rep_dirs(
    rep_dirs: list[Path], metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Same aggregation as curve_utils.aggregate_across_reps, but over an
    explicit list of rep dirs instead of a {scheme}/x{x}/ glob.

    Returns (gens, mean_of_mean, std_of_mean, mean_of_running_best,
    std_of_running_best), or None if no rep has data.
    """
    loaded = []
    max_gen = -1
    for rep_dir in rep_dirs:
        by_gen = load_run(rep_dir, metric)
        if not by_gen:
            continue
        loaded.append(by_gen)
        max_gen = max(max_gen, max(by_gen.keys()))
    if not loaded:
        return None

    gens = np.arange(max_gen + 1)
    per_rep_mean = []
    per_rep_running_best = []
    for by_gen in loaded:
        mean_curve = np.full(len(gens), np.nan)
        running_best_curve = np.full(len(gens), np.nan)
        last_mean, last_running = np.nan, -np.inf
        for i, g in enumerate(gens):
            if g in by_gen:
                last_mean = float(np.mean(by_gen[g]))
                last_running = max(last_running, float(np.max(by_gen[g])))
            mean_curve[i] = last_mean
            running_best_curve[i] = last_running if np.isfinite(last_running) else np.nan
        per_rep_mean.append(mean_curve)
        per_rep_running_best.append(running_best_curve)

    mean_arr = np.array(per_rep_mean)
    best_arr = np.array(per_rep_running_best)
    return (
        gens,
        np.nanmean(mean_arr, axis=0), np.nanstd(mean_arr, axis=0),
        np.nanmean(best_arr, axis=0), np.nanstd(best_arr, axis=0),
    )


def plot_metric(
    by_variant: dict[str, list[Path]], metric: str, ylabel: str, title: str, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    any_data = False
    for variant in VARIANT_ORDER:
        rep_dirs = by_variant.get(variant, [])
        result = aggregate_rep_dirs(rep_dirs, metric)
        if result is None:
            continue
        any_data = True
        gens, mean_of_mean, std_of_mean, mean_of_best, std_of_best = result
        color = VARIANT_COLORS[variant]
        label = f"{VARIANT_LABELS[variant]} (n={len(rep_dirs)})"

        ax.plot(gens, mean_of_mean, color=color, lw=2, label=label)
        ax.fill_between(
            gens, mean_of_mean - std_of_mean, mean_of_mean + std_of_mean,
            color=color, alpha=0.15, linewidth=0,
        )
        ax.plot(gens, mean_of_best, color=color, lw=2, ls="--")
        ax.fill_between(
            gens, mean_of_best - std_of_best, mean_of_best + std_of_best,
            color=color, alpha=0.10, linewidth=0,
        )

    if not any_data:
        print(f"  [skip] no data for {out_path.name}")
        plt.close(fig)
        return

    ax.plot([], [], color="0.3", lw=2, label="mean (solid)")
    ax.plot([], [], color="0.3", lw=2, ls="--", label="running best (dashed)")

    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="__data__/cluster_data")
    parser.add_argument("--out-dir", default=None, help="Defaults to <data-dir>/plots")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "plots"

    by_variant = discover_variant_runs(data_dir)
    for variant in VARIANT_ORDER:
        n = len(by_variant.get(variant, []))
        print(f"{VARIANT_LABELS[variant]}: {n} rep(s) found")

    plot_metric(
        by_variant, metric="fitness", ylabel="fitness",
        title="Darwinian selection (x=1.0): fitness by survivor-selection variant",
        out_path=out_dir / "fitness_comparison.png",
    )
    plot_metric(
        by_variant, metric="novelty", ylabel="novelty (diversity)",
        title="Darwinian selection (x=1.0): diversity by survivor-selection variant",
        out_path=out_dir / "diversity_comparison.png",
    )


if __name__ == "__main__":
    main()
