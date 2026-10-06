"""Compare the k_inherit pilot from run_k_inherit_sweep.sh (best_many, x=1.0,
k in {3, 5, 8}).

Expects the cluster's best_many/x10/ tree rsynced into --data-dir, i.e.
{data-dir}/k{K}/rep_{R}_novSTRUCT/database.db.

Plots raw distance (tags_['distance']) rather than fitness_: fitness_ is
min-max normalized within each generation's population (see experiment.py),
so it isn't comparable across generations or across k values.

Unlike curve_utils.aggregate_across_reps, reps aren't forward-filled to the
longest rep: each rep is NaN past its last generation and the average is a
nanmean over whichever reps reached that generation, so each k's curve runs
to its longest rep. Reps with fewer than --min-gens generations are dropped
entirely.

Produces {out-dir}/k_inherit_comparison.png: distance (mean solid, running
best dashed, mean+-std across reps) and novelty, one color per k.

Usage:
    uv run examples/d_social_learning/analysis/compare_k_inherit.py \
        --data-dir __data__/x10 --out-dir __data__/x10/plots
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from curve_utils import rep_db_parts

K_DIR_RE = re.compile(r"^k(?P<k>\d+)$")
K_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]


def load_tag(rep_dir: Path, key: str) -> dict[int, list[float]]:
    """{generation: [tags_[key] of individuals born that generation]},
    stitched across database_part*.db the same way curve_utils.load_run does.
    """
    by_gen: dict[int, list[float]] = defaultdict(list)
    covered_max = -1
    for db_path in rep_db_parts(rep_dir):
        con = sqlite3.connect(db_path)
        rows = con.execute(
            "SELECT time_of_birth, tags_ FROM individual "
            "WHERE fitness_ IS NOT NULL ORDER BY time_of_birth"
        ).fetchall()
        con.close()
        part_max = covered_max
        for birth, tags in rows:
            if birth <= covered_max:
                continue
            value = (json.loads(tags) if tags else {}).get(key)
            if value is not None and np.isfinite(value):
                by_gen[birth].append(float(value))
            part_max = max(part_max, birth)
        covered_max = part_max
    return dict(by_gen)


def rep_curves(by_gen: dict[int, list[float]], n_gens: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-generation population mean and running best, NaN past the rep's
    last generation."""
    mean = np.full(n_gens, np.nan)
    running = np.full(n_gens, np.nan)
    last_mean, best = np.nan, -np.inf
    for g in range(max(by_gen) + 1):
        if g in by_gen:
            last_mean = float(np.mean(by_gen[g]))
            best = max(best, float(np.max(by_gen[g])))
        mean[g] = last_mean
        running[g] = best if np.isfinite(best) else np.nan
    return mean, running


def discover(data_dir: Path) -> dict[int, list[Path]]:
    by_k: dict[int, list[Path]] = {}
    for child in sorted(data_dir.iterdir()):
        m = K_DIR_RE.match(child.name)
        if m and child.is_dir():
            by_k[int(m.group("k"))] = sorted(p.parent for p in child.glob("rep_*/database.db"))
    return dict(sorted(by_k.items()))


def plot_panel(ax, data: dict[int, list[dict]], ylabel: str, show_best: bool) -> None:
    n_gens = max((max(d) + 1 for runs in data.values() for d in runs), default=0)
    gens = np.arange(n_gens)
    for i, (k, runs) in enumerate(data.items()):
        if not runs:
            continue
        color = K_COLORS[i % len(K_COLORS)]
        curves = [rep_curves(d, n_gens) for d in runs]
        # Reps are NaN past their last generation, so nanmean averages over
        # whichever reps are still running.
        means = np.array([c[0] for c in curves])
        bests = np.array([c[1] for c in curves])
        label = f"k={k} (n={len(runs)})"

        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            m, s = np.nanmean(means, axis=0), np.nanstd(means, axis=0)
            bm, bs = np.nanmean(bests, axis=0), np.nanstd(bests, axis=0)

        ax.plot(gens, m, color=color, lw=2, label=label)
        ax.fill_between(gens, m - s, m + s, color=color, alpha=0.15, linewidth=0)
        if show_best:
            ax.plot(gens, bm, color=color, lw=2, ls="--")
            ax.fill_between(gens, bm - bs, bm + bs, color=color, alpha=0.10, linewidth=0)

    if show_best:
        ax.plot([], [], color="0.3", lw=2, label="pop. mean (solid)")
        ax.plot([], [], color="0.3", lw=2, ls="--", label="running best (dashed)")
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="__data__/x10")
    parser.add_argument("--out-dir", default=None, help="Defaults to <data-dir>/plots")
    parser.add_argument("--min-gens", type=int, default=10,
                        help="Drop reps that reached fewer generations than this")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "plots"

    distance: dict[int, list[dict]] = {}
    novelty: dict[int, list[dict]] = {}
    for k, rep_dirs in discover(data_dir).items():
        distance[k], novelty[k] = [], []
        for rep_dir in rep_dirs:
            dist = load_tag(rep_dir, "distance")
            last = max(dist) if dist else -1
            if last + 1 < args.min_gens:
                print(f"  [skip] k={k} {rep_dir.name}: only {last + 1} generation(s)")
                continue
            print(f"  k={k} {rep_dir.name}: {last + 1} generations")
            distance[k].append(dist)
            novelty[k].append(load_tag(rep_dir, "novelty"))

    fig, (ax_d, ax_n) = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_panel(ax_d, distance, "distance", show_best=True)
    ax_d.set_title("Distance")
    plot_panel(ax_n, novelty, "novelty (STRUCT)", show_best=False)
    ax_n.set_title("Novelty")
    fig.suptitle("k_inherit pilot: best_many, x=1.0 (mean ± std across reps)")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "k_inherit_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
