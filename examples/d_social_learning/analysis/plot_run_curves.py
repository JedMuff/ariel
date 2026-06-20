"""Per-run fitness curves: mean±std (shaded) + max (dashed) per generation.

Produces one PNG per domain with subplots for each scheme.

Usage:
    uv run examples/d_social_learning/analysis/plot_run_curves.py \
        [--data-dir __data__/social] [--out-dir __data__/social/plots] \
        [--x 0.0] [--rep 0]
"""

from __future__ import annotations

import argparse
import sqlite3
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCHEMES = [
    "darwinian", "lamarckian", "random", "random_many",
    "best", "best_many", "similar", "similar_many",
]
DOMAINS = ["ariel", "evogym"]


def _x_str(x: float) -> str:
    return str(x).replace(".", "")


def load_run(db_path: Path) -> dict[int, list[float]] | None:
    if not db_path.exists():
        return None
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "SELECT time_of_birth, fitness_ FROM individual "
        "WHERE fitness_ IS NOT NULL ORDER BY time_of_birth"
    )
    rows = cur.fetchall()
    con.close()
    if not rows:
        return None
    by_gen: dict[int, list[float]] = defaultdict(list)
    for birth, fit in rows:
        by_gen[birth].append(float(fit))
    return dict(by_gen)


def plot_domain(
    domain: str,
    data_dir: Path,
    out_dir: Path,
    x: float,
    rep: int,
) -> None:
    x_str = _x_str(x)
    present = []
    run_data = {}
    for scheme in SCHEMES:
        db_path = data_dir / domain / scheme / f"x{x_str}" / f"rep_{rep}" / "database.db"
        d = load_run(db_path)
        if d is not None:
            run_data[scheme] = d
            present.append(scheme)

    if not present:
        print(f"  [skip] no data for domain={domain} x={x} rep={rep}")
        return

    ncols = 4
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 3.2 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"{domain.upper()} — Fitness per generation  (x={x}, rep={rep})",
        fontsize=13, y=1.01,
    )

    for idx, scheme in enumerate(present):
        ax = axes[idx // ncols][idx % ncols]
        by_gen = run_data[scheme]
        gens = sorted(by_gen.keys())

        means = np.array([np.mean(by_gen[g]) for g in gens])
        stds  = np.array([np.std(by_gen[g])  for g in gens])
        maxes = np.array([np.max(by_gen[g])  for g in gens])

        ax.fill_between(gens, means - stds, means + stds, alpha=0.25, label="±1 std")
        ax.plot(gens, means, lw=2.0, label="mean")
        ax.plot(gens, maxes, lw=1.5, ls="--", label="max")

        ax.set_title(scheme, fontsize=10)
        ax.set_xlabel("generation", fontsize=8)
        ax.set_ylabel("fitness", fontsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7, loc="upper left")

    # Hide unused axes
    for idx in range(len(present), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / f"{domain}_x{x_str}_rep{rep}_curves.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="__data__/social")
    parser.add_argument("--out-dir", default="__data__/social/plots")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--rep", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain in DOMAINS:
        plot_domain(domain, Path(args.data_dir), out_dir, args.x, args.rep)


if __name__ == "__main__":
    main()
