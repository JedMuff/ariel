"""Plot social-learning experiment results.

Reads summary.csv and produces:
  1. One figure per domain: rows=x values, cols=schemes, each cell = mean±std best fitness.
  2. Supplementary bar chart: final-generation best fitness by scheme, grouped by x.

Usage:
    uv run examples/d_social_learning/analysis/plot_social_results.py \
        [--csv __data__/social/summary.csv] [--out-dir __data__/social/plots]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="__data__/social/summary.csv")
    parser.add_argument("--out-dir", default="__data__/social/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(Path(args.csv))
    if not rows:
        print("No data found in CSV.")
        return

    SCHEMES = [
        "darwinian", "lamarckian", "random", "random_many",
        "best", "best_many", "similar", "similar_many",
    ]
    X_VALUES = [0.0, 0.5, 1.0]
    DOMAINS = ["ariel", "evogym"]

    # index: domain → scheme → x → rep → {gen: best_fitness}
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for row in rows:
        domain = row["domain"]
        scheme = row["scheme"]
        x = float(row["x"])
        rep = int(row["rep"])
        gen = int(row["gen"])
        bf = float(row["best_fitness"])
        data[domain][scheme][x][rep][gen] = bf

    for domain in DOMAINS:
        if domain not in data:
            continue

        max_gen = max(
            g
            for s in data[domain].values()
            for xd in s.values()
            for rd in xd.values()
            for g in rd.keys()
        )
        gens = list(range(max_gen + 1))

        fig, axes = plt.subplots(
            len(X_VALUES), len(SCHEMES),
            figsize=(4 * len(SCHEMES), 3 * len(X_VALUES)),
            sharex=True,
        )
        fig.suptitle(f"{domain.upper()} — Best Fitness per Generation", fontsize=14)

        final_means: dict[float, dict[str, float]] = {x: {} for x in X_VALUES}

        for xi, x in enumerate(X_VALUES):
            for si, scheme in enumerate(SCHEMES):
                ax = axes[xi][si]
                rep_curves = []
                for rep_data in data[domain].get(scheme, {}).get(x, {}).values():
                    curve = [rep_data.get(g, np.nan) for g in gens]
                    # forward-fill
                    last = np.nan
                    filled = []
                    for v in curve:
                        if not np.isnan(v):
                            last = v
                        filled.append(last)
                    rep_curves.append(filled)

                if rep_curves:
                    arr = np.array(rep_curves, dtype=float)
                    mean = np.nanmean(arr, axis=0)
                    std = np.nanstd(arr, axis=0)
                    ax.plot(gens, mean, lw=1.5)
                    ax.fill_between(gens, mean - std, mean + std, alpha=0.2)
                    final_means[x][scheme] = float(np.nanmean(arr[:, -1]))

                if xi == 0:
                    ax.set_title(scheme, fontsize=9)
                if si == 0:
                    ax.set_ylabel(f"x={x}", fontsize=9)
                ax.set_xlabel("gen" if xi == len(X_VALUES) - 1 else "", fontsize=8)

        plt.tight_layout()
        fig_path = out_dir / f"{domain}_learning_curves.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"Saved {fig_path}")

        # supplementary bar chart
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        x_offsets = np.arange(len(SCHEMES))
        width = 0.25
        for xi, x in enumerate(X_VALUES):
            heights = [final_means.get(x, {}).get(s, 0.0) for s in SCHEMES]
            ax2.bar(x_offsets + xi * width, heights, width=width, label=f"x={x}")
        ax2.set_xticks(x_offsets + width)
        ax2.set_xticklabels(SCHEMES, rotation=30, ha="right")
        ax2.set_ylabel("Final-gen best fitness (mean over reps)")
        ax2.set_title(f"{domain.upper()} — Final Generation Best Fitness by Scheme")
        ax2.legend()
        plt.tight_layout()
        bar_path = out_dir / f"{domain}_final_bar.png"
        fig2.savefig(bar_path, dpi=120)
        plt.close(fig2)
        print(f"Saved {bar_path}")


if __name__ == "__main__":
    main()
