"""Plot CMA-ES fitness over evaluations for one or more zoo-benchmark runs.

Each run dir must contain history.csv and config.json (from zoo_benchmark.py).
Solid line = best-so-far x-speed, faint line = generation mean.

Usage:
  python plot_zoo_runs.py runs/gecko_* -o gecko_fitness.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BRAIN_COLORS = {"ann": "#1f77b4", "sine": "#ff7f0e", "revolve_cpg": "#2ca02c", "matsuoka": "#d62728"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("-o", "--out", type=Path, default=Path("zoo_fitness.png"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    bodies = set()
    for run_dir in args.run_dirs:
        cfg = json.loads((run_dir / "config.json").read_text())
        hist = pd.read_csv(run_dir / "history.csv")
        bodies.add(cfg["body"])
        color = BRAIN_COLORS.get(cfg["brain"])
        label = f"{cfg['brain']} ({cfg['n_params']} params, seed {cfg['seed']})"
        ax.plot(hist["evals"], hist["best"], color=color, lw=2, label=label)
        ax.plot(hist["evals"], hist["gen_mean"], color=color, lw=0.8, alpha=0.35)

    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("x-speed (m/s)")
    ax.set_title(f"CMA-ES on {', '.join(sorted(bodies))}: best so far (solid), generation mean (faint)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
