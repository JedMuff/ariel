"""Plot the NSGA-II Pareto front for the gecko brain-hyperparameter tuning study.

Scatters all completed trials (time_s vs fitness), highlights the Pareto
front, and annotates each Pareto point with its trial number.

Usage:
    uv run examples/d_social_learning/analysis/plot_tuning_pareto.py \
        [--study-name gecko_nsga2] \
        [--storage sqlite:///__data__/tuning/gecko_nsga2.db] \
        [--out __data__/tuning/plots/pareto_front.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import optuna


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-name", type=str, default="gecko_nsga2")
    parser.add_argument("--storage", type=str, default="sqlite:///__data__/tuning/gecko_nsga2.db")
    parser.add_argument("--out", type=Path, default=Path("__data__/tuning/plots/pareto_front.png"))
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pareto_nums = {t.number for t in study.best_trials}

    dominated = [t for t in complete if t.number not in pareto_nums]
    pareto = sorted((t for t in complete if t.number in pareto_nums), key=lambda t: t.values[0])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        [t.values[0] for t in dominated],
        [t.values[1] for t in dominated],
        s=28, c="#9aa5b1", alpha=0.6, label=f"dominated ({len(dominated)})", zorder=2,
    )
    ax.scatter(
        [t.values[0] for t in pareto],
        [t.values[1] for t in pareto],
        s=70, c="#d6455d", edgecolors="white", linewidths=0.8,
        label=f"Pareto front ({len(pareto)})", zorder=3,
    )
    ax.plot(
        [t.values[0] for t in pareto],
        [t.values[1] for t in pareto],
        c="#d6455d", alpha=0.5, lw=1.2, zorder=2,
    )

    for t in pareto:
        p = t.params
        label = (
            f"#{t.number}\n"
            f"σ={p['sigma']:.2f} pop={p['pop_size']}\n"
            f"gens={p['inner_gens']} h={p['hidden']}"
        )
        ax.annotate(
            label,
            (t.values[0], t.values[1]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=7, color="#333333",
        )

    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("fitness (forward displacement − penalties)")
    ax.set_title(f"Gecko brain tuning: Pareto front ({len(complete)} trials)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
