"""
Plot fitness curves from body+brain randomized-waypoints runs.

Reads one or more SQLite databases produced by 6_body_brain_randomized_waypoints.py
and plots best & mean fitness per outer generation. Each run becomes one line.

Usage
-----
    # Auto-discover runs under __data__/body_brain_*/
    uv run python examples/re_book/6_plot_fitness.py

    # Explicit DBs
    uv run python examples/re_book/6_plot_fitness.py \\
        --db __data__/body_brain_14288_0/__data__/6_body_brain_randomized_waypoints/database_*.db \\
        --db __data__/body_brain_14289_1/__data__/6_body_brain_randomized_waypoints/database_*.db

In the DB schema (one table, `individual`):
  time_of_birth = outer generation index (0..BUDGET)
  fitness_      = brain-best fitness for that body in that generation (lower is better)
"""
from __future__ import annotations

import argparse
import sqlite3
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_per_gen(db_path: Path) -> dict[str, np.ndarray]:
    """Return {gen, best, mean, worst, n} arrays (gen = time_of_birth)."""
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT time_of_birth, fitness_ FROM individual "
            "WHERE fitness_ IS NOT NULL"
        ).fetchall()

    by_gen: dict[int, list[float]] = {}
    for gen, fit in rows:
        if fit is None or not np.isfinite(fit):
            continue
        by_gen.setdefault(int(gen), []).append(float(fit))

    gens = sorted(by_gen)
    best = np.array([min(by_gen[g]) for g in gens])
    mean = np.array([float(np.mean(by_gen[g])) for g in gens])
    worst = np.array([max(by_gen[g]) for g in gens])
    n = np.array([len(by_gen[g]) for g in gens])
    return {"gen": np.array(gens), "best": best, "mean": mean, "worst": worst, "n": n}


def discover_dbs(root: Path) -> list[Path]:
    pattern = str(root / "body_brain_*" / "__data__" / "*" / "database_*.db")
    return sorted(Path(p) for p in glob(pattern))


def label_for(db_path: Path) -> str:
    # .../body_brain_<JOB>_<TASK>/__data__/<script>/database_*.db
    for part in db_path.parts:
        if part.startswith("body_brain_"):
            return part
    return db_path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", action="append", default=[],
                        help="Path to a database_*.db (repeatable). If omitted, auto-discovers under __data__/.")
    parser.add_argument("--data-root", default="__data__",
                        help="Root for auto-discovery (default: __data__).")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: <data-root>/6_fitness_curves.png).")
    parser.add_argument("--show", action="store_true", help="Open the plot interactively.")
    args = parser.parse_args()

    if args.db:
        dbs = [Path(p) for p in args.db]
    else:
        dbs = discover_dbs(Path(args.data_root))

    if not dbs:
        raise SystemExit(f"No DBs found. Pass --db <path> or check {args.data_root}/")

    out_path = Path(args.out) if args.out else Path(args.data_root) / "6_fitness_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    ax_best, ax_mean = axes

    cmap = plt.get_cmap("tab10")
    print(f"Loading {len(dbs)} run(s):")
    for i, db in enumerate(dbs):
        stats = load_per_gen(db)
        lbl = label_for(db)
        color = cmap(i % 10)
        ax_best.plot(stats["gen"], stats["best"], "-o", ms=3, color=color, label=lbl)
        ax_mean.plot(stats["gen"], stats["mean"], "-",  color=color, label=lbl)
        ax_mean.fill_between(stats["gen"], stats["best"], stats["worst"],
                             color=color, alpha=0.12)
        print(f"  {lbl}  gens={len(stats['gen'])}  best={stats['best'].min():.3f}  "
              f"final-best={stats['best'][-1]:.3f}")

    ax_best.set_title("Best fitness per generation (lower is better)")
    ax_best.set_xlabel("Outer generation")
    ax_best.set_ylabel("Fitness")
    ax_best.axhline(-30.0, color="gray", lw=0.6, ls=":", label="theoretical max (3 wp + full bonus)")
    ax_best.grid(alpha=0.3)
    ax_best.legend(fontsize=8)

    ax_mean.set_title("Mean fitness per generation (band: min–max)")
    ax_mean.set_xlabel("Outer generation")
    ax_mean.set_ylabel("Fitness")
    ax_mean.grid(alpha=0.3)
    ax_mean.legend(fontsize=8)

    fig.suptitle("Body+brain co-evolution — randomised waypoints")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nSaved plot → {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
