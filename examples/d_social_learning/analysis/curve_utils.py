"""Shared helpers for aggregating per-generation metrics across replicate runs
of a social-learning experiment.

Used by plot_run_curves.py, plot_diversity_curves.py, and compare_schemes.py.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

SCHEMES = [
    "darwinian", "lamarckian", "random", "random_many",
    "best", "best_many", "similar", "similar_many",
]
X_VALUES = [0.0, 0.5, 1.0]

# Distinct color per scheme, shared across all plots for consistency.
SCHEME_COLORS = {
    scheme: color for scheme, color in zip(
        SCHEMES,
        ["#4c72b0", "#dd8452", "#55a868", "#c44e52",
         "#8172b2", "#937860", "#da8bc3", "#8c8c8c"],
    )
}


def x_str(x: float) -> str:
    return str(x).replace(".", "")


def rep_db_parts(rep_dir: Path) -> list[Path]:
    """Return a rep's db files in stitching order: database.db first, then
    database_part2.db, database_part3.db, ... (numeric part order).

    experiment.py's --resume flag restarts an EA from database.db's last
    saved generation and writes the continuation to database_part2.db (see
    experiment.py's main()); the restored survivor population is re-persisted
    into the new file under its *original* time_of_birth, so the two files'
    generation axes overlap rather than chain — see load_run's stitching.
    """
    parts = [rep_dir / "database.db"]
    numbered = sorted(
        rep_dir.glob("database_part*.db"),
        key=lambda p: int(re.search(r"database_part(\d+)\.db", p.name).group(1)),
    )
    parts.extend(numbered)
    return [p for p in parts if p.exists()]


def load_run(rep_dir: Path, metric: str) -> dict[int, list[float]] | None:
    """Return {generation: [values]} for one rep, stitched across
    database.db and any database_part*.db continuation files, or None if
    missing/empty.

    metric="fitness" reads individual.fitness_; metric="novelty" reads
    tags_['novelty']. Both are restricted to individuals with fitness_ IS NOT
    NULL, matching how the outer EA marks an individual as evaluated.

    A resumed run's continuation file re-persists the restored survivor
    population under its original time_of_birth (see rep_db_parts), so
    stitching keeps, from each part in order, only rows at generations beyond
    the highest generation already covered by earlier parts — this drops the
    duplicated restart-generation rows instead of double-counting them.
    """
    db_parts = rep_db_parts(rep_dir)
    if not db_parts:
        return None

    by_gen: dict[int, list[float]] = defaultdict(list)
    covered_max = -1
    for db_path in db_parts:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(
            "SELECT time_of_birth, fitness_, tags_ FROM individual "
            "WHERE fitness_ IS NOT NULL ORDER BY time_of_birth"
        )
        rows = cur.fetchall()
        con.close()

        part_max = covered_max
        for birth, fit, tags in rows:
            if birth <= covered_max:
                continue
            if metric == "fitness":
                value = fit
            else:
                tags_d = json.loads(tags) if tags else {}
                value = tags_d.get("novelty")
            if value is not None and np.isfinite(value):
                by_gen[birth].append(float(value))
            part_max = max(part_max, birth)
        covered_max = part_max

    return dict(by_gen) if by_gen else None


def single_run_curve(
    rep_dir: Path, metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-generation stats for one rep, no cross-rep aggregation: population
    mean, population std, best-of-generation (max among that generation's
    newly-evaluated individuals only — noisy, since it misses elite
    individuals that survived from an earlier generation without being
    re-born), and running-best (cumulative max of best-of-generation across
    generations).

    running-best reconstructs "best individual seen by generation g" without
    relying on time_of_death, so it stays correct across a resumed run's
    database.db/database_part*.db boundary (time_of_death in the earlier
    file isn't updated by the later file — see rep_db_parts/load_run).
    Under this framework's (mu+lambda) elitism (fitness_ is maximized, see
    Population.best(sort="max") in src/ariel/ec/population.py), the true
    best-ever individual is always alive going forward, so for metric="fitness"
    this line coincides with the actual best-alive fitness at every
    generation — unlike best-of-generation, it can only go up.

    Missing generations (e.g. the one-generation gap a resume leaves at its
    restart point) are forward-filled from the previous generation, matching
    aggregate_across_reps.

    Returns (gens, mean, std, best_of_gen, running_best), or None if no data.
    """
    by_gen = load_run(rep_dir, metric)
    if not by_gen:
        return None

    max_gen = max(by_gen.keys())
    gens = np.arange(max_gen + 1)
    mean = np.full(len(gens), np.nan)
    std = np.full(len(gens), np.nan)
    best_of_gen = np.full(len(gens), np.nan)
    running_best = np.full(len(gens), np.nan)

    last_mean, last_std, last_best, last_running = np.nan, np.nan, np.nan, -np.inf
    for i, g in enumerate(gens):
        if g in by_gen:
            vals = by_gen[g]
            last_mean = float(np.mean(vals))
            last_std = float(np.std(vals))
            last_best = float(np.max(vals))
            last_running = max(last_running, last_best)
        mean[i] = last_mean
        std[i] = last_std
        best_of_gen[i] = last_best
        running_best[i] = last_running if np.isfinite(last_running) else np.nan

    return gens, mean, std, best_of_gen, running_best


def discover_reps(exp_dir: Path, scheme: str, xv: float) -> list[Path]:
    base = exp_dir / scheme / f"x{x_str(xv)}"
    if not base.exists():
        return []
    return sorted(p.parent for p in base.glob("rep_*/database.db"))


@lru_cache(maxsize=None)
def aggregate_across_reps(
    exp_dir: Path, scheme: str, xv: float, metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Aggregate a metric across all reps found for (scheme, x).

    Within each rep, per-generation population mean and max are computed and
    forward-filled across generations. Those per-rep curves are then averaged
    (with std) across reps.

    Returns (gens, mean_of_mean, std_of_mean, mean_of_max, std_of_max), or
    None if no reps have data for (scheme, x).
    """
    rep_dirs = discover_reps(exp_dir, scheme, xv)
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
    per_rep_max = []
    for by_gen in loaded:
        mean_curve = np.full(len(gens), np.nan)
        max_curve = np.full(len(gens), np.nan)
        last_mean, last_max = np.nan, np.nan
        for i, g in enumerate(gens):
            if g in by_gen:
                last_mean = float(np.mean(by_gen[g]))
                last_max = float(np.max(by_gen[g]))
            mean_curve[i] = last_mean
            max_curve[i] = last_max
        per_rep_mean.append(mean_curve)
        per_rep_max.append(max_curve)

    mean_arr = np.array(per_rep_mean)
    max_arr = np.array(per_rep_max)
    return (
        gens,
        np.nanmean(mean_arr, axis=0), np.nanstd(mean_arr, axis=0),
        np.nanmean(max_arr, axis=0), np.nanstd(max_arr, axis=0),
    )


def _pad_range(lo: float, hi: float, pad_frac: float) -> tuple[float, float] | None:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def compute_global_ylim(
    exp_dirs: list[Path],
    metric: str,
    schemes: list[str] = SCHEMES,
    x_values: list[float] = X_VALUES,
    pad_frac: float = 0.05,
) -> tuple[float, float] | None:
    """Min/max y-range (mean±std band and best±std band) across every
    (exp_dir, scheme, x) combination in `x_values`, so multiple
    figures/subplots for the same metric can share one y-scale.

    Pass a single-element `x_values` to get a per-row (per-x) range instead
    of one shared across all x values — see compute_row_ylims.
    """
    lo, hi = np.inf, -np.inf
    for exp_dir in exp_dirs:
        for scheme in schemes:
            for xv in x_values:
                result = aggregate_across_reps(exp_dir, scheme, xv, metric)
                if result is None:
                    continue
                _, mean_of_mean, std_of_mean, mean_of_max, std_of_max = result
                lo = min(
                    lo,
                    np.nanmin(mean_of_mean - std_of_mean),
                    np.nanmin(mean_of_max - std_of_max),
                )
                hi = max(
                    hi,
                    np.nanmax(mean_of_mean + std_of_mean),
                    np.nanmax(mean_of_max + std_of_max),
                )

    return _pad_range(lo, hi, pad_frac)


def compute_row_ylims(
    exp_dirs: list[Path],
    metric: str,
    schemes: list[str] = SCHEMES,
    x_values: list[float] = X_VALUES,
    pad_frac: float = 0.05,
) -> dict[float, tuple[float, float] | None]:
    """Per-x y-range (see compute_global_ylim), so each row of a grid keyed
    by x gets its own shared scale instead of one scale for the whole grid.
    """
    return {
        xv: compute_global_ylim(exp_dirs, metric, schemes, [xv], pad_frac)
        for xv in x_values
    }


def compute_global_ylim_split(
    exp_dirs: list[Path],
    metric: str,
    schemes: list[str] = SCHEMES,
    x_values: list[float] = X_VALUES,
    pad_frac: float = 0.05,
) -> dict[str, tuple[float, float] | None]:
    """Separate y-ranges for the "mean" panel and the "best/max" panel
    (each including its own ±std band) across every (exp_dir, scheme, x)
    combination in `x_values`.
    """
    mean_lo, mean_hi = np.inf, -np.inf
    max_lo, max_hi = np.inf, -np.inf
    for exp_dir in exp_dirs:
        for scheme in schemes:
            for xv in x_values:
                result = aggregate_across_reps(exp_dir, scheme, xv, metric)
                if result is None:
                    continue
                _, mean_of_mean, std_of_mean, mean_of_max, std_of_max = result
                mean_lo = min(mean_lo, np.nanmin(mean_of_mean - std_of_mean))
                mean_hi = max(mean_hi, np.nanmax(mean_of_mean + std_of_mean))
                max_lo = min(max_lo, np.nanmin(mean_of_max - std_of_max))
                max_hi = max(max_hi, np.nanmax(mean_of_max + std_of_max))

    return {
        "mean": _pad_range(mean_lo, mean_hi, pad_frac),
        "max": _pad_range(max_lo, max_hi, pad_frac),
    }


def compute_row_ylim_splits(
    exp_dirs: list[Path],
    metric: str,
    schemes: list[str] = SCHEMES,
    x_values: list[float] = X_VALUES,
    pad_frac: float = 0.05,
) -> dict[float, dict[str, tuple[float, float] | None]]:
    """Per-x version of compute_global_ylim_split."""
    return {
        xv: compute_global_ylim_split(exp_dirs, metric, schemes, [xv], pad_frac)
        for xv in x_values
    }
