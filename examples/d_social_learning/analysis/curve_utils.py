"""Shared helpers for aggregating per-generation metrics across replicate runs
of a social-learning experiment.

Used by plot_run_curves.py, plot_diversity_curves.py, and compare_schemes.py.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.novelty_history import read_run_history  # noqa: E402

SCHEMES = [
    "darwinian", "lamarckian", "random", "random_many",
    "best", "best_many", "similar_DESCR", "similar_many_DESCR",
    "similar_STRUCT", "similar_many_STRUCT",
    "similar_TOPDOWN", "similar_many_TOPDOWN",
]
X_VALUES = [0.0, 0.5, 1.0]

# novelty_history metric (core/novelty_history.py) that metric="novelty" and
# metric="fitness" curves read; runs made before the table existed get it
# from analysis/backfill_novelty.py.
NOVELTY_METRIC = "TOPDOWN"

# Distinct color per scheme, shared across all plots for consistency.
SCHEME_COLORS = {
    scheme: color for scheme, color in zip(
        SCHEMES,
        ["#4c72b0", "#dd8452", "#55a868", "#c44e52",
         "#8172b2", "#937860", "#da8bc3", "#8c8c8c",
         "#ccb974", "#64b5cd", "#ccb974", "#64b5cd"],
    )
}


def x_str(x: float) -> str:
    return str(x).replace(".", "")


# Must match sim_ariel/evaluator.py's values at the time the data was generated.
JERK_PENALTY_WEIGHT = 3.0
JERK_THRESHOLD = 0.15


def pure_displacement(tags: dict) -> float | None:
    """Forward displacement with the jerk penalty added back onto the
    learned (penalised) fitness stored in tags['distance'].

    Only the jerk penalty can be undone: the spawn-height penalty depends on
    core height, which isn't stored, so it stays included (it's zero unless
    the core spawns above HEIGHT_PENALTY_THRESHOLD).
    """
    distance = tags.get("distance")
    mean_jerk = tags.get("mean_jerk")
    if distance is None or mean_jerk is None:
        return None
    jerk_penalty = JERK_PENALTY_WEIGHT * mean_jerk if mean_jerk >= JERK_THRESHOLD else 0.0
    return distance + jerk_penalty


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


def read_individuals(db: Path | str, columns: str, where: str = "fitness_ IS NOT NULL") -> "pd.DataFrame":
    """``SELECT id, <columns> FROM individual WHERE <where>`` over a run's
    database.db and every database_part*.db continuation (`db` is the run's
    database.db). A continuation re-stores the restored survivors under their
    original ids; each individual's row from the newest file wins, as only
    there are its time_of_death and alive current."""
    import pandas as pd
    frames = []
    for part in rep_db_parts(Path(db).parent):
        con = sqlite3.connect(f"file:{part}?mode=ro", uri=True)
        frames.append(pd.read_sql(f"SELECT id, {columns} FROM individual WHERE {where}", con))
        con.close()
    return pd.concat(frames, ignore_index=True).drop_duplicates("id", keep="last").reset_index(drop=True)


def load_history_by_gen(rep_dir: Path, column: str, metric: str = NOVELTY_METRIC) -> dict[int, list[float]] | None:
    """{generation: [novelty_history `column` of every selection-pool member]}
    for one rep (stitched across database_part*.db), or None if the rep has
    no history for `metric`."""
    hist = read_run_history(rep_dir, metric)
    if hist.empty:
        return None
    hist = hist[np.isfinite(hist[column].astype(float))]
    return {int(g): grp[column].astype(float).tolist() for g, grp in hist.groupby("generation")}


def load_run(rep_dir: Path, metric: str) -> dict[int, list[float]] | None:
    """Return {generation: [values]} for one rep, stitched across
    database.db and any database_part*.db continuation files, or None if
    missing/empty.

    metric="fitness"/"novelty" read the novelty_history table for
    NOVELTY_METRIC: every selection-pool member's value at each generation
    (individual.fitness_ / tags_['novelty'] only keep each individual's last
    value, see core/novelty_history.py). metric="displacement" reads
    pure_displacement(tags_) of the individuals born each generation,
    restricted to fitness_ IS NOT NULL, matching how the outer EA marks an
    individual as evaluated.

    A resumed run's continuation file re-persists the restored survivor
    population under its original time_of_birth (see rep_db_parts), so
    stitching keeps, from each part in order, only rows at generations beyond
    the highest generation already covered by earlier parts — this drops the
    duplicated restart-generation rows instead of double-counting them.
    """
    if metric in ("fitness", "novelty"):
        return load_history_by_gen(rep_dir, metric)

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
        for birth, _fitness, tags in rows:
            if birth <= covered_max:
                continue
            value = pure_displacement(json.loads(tags) if tags else {})
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
    """Rep dirs for (scheme, x): ``{scheme}/x{x}/rep_*`` (older layout) and
    ``{scheme}/x{x}/k{k}/rep_*`` (experiment.py's current layout)."""
    base = exp_dir / scheme / f"x{x_str(xv)}"
    if not base.exists():
        return []
    dbs = [*base.glob("rep_*/database.db"), *base.glob("k*/rep_*/database.db")]
    k_dirs = {p.parent.parent.name for p in dbs if p.parent.parent != base}
    if len(k_dirs) > 1:
        print(f"[curve_utils] warning: {base} mixes reps from {sorted(k_dirs)}")
    return sorted(p.parent for p in dbs)


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
