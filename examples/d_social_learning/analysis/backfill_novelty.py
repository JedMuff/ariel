"""Backfill the novelty_history table (core/novelty_history.py) of existing
runs for a structural novelty metric, from their stored morphologies.

Runs made before novelty_history existed only keep each individual's last
novelty/fitness (experiment.py overwrote them every generation). This rebuilds
every generation's selection pool -- the previous generation's survivors plus
its offspring (time_of_birth <= g-1 < time_of_death, or time_of_birth == g) --
and recomputes, within each pool, what evaluate_and_select computes:

  - novelty: mean distance to the K_NOVELTY nearest pool members under
    --metric (core.novelty.compute_novelty_ted)
  - fitness: combined_fitness of the pool-normalised distance and novelty

Rows are written with source="backfill": when --metric differs from the
metric the run selected with, this fitness is what selection *would* have
seen, not what it used; the survivor column is always the actual selection.

Writing into a database changes it, so re-syncing that run from the cluster
replaces the file and drops the table: re-run this script afterwards (runs
that already have rows for --metric are skipped, so it only redoes those).

Usage:
    uv run examples/d_social_learning/analysis/backfill_novelty.py \
        --data-dir __data__/social/ariel --metric TOPDOWN --workers 16
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_SOCIAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SOCIAL_DIR))

import simulator_dependent_functions  # noqa: E402
from core.fitness import combined_fitness  # noqa: E402
from core.novelty import compute_novelty_ted  # noqa: E402
from core.novelty_history import COLUMNS, read_history, write_rows  # noqa: E402


def _run_x(db: Path) -> float:
    config = db.parent / "run_config.json"
    if config.exists():
        return float(json.loads(config.read_text())["args"]["x"])
    return int(db.parents[2].name[1:]) / 10  # .../x05/k5/rep_*/database.db


def backfill_db(job: tuple[str, str, str, bool]) -> tuple[str, int, float]:
    db_str, platform, metric, force = job
    db = Path(db_str)
    t0 = time.time()
    if not force and not read_history(db, metric).empty:
        return db_str, 0, 0.0
    simulator_dependent_functions.simulator = platform
    dist_fn = simulator_dependent_functions.similarity_function(metric)
    x = _run_x(db)

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    df = pd.read_sql(
        """SELECT id, time_of_birth AS born, time_of_death AS tod, alive,
                  json_extract(genotype_, '$.morph') AS morph,
                  json_extract(tags_, '$.distance') AS distance
           FROM individual WHERE fitness_ IS NOT NULL""",
        con,
    ).set_index("id")
    con.close()
    last_gen = int(df.tod.max())
    morph = {i: json.loads(m) for i, m in df.morph.items()}
    canon = {i: json.dumps(m, sort_keys=True) for i, m in morph.items()}
    cache: dict[tuple[str, str], float] = {}

    def cached_dist(a: int, b: int) -> float:
        ka, kb = canon[a], canon[b]
        if ka == kb:
            return 0.0
        key = (ka, kb) if ka < kb else (kb, ka)
        if key not in cache:
            cache[key] = dist_fn(morph[a], morph[b])
        return cache[key]

    rows = []
    for g in range(1, last_gen + 1):
        members = df.index[((df.born <= g - 1) & (df.tod > g - 1)) | (df.born == g)].tolist()
        if not members:
            continue
        novelty = compute_novelty_ted(members, cached_dist)
        distance = df.loc[members, "distance"].to_numpy(dtype=float)
        fin_d = distance[np.isfinite(distance)]
        fin_n = novelty[np.isfinite(novelty)]
        lo_d, hi_d = (fin_d.min(), fin_d.max()) if len(fin_d) else (0.0, 0.0)
        lo_n, hi_n = (fin_n.min(), fin_n.max()) if len(fin_n) else (0.0, 0.0)
        for i, ind in enumerate(members):
            nd = (distance[i] - lo_d) / (hi_d - lo_d) if hi_d > lo_d else 0.0
            nn = (novelty[i] - lo_n) / (hi_n - lo_n) if hi_n > lo_n else 0.0
            fit = combined_fitness(nd, nn, x)
            survivor = df.at[ind, "tod"] > g or (bool(df.at[ind, "alive"]) and g == last_gen)
            rows.append((g, ind, metric, float(novelty[i]), float(fit) if np.isfinite(fit) else None,
                         int(survivor), "backfill"))
    write_rows(db, pd.DataFrame(rows, columns=COLUMNS))
    return db_str, len(rows), time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="__data__/social/ariel")
    parser.add_argument("--platform", default="ariel", choices=["ariel", "evogym"])
    parser.add_argument("--metric", default="TOPDOWN", choices=["STRUCT", "TOPDOWN"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="Recompute runs that already have rows.")
    args = parser.parse_args()

    dbs = sorted(Path(args.data_dir).glob("*/x*/k*/rep_*/database.db"))
    resumed = [db for db in dbs if any(db.parent.glob("database_part*.db"))]
    if resumed:
        # A continuation file's pools for generations up to its restart point
        # only contain the restored survivors; not handled yet.
        print(f"Skipping {len(resumed)} resumed runs (database_part*.db): {[str(d.parent) for d in resumed]}")
        dbs = [db for db in dbs if db not in resumed]
    print(f"Backfilling {args.metric} novelty_history for {len(dbs)} databases", flush=True)

    t0 = time.time()
    done = skipped = 0
    with Pool(args.workers) as pool:
        jobs = [(str(db), args.platform, args.metric, args.force) for db in dbs]
        for db, n_rows, secs in pool.imap_unordered(backfill_db, jobs):
            if n_rows:
                done += 1
            else:
                skipped += 1
    print(f"Done in {time.time() - t0:.0f}s: {done} backfilled, {skipped} already had rows")


if __name__ == "__main__":
    main()
