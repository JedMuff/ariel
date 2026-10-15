"""Per-generation novelty/fitness history of every selection-pool member.

experiment.py recomputes novelty and the blended fitness of every alive
individual each generation, and Individual.fitness_/tags['novelty'] only keep
the latest values. So each generation's values are also appended to the
individual's tags['novelty_history'] (a list of
{"gen", "metric", "novelty", "fitness", "survivor"} dicts), which the EA
persists with the individual on every commit.

The ``novelty_history`` table holds the same data in queryable form, one row
per (generation, individual, metric):

    generation    selection step (pool = generation-1 survivors + generation's offspring)
    individual_id individual.id
    metric        novelty metric: DESCR, STRUCT or TOPDOWN
    novelty       k-NN novelty within that generation's pool
    fitness       blended fitness within that pool (core/fitness.py)
    survivor      1 if the individual survived that generation's selection
    source        "live": computed during the run, used for selection
                  "backfill": computed afterwards (analysis/backfill_novelty.py)
                  from the stored morphologies, NOT what selection used

experiment.py writes the table from the tags when a run finishes; readers use
``read_history``, which falls back to the tags for runs that stopped early.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

TABLE = "novelty_history"
COLUMNS = ["generation", "individual_id", "metric", "novelty", "fitness", "survivor", "source"]

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    generation INTEGER NOT NULL,
    individual_id INTEGER NOT NULL,
    metric TEXT NOT NULL,
    novelty REAL,
    fitness REAL,
    survivor INTEGER NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (generation, individual_id, metric)
)
"""


def _finite_or_none(v: float) -> float | None:
    return float(v) if math.isfinite(v) else None


def history_entry(gen: int, metric: str, novelty: float, fitness: float, survivor: bool) -> dict:
    """One tags['novelty_history'] item. Non-finite values (a failed
    evaluation's -inf fitness) are stored as None: -inf/NaN aren't valid JSON."""
    return {"gen": gen, "metric": metric, "novelty": _finite_or_none(novelty),
            "fitness": _finite_or_none(fitness), "survivor": bool(survivor)}


def _from_tags(con: sqlite3.Connection) -> pd.DataFrame:
    rows = []
    for ind_id, hist in con.execute(
        "SELECT id, json_extract(tags_, '$.novelty_history') FROM individual "
        "WHERE json_extract(tags_, '$.novelty_history') IS NOT NULL"
    ):
        for h in json.loads(hist):
            rows.append((h["gen"], ind_id, h["metric"], h["novelty"], h["fitness"], int(h["survivor"]), "live"))
    return pd.DataFrame(rows, columns=COLUMNS)


def write_rows(db_path: Path | str, df: pd.DataFrame) -> None:
    """Insert/replace history rows (columns = COLUMNS) into db_path's table."""
    con = sqlite3.connect(db_path)
    try:
        con.execute(_SCHEMA)
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE} ({', '.join(COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?, ?)",
            df[COLUMNS].itertuples(index=False, name=None),
        )
        con.commit()
    finally:
        con.close()


def write_table_from_tags(db_path: Path | str) -> int:
    """Materialise tags['novelty_history'] into the table; returns row count."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = _from_tags(con)
    finally:
        con.close()
    if not df.empty:
        write_rows(db_path, df)
    return len(df)


def read_history(db_path: Path | str, metric: str | None = None) -> pd.DataFrame:
    """History rows for one database file (table, else tags), optionally
    filtered to one metric. Empty DataFrame if there is no history."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        has_table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)
        ).fetchone()
        df = pd.read_sql(f"SELECT * FROM {TABLE}", con) if has_table else _from_tags(con)
        if has_table and df.empty:
            df = _from_tags(con)
    finally:
        con.close()
    if metric is not None:
        df = df[df.metric == metric]
    return df.reset_index(drop=True)


def read_run_history(rep_dir: Path | str, metric: str | None = None) -> pd.DataFrame:
    """History for a whole run, stitched across database.db and any
    database_part{N}.db continuation files. A continuation re-persists the
    restored survivors with their earlier history under the same ids, so
    duplicated (generation, individual_id, metric) rows are dropped."""
    rep_dir = Path(rep_dir)
    parts = [rep_dir / "database.db", *sorted(
        rep_dir.glob("database_part*.db"), key=lambda p: int(p.stem.removeprefix("database_part")),
    )]
    frames = [read_history(p, metric) for p in parts if p.exists()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["generation", "individual_id", "metric"], keep="first").reset_index(drop=True)
