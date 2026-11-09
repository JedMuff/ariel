"""
Rest-pose body measures (bodies.body_measures) of the bodies of chosen episodes.

Each distinct body of a run is measured once (3 s settle with zero control, no
controller), so a body that survives many generations costs one measurement.

Writes <out-dir>/<run name>/rest_measures.csv: genome_hash and the measures. A
run whose file exists is skipped.

Usage:
    python rest_measures.py --run-dir RUN [RUN ...] [--which s:0,10,50] [--workers 32] -o OUT
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import runs
from bodies import body_measures
from sim import genome_hash


def _measure(item: tuple[str, dict]) -> dict:
    body_hash, genome = item
    try:
        return {"genome_hash": body_hash, **body_measures(genome)}
    except Exception as exc:  # noqa: BLE001 -- one bad body must not stop the run
        return {"genome_hash": body_hash, "error": f"{type(exc).__name__}: {exc}"}


def measure_run(run_dir: Path, out: Path, which: str, workers: int) -> str:
    name = runs.run_name(run_dir)
    target = out / "rest_measures.csv"
    if target.exists():
        return f"{name}: already measured"
    bodies = {genome_hash(ep.genome): ep.genome for ep in runs.episodes(run_dir, which)}
    with Pool(processes=min(workers, len(bodies))) as pool:
        rows = pool.map(_measure, list(bodies.items()), chunksize=1)
    out.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(target)
    errors = sum("error" in r for r in rows)
    return f"{name}: {len(rows)} bodies, {errors} errors"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--which", default="final", help="as replay.py: final, champion, g:<gens> or s:<gens>")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    args = parser.parse_args()
    for run_dir in args.run_dir:
        print(measure_run(run_dir, args.out_dir / runs.run_name(run_dir), args.which, args.workers), flush=True)


if __name__ == "__main__":
    main()
