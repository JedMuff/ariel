"""
Experiment 4: retrain final-generation bodies of Experiment 1 from scratch with a long budget.

Takes lines --first .. --first+--count-1 of a bodies file (export_final_bodies.py) and trains
each body's own controller kind (its "brain" field) for --evals evaluations with one seed,
--concurrent bodies at a time on one pool of --workers processes. Each body is written as
soon as its training finishes, so a task stopped by the time limit keeps what it finished,
and a rerun trains only the bodies not yet in results.jsonl.

Output (--out-root/):
  results/<first>.jsonl  one line per body: body_id, genome_type, brain, e1_evals, e1_seed,
                         e1_id, seed, score (best displacement, m), evals, stopped_early,
                         wall_s, curve {best, median, sigma per CMA-ES generation}
  weights/<body_id>.npy  best weights
  config/<first>.json    evals, seed, protocol, git commit, bodies file

Usage:
    python train_long.py --bodies e4_bodies.jsonl --first 0 --count 48 --evals 10000 \\
        --workers 32 --concurrent 8 --out-root runs/e4
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import PROTOCOL
from evolve import git_commit
from sim import make_pool, train_controller

META = ("body_id", "genome_type", "brain", "e1_evals", "e1_seed", "e1_id", "e1_score", "rank", "copies")


def done_ids(out_root: Path) -> set[str]:
    ids = set()
    for f in (out_root / "results").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                ids.add(json.loads(line)["body_id"])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 4: long retraining of evolved bodies")
    ap.add_argument("--bodies", type=Path, required=True)
    ap.add_argument("--first", type=int, required=True, help="0-based first line of --bodies")
    ap.add_argument("--count", type=int, required=True, help="lines to train")
    ap.add_argument("--evals", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--concurrent", type=int, default=8, help="bodies trained at once")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="seconds; start no new body after this (one still running is lost)")
    ap.add_argument("--out-root", type=Path, required=True)
    a = ap.parse_args()
    t_start = time.monotonic()

    lines = a.bodies.read_text().splitlines()[a.first:a.first + a.count]
    bodies = [json.loads(x) for x in lines]
    for d in ("results", "weights", "config"):
        (a.out_root / d).mkdir(parents=True, exist_ok=True)
    done = done_ids(a.out_root)
    todo = [b for b in bodies if b["body_id"] not in done]
    print(f"lines {a.first}..{a.first + len(bodies) - 1}: {len(bodies) - len(todo)} done, {len(todo)} to train",
          flush=True)
    if not todo:
        return
    (a.out_root / "config" / f"{a.first}.json").write_text(json.dumps(
        {"experiment": "e4", "bodies": str(a.bodies), "first": a.first, "count": a.count, "evals": a.evals,
         "seed": a.seed, "protocol": asdict(PROTOCOL), "git_commit": git_commit()}, indent=2))
    results = a.out_root / "results" / f"{a.first}.jsonl"
    lock = threading.Lock()

    def one(body: dict) -> str:
        if a.time_limit and time.monotonic() - t_start > a.time_limit:
            return f"{body['body_id']}: skipped (time limit)"
        r = train_controller(body["genome"], body["brain"], a.evals, a.seed, pool=pool)
        np.save(a.out_root / "weights" / f"{body['body_id']}.npy", r.weights)
        rec = {**{k: body.get(k) for k in META}, "seed": a.seed, "score": r.score, "evals": r.evals,
               "stopped_early": r.stopped_early, "wall_s": round(r.wall_s, 2),
               "curve": {k: [round(x, 5) for x in v] for k, v in r.curve.items()}}
        with lock, results.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return f"{body['body_id']}: {r.score:.3f} m (E1 {body.get('e1_score', float('nan')):.3f}), " \
               f"{r.evals} evals, {r.wall_s:.0f} s"

    with make_pool(a.workers) as pool, ThreadPoolExecutor(max_workers=max(1, a.concurrent)) as threads:
        for f in as_completed([threads.submit(one, b) for b in todo]):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
