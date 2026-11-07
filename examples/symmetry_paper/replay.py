"""
Replay trained controllers and measure their gaits.

Each episode (runs.episodes: an E1 run's final population, its champion or
chosen generations; every seed of an E2 training) is replayed with the
training episode itself (sim.run_episode), so the replayed score reproduces
the stored one on the machine that trained it. Body poses are recorded every
RECORD_EVERY_N physics steps (50 Hz), stored as float16 / float32 time as the
investigation's traces the gait detector was validated on; gait.py then gives
the gait class, phase relations, phase quality and mean gait cycle.

Writes to <out-dir>/<run name>/:
  episodes.csv   one row per episode: run condition, key, stored and replayed
                 score, body descriptors (bodies.descriptors), gait metrics
  pairs.csv      one row per left/right pair (gait.analyse_episode)
  cycles.npz     mean gait cycle per episode

Usage:
    python replay.py --run-dir RUN [RUN ...] [--which final|champion|g:0,49] [--workers 32] -o OUT
    python replay.py --list-runs ROOT [ROOT ...]     # E1 and E2 run dirs, sorted (for Slurm arrays)
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np
import pandas as pd

import gait
import runs
from bodies import descriptors
from brains import make_brain_for_model
from config import PROTOCOL
from sim import build_world, robot_bodies, run_episode

RECORD_EVERY_N = 10     # physics steps between recorded poses: 50 Hz
SCORE_TOL = 1e-6        # replayed vs stored score


def record_episode(genome: dict, brain_kind: str, weights: np.ndarray,
                   perturb: Optional[Callable] = None) -> tuple[float, np.ndarray, np.ndarray, float]:
    """(score, P (T, B, 3), Q (T, B, 4), dt) of one replayed episode: post-settle
    poses of the bodies with geoms, core first. perturb(brain, model, data) ->
    brain changes the controller before the episode."""
    model, data = build_world(genome)
    brain = make_brain_for_model(brain_kind, model, data, np.asarray(weights, dtype=np.float32))
    if perturb is not None:
        brain = perturb(brain, model, data)
    bodies = robot_bodies(model)
    t, pos, quat = [], [], []
    step = itertools.count()

    def record(d: mujoco.MjData) -> None:
        if next(step) % RECORD_EVERY_N == 0:
            t.append(d.time)
            pos.append(d.xpos[bodies].copy())
            quat.append(d.xquat[bodies].copy())

    score = run_episode(model, data, brain, recorder=record, on_settle_step=record)
    t32 = np.asarray(t, dtype=np.float32)
    keep = t32 >= PROTOCOL.settle_s
    P = np.asarray(pos, dtype=np.float16)[keep].astype(np.float64)
    Q = np.asarray(quat, dtype=np.float16)[keep].astype(np.float64)
    return score, P, Q, float(np.median(np.diff(t32[keep])))


def _replay_job(ep: runs.Episode) -> tuple[dict, list[dict], Optional[np.ndarray], str]:
    try:
        score, P, Q, dt = record_episode(ep.genome, ep.brain, ep.weights)
        ind, pairs = gait.analyse_episode(P, Q, dt)
        cycle, cyc_stats = gait.episode_cycle(P, Q, dt)
        row = {"key": ep.key, **ep.meta, "stored_score": ep.score, "score": score,
               **descriptors(ep.genome), **ind, **cyc_stats}
        return row, pairs, cycle.astype(np.float32), ""
    except Exception as exc:  # noqa: BLE001 -- one bad body must not stop the run
        return {"key": ep.key}, [], None, f"{type(exc).__name__}: {exc}"


def replay_run(run_dir: Path, out: Path, which: str, workers: int) -> str:
    eps = list(runs.episodes(run_dir, which))
    if not eps:
        return f"{run_dir.name}: nothing to replay"
    ident = runs.identity(run_dir)
    if workers > 1 and len(eps) > 1:
        with Pool(processes=min(workers, len(eps))) as pool:
            results = pool.map(_replay_job, eps, chunksize=1)
    else:
        results = [_replay_job(e) for e in eps]

    rows, pair_rows, cycles, keys, errors = [], [], [], [], []
    for row, pairs, cycle, err in results:
        if err:
            errors.append(f"{row['key']}: {err}")
            continue
        rows.append({**ident, **row})
        pair_rows += [{"key": row["key"], **p} for p in pairs]
        cycles.append(cycle)
        keys.append(row["key"])

    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "pairs.csv", index=False)
    if cycles:
        np.savez_compressed(out / "cycles.npz", key=np.array(keys), cycles=np.concatenate(cycles),
                            n_modules=np.array([len(c) for c in cycles]))
    for e in errors:
        print(f"  ERROR {e}", flush=True)
    mism = int((np.abs(df["score"] - df["stored_score"]) > SCORE_TOL).sum()) if len(df) else 0
    labels = ", ".join(f"{k}={v}" for k, v in df["rowing_label"].value_counts().items()) if len(df) else ""
    return f"{run_dir.name}: {len(df)} episodes ({mism} not reproduced, {len(errors)} errors) {labels}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", help="run directories to replay")
    parser.add_argument("--list-runs", type=Path, nargs="+", help="print the run dirs under these roots and exit")
    parser.add_argument("--which", default="final", help="E1: final, champion or g:<gens>; E2: every seed or champion")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("-o", "--out-dir", type=Path, help="results go to <out-dir>/<run name>/")
    args = parser.parse_args()

    if args.list_runs:
        for d in runs.find(args.list_runs, "e1") + runs.find(args.list_runs, "e2"):
            print(d)
        return
    if not args.run_dir or not args.out_dir:
        parser.error("--run-dir and --out-dir are required (or --list-runs)")
    for run_dir in args.run_dir:
        print(replay_run(run_dir, args.out_dir / run_dir.name, args.which, args.workers), flush=True)


if __name__ == "__main__":
    main()
