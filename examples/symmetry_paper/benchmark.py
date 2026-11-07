"""
Cost of the experiment on this machine: core-seconds per evaluation for every
genome and brain, and the projected core-hours of Experiments 1 and 2.

Trains --bodies random bodies per genome with every brain for --evals
evaluations on a pool of --workers processes (all trainings of a genome and
brain concurrently, as evolve.py does), and divides wall time x workers by the
evaluations done. Run it on the node type the experiment will use; per-machine
numbers are what the budget needs.

Writes --out/benchmark.csv (one row per genome and brain) and prints the
projection: E1 at --seeds seeds, (1 + generations) x mu + generations x lam
trainings per run; E2 at 80 bodies x 20 seeds x 800 evaluations.

Usage:
    python benchmark.py --workers 32 --bodies 4 --evals 60 -o bench/
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import genomes
from config import BRAIN_KINDS, BUDGETS, GENOME_TYPES, PROTOCOL
from sim import make_pool, train_controllers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--bodies", type=int, default=4, help="random bodies per genome")
    ap.add_argument("--evals", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=20, help="E1 seeds, for the projection")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    bodies = {g: [genomes.random_body(genomes.genome_ops(g))[0] for _ in range(args.bodies)] for g in GENOME_TYPES}
    rows = []
    with make_pool(args.workers) as pool:
        train_controllers([(bodies["tree"][0], "bang_bang", 20, 0)], pool, 1)  # warm the pool up
        for g in GENOME_TYPES:
            for b in BRAIN_KINDS:
                t0 = time.perf_counter()
                res = train_controllers([(body, b, args.evals, k) for k, body in enumerate(bodies[g])], pool,
                                        len(bodies[g]))
                wall = time.perf_counter() - t0
                n = sum(r.evals for r in res)
                rows.append({"genome": g, "brain": b, "evals": n, "wall_s": wall,
                             "core_s_per_eval": wall * args.workers / n,
                             "n_hinges": float(np.mean([genomes.hinge_count(x) for x in bodies[g]]))})
                print(f"{g:15s} {b:10s} {rows[-1]['core_s_per_eval']:.3f} core-s / evaluation", flush=True)
    df = pd.DataFrame(rows)
    df["workers"] = args.workers
    df.to_csv(args.out / "benchmark.csv", index=False)

    p = PROTOCOL
    trainings = (1 + p.generations) * p.mu + p.generations * p.lam
    per_eval = df.groupby("brain")["core_s_per_eval"].mean()
    print(f"\nProjection ({trainings} trainings per E1 run, {args.seeds} seeds, 5 genomes):")
    total = 0.0
    for b in BRAIN_KINDS:
        for e in BUDGETS:
            ch = per_eval[b] * e * trainings * args.seeds * len(GENOME_TYPES) / 3600
            total += ch
            print(f"  E1 {b:10s} {e:4d} evals: {ch:9,.0f} core-h  ({ch / args.seeds / len(GENOME_TYPES):6.1f} per run)")
    e2 = sum(per_eval[b] for b in BRAIN_KINDS) * 80 * len(GENOME_TYPES) * 20 * 800 / 3600
    print(f"  E1 total {total:,.0f} core-h;  E2 {e2:,.0f} core-h")


if __name__ == "__main__":
    main()
