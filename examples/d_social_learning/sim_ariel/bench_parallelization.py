"""Benchmark outer- vs inner-parallelization for gecko brain training.

Trains N gecko brains (fixed morphology, varying CMA-ES random init only) via
one of two parallelization schemes:

  outer:  N processes in parallel, each running one full CMA-ES sequentially.
          Mirrors experiment.py's Pool(workers).map(evaluate_individual, ...).
  inner:  N gecko trainings run one after another; within each, the CMA-ES
          candidates for a generation are evaluated in parallel via a pool.

Usage:
    uv run examples/d_social_learning/ariel/bench_parallelization.py \
        --scheme outer --n-individuals 16 --inner-gens 20 --pop-size 16 \
        --workers 16 [--sigma 0.5] [--hidden 32] [--out path.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np
from rich.console import Console

import gecko_evaluator as ge

console = Console()


def _train_one_outer(args: tuple) -> dict:
    """Picklable worker for the outer scheme: build a world once, train sequentially."""
    seed, hidden, sigma, pop_size, inner_gens, duration, ctrl_every = args

    model, data, adapter, brain, hinge_ids, floor_id = ge.build_gecko_world(hidden)
    run_episode = ge.make_run_episode(
        model, data, adapter, brain, hinge_ids, floor_id,
        duration=duration, ctrl_every=ctrl_every,
    )

    rng = np.random.default_rng(seed)
    init_mean = rng.normal(scale=0.01, size=brain.n_params)

    start = time.perf_counter()
    result = ge.run_cma_training(
        n_params=brain.n_params,
        run_episode_fn=run_episode,
        init_mean=init_mean,
        sigma=sigma,
        pop_size=pop_size,
        inner_gens=inner_gens,
        parallel_inner=False,
    )
    elapsed = time.perf_counter() - start

    return {"seed": seed, "time_s": elapsed, "best_fitness": result.best_fitness}


def run_outer_scheme(
    n_individuals: int, hidden: int, sigma: float, pop_size: int,
    inner_gens: int, duration: float, ctrl_every: int, workers: int,
) -> dict:
    worker_args = [
        (seed, hidden, sigma, pop_size, inner_gens, duration, ctrl_every)
        for seed in range(n_individuals)
    ]
    start = time.perf_counter()
    with Pool(processes=workers) as pool:
        individuals = pool.map(_train_one_outer, worker_args)
    total_time_s = time.perf_counter() - start
    return {"scheme": "outer", "total_time_s": total_time_s, "individuals": individuals}


def run_inner_scheme(
    n_individuals: int, hidden: int, sigma: float, pop_size: int,
    inner_gens: int, duration: float, ctrl_every: int, workers: int,
) -> dict:
    # n_params depends only on hidden (and the fixed N_NEIGHBORS/features_per_node),
    # so it can be computed once without building a full world.
    from ariel.simulation.controllers.distributed_mlp import DistributedMLP
    n_params = DistributedMLP(n_neighbors=ge.N_NEIGHBORS, hidden=hidden).n_params

    individuals = []
    start = time.perf_counter()
    for seed in range(n_individuals):
        rng = np.random.default_rng(seed)
        init_mean = rng.normal(scale=0.01, size=n_params)

        indiv_start = time.perf_counter()
        result = ge.run_cma_training(
            n_params=n_params,
            run_episode_fn=None,
            init_mean=init_mean,
            sigma=sigma,
            pop_size=pop_size,
            inner_gens=inner_gens,
            parallel_inner=True,
            inner_workers=workers,
            hidden=hidden,
            duration=duration,
            ctrl_every=ctrl_every,
        )
        indiv_elapsed = time.perf_counter() - indiv_start
        individuals.append({"seed": seed, "time_s": indiv_elapsed, "best_fitness": result.best_fitness})
    total_time_s = time.perf_counter() - start
    return {"scheme": "inner", "total_time_s": total_time_s, "individuals": individuals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scheme", required=True, choices=["outer", "inner"])
    parser.add_argument("--n-individuals", type=int, default=16)
    parser.add_argument("--inner-gens", type=int, default=20)
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=ge.HIDDEN)
    parser.add_argument("--duration", type=float, default=ge.DURATION)
    parser.add_argument("--ctrl-every", type=int, default=ge.CTRL_EVERY)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    console.rule(f"[bold cyan]Bench parallelization | scheme={args.scheme} n={args.n_individuals} workers={args.workers}")

    run_fn = run_outer_scheme if args.scheme == "outer" else run_inner_scheme
    summary = run_fn(
        n_individuals=args.n_individuals,
        hidden=args.hidden,
        sigma=args.sigma,
        pop_size=args.pop_size,
        inner_gens=args.inner_gens,
        duration=args.duration,
        ctrl_every=args.ctrl_every,
        workers=args.workers,
    )
    summary["config"] = vars(args)

    out_path = Path(args.out) if args.out else Path(
        f"__data__/benchmarks/{args.scheme}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    fitnesses = [ind["best_fitness"] for ind in summary["individuals"]]
    console.log(f"Total time: {summary['total_time_s']:.1f}s")
    console.log(f"Best fitness: mean={np.mean(fitnesses):.4f} std={np.std(fitnesses):.4f}")
    console.log(f"Written to {out_path}")
    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
