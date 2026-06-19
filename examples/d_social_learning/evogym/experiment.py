"""EvoGym social-learning experiment: (mu+lambda) EA + CMA-ES brain learning.

Usage (evogym-venv, Python 3.10):
    evogym-venv/bin/python examples/d_social_learning/evogym/experiment.py \
        --scheme lamarckian --x 0.5 --rep 0 [--gens 100] [--pop 20] [--lam 100] \
        [--inner-gens 20] [--inner-pop 16] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

_THIS_DIR = Path(__file__).parent       # d_social_learning/evogym/
_SOCIAL_DIR = _THIS_DIR.parent          # d_social_learning/
_CORE_DIR = _SOCIAL_DIR / "core"

for _p in [str(_THIS_DIR), str(_CORE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from db import Individual, Population, SimpleEA
from descriptor import voxel_descriptor
from morphology_ops import body_from_list, body_to_list, mutate_body, random_body
from evaluator import evaluate_individual
from fitness import combined_fitness
from inheritance import SCHEMES
from novelty import compute_novelty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pop_state(individuals: list[Individual]) -> list[dict]:
    states = []
    for ind in individuals:
        theta = ind.tags_.get("theta") if ind.tags_ else None
        fitness = ind.fitness_ if not ind.requires_eval else None
        descriptor = ind.tags_.get("descriptor") if ind.tags_ else None
        if descriptor is None:
            descriptor = [0.0] * 8
        states.append({
            "descriptor": np.array(descriptor, dtype=np.float64),
            "theta": np.array(theta, dtype=np.float64) if theta else None,
            "fitness": fitness,
            "db_id": ind.id,
        })
    return states


def _compute_descriptors(individuals: list[Individual]) -> list[np.ndarray]:
    descs = []
    for ind in individuals:
        stored = ind.tags_.get("descriptor") if ind.tags_ else None
        if stored:
            descs.append(np.array(stored, dtype=np.float64))
        else:
            body = body_from_list(ind.genotype_["body"])
            descs.append(voxel_descriptor(body))
    return descs


# ---------------------------------------------------------------------------
# Step function
# ---------------------------------------------------------------------------


def _compute_n_params(n_neighbors: int = 8, features_per_node: int = 8, hidden: int = 32) -> int:
    """Replicate DistributedMLP n_params calculation without importing ariel."""
    input_size = (1 + n_neighbors) * features_per_node + 1
    return (input_size * hidden + hidden) + (hidden * 1 + 1)


def make_step_fn(
    scheme_name: str,
    x_val: float,
    mu: int,
    lam: int,
    inner_gens: int,
    inner_pop: int,
    num_workers: int,
):
    n_params = _compute_n_params(n_neighbors=8)
    scheme_fn = SCHEMES[scheme_name]

    def step(population: Population, current_gen: int) -> Population:
        parents = [ind for ind in population if ind.alive and not ind.requires_eval]
        if not parents:
            parents = [ind for ind in population if ind.alive]

        offspring_list = []
        import random as _random
        parent_pool = list(parents)
        for i in range(lam):
            if i % len(parent_pool) == 0:
                _random.shuffle(parent_pool)
            parent = parent_pool[i % len(parent_pool)]
            child_body = mutate_body(body_from_list(parent.genotype_["body"]))
            child = Individual()
            child.genotype = {"body": body_to_list(child_body), "brain": []}
            offspring_list.append(child)

        all_alive = parents + offspring_list

        descs = _compute_descriptors(all_alive)
        novelties = compute_novelty(descs)
        all_state = _pop_state(all_alive)

        worker_args = []
        for i, ind in enumerate(all_alive):
            init_mean_arr, donor_ids = scheme_fn(all_state, i, n_params)
            worker_args.append((
                ind.genotype_["body"],
                init_mean_arr.tolist(),
                donor_ids,
                inner_gens,
                inner_pop,
            ))

        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results = pool.map(evaluate_individual, worker_args)
        else:
            results = [evaluate_individual(a) for a in worker_args]

        for i, ind in enumerate(all_alive):
            r = results[i]
            distance = r["distance"]
            theta_list = r["best_theta"]
            novelty = float(novelties[i])
            desc = descs[i]
            fitness = combined_fitness(distance, novelty, x_val)
            ind.fitness = fitness
            ind.tags = {
                "distance": distance,
                "novelty": novelty,
                "descriptor": desc.tolist(),
                "theta": theta_list,
                "init_fitness": r["init_fitness"],
                "learning_curve": r["learning_curve"],
                "donor_ids": r["donor_ids"],
            }
            ind.genotype_ = {"body": ind.genotype_["body"], "brain": theta_list}

        combined = Population(all_alive)
        survivors = combined.best(n=mu).to_list()
        survivor_ids = {id(s) for s in survivors}
        for ind in all_alive:
            ind.alive = id(ind) in survivor_ids

        return Population(all_alive)

    return step


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", required=True, choices=list(SCHEMES.keys()))
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--pop", type=int, default=20, help="mu")
    parser.add_argument("--lam", type=int, default=100)
    parser.add_argument("--inner-gens", type=int, default=20)
    parser.add_argument("--inner-pop", type=int, default=16)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    x_str = str(args.x).replace(".", "")
    out_dir = Path(f"__data__/social/evogym/{args.scheme}/x{x_str}/rep_{args.rep}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"EvoGym | scheme={args.scheme} x={args.x} rep={args.rep}", flush=True)

    # initial population
    inds = []
    for _ in range(args.pop):
        ind = Individual()
        body = random_body()
        ind.genotype = {"body": body_to_list(body), "brain": []}
        inds.append(ind)
    pop = Population(inds)

    step_fn = make_step_fn(
        scheme_name=args.scheme,
        x_val=args.x,
        mu=args.pop,
        lam=args.lam,
        inner_gens=args.inner_gens,
        inner_pop=args.inner_pop,
        num_workers=args.workers,
    )

    ea = SimpleEA(
        population=pop,
        db_file_path=out_dir / "database.db",
        db_handling="delete",
    )
    ea.run(step_fn, num_steps=args.gens)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
