"""ARIEL social-learning experiment: (mu+lambda) morphology EA + CMA-ES brain learning.

Usage:
    uv run examples/d_social_learning/ariel/experiment.py \
        --scheme lamarckian --x 0.5 --rep 0 [--gens 100] [--pop 20] [--lam 100] \
        [--inner-gens 20] [--inner-pop 16] [--sigma 0.5] [--hidden 32] [--workers N]
"""

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

# Add local module dirs to sys.path WITHOUT adding d_social_learning/ itself,
# because that directory contains an ariel/ subdirectory that would shadow the
# installed ariel package.
_THIS_DIR = Path(__file__).parent          # d_social_learning/ariel/
_SOCIAL_DIR = _THIS_DIR.parent             # d_social_learning/
_CORE_DIR = _SOCIAL_DIR / "core"
for _p in [str(_THIS_DIR), str(_CORE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from rich.console import Console

from ariel.ec import EA, EAOperation, Individual, Population

# Local social-learning modules (resolved via _THIS_DIR on sys.path)
from descriptor import tree_descriptor       # d_social_learning/ariel/descriptor.py
from morphology_ops import mutate, random_individual
from evaluator import evaluate_individual

from fitness import combined_fitness
from inheritance import SCHEMES
from novelty import compute_novelty

console = Console()

N_NEIGHBORS = 6  # must match evaluator.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pop_state(individuals: list[Individual]) -> list[dict]:
    """Build lightweight state list for inheritance scheme functions."""
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
            try:
                d = tree_descriptor(ind.genotype_["morph"])
            except Exception:  # noqa: BLE001
                d = np.zeros(8, dtype=np.float64)
            descs.append(d)
    return descs


# ---------------------------------------------------------------------------
# (mu+lambda) EA operations as EAOperation functions
# ---------------------------------------------------------------------------

def build_ops(
    scheme_name: str,
    x_val: float,
    mu: int,
    lam: int,
    inner_gens: int,
    inner_pop: int,
    num_workers: int,
    sigma: float,
    hidden: int,
    comma_selection: bool = False,
) -> list[EAOperation]:
    """Return the ordered list of EAOperation steps for the outer EA."""
    @EAOperation
    def generate_offspring(population: Population) -> Population:
        parents = [ind for ind in population if ind.alive and not ind.requires_eval]
        if not parents:
            parents = [ind for ind in population if ind.alive]

        offspring_list = []
        # sample parents without replacement where possible, cycling if lam > mu
        import random as _random
        parent_pool = list(parents)
        for i in range(lam):
            if i % len(parent_pool) == 0:
                _random.shuffle(parent_pool)
            parent = parent_pool[i % len(parent_pool)]
            child_morph = mutate(parent.genotype_["morph"])
            child = Individual()
            parent_brain = parent.genotype_.get("brain") or []
            child.genotype = {"morph": child_morph, "brain": parent_brain}
            child.tags = {"parent_id": parent.id}
            offspring_list.append(child)

        pop_out = Population(list(population))
        pop_out.extend(offspring_list)
        return pop_out

    @EAOperation
    def evaluate_and_select(population: Population) -> Population:
        parents = [ind for ind in population if ind.alive and not ind.requires_eval]
        offspring = [ind for ind in population if ind.alive and ind.requires_eval]
        all_alive = parents + offspring

        descs = _compute_descriptors(all_alive)
        novelties = compute_novelty(descs)

        if x_val == 0.0:
            # Pure novelty — no simulation needed
            for i, ind in enumerate(all_alive):
                novelty = float(novelties[i])
                desc = descs[i]
                prior = ind.tags_ or {}
                ind.fitness = novelty
                ind.tags = {
                    "parent_id": prior.get("parent_id"),
                    "distance": 0.0,
                    "novelty": novelty,
                    "descriptor": desc.tolist(),
                    "theta": prior.get("theta", []),
                    "init_fitness": 0.0,
                    "learning_curve": [],
                    "donor_ids": [],
                    "mean_jerk": 0.0,
                    "c_hinge": 0,
                }
        else:
            all_state = _pop_state(all_alive)
            scheme_fn = SCHEMES[scheme_name]

            from ariel.simulation.controllers.distributed_mlp import DistributedMLP
            n_params = DistributedMLP(n_neighbors=N_NEIGHBORS, hidden=hidden).n_params

            worker_args = []
            for i, ind in enumerate(all_alive):
                init_mean_arr, donor_ids = scheme_fn(all_state, i, n_params)
                worker_args.append((
                    ind.genotype_["morph"],
                    init_mean_arr.tolist(),
                    donor_ids,
                    inner_gens,
                    inner_pop,
                    sigma,
                    hidden,
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
                prior = ind.tags_ or {}
                ind.fitness = combined_fitness(distance, novelty, x_val)
                ind.tags = {
                    "parent_id": prior.get("parent_id"),
                    "distance": distance,
                    "novelty": novelty,
                    "descriptor": desc.tolist(),
                    "theta": theta_list,
                    "init_fitness": r["init_fitness"],
                    "learning_curve": r["learning_curve"],
                    "donor_ids": r["donor_ids"],
                    "mean_jerk": r.get("mean_jerk", 0.0),
                    "c_hinge": r.get("c_hinge", 0),
                }
                ind.genotype_ = {"morph": ind.genotype_["morph"], "brain": theta_list}

        if comma_selection:
            # (mu,lambda): survivors drawn only from offspring, parents always die.
            selection_pool = Population(offspring)
        else:
            # (mu+lambda): survivors drawn from parents + offspring.
            selection_pool = Population(all_alive)
        survivors = selection_pool.best(n=mu).to_list()
        survivor_ids = {id(s) for s in survivors}
        for ind in all_alive:
            ind.alive = id(ind) in survivor_ids

        return Population(all_alive)

    return [generate_offspring, evaluate_and_select]


# ---------------------------------------------------------------------------
# Initial population
# ---------------------------------------------------------------------------

def make_initial_population(mu: int) -> Population:
    inds = []
    for _ in range(mu):
        ind = Individual()
        ind.genotype = {"morph": random_individual(), "brain": []}
        inds.append(ind)
    return Population(inds)


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
    parser.add_argument("--comma-selection", action="store_true",
                        help="Use (mu,lambda) selection (survivors from offspring only) "
                             "instead of the default (mu+lambda) (survivors from parents+offspring). "
                             "Requires --lam >= --pop.")
    parser.add_argument("--inner-gens", type=int, default=20)
    parser.add_argument("--inner-pop", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.5, help="CMA-ES initial step size")
    parser.add_argument("--hidden", type=int, default=32, help="DistributedMLP hidden width")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing database.db at the last saved generation")
    args = parser.parse_args()

    if args.comma_selection and args.lam < args.pop:
        parser.error("--comma-selection requires --lam >= --pop (not enough offspring to fill mu)")

    x_str = str(args.x).replace(".", "")
    out_dir = Path(f"__data__/social/ariel/{args.scheme}/x{x_str}/rep_{args.rep}")
    out_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold cyan]ARIEL | scheme={args.scheme} x={args.x} rep={args.rep}")

    ops = build_ops(
        scheme_name=args.scheme,
        x_val=args.x,
        mu=args.pop,
        lam=args.lam,
        inner_gens=args.inner_gens,
        inner_pop=args.inner_pop,
        num_workers=args.workers,
        sigma=args.sigma,
        hidden=args.hidden,
        comma_selection=args.comma_selection,
    )

    db_path = out_dir / "database.db"

    if args.resume:
        resume_db_path = out_dir / "database_part2.db"
        ea = EA(
            restart=db_path,
            operations=ops,
            num_steps=args.gens,
            db_file_path=resume_db_path,
            db_handling="delete",
        )
    else:
        ea = EA(
            population=make_initial_population(args.pop),
            operations=ops,
            num_steps=args.gens,
            db_file_path=db_path,
            db_handling="delete",
        )
    ea.run()

    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
