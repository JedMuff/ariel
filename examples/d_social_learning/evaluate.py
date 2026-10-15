"""Picklable ARIEL evaluator: inner CMA-ES + DistributedMLP on MuJoCo sim."""

from __future__ import annotations

import os
import random

import simulator_dependent_functions

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import simulator_dependent_functions
import numpy as np

def init_worker(simulator, sim_preset="legacy"):
    simulator_dependent_functions.simulator = simulator
    simulator_dependent_functions.set_sim_preset(sim_preset)

def evaluate_individual(args: tuple) -> dict:
    """Picklable worker: run inner CMA-ES for one individual.

    Parameters
    ----------
    args : tuple
        (genome_dict, init_mean_list, donor_ids, inner_gens, pop_size, sigma, hidden, seed)

        ``seed`` seeds this worker's ``random`` and ``np.random`` (which
        nevergrad's CMA draws its random_state from), so an individual's
        learning is reproducible regardless of which pool worker runs it.

    Returns
    -------
    dict with keys:
        distance       : float  — best fitness achieved
        best_theta     : list[float]  — best θ weights
        init_fitness   : float  — episode fitness of inherited theta before any learning
        learning_curve : list[list[float]]  — per inner-gen, fitness of every candidate
        donor_ids      : list[int]  — db ids of individuals whose theta was inherited
    """
    from ariel.simulation.controllers.cmaes_learner import CMAESLearner
    from ariel.simulation.controllers.distributed_mlp import DistributedMLP

    genome, init_mean_list, donor_ids, inner_gens, pop_size, sigma, hidden, seed = args
    random.seed(seed)
    np.random.seed(seed)

    # Returned if the evaluation raises. NaN, not 0.0: most fitnesses are
    # negative, so 0.0 would rank a crashed individual near the top. Matches
    # experiment._timeout_result; experiment.py turns non-finite into -inf.
    _empty = {
        "distance": float("nan"),
        "best_theta": [],
        "init_fitness": float("nan"),
        "learning_curve": [],
        "donor_ids": donor_ids,
        "mean_jerk": float("nan"),
        "c_hinge": 0,
    }

    try:
        simulator_specifics = simulator_dependent_functions.initialize_world(genome)
        brain = DistributedMLP(n_neighbors=simulator_dependent_functions.n_neighbours(), hidden=hidden)

        init_mean = (
            np.asarray(init_mean_list, dtype=np.float64)
            if init_mean_list
            else None
        )

        # Evaluate inherited theta before any learning
        init_theta = init_mean if init_mean is not None else np.zeros(brain.n_params, dtype=np.float64)
        init_ep = simulator_dependent_functions.run_episode(init_theta, brain, simulator_specifics)
        init_fitness = init_ep["fitness"]

        learner = CMAESLearner(
            n_params=brain.n_params,
            init_mean=init_mean,
            sigma=sigma,
            pop_size=pop_size,
        )

        learning_curve: list[list[float]] = []
        for _ in range(inner_gens):
            candidates = learner.ask()
            eps = [simulator_dependent_functions.run_episode(theta, brain, simulator_specifics) for theta in candidates]
            fitnesses = [ep["fitness"] for ep in eps]
            learner.tell(candidates, fitnesses)
            learning_curve.append(fitnesses)

        # Diagnostics from best theta re-evaluation
        best_ep = simulator_dependent_functions.run_episode(np.asarray(learner.best_theta, dtype=np.float64), brain, simulator_specifics)

        simulator_dependent_functions.stop_simulator(simulator_specifics)

        return {
            "distance": learner.best_fitness,
            "best_theta": learner.best_theta.tolist(),
            "init_fitness": init_fitness,
            "learning_curve": learning_curve,
            "donor_ids": donor_ids
        } | simulator_dependent_functions.extra_results(best_ep)

    except Exception as exc:  # noqa: BLE001
        print(f"[evaluator] worker error: {exc}")
        return _empty