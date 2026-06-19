"""Picklable ARIEL evaluator: inner CMA-ES + DistributedMLP on MuJoCo sim."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

DURATION = 10.0
SPAWN_POS = (-0.8, 0.0, 0.1)
CTRL_EVERY = 50
N_NEIGHBORS = 6


def _scale_actions(raw: np.ndarray) -> np.ndarray:
    import math
    return raw * (math.pi / 2)


def evaluate_individual(args: tuple) -> dict:
    """Picklable worker: run inner CMA-ES for one individual.

    Parameters
    ----------
    args : tuple
        (genome_dict, init_mean_list, donor_ids, inner_gens, pop_size)

    Returns
    -------
    dict with keys:
        distance       : float  — best x-displacement achieved
        best_theta     : list[float]  — best θ weights
        init_fitness   : float  — episode fitness of inherited theta before any learning
        learning_curve : list[list[float]]  — per inner-gen, fitness of every candidate
        donor_ids      : list[int]  — db ids of individuals whose theta was inherited
    """
    import mujoco

    from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
    from ariel.ec.genotypes.tree.tree_genome import TreeGenome
    from ariel.simulation.controllers.cmaes_learner import CMAESLearner
    from ariel.simulation.controllers.distributed_mlp import DistributedMLP
    from ariel.simulation.controllers.morphology_adapter import MorphologyAdapter
    from ariel.simulation.environments import SimpleFlatWorld

    genome_dict, init_mean_list, donor_ids, inner_gens, pop_size = args

    _empty = {"distance": 0.0, "best_theta": [], "init_fitness": 0.0, "learning_curve": [], "donor_ids": donor_ids}

    try:
        genome = TreeGenome.from_dict(genome_dict)
        graph = genome.to_networkx()

        core = construct_mjspec_from_graph(graph)
        world = SimpleFlatWorld()
        world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
        model = world.spec.compile()
        data = mujoco.MjData(model)

        adapter = MorphologyAdapter.from_graph(graph)
        brain = DistributedMLP(n_neighbors=N_NEIGHBORS)

        init_mean = (
            np.asarray(init_mean_list, dtype=np.float64)
            if init_mean_list
            else None
        )

        def run_episode(theta: np.ndarray) -> float:
            brain.set_theta(theta)
            mujoco.mj_resetData(model, data)
            ctrl_step = sim_step = 0
            while data.time < DURATION:
                if sim_step % CTRL_EVERY == 0:
                    node_inputs, t = adapter.get_node_inputs(model, data, ctrl_step)
                    raw = brain.forward_all(node_inputs, t)
                    data.ctrl[:] = _scale_actions(raw)
                    ctrl_step += 1
                mujoco.mj_step(model, data)
                sim_step += 1
            return float(data.qpos[0])

        # Evaluate inherited theta before any learning
        if init_mean is not None:
            init_fitness = run_episode(init_mean)
        else:
            init_fitness = run_episode(np.zeros(brain.n_params, dtype=np.float64))

        learner = CMAESLearner(
            n_params=brain.n_params,
            init_mean=init_mean,
            sigma=0.5,
            pop_size=pop_size,
        )

        learning_curve: list[list[float]] = []
        for _ in range(inner_gens):
            candidates = learner.ask()
            fitnesses = [run_episode(theta) for theta in candidates]
            learner.tell(candidates, fitnesses)
            learning_curve.append(fitnesses)

        return {
            "distance": learner.best_fitness,
            "best_theta": learner.best_theta.tolist(),
            "init_fitness": init_fitness,
            "learning_curve": learning_curve,
            "donor_ids": donor_ids,
        }

    except Exception as exc:  # noqa: BLE001
        print(f"[evaluator] worker error: {exc}")
        return _empty
