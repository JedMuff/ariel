"""Picklable ARIEL evaluator: inner CMA-ES + DistributedMLP on MuJoCo sim."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

DURATION = 30.0
SETTLE_TIME = 3.0
SPAWN_POS = (-0.8, 0.0, 0.1)
CTRL_EVERY = 100
N_NEIGHBORS = 6
HINGE_CONTACT_LIMIT = 200
HINGE_CONTACT_PENALTY = 0.005
JERK_PENALTY_WEIGHT = 0.01   # penalty per unit of mean absolute ctrl delta
CTRL_ALPHA = 0.5   # control blending factor (0=no change, 1=instant) — damps resonant ground-jitter exploit
HEIGHT_PENALTY_THRESHOLD = 0.21  # m — only penalise spawn height above this


def _scale_actions(raw: np.ndarray) -> np.ndarray:
    import math
    return raw * (math.pi / 2)


def evaluate_individual(args: tuple) -> dict:
    """Picklable worker: run inner CMA-ES for one individual.

    Parameters
    ----------
    args : tuple
        (genome_dict, init_mean_list, donor_ids, inner_gens, pop_size, sigma, hidden)

    Returns
    -------
    dict with keys:
        distance       : float  — best fitness achieved
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

    genome_dict, init_mean_list, donor_ids, inner_gens, pop_size, sigma, hidden = args

    _empty = {"distance": 0.0, "best_theta": [], "init_fitness": 0.0, "learning_curve": [], "donor_ids": donor_ids}

    try:
        genome = TreeGenome.from_dict(genome_dict)
        graph = genome.to_networkx()

        core = construct_mjspec_from_graph(graph)
        world = SimpleFlatWorld()
        world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
        model = world.spec.compile()
        data = mujoco.MjData(model)

        # Build rotor (not stator) and floor geom ID sets once per model
        hinge_geom_ids: set[int] = set()
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name.endswith("-rotor"):
                hinge_geom_ids.add(i)
        floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        adapter = MorphologyAdapter.from_graph(graph)
        brain = DistributedMLP(n_neighbors=N_NEIGHBORS, hidden=hidden)

        init_mean = (
            np.asarray(init_mean_list, dtype=np.float64)
            if init_mean_list
            else None
        )

        def run_episode(theta: np.ndarray) -> dict:
            """Run one episode; return fitness and diagnostics."""
            brain.set_theta(theta)
            mujoco.mj_resetData(model, data)

            # Record spawn height before any steps (penalises falling)
            core_height = float(data.qpos[2])

            # Settle phase: let robot fall into place, no control, no counting
            sim_step = 0
            while data.time < SETTLE_TIME:
                mujoco.mj_step(model, data)
                sim_step += 1

            # Rollout phase
            c_hinge = 0
            ctrl_step = 0
            active_hinge_contacts: set[frozenset[int]] = set()
            prev_ctrl = np.zeros(model.nu, dtype=np.float32)
            jerk_sum = 0.0
            rollout_end = SETTLE_TIME + DURATION
            while data.time < rollout_end:
                if sim_step % CTRL_EVERY == 0:
                    node_inputs, t = adapter.get_node_inputs(model, data, ctrl_step)
                    raw = brain.forward_all(node_inputs, t)
                    target_ctrl = _scale_actions(raw)
                    # Alpha-blend towards new action and clip to servo range
                    # to prevent resonant ground-jitter exploitation.
                    new_ctrl = np.clip(
                        prev_ctrl * (1.0 - CTRL_ALPHA) + target_ctrl * CTRL_ALPHA,
                        -np.pi / 2, np.pi / 2,
                    ).astype(np.float32)
                    if ctrl_step > 0 and model.nu > 0:
                        jerk_sum += float(np.mean(np.abs(new_ctrl - prev_ctrl)))
                    prev_ctrl = new_ctrl.copy()
                    data.ctrl[:] = new_ctrl
                    ctrl_step += 1

                mujoco.mj_step(model, data)

                # Rising-edge hinge-floor contact events
                current: set[frozenset[int]] = set()
                for k in range(data.ncon):
                    c = data.contact[k]
                    if ((c.geom1 == floor_geom_id and c.geom2 in hinge_geom_ids) or
                            (c.geom2 == floor_geom_id and c.geom1 in hinge_geom_ids)):
                        current.add(frozenset((c.geom1, c.geom2)))
                c_hinge += len(current - active_hinge_contacts)
                active_hinge_contacts = current
                sim_step += 1

            mean_jerk = jerk_sum / max(ctrl_step - 1, 1)

            d = float(data.qpos[0])
            if c_hinge > HINGE_CONTACT_LIMIT or not np.isfinite(d):
                fitness = -1.0
            else:
                height_penalty = core_height if core_height > HEIGHT_PENALTY_THRESHOLD else 0.0
                fitness = (
                    d
                    - height_penalty
                    - HINGE_CONTACT_PENALTY * c_hinge
                    - JERK_PENALTY_WEIGHT * mean_jerk
                )

            return {
                "fitness": fitness,
                "mean_jerk": mean_jerk,
                "c_hinge": c_hinge,
            }

        # Evaluate inherited theta before any learning
        init_theta = init_mean if init_mean is not None else np.zeros(brain.n_params, dtype=np.float64)
        init_ep = run_episode(init_theta)
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
            eps = [run_episode(theta) for theta in candidates]
            fitnesses = [ep["fitness"] for ep in eps]
            learner.tell(candidates, fitnesses)
            learning_curve.append(fitnesses)

        # Diagnostics from best theta re-evaluation
        best_ep = run_episode(np.asarray(learner.best_theta, dtype=np.float64))

        return {
            "distance": learner.best_fitness,
            "best_theta": learner.best_theta.tolist(),
            "init_fitness": init_fitness,
            "learning_curve": learning_curve,
            "donor_ids": donor_ids,
            "mean_jerk": best_ep["mean_jerk"],
            "c_hinge": best_ep["c_hinge"],
        }

    except Exception as exc:  # noqa: BLE001
        print(f"[evaluator] worker error: {exc}")
        return _empty
