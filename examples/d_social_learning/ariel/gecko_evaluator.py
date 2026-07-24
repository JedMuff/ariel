"""Gecko-fixed evaluation primitives shared by the parallelization benchmark
and the NSGA-II brain-hyperparameter tuning script.

Bypasses TreeGenome/the outer morphology EA entirely — always builds the
prebuilt gecko topology via ``gecko_graph()``. Rollout logic (rotor-only
contact counting, CTRL_ALPHA control smoothing, conditional height penalty,
jerk penalty, glitch sentinel) mirrors ``evaluator.py::evaluate_individual``.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

# Append (not insert at position 0) so the installed ariel package always
# resolves before d_social_learning/ariel/, which would otherwise shadow it.
_SOCIAL_DIR = Path(__file__).parent.parent  # d_social_learning/
if str(_SOCIAL_DIR) not in sys.path:
    sys.path.append(str(_SOCIAL_DIR))

DURATION = 30.0
SETTLE_TIME = 3.0
SPAWN_POS = (-0.8, 0.0, 0.1)
CTRL_EVERY = 100
N_NEIGHBORS = 6
HIDDEN = 32
HINGE_CONTACT_LIMIT = 200
HINGE_CONTACT_PENALTY = 0.005
JERK_PENALTY_WEIGHT = 0.01
CTRL_ALPHA = 0.5
HEIGHT_PENALTY_THRESHOLD = 0.21


def _scale_actions(raw: np.ndarray) -> np.ndarray:
    import math
    return raw * (math.pi / 2)


def build_gecko_world(hidden: int = HIDDEN):
    """Build a fresh gecko MuJoCo world + brain + adapter.

    Returns
    -------
    model, data, adapter, brain, hinge_geom_ids, floor_geom_id
    """
    import mujoco

    from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
    from ariel.simulation.controllers.distributed_mlp import DistributedMLP
    from ariel.simulation.controllers.morphology_adapter import MorphologyAdapter
    from ariel.simulation.environments import SimpleFlatWorld
    from morphology_adapter import gecko_graph

    graph = gecko_graph()

    core = construct_mjspec_from_graph(graph)
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
    model = world.spec.compile()
    data = mujoco.MjData(model)

    hinge_geom_ids: set[int] = set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and name.endswith("-rotor"):
            hinge_geom_ids.add(i)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    adapter = MorphologyAdapter.from_graph(graph)
    brain = DistributedMLP(n_neighbors=N_NEIGHBORS, hidden=hidden)

    return model, data, adapter, brain, hinge_geom_ids, floor_geom_id


def make_run_episode(
    model: Any,
    data: Any,
    adapter: Any,
    brain: Any,
    hinge_geom_ids: set[int],
    floor_geom_id: int,
    *,
    duration: float = DURATION,
    ctrl_every: int = CTRL_EVERY,
) -> Callable[[np.ndarray], dict]:
    """Return a ``run_episode(theta) -> dict`` closure, matching evaluator.py's."""
    import mujoco

    def run_episode(theta: np.ndarray) -> dict:
        brain.set_theta(theta)
        mujoco.mj_resetData(model, data)

        core_height = float(data.qpos[2])

        sim_step = 0
        while data.time < SETTLE_TIME:
            mujoco.mj_step(model, data)
            sim_step += 1

        c_hinge = 0
        ctrl_step = 0
        active_hinge_contacts: set[frozenset[int]] = set()
        prev_ctrl = np.zeros(model.nu, dtype=np.float32)
        jerk_sum = 0.0
        rollout_end = SETTLE_TIME + duration
        while data.time < rollout_end:
            if sim_step % ctrl_every == 0:
                node_inputs, t = adapter.get_node_inputs(model, data, ctrl_step)
                raw = brain.forward_all(node_inputs, t)
                target_ctrl = _scale_actions(raw)
                new_ctrl = np.clip(
                    prev_ctrl * (1.0 - CTRL_ALPHA) + target_ctrl * CTRL_ALPHA,
                    -np.pi / 2, np.pi / 2,
                ).astype(np.float32)
                if ctrl_step > 0:
                    jerk_sum += float(np.mean(np.abs(new_ctrl - prev_ctrl)))
                prev_ctrl = new_ctrl.copy()
                data.ctrl[:] = new_ctrl
                ctrl_step += 1

            mujoco.mj_step(model, data)

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

        return {"fitness": fitness, "mean_jerk": mean_jerk, "c_hinge": c_hinge}

    return run_episode


# ---------------------------------------------------------------------------
# Inner-parallel worker pool: build the gecko world once per worker process
# (via Pool(initializer=...)), so only theta crosses the pickling boundary —
# matching the "build once, reuse across candidates" cost profile of the
# outer-parallel scheme for a fair comparison.
# ---------------------------------------------------------------------------

_worker_run_episode: Callable[[np.ndarray], dict] | None = None


def _init_worker(hidden: int, duration: float, ctrl_every: int) -> None:
    global _worker_run_episode
    model, data, adapter, brain, hinge_geom_ids, floor_geom_id = build_gecko_world(hidden)
    _worker_run_episode = make_run_episode(
        model, data, adapter, brain, hinge_geom_ids, floor_geom_id,
        duration=duration, ctrl_every=ctrl_every,
    )


def _eval_theta(theta: np.ndarray) -> dict:
    assert _worker_run_episode is not None
    return _worker_run_episode(theta)


@dataclass
class CMATrainingResult:
    best_theta: np.ndarray
    best_fitness: float
    learning_curve: list[list[float]] = field(default_factory=list)
    wall_time_s: float = 0.0


def run_cma_training(
    n_params: int,
    run_episode_fn: Callable[[np.ndarray], dict] | None,
    *,
    init_mean: np.ndarray | None = None,
    sigma: float = 0.5,
    pop_size: int = 16,
    inner_gens: int = 20,
    parallel_inner: bool = False,
    inner_workers: int | None = None,
    hidden: int = HIDDEN,
    duration: float = DURATION,
    ctrl_every: int = CTRL_EVERY,
) -> CMATrainingResult:
    """Run the CMA-ES ask/eval/tell loop, timed end to end.

    When ``parallel_inner`` is False, ``run_episode_fn`` must be provided and
    is called sequentially for every candidate (matches evaluator.py today).

    When ``parallel_inner`` is True, ``run_episode_fn`` is ignored — a
    ``Pool(initializer=_init_worker, ...)`` builds one gecko world per worker
    process (paying the MuJoCo compile cost once, not once per candidate),
    and each generation's candidates are evaluated via ``pool.map``.
    """
    from ariel.simulation.controllers.cmaes_learner import CMAESLearner

    learner = CMAESLearner(
        n_params=n_params,
        init_mean=init_mean,
        sigma=sigma,
        pop_size=pop_size,
    )

    learning_curve: list[list[float]] = []
    start = time.perf_counter()

    if parallel_inner:
        with Pool(
            processes=inner_workers,
            initializer=_init_worker,
            initargs=(hidden, duration, ctrl_every),
        ) as pool:
            for _ in range(inner_gens):
                candidates = learner.ask()
                eps = pool.map(_eval_theta, candidates)
                fitnesses = [ep["fitness"] for ep in eps]
                learner.tell(candidates, fitnesses)
                learning_curve.append(fitnesses)
    else:
        if run_episode_fn is None:
            raise ValueError("run_episode_fn is required when parallel_inner=False")
        for _ in range(inner_gens):
            candidates = learner.ask()
            eps = [run_episode_fn(theta) for theta in candidates]
            fitnesses = [ep["fitness"] for ep in eps]
            learner.tell(candidates, fitnesses)
            learning_curve.append(fitnesses)

    wall_time_s = time.perf_counter() - start

    return CMATrainingResult(
        best_theta=learner.best_theta,
        best_fitness=learner.best_fitness,
        learning_curve=learning_curve,
        wall_time_s=wall_time_s,
    )
