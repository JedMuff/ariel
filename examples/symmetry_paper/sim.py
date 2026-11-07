"""
Simulation and controller learning: zoo bodies on a flat floor, the forward
episode, and the inner CMA-ES (config.py has the settings).

Bodies are built from the local module classes with kgd's zoo sizes, masses and
hinge servo (the zoo benchmark's zoo_modules.py). The module classes read their
sizes and masses from module-level constants, which this module sets to the zoo
values on import, for the whole process and its forked workers, so the genome
operators' collision checks see the same geometry as the simulator.

The episode is the v2 paper's (legacy/sim.py) with the displacement fitness
only: the same body and weights give the same displacement. CMA-ES draws from
its own generator per training (v2 used numpy's global one), so seeded
trainings do not reproduce v2's.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cma
import mujoco
import numpy as np

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld

from brains import BRAIN_X0, make_brain_for_model
from config import PROTOCOL, Protocol, cma_generations

SPAWN_POSITION = (0.0, 0.0, 0.1)
# Every body spawns unrotated, so the core camera's forward direction (world -Y)
# is the same for every body: the reward axis.
FORWARD_AXIS = np.array([0.0, -1.0])

# ── Zoo modules ───────────────────────────────────────────────────────────────

ZOO_HINGE_KP          = 1.3605113362038874
ZOO_HINGE_KV          = 0.3555497840137177
ZOO_HINGE_ARMATURE    = 0.08117047736037042
ZOO_HINGE_RANGE       = (-math.pi / 2, math.pi / 2)  # hard joint limit, same as the ctrl range
ZOO_CORE_MASS         = 1.363
ZOO_STATOR_MASS       = 0.065
ZOO_ROTOR_MASS        = 0.040
ZOO_BRICK_MASS        = 0.055
ZOO_CORE_DIMENSIONS   = (0.075, 0.075, 0.075)  # half-extents (m)
ZOO_BRICK_DIMENSIONS  = (0.0375, 0.0375, 0.0375)
ZOO_STATOR_DIMENSIONS = (0.026, 0.022, 0.026)
ZOO_ROTOR_DIMENSIONS  = (0.026, 0.0155, 0.026)


def use_zoo_modules() -> None:
    """Set the module classes' size and mass constants to the zoo values (and
    drop the cached collision geometry compiled from the old ones)."""
    from ariel.body_phenotypes.robogen_lite.collision_utils import module_template  # noqa: PLC0415
    from ariel.body_phenotypes.robogen_lite.modules import brick, core, hinge  # noqa: PLC0415

    core.CORE_MASS, core.CORE_DIMENSIONS = ZOO_CORE_MASS, ZOO_CORE_DIMENSIONS
    hinge.STATOR_MASS, hinge.ROTOR_MASS = ZOO_STATOR_MASS, ZOO_ROTOR_MASS
    hinge.STATOR_DIMENSIONS, hinge.ROTOR_DIMENSIONS = ZOO_STATOR_DIMENSIONS, ZOO_ROTOR_DIMENSIONS
    brick.ariel_modules_config = brick.ariel_modules_config.model_copy(
        update={"BRICK_MASS": ZOO_BRICK_MASS, "BRICK_DIMENSIONS": ZOO_BRICK_DIMENSIONS},
    )
    module_template.cache_clear()


use_zoo_modules()


def apply_zoo_hinge(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """The zoo hinge servo (gain, damping, armature, hard +-90 deg limit), in
    place: hard-coded in hinge.py, so set after construction."""
    for a in spec.actuators:
        a.gainprm[0] = ZOO_HINGE_KP
        a.biasprm[1] = -ZOO_HINGE_KP
        a.biasprm[2] = -ZOO_HINGE_KV
    scale = 180.0 / math.pi if spec.compiler.degree else 1.0
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.armature = ZOO_HINGE_ARMATURE
            j.range = [r * scale for r in ZOO_HINGE_RANGE]
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    return spec


def body_spec(body: dict) -> Optional[mujoco.MjSpec]:
    """The MjSpec of a body (tree genome dict: nodes and edges; other keys are
    ignored), or None if it cannot be built."""
    try:
        graph = TreeGenome.from_dict(body).to_networkx()
        if graph.number_of_nodes() == 0:
            return None
        spec = construct_mjspec_from_graph(graph).spec
    except Exception:
        return None
    return apply_zoo_hinge(spec)


def build_world(body: dict) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """The body on a flat floor."""
    spec = body_spec(body)
    if spec is None:
        raise ValueError("Could not build the body")
    world = SimpleFlatWorld()
    try:
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=False)
    model = world.spec.compile()
    return model, mujoco.MjData(model)


def genome_hash(body: dict) -> str:
    return hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()


def robot_bodies(model: mujoco.MjModel) -> list[int]:
    """Robot bodies that own at least one geom (core, stators, rotors, bricks)."""
    has_geom = {int(b) for b in model.geom_bodyid}
    return [b for b in range(model.nbody) if model.body(b).name.startswith("robot1_") and b in has_geom]


# ── Episode ───────────────────────────────────────────────────────────────────


def run_episode(
    model: mujoco.MjModel, data: mujoco.MjData, brain: Any,
    protocol: Protocol = PROTOCOL,
    recorder: Optional[Callable[[mujoco.MjData], None]] = None,
    on_settle_step: Optional[Callable[[mujoco.MjData], None]] = None,
) -> float:
    """Forward displacement in m (along FORWARD_AXIS, unclipped) of one episode
    of a brain whose params are set. recorder(data) is called after every
    post-settle physics step and on_settle_step(data) after every settle step
    (replay.py's recording); neither may change data."""
    brain.reset()
    mujoco.mj_resetData(model, data)
    while data.time < protocol.settle_s:
        mujoco.mj_step(model, data)
        if on_settle_step is not None:
            on_settle_step(data)

    step   = 0
    action = np.zeros(model.nu)
    xy0    = np.array([data.qpos[0], data.qpos[1]])
    end    = protocol.settle_s + protocol.episode_s
    while data.time < end:
        if step % protocol.control_every == 0:
            target = brain.act(data, get_state_from_data(data).astype(np.float32))
            action = np.clip(action * (1.0 - protocol.ctrl_alpha) + target * protocol.ctrl_alpha,
                             -math.pi / 2, math.pi / 2)
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        if recorder is not None:
            recorder(data)
        step += 1
    return float(np.dot(np.array([data.qpos[0], data.qpos[1]]) - xy0, FORWARD_AXIS))


# ── Inner CMA-ES ──────────────────────────────────────────────────────────────
#
# Candidates are evaluated by a pool of worker processes shared by every
# training in the run: train_controllers trains several bodies at once (one
# thread per body drives its CMA-ES), so a generation's bodies keep all
# workers busy whatever the core count. Each worker caches the worlds it has
# built; an episode resets the world and the brain, so its result does not
# depend on which worker ran it.

_WORLD_CACHE_SIZE = 64
_worlds: dict[tuple[str, str], tuple[mujoco.MjModel, mujoco.MjData, Any]] = {}


def _world(body: dict, brain_kind: str, protocol: Protocol) -> tuple[mujoco.MjModel, mujoco.MjData, Any]:
    key = (genome_hash(body), brain_kind)
    if key not in _worlds:
        if len(_worlds) >= _WORLD_CACHE_SIZE:
            _worlds.pop(next(iter(_worlds)))
        model, data = build_world(body)
        _worlds[key] = (model, data, make_brain_for_model(brain_kind, model, data, duration=protocol.episode_s))
    return _worlds[key]


def evaluate(body: dict, brain_kind: str, weights: np.ndarray, protocol: Protocol = PROTOCOL) -> float:
    """Displacement in m of one episode of a controller (in this process)."""
    model, data, brain = _world(body, brain_kind, protocol)
    brain.set_params(np.asarray(weights, dtype=np.float32))
    return run_episode(model, data, brain, protocol)


def _evaluate_task(args: tuple) -> float:
    return evaluate(*args)


def _worker_init() -> None:
    np.random.seed(os.getpid() % (2**32 - 1))


def make_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=max(1, workers), initializer=_worker_init)


@dataclass
class Training:
    """One controller training. Scores are displacements in m (higher is better).
    curve has one entry per CMA-ES generation: the generation's best and median
    score and CMA-ES's sigma after it (evaluations = (index + 1) x popsize)."""

    weights: np.ndarray
    score: float
    curve: dict[str, list[float]] = field(default_factory=lambda: {"best": [], "median": [], "sigma": []})
    evals: int = 0
    stopped_early: bool = False
    wall_s: float = 0.0


GenerationHook = Callable[[int, int, np.ndarray, float], None]


def train_controller(
    body: dict, brain_kind: str, evals: int, seed: int, pool: ProcessPoolExecutor,
    protocol: Protocol = PROTOCOL,
    on_generation: Optional[GenerationHook] = None,
) -> Training:
    """Train one body's forward controller with CMA-ES for evals evaluations
    (rounded up to whole generations), evaluating candidates on pool.
    on_generation(gen, evals_so_far, best_weights, best_score) is called after
    every generation.

    CMA-ES stops early when its own criteria say so (e.g. every candidate
    scores exactly the same); the result is then flagged stopped_early."""
    t0 = time.perf_counter()
    model, data = build_world(body)
    n_params = make_brain_for_model(brain_kind, model, data, duration=protocol.episode_s).num_params
    x0 = [BRAIN_X0.get(brain_kind, 0.0)] * n_params
    # Its own generator, not cma's default np.random: trainings run concurrently.
    rng = np.random.default_rng(seed)
    es = cma.CMAEvolutionStrategy(
        x0, protocol.cma_sigma,
        {"popsize": protocol.cma_popsize, "randn": lambda *shape: rng.standard_normal(shape),
         "verbose": -9, "maxiter": 10**9},
    )

    out = Training(weights=np.array(x0, dtype=np.float32), score=-math.inf)
    for gen in range(cma_generations(evals, protocol)):
        if es.stop():
            out.stopped_early = True
            break
        solutions = es.ask()
        scores = list(pool.map(_evaluate_task, [(body, brain_kind, s.astype(np.float32), protocol) for s in solutions]))
        es.tell(solutions, [-x for x in scores])
        out.evals += len(solutions)
        i = int(np.argmax(scores))
        if scores[i] > out.score:
            out.score, out.weights = scores[i], np.array(solutions[i], dtype=np.float32)
        out.curve["best"].append(scores[i])
        out.curve["median"].append(float(np.median(scores)))
        out.curve["sigma"].append(float(es.sigma))
        if on_generation is not None:
            on_generation(gen, out.evals, out.weights, out.score)
    out.wall_s = time.perf_counter() - t0
    return out


def train_controllers(
    jobs: list[tuple[dict, str, int, int]], pool: ProcessPoolExecutor, concurrent: int,
    protocol: Protocol = PROTOCOL,
) -> list[Training]:
    """train_controller for every (body, brain_kind, evals, seed) job, up to
    concurrent at a time, in job order."""
    with ThreadPoolExecutor(max_workers=max(1, concurrent)) as threads:
        return list(threads.map(lambda job: train_controller(*job, pool=pool, protocol=protocol), jobs))
