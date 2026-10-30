"""
Simulation, skill training and body operators for the symmetry paper.

Pruned from the symmetry-pressure investigation's shared.py down to what the
paper runs: forward locomotion under the dsocial_zoo profile (zoo module
sizes and masses, zoo hinge or a chosen HingeActuator, 15 s episodes at 20 Hz
control, inner CMA-ES from a fixed x0 with sigma 0.5 and popsize 20), the
tree / tree_symmetric / tree_tied body operators, and the brains of
brains.py. What is left is unchanged, so runs reproduce the investigation's
runs exactly for the same seed.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cma
import mujoco
import numpy as np

from ariel.body_phenotypes.robogen_lite.config import ALLOWED_ROTATIONS, IDX_OF_CORE, ModuleType
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec import Individual
from ariel.ec.genotypes.tree.collision import prune_colliding_subtrees, prune_colliding_subtrees_symmetric
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    crossover_subtree,
    crossover_subtree_symmetric,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    random_tree_symmetric,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.symmetry import MirrorAxis, mirror_face, symmetrize_genome
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.tying import crossover_tied, mutate_morph_tied, random_tree_tied
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

from brains import BRAIN_X0, SETTLE_DURATION, make_brain_for_model

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_HINGES = 4
SPAWN_POSITION = (0.0, 0.0, 0.1)

# Thread-safe counter for assigning individual IDs (Individual.id is a DB
# primary key and stays None when not persisted to SQLite).
_ind_id_lock = threading.Lock()
_ind_id_counter = 0


def _next_ind_id() -> int:
    global _ind_id_counter  # noqa: PLW0603
    with _ind_id_lock:
        _ind_id_counter += 1
        return _ind_id_counter


# ── Body construction ─────────────────────────────────────────────────────────


def genome_to_spec(genome_dict: dict) -> Optional[mujoco.MjSpec]:
    try:
        graph = TreeGenome.from_dict(genome_dict).to_networkx()
        if graph.number_of_nodes() == 0:
            return None
        return construct_mjspec_from_graph(graph).spec
    except Exception:
        return None


# kgd's zoo modules (zoo benchmark zoo_modules.py). Only the values differ:
# the local module classes, and so their face layout, are kept, and every
# site position scales with the dimensions.
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


@dataclass(frozen=True)
class HingeActuator:
    """Hinge servo applied to zoo bodies (apply_hinge_actuator): a position
    actuator with gain kp and damping kv, the joint's armature, and whether
    the joint has a hard +-90 deg limit (the ctrl range is +-90 deg either way)."""

    kp: float
    kv: float
    armature: float
    limited: bool

    @property
    def label(self) -> str:
        return f"kp{self.kp:g}_kv{self.kv:g}_arm{self.armature:g}_lim{int(self.limited)}"


ZOO_HINGE = HingeActuator(ZOO_HINGE_KP, ZOO_HINGE_KV, ZOO_HINGE_ARMATURE, True)

_ZOO_LOCK = threading.RLock()


def _zoo_module_overrides() -> list[tuple[Any, str, Any]]:
    """(module, attribute, zoo value) for the module-level constants the local
    module classes read when they are instantiated."""
    from ariel.body_phenotypes.robogen_lite.modules import brick, core, hinge  # noqa: PLC0415

    brick_config = brick.ariel_modules_config.model_copy(
        update={"BRICK_MASS": ZOO_BRICK_MASS, "BRICK_DIMENSIONS": ZOO_BRICK_DIMENSIONS},
    )
    return [
        (core, "CORE_MASS", ZOO_CORE_MASS),
        (core, "CORE_DIMENSIONS", ZOO_CORE_DIMENSIONS),
        (hinge, "STATOR_MASS", ZOO_STATOR_MASS),
        (hinge, "ROTOR_MASS", ZOO_ROTOR_MASS),
        (hinge, "STATOR_DIMENSIONS", ZOO_STATOR_DIMENSIONS),
        (hinge, "ROTOR_DIMENSIONS", ZOO_ROTOR_DIMENSIONS),
        (brick, "ariel_modules_config", brick_config),
    ]


def _set_module_constants(values: list[tuple[Any, str, Any]]) -> None:
    from ariel.body_phenotypes.robogen_lite.collision_utils import module_template  # noqa: PLC0415

    for mod, name, value in values:
        setattr(mod, name, value)
    module_template.cache_clear()  # collision geometry is compiled from the module classes and cached


@contextlib.contextmanager
def zoo_module_constants():
    """Build modules with the zoo sizes and masses inside this block; the
    previous values are restored on exit."""
    with _ZOO_LOCK:
        overrides = _zoo_module_overrides()
        saved = [(mod, name, getattr(mod, name)) for mod, name, _ in overrides]
        _set_module_constants(overrides)
        try:
            yield
        finally:
            _set_module_constants(saved)


def use_zoo_modules_globally() -> None:
    """Zoo sizes and masses for the rest of this process (and forked workers),
    so the genome operators' collision checks use the zoo geometry too."""
    _set_module_constants(_zoo_module_overrides())


def apply_hinge_actuator(spec: mujoco.MjSpec, actuator: HingeActuator = ZOO_HINGE) -> mujoco.MjSpec:
    """Give a robot spec a hinge actuator (kp, kv, armature, joint limit), in
    place. Hard-coded in hinge.py, so set after construction."""
    for a in spec.actuators:
        a.gainprm[0] = actuator.kp
        a.biasprm[1] = -actuator.kp
        a.biasprm[2] = -actuator.kv
    scale = 180.0 / math.pi if spec.compiler.degree else 1.0
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.armature = actuator.armature
            if actuator.limited:
                j.range = [r * scale for r in ZOO_HINGE_RANGE]
                j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            else:
                j.limited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec


def zoo_spec(genome_dict: dict, actuator: HingeActuator = ZOO_HINGE) -> Optional[mujoco.MjSpec]:
    """The body of a tree genome, built with zoo modules and this hinge actuator."""
    with zoo_module_constants():
        spec = genome_to_spec(genome_dict)
    return apply_hinge_actuator(spec, actuator) if spec is not None else None


def zoo_spec_fn(actuator: HingeActuator = ZOO_HINGE) -> Callable[[dict], Optional[mujoco.MjSpec]]:
    """A picklable genome -> spec function for this hinge actuator."""
    return functools.partial(zoo_spec, actuator=actuator)


def build_world(genome_dict: dict, to_spec_fn=zoo_spec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """The body on a flat floor."""
    spec = to_spec_fn(genome_dict)
    if spec is None:
        raise ValueError("Could not decode morphology")
    world = SimpleFlatWorld()
    try:
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=False)
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    return model, data


def genome_hash(genome_dict: dict) -> str:
    return hashlib.sha1(json.dumps(genome_dict, sort_keys=True).encode()).hexdigest()


def robot_bodies(model: mujoco.MjModel) -> list[int]:
    """Robot bodies that own at least one geom (core, stators, rotors, bricks)."""
    has_geom = set(int(b) for b in model.geom_bodyid)
    return [
        b for b in range(model.nbody)
        if model.body(b).name.startswith("robot1_") and b in has_geom
    ]


# ── Morphological metrics ─────────────────────────────────────────────────────


def bilateral_symmetry_score(genome_dict: dict, axis: str = "y_zero") -> float:
    """Bilateral symmetry of a TreeGenome about ``axis`` ("y_zero" or "x_equals_y").

    Modules are identified by their path of faces from the root. Two modules
    are mirror partners if one's root-path is the other's with faces mirrored
    per ``axis`` (the rule symmetrize_genome enforces). Returns the fraction
    of off-midline modules that have a same-type mirror partner, in [0, 1];
    0.0 on failure.
    """
    try:
        mirror_axis = MirrorAxis.Y_ZERO if axis == "y_zero" else MirrorAxis.X_EQUALS_Y

        graph = TreeGenome.from_dict(genome_dict).to_networkx()
        if graph.number_of_nodes() == 0:
            return 0.0

        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        if not roots:
            return 0.0
        root = roots[0]

        # BFS to assign each node its path of faces from the root.
        paths: dict[Any, tuple[str, ...]] = {root: ()}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in graph.successors(node):
                if child in paths:
                    continue
                face = (graph.get_edge_data(node, child) or {}).get("face", "FRONT")
                paths[child] = (*paths[node], face)
                queue.append(child)

        def mirror_path(path: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(mirror_face(face, mirror_axis, is_outer=(i == 0)) for i, face in enumerate(path))

        by_path: dict[tuple[str, ...], str] = {p: graph.nodes[n].get("type", "UNKNOWN") for n, p in paths.items()}

        num_symmetrical = 0
        num_off_midline = 0
        seen: set[tuple[str, ...]] = set()
        for path, t in by_path.items():
            mpath = mirror_path(path)
            if path == mpath or path in seen:
                continue  # midline module, or already counted with its partner
            seen.add(path)
            seen.add(mpath)
            num_off_midline += 1
            if by_path.get(mpath) == t:
                num_symmetrical += 1

        return num_symmetrical / num_off_midline if num_off_midline > 0 else 0.0
    except Exception:
        return 0.0


def joint_count(genome: TreeGenome) -> int:
    """Count hinge joints by walking the genome graph — no MuJoCo compilation needed."""
    return sum(1 for d in genome.nodes.values() if d.get("type") == "HINGE")


# ── Body mutation + generation ────────────────────────────────────────────────


def mutate_morph(genome: TreeGenome, rng: np.random.Generator, num_modules: int) -> TreeGenome:
    new = copy.deepcopy(genome)
    mutation_type = rng.choice(["point", "subtree", "shrink", "hoist"], p=[0.45, 0.35, 0.1, 0.1])

    if mutation_type == "point":
        mutate_replace_node(new)
    elif mutation_type == "subtree":
        mutate_subtree_replacement(new, max_modules=num_modules)
    elif mutation_type == "shrink":
        mutate_shrink(new)
    else:
        mutate_hoist(new)

    if rng.random() < 0.25:
        noncore = [nid for nid in new.nodes if nid != IDX_OF_CORE]
        if noncore:
            nid = random.choice(noncore)
            mtype = ModuleType[new.nodes[nid]["type"]]
            rots = [r.name for r in ALLOWED_ROTATIONS[mtype]]
            if rots:
                new.nodes[nid]["rotation"] = random.choice(rots)

    _prune_invalid_edges(new)
    # Point/rotation changes and grafted subtrees can make parts overlap;
    # drop any subtree that now collides (FCL, real module geometry).
    prune_colliding_subtrees(new)
    with contextlib.suppress(ValueError):
        validate_genome_dict(new.to_dict())
    return new


def mutate_morph_symmetric(
    genome: TreeGenome, rng: np.random.Generator, num_modules: int, axis: MirrorAxis
) -> TreeGenome:
    """mutate_morph, then re-symmetrize about ``axis``. Mirroring can create new
    overlaps (an arm hitting its own mirror image), so colliding subtrees are
    pruned again, pairwise with their mirror images."""
    mutated = mutate_morph(genome, rng, num_modules)
    symmetric = symmetrize_genome(mutated, axis)
    prune_colliding_subtrees_symmetric(symmetric, axis)
    return symmetric


def _morph_ops(
    num_modules: int, symmetry_axis: Optional[MirrorAxis], tie_prob: Optional[float]
) -> tuple[Callable[[], dict], Callable[[dict, dict], tuple[dict, dict]], Callable[[dict, np.random.Generator], dict]]:
    """(random genome, crossover, mutate) on genome dicts for the body genome in use:
    tree, tree_symmetric (symmetry_axis) or tree_tied (tie_prob, ariel...tree.tying)."""
    if tie_prob is not None:
        return (lambda: random_tree_tied(num_modules, tie_prob).to_dict(),
                crossover_tied,
                lambda d, rng: mutate_morph_tied(d, rng, num_modules, tie_prob))

    def random_genome() -> dict:
        if symmetry_axis is not None:
            return random_tree_symmetric(num_modules, symmetry_axis).to_dict()
        return random_tree(num_modules).to_dict()

    def crossover(a: dict, b: dict) -> tuple[dict, dict]:
        t1, t2 = TreeGenome.from_dict(a), TreeGenome.from_dict(b)
        if symmetry_axis is not None:
            c1, c2 = crossover_subtree_symmetric(t1, t2, symmetry_axis)
        else:
            c1, c2 = crossover_subtree(t1, t2)
        return c1.to_dict(), c2.to_dict()

    def mutate(d: dict, rng: np.random.Generator) -> dict:
        t = TreeGenome.from_dict(d)
        if symmetry_axis is not None:
            return mutate_morph_symmetric(t, rng, num_modules, symmetry_axis).to_dict()
        return mutate_morph(t, rng, num_modules).to_dict()

    return random_genome, crossover, mutate


def _morph_ok(genome_dict: dict, max_depth: int) -> bool:
    genome = TreeGenome.from_dict(genome_dict)
    return joint_count(genome) >= MIN_HINGES and validate_tree_depth(genome, max_depth)


def create_individual(
    num_modules: int, max_depth: int, symmetry_axis: Optional[MirrorAxis] = None,
    tie_prob: Optional[float] = None,
) -> Individual:
    random_genome, _, _ = _morph_ops(num_modules, symmetry_axis, tie_prob)
    while True:
        genome = random_genome()
        if _morph_ok(genome, max_depth):
            break
    ind = Individual()
    ind.id = _next_ind_id()
    ind.genotype = {"morph": genome}
    ind.tags = {"ps": False, "valid": True, "best_brain": []}
    return ind


def make_offspring(
    parents: list[Individual],
    lam: int,
    rng: np.random.Generator,
    num_modules: int,
    max_depth: int,
    symmetry_axis: Optional[MirrorAxis] = None,
    tie_prob: Optional[float] = None,
) -> list[Individual]:
    """Generate lam offspring via subtree crossover + mutation."""
    _, crossover, mutate = _morph_ops(num_modules, symmetry_axis, tie_prob)
    offspring: list[Individual] = []
    while len(offspring) < lam:
        use_sexual = len(parents) >= 2 and rng.random() < 0.6
        if use_sexual:
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1.genotype["morph"], p2.genotype["morph"])
            child_morph = c1 if rng.random() < 0.5 else c2
            parent_ids = [p1.id, p2.id]
        else:
            p = random.choice(parents)
            child_morph = p.genotype["morph"]
            parent_ids = [p.id]

        child_morph = mutate(child_morph, rng)

        attempts = 0
        valid = False
        while attempts < 50:
            if _morph_ok(child_morph, max_depth):
                valid = True
                break
            child_morph = mutate(child_morph, rng)
            attempts += 1

        child = Individual()
        child.id = _next_ind_id()
        child.genotype = {"morph": child_morph, "parent_ids": parent_ids}
        child.tags = {"ps": False, "valid": valid, "best_brain": []}
        child.requires_eval = True
        if not valid:
            child.fitness = float("inf")
            child.requires_eval = False
        offspring.append(child)
    return offspring


# ── Evaluation profile (dsocial_zoo) ──────────────────────────────────────────
#
# d_social_learning's sim_ariel/evaluator.py "zoo" preset: 15 s translate
# episodes at 20 Hz control (every 25 physics steps at 500 Hz), the legacy
# 9-step alpha and jerk constants rescaled to the 25-step control period, a
# spawn-height penalty above 0.5 m, and inner CMA-ES from a fixed x0 with
# sigma 0.5, popsize 20, 10 generations by default. The skill is normalised
# (translate_score, [0, 2]) before the penalties are subtracted.

_LEGACY_CTRL_ALPHA     = 0.5
_LEGACY_JERK_WEIGHT    = 3.0   # penalty per unit of mean absolute ctrl delta, above the threshold
_LEGACY_JERK_THRESHOLD = 0.15  # hurdle: mean_jerk below this is free
_STEP_RATIO            = 25 / 9  # zoo control period / legacy control period


@dataclass(frozen=True)
class EvalProfile:
    loco_duration: float = 15.0
    control_step_freq: int = 25          # physics steps between control updates
    skill_budget: int = 10               # inner CMA-ES generations
    ctrl_alpha: float = 1.0 - (1.0 - _LEGACY_CTRL_ALPHA) ** _STEP_RATIO
    jerk_weight: float = _LEGACY_JERK_WEIGHT / _STEP_RATIO
    jerk_threshold: float = _LEGACY_JERK_THRESHOLD * _STEP_RATIO
    height_threshold: float = 0.5        # core height at spawn above this is penalised
    cma_sigma: float = 0.5
    cma_popsize: int = 20


PROFILE = EvalProfile()

TRANSLATE_REF_SPEED = 0.5        # m/s; a normalised translate skill scores TRANSLATE_MAX at this mean speed
TRANSLATE_MAX       = 2.0        # normalised translate score cap

# Every body spawns with no rotation, so the core camera's forward direction
# (world -Y) is the same for every body: the forward skill's reward axis.
FORWARD_AXIS = np.array([0.0, -1.0])


def translate_score(disp: float, duration: float) -> float:
    """Normalised translate skill: 2 x displacement along the reward axis over
    TRANSLATE_REF_SPEED * duration, clipped to [0, TRANSLATE_MAX]."""
    return float(np.clip(TRANSLATE_MAX * disp / (TRANSLATE_REF_SPEED * duration), 0.0, TRANSLATE_MAX))


# ── Episode ───────────────────────────────────────────────────────────────────


def run_skill_episode(
    model: mujoco.MjModel, data: mujoco.MjData, brain: Any,
    profile: EvalProfile = PROFILE,
    recorder: Optional[Callable[[mujoco.MjData], None]] = None,
    on_settle_step: Optional[Callable[[mujoco.MjData], None]] = None,
) -> tuple[float, dict[str, Any]]:
    """(fitness, info) of one forward episode of a brain whose params are set.

    Fitness is minimised: -(normalised score - height penalty - jerk penalty).
    info holds raw_score (displacement along FORWARD_AXIS, m), score, and the
    two penalties. recorder(data) is called after every post-settle physics
    step and on_settle_step(data) after every settle step (replay.py's
    recording); neither may change data.
    """
    brain.reset()
    mujoco.mj_resetData(model, data)
    initial_height = float(data.qpos[2])  # core height at spawn

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)
        if on_settle_step is not None:
            on_settle_step(data)

    step           = 0
    current_action = np.zeros(model.nu)
    prev_ctrl: Optional[np.ndarray] = None
    jerk_sum       = 0.0
    ctrl_step      = 0
    xy0            = np.array([data.qpos[0], data.qpos[1]])

    episode_end = SETTLE_DURATION + profile.loco_duration
    while data.time < episode_end:
        if step % profile.control_step_freq == 0:
            state      = get_robot_state(data).astype(np.float32)
            raw_action = brain.act(data, state)
            current_action = np.clip(
                current_action * (1.0 - profile.ctrl_alpha) + raw_action * profile.ctrl_alpha,
                -math.pi / 2, math.pi / 2,
            )
            if prev_ctrl is not None:
                jerk_sum += float(np.mean(np.abs(current_action - prev_ctrl)))
            prev_ctrl = current_action.copy()
            ctrl_step += 1
        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        if recorder is not None:
            recorder(data)
        step += 1

    mean_jerk    = jerk_sum / max(ctrl_step - 1, 1)
    jerk_penalty = profile.jerk_weight * mean_jerk if mean_jerk >= profile.jerk_threshold else 0.0

    height_penalty = initial_height if initial_height > profile.height_threshold else 0.0
    raw = float(np.dot(np.array([data.qpos[0], data.qpos[1]]) - xy0, FORWARD_AXIS))
    score = translate_score(raw, profile.loco_duration)
    info = {"raw_score": raw, "height_penalty": height_penalty, "jerk_penalty": jerk_penalty, "score": score}
    return -(score - height_penalty - jerk_penalty), info


# ── Inner CMA-ES (one body, candidates evaluated in parallel) ─────────────────

_skill_worker_ctx: Optional[dict[str, Any]] = None


def _skill_worker_init(seed: int, genome_dict: dict, to_spec_fn, brain_kind: str, profile: EvalProfile) -> None:
    global _skill_worker_ctx  # noqa: PLW0603
    np.random.seed((seed + os.getpid()) % (2**32 - 1))
    model, data = build_world(genome_dict, to_spec_fn)
    brain       = make_brain_for_model(brain_kind, model, data, duration=profile.loco_duration)
    _skill_worker_ctx = {"model": model, "data": data, "brain": brain, "profile": profile}


def _skill_worker_eval(weights_list: list[float]) -> tuple[float, dict[str, Any]]:
    """run_skill_episode for one candidate in a skill-training worker."""
    assert _skill_worker_ctx is not None
    ctx = _skill_worker_ctx
    ctx["brain"].set_params(np.array(weights_list, dtype=np.float32))
    return run_skill_episode(ctx["model"], ctx["data"], ctx["brain"], ctx["profile"])


GenerationHook = Callable[[int, int, cma.CMAEvolutionStrategy, list[float], np.ndarray, float], None]


def train_skill_for_body(
    genome_dict: dict,
    budget: int,
    workers: int,
    seed: int,
    to_spec_fn=zoo_spec,
    brain_kind: str = "bang_bang",
    profile: EvalProfile = PROFILE,
    on_generation: Optional[GenerationHook] = None,
) -> tuple[np.ndarray, list[float], float, dict[str, Any]]:
    """Train the forward skill of one body with CMA-ES for budget generations.

    on_generation(gen, n_evals, es, gen_fits, best_w, best_fit), if given, is
    called after every generation (train_fixed_body.py's per-generation curve
    and checkpoints); it must not change its arguments.

    Returns (best_weights, learning_curve, eval_time_s, best_info): the
    learning curve holds the best-so-far fitness at each improvement, and
    best_info is run_skill_episode's info for best_weights.
    """
    t_start = time.perf_counter()

    model, data = build_world(genome_dict, to_spec_fn)
    num_params  = make_brain_for_model(brain_kind, model, data, duration=profile.loco_duration).num_params
    x0 = [BRAIN_X0.get(brain_kind, 0.0)] * num_params

    es = cma.CMAEvolutionStrategy(
        x0, profile.cma_sigma,
        {"popsize": profile.cma_popsize, "seed": seed, "verbose": -9, "maxiter": 10**9},
    )

    best_fit = float("inf")
    best_w   = np.array(x0, dtype=np.float32)
    best_info: dict[str, Any] = {}
    learning_curve: list[float] = []
    n_evals = 0

    effective_workers = min(profile.cma_popsize, workers)
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=_skill_worker_init,
        initargs=(seed, genome_dict, to_spec_fn, brain_kind, profile),
    ) as pool:
        gen = 0
        while gen < budget:
            if es.stop():
                break

            solutions = es.ask()
            results   = list(pool.map(_skill_worker_eval, [s.tolist() for s in solutions]))
            fits      = [f for f, _ in results]
            es.tell(solutions, fits)
            n_evals  += len(solutions)

            gen_best = min(fits)
            if gen_best < best_fit:
                best_fit  = gen_best
                i_best    = fits.index(gen_best)
                best_w    = np.array(solutions[i_best], dtype=np.float32)
                best_info = results[i_best][1]
                learning_curve.append(best_fit)

            if on_generation is not None:
                on_generation(gen, n_evals, es, fits, best_w, best_fit)
            gen += 1

    return best_w, learning_curve, time.perf_counter() - t_start, best_info
