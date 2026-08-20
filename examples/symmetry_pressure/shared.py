"""
Shared helpers for the symmetry-pressure investigation.

Adapted from examples/re_book/6_body_brain_randomized_waypoints.py with
additions for extended data logging, NSGA-II selection, and repeat evaluations.
"""

import contextlib
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, cast

import cma
import cv2
import mujoco
import nevergrad as ng
import networkx as nx
import numpy as np
import torch
from torch import nn

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec import Individual, Population
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
from ariel.ec.genotypes.tree.symmetry import MirrorAxis, symmetrize_genome
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

# ── Constants ─────────────────────────────────────────────────────────────────

# HPO-tuned defaults (see HPO_FINDINGS.md): 1 hidden layer, 32 units, σ=0.35,
# pop=20, budget=900, init_scale=1.3 — Pareto-optimal for ≤100 s/run on both tasks.
HIDDEN_SIZE = 32
MIN_HINGES = 4
RING_R_MIN = 0.5
RING_R_MAX = 1.0
GATE_HALF_HEIGHT = 0.15
SPAWN_POSITION = (0.0, 0.0, 0.1)

# Thread-safe counter for assigning individual IDs (Individual.id is a DB
# primary key and stays None when not persisted to SQLite).
_ind_id_lock = threading.Lock()
_ind_id_counter = 0


def _next_ind_id() -> int:
    global _ind_id_counter
    with _ind_id_lock:
        _ind_id_counter += 1
        return _ind_id_counter


# ── Neural network ────────────────────────────────────────────────────────────


class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.fc1    = nn.Linear(input_size, hidden_size)
        self.fc2    = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.hidden_act = nn.ELU()
        self.out_act    = nn.Tanh()
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, model: Any, data: Any, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        x = torch.as_tensor(state, dtype=torch.float32)
        x = self.hidden_act(self.fc1(x))
        x = self.hidden_act(self.fc2(x))
        return (self.out_act(self.fc_out(x)) * (torch.pi / 2)).numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: np.ndarray) -> None:
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n])
        address += n
    if address != len(vector):
        raise IndexError("Parameter vector length mismatch")


# ── Vision helpers ────────────────────────────────────────────────────────────


def isolate_green(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))


def analyze_sections(mask: np.ndarray) -> list[float]:
    sections = np.array_split(mask, 3, axis=1)
    return [cv2.countNonZero(s) / max(s.size, 1) for s in sections]


# ── Body construction ─────────────────────────────────────────────────────────


def genome_to_spec(genome_dict: dict) -> Optional[mujoco.MjSpec]:
    try:
        graph = TreeGenome.from_dict(genome_dict).to_networkx()
        if graph.number_of_nodes() == 0:
            return None
        return construct_mjspec_from_graph(graph).spec
    except Exception:
        return None


def build_world_for_body(
    genome_dict: dict,
    reach_radius: float,
    arena_radius: float,
    add_food_target: bool = False,
) -> tuple[mujoco.MjModel, mujoco.MjData, Optional[int], Optional[str]]:
    """Build SimpleFlatWorld + TreeGenome body.

    When add_food_target=True, adds a green mocap marker (food/waypoint goal).
    Returns (model, data, target_mocap_id, cam_name).
    target_mocap_id is None when add_food_target=False.
    """
    spec = genome_to_spec(genome_dict)
    if spec is None:
        raise ValueError("Could not decode morphology")

    world = SimpleFlatWorld()
    try:
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=False)

    target_mocap_id: Optional[int] = None
    if add_food_target:
        marker = world.spec.worldbody.add_body(
            name="green_target", mocap=True, pos=[0.0, 0.0, GATE_HALF_HEIGHT]
        )
        marker.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[reach_radius, GATE_HALF_HEIGHT],
            rgba=[0, 1, 0, 0.7],
            contype=0,
            conaffinity=0,
        )

    world.spec.worldbody.add_camera(
        name="overview_cam",
        pos=[0.0, 0.0, arena_radius * 3.5],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )

    model = world.spec.compile()
    data  = mujoco.MjData(model)

    if add_food_target:
        target_mocap_id = model.body("green_target").mocapid[0]

    cam_name: Optional[str] = None
    if add_food_target:
        for i in range(model.ncam):
            name = model.camera(i).name
            if ("camera" in name or "core" in name) and "overview" not in name:
                cam_name = name
                break

    return model, data, target_mocap_id, cam_name


def genome_input_dim(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    use_vision: bool,
    use_phase: bool = False,
) -> int:
    base = len(get_robot_state(data))
    if use_vision:
        n_phase = 2 if use_phase else 0
        return base + 3 + n_phase + 1  # 3 vision bins + optional 2 phase + 1 progress
    if use_phase:
        return base + 2  # sin + cos
    return base


def genome_hash(genome_dict: dict) -> str:
    return hashlib.sha1(json.dumps(genome_dict, sort_keys=True).encode()).hexdigest()


# ── Morphological metrics ─────────────────────────────────────────────────────


def bilateral_symmetry_score(genome_dict: dict, axis: str = "y_zero") -> float:
    """Bilateral symmetry of a TreeGenome about ``axis`` ("y_zero" or "x_equals_y").

    Identifies each module by its path of faces from the root (rather than a
    collapsed integer (x, y, z) grid position, which can alias distinct
    branches onto the same cell on deep/branchy trees and produce false
    mismatches). Two modules are mirror partners if one's root-path is the
    other's root-path with faces mirrored per ``axis`` — using the same rule
    `ariel.ec.genotypes.tree.symmetry.symmetrize_genome` enforces: for
    "y_zero", RIGHT<->LEFT at every hop; for "x_equals_y", FRONT<->RIGHT and
    BACK<->LEFT at the first hop off the root only, RIGHT<->LEFT at every hop
    after that. Returns the fraction of off-midline modules that have a
    same-type mirror partner, in [0, 1]. Returns 0.0 on failure.
    """
    try:
        from ariel.ec.genotypes.tree.symmetry import MirrorAxis, mirror_face

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
            mirrored = []
            for i, face in enumerate(path):
                mirrored.append(mirror_face(face, mirror_axis, is_outer=(i == 0)))
            return tuple(mirrored)

        def is_off_midline(path: tuple[str, ...]) -> bool:
            return path != mirror_path(path)

        def node_type(n: Any) -> str:
            return graph.nodes[n].get("type", "UNKNOWN")

        # Lookup: path -> type (paths are unique per node by construction).
        by_path: dict[tuple[str, ...], str] = {p: node_type(n) for n, p in paths.items()}

        num_symmetrical = 0
        num_off_midline = 0
        seen: set[tuple[str, ...]] = set()
        for path, t in by_path.items():
            if not is_off_midline(path):
                continue  # midline module, skip
            if path in seen:
                continue
            mpath = mirror_path(path)
            seen.add(path)
            seen.add(mpath)
            num_off_midline += 1
            if by_path.get(mpath) == t:
                num_symmetrical += 1

        return num_symmetrical / num_off_midline if num_off_midline > 0 else 0.0
    except Exception:
        return 0.0


def action_control_cost(ctrl_history: list[np.ndarray]) -> float:
    """Mean squared control effort over an episode."""
    if not ctrl_history:
        return 0.0
    arr = np.stack(ctrl_history)
    return float(np.mean(arr ** 2))


def signed_vertical_yaw_delta(r_prev: np.ndarray, r_curr: np.ndarray) -> float:
    """Signed rotation-about-world-Z component of the incremental rotation
    r_prev -> r_curr (each a flattened row-major 3x3, e.g. mujoco's xmat).

    Convention-free: does not assume which local body axis is "up" or
    "forward", unlike atan2(xmat[3], xmat[0]) (which is only a valid yaw
    reading if the body's core happens to stay aligned with the "standard"
    spawn orientation). A body whose core is rotated relative to that
    orientation, or that tumbles, can make atan2(xmat[3], xmat[0]) swing
    through large fake "yaw" deltas from roll/pitch alone -- evolved bodies
    have been found exploiting exactly this to rack up huge fake turn-skill
    fitness while barely moving. This measures the true geodesic rotation
    between frames and keeps only its vertical-axis component, so tumbling
    or rotation about a horizontal axis contributes ~0.
    """
    r_prev_mat = np.asarray(r_prev, dtype=np.float64).reshape(3, 3)
    r_curr_mat = np.asarray(r_curr, dtype=np.float64).reshape(3, 3)
    r_rel = r_curr_mat @ r_prev_mat.T
    cos_theta = float(np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos_theta)
    if angle < 1e-9:
        return 0.0
    # Rotation axis from the antisymmetric part: (R - R^T) = 2 sin(theta) [axis]_x
    sin_theta = math.sin(angle)
    if abs(sin_theta) < 1e-9:
        return 0.0
    axis_z = (r_rel[1, 0] - r_rel[0, 1]) / (2.0 * sin_theta)
    return angle * axis_z


# ── Waypoint sampling ─────────────────────────────────────────────────────────


def sample_waypoints(
    rng: np.random.Generator,
    n: int,
    r_min: float = RING_R_MIN,
    r_max: float = RING_R_MAX,
) -> list[np.ndarray]:
    r2_min, r2_max = r_min * r_min, r_max * r_max
    wps: list[np.ndarray] = []
    prev_xy = np.array([0.0, 0.0])
    for _ in range(n):
        r = float(np.sqrt(rng.uniform(r2_min, r2_max)))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        offset = np.array([r * np.cos(theta), r * np.sin(theta)])
        new_xy = prev_xy + offset
        wps.append(np.array([new_xy[0], new_xy[1], GATE_HALF_HEIGHT]))
        prev_xy = new_xy
    return wps


# ── Worker context ─────────────────────────────────────────────────────────────

_RENDER_INIT_LOCK = threading.Lock()
_process_local_ctx: Optional[dict[str, Any]] = None


def ensure_ctx_for_body(
    body_hash: str,
    genome_dict: dict,
    reach_radius: float,
    arena_radius: float,
    use_vision: bool,
    use_phase: bool = False,
) -> dict[str, Any]:
    global _process_local_ctx  # noqa: PLW0603
    if _process_local_ctx is not None and _process_local_ctx.get("body_hash") == body_hash:
        return cast(dict[str, Any], _process_local_ctx)

    if _process_local_ctx is not None:
        with contextlib.suppress(Exception):
            _process_local_ctx["renderer"].close()
        _process_local_ctx = None

    model, data, target_mocap_id, cam_name = build_world_for_body(
        genome_dict, reach_radius, arena_radius, add_food_target=use_vision
    )
    input_dim = genome_input_dim(model, data, use_vision, use_phase=use_phase)
    network = Network(input_size=input_dim, output_size=model.nu)

    renderer: Optional[mujoco.Renderer] = None
    if use_vision:
        with _RENDER_INIT_LOCK:
            renderer = mujoco.Renderer(model, height=48 * 2, width=64 * 2)

    _process_local_ctx = {
        "body_hash":       body_hash,
        "model":           model,
        "data":            data,
        "network":         network,
        "renderer":        renderer,
        "cam_name":        cam_name,
        "target_mocap_id": target_mocap_id,
        "input_dim":       input_dim,
        "num_params":      sum(p.numel() for p in network.parameters()),
        "use_vision":      use_vision,
    }
    return cast(dict[str, Any], _process_local_ctx)


def init_worker(base_seed: int) -> None:
    torch.set_num_threads(1)
    seed = (base_seed + os.getpid()) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── NSGA-II utilities ─────────────────────────────────────────────────────────


def pareto_rank(objectives: list[tuple[float, ...]]) -> list[int]:
    """Fast non-dominated sort. Returns rank per individual (0 = Pareto front)."""
    n = len(objectives)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count = [0] * n
    ranks = [0] * n
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i dominates j if i is no worse on all objectives and strictly better on one
            # Objectives here are (minimise task_fitness, minimise second_objective)
            i_dom_j = all(objectives[i][k] <= objectives[j][k] for k in range(len(objectives[i])))
            i_strict = any(objectives[i][k] < objectives[j][k] for k in range(len(objectives[i])))
            if i_dom_j and i_strict:
                dominated_by[i].append(j)
                domination_count[j] += 1

    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    ranks[j] = current + 1
                    next_front.append(j)
        fronts.append(next_front)
        current += 1

    return ranks


def crowding_distance(front_indices: list[int], objectives: list[tuple[float, ...]]) -> list[float]:
    """Crowding distance for individuals in one front."""
    n = len(front_indices)
    if n <= 2:
        return [float("inf")] * n

    num_obj = len(objectives[0])
    distances = [0.0] * n

    for k in range(num_obj):
        vals = [(objectives[front_indices[i]][k], i) for i in range(n)]
        vals.sort()
        distances[vals[0][1]]  = float("inf")
        distances[vals[-1][1]] = float("inf")
        obj_range = vals[-1][0] - vals[0][0]
        if obj_range == 0:
            continue
        for pos in range(1, n - 1):
            distances[vals[pos][1]] += (vals[pos + 1][0] - vals[pos - 1][0]) / obj_range

    return distances


def nsga2_survivor_selection(
    individuals: list[Individual],
    n_survivors: int,
    objectives: list[tuple[float, ...]],
) -> list[Individual]:
    """Select n_survivors from individuals using NSGA-II crowded-comparison.

    objectives[i] corresponds to individuals[i]. Both are minimisation objectives.
    Returns sorted list (best Pareto rank + highest crowding distance first).
    """
    n = len(individuals)
    ranks = pareto_rank(objectives)

    # Group by rank
    rank_to_indices: dict[int, list[int]] = {}
    for i, r in enumerate(ranks):
        rank_to_indices.setdefault(r, []).append(i)

    survivors: list[Individual] = []
    for rank in sorted(rank_to_indices.keys()):
        front = rank_to_indices[rank]
        if len(survivors) + len(front) <= n_survivors:
            survivors.extend(individuals[i] for i in front)
        else:
            needed = n_survivors - len(survivors)
            cd = crowding_distance(front, objectives)
            # Sort by descending crowding distance
            sorted_front = sorted(zip(cd, front), key=lambda x: -x[0])
            survivors.extend(individuals[sorted_front[i][1]] for i in range(needed))
            break

    return survivors


# ── Body mutation + generation ────────────────────────────────────────────────


def symmetry_axis_from_cli(value: Optional[str]) -> Optional[MirrorAxis]:
    """Map a ``--symmetry-axis`` CLI value ("none"/"y_zero"/"x_equals_y"/None) to a MirrorAxis."""
    if value is None or value == "none":
        return None
    return MirrorAxis(value)


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
    with contextlib.suppress(ValueError):
        validate_genome_dict(new.to_dict())
    return new


def mutate_morph_symmetric(
    genome: TreeGenome, rng: np.random.Generator, num_modules: int, axis: MirrorAxis
) -> TreeGenome:
    """Like `mutate_morph`, but guarantees the result stays symmetric about ``axis``.

    Applies the ordinary (potentially symmetry-breaking) mutation, then
    re-symmetrizes — reusing all four existing mutation operators unchanged.
    Assumes ``genome`` is already symmetric about ``axis``.
    """
    mutated = mutate_morph(genome, rng, num_modules)
    return symmetrize_genome(mutated, axis)


def joint_count(genome: TreeGenome) -> int:
    """Count hinge joints by walking the genome graph — no MuJoCo compilation needed."""
    return sum(1 for d in genome.nodes.values() if d.get("type") == "HINGE")


def create_individual(
    num_modules: int, max_depth: int, symmetry_axis: Optional[MirrorAxis] = None
) -> Individual:
    while True:
        genome = (
            random_tree_symmetric(num_modules, symmetry_axis)
            if symmetry_axis is not None
            else random_tree(num_modules)
        )
        if joint_count(genome) >= MIN_HINGES and validate_tree_depth(genome, max_depth):
            break
    ind = Individual()
    ind.id = _next_ind_id()
    ind.genotype = {"morph": genome.to_dict()}
    ind.tags = {"ps": False, "valid": True, "best_brain": []}
    return ind


def make_offspring(
    parents: list[Individual],
    lam: int,
    rng: np.random.Generator,
    num_modules: int,
    max_depth: int,
    symmetry_axis: Optional[MirrorAxis] = None,
) -> list[Individual]:
    """Generate lam offspring via subtree crossover + mutation."""
    offspring: list[Individual] = []
    while len(offspring) < lam:
        use_sexual = len(parents) >= 2 and rng.random() < 0.6
        if use_sexual:
            p1, p2 = random.sample(parents, 2)
            t1 = TreeGenome.from_dict(p1.genotype["morph"])
            t2 = TreeGenome.from_dict(p2.genotype["morph"])
            if symmetry_axis is not None:
                c1, c2 = crossover_subtree_symmetric(t1, t2, symmetry_axis)
            else:
                c1, c2 = crossover_subtree(t1, t2)
            child_morph = c1 if rng.random() < 0.5 else c2
            parent_ids = [p1.id, p2.id]
        else:
            p = random.choice(parents)
            child_morph = TreeGenome.from_dict(p.genotype["morph"])
            parent_ids = [p.id]

        if symmetry_axis is not None:
            child_morph = mutate_morph_symmetric(child_morph, rng, num_modules, symmetry_axis)
        else:
            child_morph = mutate_morph(child_morph, rng, num_modules)

        attempts = 0
        valid = False
        while attempts < 50:
            if joint_count(child_morph) >= MIN_HINGES and validate_tree_depth(child_morph, max_depth):
                valid = True
                break
            if symmetry_axis is not None:
                child_morph = mutate_morph_symmetric(child_morph, rng, num_modules, symmetry_axis)
            else:
                child_morph = mutate_morph(child_morph, rng, num_modules)
            attempts += 1

        child = Individual()
        child.id = _next_ind_id()
        child.genotype = {"morph": child_morph.to_dict(), "parent_ids": parent_ids}
        child.tags = {"ps": False, "valid": valid, "best_brain": []}
        child.requires_eval = True
        if not valid:
            child.fitness = float("inf")
            child.requires_eval = False
        offspring.append(child)
    return offspring


# ── Inner CMA-ES (train one body, candidates evaluated in parallel) ───────────

# Registry so subprocess workers can look up episode functions by name without
# pickling the callable itself.
_EPISODE_FN_REGISTRY: dict[str, Any] = {}


def register_episode_fn(name: str, fn: Any) -> None:
    _EPISODE_FN_REGISTRY[name] = fn


def _eval_candidate_worker(job: dict[str, Any]) -> float:
    """Evaluate one candidate weight vector in a subprocess.

    Rebuilds (or reuses cached) ctx for the body, then runs the episode
    function looked up by name from the registry.
    """
    body_hash    = job["body_hash"]
    genome_dict  = job["genome_dict"]
    weights      = job["weights"]
    waypoints    = job["waypoints"]
    reach_radius = job["reach_radius"]
    arena_radius = job["arena_radius"]
    use_vision   = job["use_vision"]
    use_phase    = job.get("use_phase", False)
    episode_fn_name = job["episode_fn_name"]
    episode_fn = _EPISODE_FN_REGISTRY[episode_fn_name]
    try:
        ctx = ensure_ctx_for_body(body_hash, genome_dict, reach_radius, arena_radius, use_vision, use_phase)
    except Exception:
        return float("inf")

    result = episode_fn(ctx, waypoints, weights)
    return result["fitness"]


def train_body_parallel(task: dict[str, Any]) -> dict[str, Any]:
    """Run the full inner CMA-ES brain search with parallelised candidate evaluation.

    Returns dict with keys:
      fitness, weights, learning_curve, trajectory, control_cost
    """
    body_hash       = task["body_hash"]
    genome_dict     = task["genome_dict"]
    waypoints       = task["waypoints"]
    rng_seed        = task["rng_seed"]
    brain_budget    = task["brain_budget"]
    brain_pop       = task["brain_pop"]
    duration        = task["duration"]
    reach_radius    = task["reach_radius"]
    arena_radius    = task["arena_radius"]
    episode_fn_name = task["episode_fn_name"]
    episode_fn      = task["episode_fn"]
    use_vision      = task["use_vision"]
    use_phase       = task.get("use_phase", False)
    brain_workers   = task.get("brain_workers", 1)

    t_start = time.perf_counter()

    # Build ctx in the main process to validate the body and get num_params,
    # but each subprocess will build its own copy via ensure_ctx_for_body.
    try:
        ctx = ensure_ctx_for_body(body_hash, genome_dict, reach_radius, arena_radius, use_vision, use_phase)
    except Exception:
        return {"fitness": float("inf"), "weights": None,
                "learning_curve": [], "trajectory": [], "control_cost": 0.0,
                "eval_time_s": time.perf_counter() - t_start}

    model: mujoco.MjModel = ctx["model"]
    if model.nu == 0:
        return {"fitness": float("inf"), "weights": None,
                "learning_curve": [], "trajectory": [], "control_cost": 0.0,
                "eval_time_s": time.perf_counter() - t_start}

    num_params: int = ctx["num_params"]

    rng = np.random.default_rng(rng_seed)
    initial_guess = rng.uniform(-1.3, 1.3, size=num_params)
    param = ng.p.Array(init=initial_guess).set_mutation(sigma=0.35)
    optimizer = ng.optimizers.CMA(
        parametrization=param,
        budget=brain_budget,
        num_workers=1,
    )

    best_fit: float = float("inf")
    best_w: Optional[np.ndarray] = None
    learning_curve: list[float] = []

    # Sequential evaluation: one candidate per CMA-ES step, matching WCCI.
    # Parallelism happens at the outer level (one CPU per morphology).
    for _ in range(brain_budget):
        candidate = optimizer.ask()
        fit = episode_fn(ctx, waypoints, candidate.value)["fitness"]
        optimizer.tell(candidate, fit)

        if fit < best_fit:
            best_fit = fit
            best_w = candidate.value.copy()
            learning_curve.append(best_fit)

    # Single recording pass with the best weights found
    final_result = episode_fn(ctx, waypoints, best_w if best_w is not None else np.zeros(num_params), record=True)

    return {
        "fitness":        best_fit,
        "weights":        best_w.tolist() if best_w is not None else None,
        "learning_curve": learning_curve,
        "trajectory":     final_result.get("trajectory", []),
        "control_cost":   final_result.get("control_cost", 0.0),
        "eval_time_s":    time.perf_counter() - t_start,
    }


# ── Single-skill CMA-ES trainer (loco / turn / directional skills) ────────────
#
# Generalizes the loco / turn-left / turn-right skill-training pipeline
# originally written for gecko_food_skills.py's SkillController so it can be
# shared by any task script that needs "train a small CMA-ES-optimized brain
# whose reward is either forward-translation along some direction, or
# rotation about the vertical axis" (gecko_food_skills.py's loco/left/right,
# gecko_skill_tasks.py's forward/multidirection/turn_avg tasks).

CMA_SIGMA          = 0.7
CMA_POPSIZE        = 16
CMA_INIT_SCALE     = 1.3
CONTROL_STEP_FREQ  = 9       # steps between control recomputes; physics runs at
                              # MuJoCo's default 500Hz (MujocoConfig.timestep is
                              # dead code -- never applied to spec.option.timestep
                              # in _base_world.py), so this is ~55.6Hz control
                              # (500/9), the closest integer divisor to 60Hz.
SETTLE_DURATION    = 3.0
LOCO_DURATION      = 30.0
TURN_DURATION      = 15.0
CTRL_ALPHA               = 0.5
HINGE_CONTACT_LIMIT       = 200
HINGE_CONTACT_PENALTY     = 0.005
HINGE_GLITCH_FITNESS      = 1.0
HEIGHT_PENALTY_THRESHOLD  = 0.21
JERK_PENALTY_WEIGHT       = 3.0   # penalty per unit of mean absolute ctrl delta, above JERK_THRESHOLD
JERK_THRESHOLD            = 0.15  # hurdle: mean_jerk below this is free

# The core-mounted camera's local mount (euler + position) is fixed in
# core.py and never rotated relative to the world, since every body here
# spawns with rotation=None (SimpleFlatWorld default = no rotation). Its
# world-frame forward direction is therefore this constant for every
# individual, not something that needs re-measuring per body: confirmed via
# data.cam_xmat at spawn (camera looks down world -Y, not +X). angle_deg=0
# on a SkillReward means "this same axis", so the vision-driven
# SkillController's notion of forward and a 0-degree translation skill agree.
FORWARD_AXIS = np.array([0.0, -1.0])


@dataclass
class SkillReward:
    """What a single trained skill is rewarded for.

    kind="translate": reward is displacement projected onto FORWARD_AXIS
    rotated by angle_deg (degrees, counter-clockwise). angle_deg=0.0
    reproduces the original "loco" skill exactly.
    kind="rotate": reward is accumulated signed yaw in the direction
    turn_sign (+1 = left, -1 = right), reproducing the original "left"/
    "right" skills exactly.
    """

    kind: Literal["translate", "rotate"]
    duration: float
    angle_deg: float = 0.0
    turn_sign: int = 1


def _rotate_2d(v: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def rotor_geom_ids(model: mujoco.MjModel) -> set[int]:
    return {i for i in range(model.ngeom) if model.geom(i).name.endswith("-rotor")}


def floor_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


def build_loco_world_for_body(genome_dict: dict, to_spec_fn=genome_to_spec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Plain flat-world body, no food target/camera — used for skill training."""
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


_skill_worker_ctx: Optional[dict[str, Any]] = None


def _skill_worker_init(seed: int, reward: SkillReward, genome_dict: dict, to_spec_fn) -> None:
    global _skill_worker_ctx  # noqa: PLW0603
    torch.set_num_threads(1)
    np.random.seed((seed + os.getpid()) % (2**32 - 1))
    model, data = build_loco_world_for_body(genome_dict, to_spec_fn)
    input_dim   = len(get_robot_state(data))
    output_dim  = model.nu
    network     = Network(input_size=input_dim, output_size=output_dim, hidden_size=HIDDEN_SIZE)
    _skill_worker_ctx = {
        "model": model, "data": data, "network": network,
        "rotor_ids": rotor_geom_ids(model), "floor": floor_id(model), "reward": reward,
    }


def _skill_worker_eval(weights_list: list[float]) -> float:
    assert _skill_worker_ctx is not None
    ctx       = _skill_worker_ctx
    model     = ctx["model"]
    data      = ctx["data"]
    network   = ctx["network"]
    rotor_ids = ctx["rotor_ids"]
    floor     = ctx["floor"]
    reward: SkillReward = ctx["reward"]
    weights   = np.array(weights_list, dtype=np.float32)
    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    core_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    initial_height = float(data.xpos[core_id, 2])
    step           = 0
    current_action = np.zeros(model.nu)
    c_hinge        = 0
    prev_contacts: set[int] = set()
    prev_ctrl: Optional[np.ndarray] = None
    jerk_sum       = 0.0
    ctrl_step      = 0

    if reward.kind == "translate":
        xy0           = np.array([data.qpos[0], data.qpos[1]])
        reward_axis   = _rotate_2d(FORWARD_AXIS, reward.angle_deg)
    else:
        r_prev      = np.array(data.xmat[core_id]).reshape(3, 3).copy()
        accumulated = 0.0

    episode_end = SETTLE_DURATION + reward.duration
    while data.time < episode_end:
        if step % CONTROL_STEP_FREQ == 0:
            state      = get_robot_state(data).astype(np.float32)
            raw_action = network.forward(model, data, state)
            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -math.pi / 2, math.pi / 2,
            )
            if prev_ctrl is not None:
                jerk_sum += float(np.mean(np.abs(current_action - prev_ctrl)))
            prev_ctrl = current_action.copy()
            ctrl_step += 1
        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

        curr: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor and g2 in rotor_ids:
                curr.add(g2)
            elif g2 == floor and g1 in rotor_ids:
                curr.add(g1)
        c_hinge += len(curr - prev_contacts)
        prev_contacts = curr

        if reward.kind == "rotate":
            r_curr = np.array(data.xmat[core_id]).reshape(3, 3)
            delta  = signed_vertical_yaw_delta(r_prev, r_curr)
            accumulated += max(0.0, delta * reward.turn_sign)
            r_prev = r_curr.copy()

        step += 1

    if c_hinge > HINGE_CONTACT_LIMIT:
        return HINGE_GLITCH_FITNESS

    mean_jerk    = jerk_sum / max(ctrl_step - 1, 1)
    jerk_penalty = JERK_PENALTY_WEIGHT * mean_jerk if mean_jerk >= JERK_THRESHOLD else 0.0

    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    if reward.kind == "translate":
        xy_now   = np.array([data.qpos[0], data.qpos[1]])
        fwd_disp = float(np.dot(xy_now - xy0, reward_axis))
        return -(fwd_disp - height_penalty - jerk_penalty)
    else:
        return -(accumulated - height_penalty - jerk_penalty)


def train_skill_for_body(
    reward: SkillReward,
    genome_dict: dict,
    budget: int,
    workers: int,
    seed: int,
    to_spec_fn=genome_to_spec,
) -> tuple[np.ndarray, list[float], float]:
    """Train one skill via CMA-ES for a specific body.

    Returns (best_weights, learning_curve, eval_time_s).
    """
    t_start = time.perf_counter()

    model, data = build_loco_world_for_body(genome_dict, to_spec_fn)
    input_dim   = len(get_robot_state(data))
    output_dim  = model.nu
    network     = Network(input_size=input_dim, output_size=output_dim, hidden_size=HIDDEN_SIZE)
    num_params  = sum(p.numel() for p in network.parameters())

    rng = np.random.default_rng(seed)
    x0  = rng.uniform(-CMA_INIT_SCALE, CMA_INIT_SCALE, size=num_params).tolist()

    es = cma.CMAEvolutionStrategy(
        x0, CMA_SIGMA,
        {"popsize": CMA_POPSIZE, "seed": seed, "verbose": -9, "maxiter": 10**9},
    )

    best_fit = float("inf")
    best_w   = np.array(x0, dtype=np.float32)
    learning_curve: list[float] = []

    effective_workers = min(CMA_POPSIZE, workers)
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=_skill_worker_init,
        initargs=(seed, reward, genome_dict, to_spec_fn),
    ) as pool:
        gen = 0
        while gen < budget:
            if es.stop():
                break

            solutions = es.ask()
            fits      = list(pool.map(_skill_worker_eval, [s.tolist() for s in solutions]))
            es.tell(solutions, fits)

            gen_best = min(fits)
            if gen_best < best_fit:
                best_fit = gen_best
                best_w   = np.array(solutions[fits.index(gen_best)], dtype=np.float32)
                learning_curve.append(best_fit)

            gen += 1

    return best_w, learning_curve, time.perf_counter() - t_start
