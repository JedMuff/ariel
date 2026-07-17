"""
Shared helpers for the symmetry-pressure investigation.

Adapted from examples/re_book/6_body_brain_randomized_waypoints.py with
additions for extended data logging, NSGA-II selection, and repeat evaluations.
"""

import contextlib
import copy
import hashlib
import json
import multiprocessing as mp
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional, cast

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
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

# ── Constants ─────────────────────────────────────────────────────────────────

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


def bilateral_symmetry_score(genome_dict: dict) -> float:
    """Left-right (XZ-plane) bilateral symmetry of a TreeGenome.

    Computes positions of all modules relative to the root via BFS, then
    counts how many modules at (x, y, z) have a mirror at (x, -y, z) of
    the same type. Returns fraction of non-midline modules that are matched,
    in [0, 1]. Returns 0.0 on failure.
    """
    try:
        graph = TreeGenome.from_dict(genome_dict).to_networkx()
        if graph.number_of_nodes() == 0:
            return 0.0

        # Find root (no predecessors)
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        if not roots:
            return 0.0
        root = roots[0]

        # BFS to assign integer (x, y, z) grid positions.
        # Edge attribute 'face' encodes direction: FRONT=+x, BACK=-x,
        # RIGHT=+y, LEFT=-y, TOP=+z, BOTTOM=-z.
        face_to_delta = {
            "FRONT":  ( 1,  0,  0),
            "BACK":   (-1,  0,  0),
            "RIGHT":  ( 0,  1,  0),
            "LEFT":   ( 0, -1,  0),
            "TOP":    ( 0,  0,  1),
            "BOTTOM": ( 0,  0, -1),
        }
        pos: dict[Any, tuple[int, int, int]] = {root: (0, 0, 0)}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in graph.successors(node):
                if child in pos:
                    continue
                face = (graph.get_edge_data(node, child) or {}).get("face", "FRONT")
                dx, dy, dz = face_to_delta.get(face, (1, 0, 0))
                px, py, pz = pos[node]
                pos[child] = (px + dx, py + dy, pz + dz)
                queue.append(child)

        def node_type(n: Any) -> str:
            return graph.nodes[n].get("type", "UNKNOWN")

        # Build lookup: (x, y, z) -> type
        grid: dict[tuple[int, int, int], str] = {p: node_type(n) for n, p in pos.items()}

        num_symmetrical = 0
        num_off_midline = 0
        seen: set[tuple[int, int, int]] = set()
        for (x, y, z), t in grid.items():
            if y == 0:
                continue  # midline module, skip
            if (x, y, z) in seen:
                continue
            seen.add((x, y, z))
            seen.add((x, -y, z))
            num_off_midline += 1
            mirror = grid.get((x, -y, z))
            if mirror == t:
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
    input_dim = genome_input_dim(model, data, use_vision)
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


def joint_count(genome: TreeGenome) -> int:
    """Count hinge joints by walking the genome graph — no MuJoCo compilation needed."""
    return sum(1 for d in genome.nodes.values() if d.get("type") == "HINGE")


def create_individual(num_modules: int, max_depth: int) -> Individual:
    while True:
        genome = random_tree(num_modules)
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
) -> list[Individual]:
    """Generate lam offspring via subtree crossover + mutation."""
    offspring: list[Individual] = []
    while len(offspring) < lam:
        use_sexual = len(parents) >= 2 and rng.random() < 0.6
        if use_sexual:
            p1, p2 = random.sample(parents, 2)
            t1 = TreeGenome.from_dict(p1.genotype["morph"])
            t2 = TreeGenome.from_dict(p2.genotype["morph"])
            c1, c2 = crossover_subtree(t1, t2)
            child_morph = c1 if rng.random() < 0.5 else c2
            parent_ids = [p1.id, p2.id]
        else:
            p = random.choice(parents)
            child_morph = TreeGenome.from_dict(p.genotype["morph"])
            parent_ids = [p.id]

        child_morph = mutate_morph(child_morph, rng, num_modules)

        attempts = 0
        valid = False
        while attempts < 50:
            if joint_count(child_morph) >= MIN_HINGES and validate_tree_depth(child_morph, max_depth):
                valid = True
                break
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
    episode_fn_name = job["episode_fn_name"]
    episode_fn = _EPISODE_FN_REGISTRY[episode_fn_name]
    try:
        ctx = ensure_ctx_for_body(body_hash, genome_dict, reach_radius, arena_radius, use_vision)
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
    brain_workers   = task.get("brain_workers", 1)

    t_start = time.perf_counter()

    # Build ctx in the main process to validate the body and get num_params,
    # but each subprocess will build its own copy via ensure_ctx_for_body.
    try:
        ctx = ensure_ctx_for_body(body_hash, genome_dict, reach_radius, arena_radius, use_vision)
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
    initial_guess = rng.uniform(-0.5, 0.5, size=num_params)
    param = ng.p.Array(init=initial_guess).set_mutation(sigma=0.3)
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
