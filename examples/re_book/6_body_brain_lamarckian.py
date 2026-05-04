"""
Body + brain co-evolution on the randomised-waypoints task — Lamarckian variant.

Same as `6_body_brain_randomized_waypoints.py`, except the inner CMA-ES brain
search **warm-starts** from the parent's best brain (with weights for new
joints / rearranged joints reinitialised). Body-invariant network parts
(fc2, fc1.bias, the input columns for non-body inputs, etc.) are inherited
verbatim. See `_lamarckian_warm_start` for the adaptation rules.

Joint-mapping uses a structural path signature (face, type, rotation) from
IDX_OF_CORE to each hinge — robust to TreeGenome ID reassignment in
`mutate_subtree_replacement` and `crossover_subtree`.

Outer loop:  (mu+lambda) ES over TreeGenome bodies (uses ariel.ec engine).
Inner loop:  CMA-ES (nevergrad) over NN brain weights, warm-started from parent.
Task:        randomised waypoints (vision-based), 60 s episodes, top-down camera.
"""

# Standard library
import argparse
import contextlib
import copy
import gc
import hashlib
import json
import multiprocessing as mp
import os
import random
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional, cast

# Third-party
import cv2
import mujoco
import nevergrad as ng
import numpy as np
import torch
from rich.console import Console
from rich.traceback import install
from torch import nn

# ARIEL
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec import EA, EAOperation, EASettings, Individual, Population
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
from ariel.utils.renderers import VideoRecorder

install()
console = Console()

# TPA fires check_consistency on noisy stochastic simulators — harmless.
warnings.filterwarnings(
    "ignore",
    message="TPA: apparent inconsistency",
    category=UserWarning,
    module="cma",
)

# ── CLI args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Body + brain co-evolution on randomised waypoints")
parser.add_argument("--budget",        type=int,   default=20,   help="Outer body generations")
parser.add_argument("--pop",           type=int,   default=10,   help="mu (parents)")
parser.add_argument("--lam",           type=int,   default=10,   help="lambda (offspring per gen)")
parser.add_argument("--brain-budget",  type=int,   default=30,   help="Inner CMA generations per body")
parser.add_argument("--brain-pop",     type=int,   default=32,   help="Inner CMA population per generation")
parser.add_argument("--brain-workers", type=int,   default=max(1, os.cpu_count() or 1), help="Inner CMA worker processes")
parser.add_argument("--dur",           type=float, default=60.0, help="Episode duration (s)")
parser.add_argument("--reach-radius",  type=float, default=0.20,
                    help="Planar reach radius (m). The gate is a vertical cylinder of this radius; "
                         "trigger fires when the core's centre is within `reach_radius` of the gate centre "
                         "in the xy-plane.")
parser.add_argument("--num-waypoints", type=int,   default=10,   help="Waypoints per episode")
parser.add_argument("--arena-radius",  type=float, default=3.0,  help="Arena radius (m)")
parser.add_argument("--max-modules",   type=int,   default=25,   help="Max modules per body")
parser.add_argument("--max-depth",     type=int,   default=25,   help="Max tree depth")
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--strategy",      choices=["within", "across"], default="across",
                    help="'within' parallelises inner-CMA candidates of one body across workers (bodies serial); "
                         "'across' trains one body per worker, (mu+lambda) bodies in parallel")
parser.add_argument("--no-video",      action="store_true", help="Skip best-of-run video")
args = parser.parse_args()

BUDGET        = args.budget
MU            = args.pop
LAM           = args.lam
BRAIN_BUDGET  = args.brain_budget
BRAIN_POP     = args.brain_pop
BRAIN_WORKERS = max(1, args.brain_workers)
DURATION      = args.dur
REACH_RADIUS  = max(0.05, args.reach_radius)
NUM_WAYPOINTS = args.num_waypoints
ARENA_RADIUS  = max(1.0, args.arena_radius)
NUM_MODULES   = args.max_modules
MAX_DEPTH     = args.max_depth
BASE_SEED     = args.seed
STRATEGY      = args.strategy

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)
CHECKPOINTS = DATA / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True, parents=True)

SPAWN_POSITION = (0.0, 0.0, 0.1)
HIDDEN_SIZE = 32
MIN_HINGES = 4  # bodies with fewer hinges are rejected at construction time

# ── Waypoint sampling (copied from 5_randomized_waypoints.py) ─────────────────

RING_R_MIN = 0.5
RING_R_MAX = 1.0
GATE_HALF_HEIGHT = 0.15  # cylinder half-height; waypoint z is set so base sits on floor


def sample_waypoints(
    rng: np.random.Generator,
    n: int = NUM_WAYPOINTS,
    r_min: float = RING_R_MIN,
    r_max: float = RING_R_MAX,
) -> list[np.ndarray]:
    """Sample n waypoints sequentially: each one is offset from the previous
    waypoint by a random vector in the annulus r ∈ [r_min, r_max], θ ∈ [0, 2π].
    The first waypoint is offset from the origin (0, 0). Uniform in area
    (r = sqrt(uniform(r_min², r_max²))) so the distribution is flat over the
    ring, not biased toward the inner edge.
    """
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


# ── Network (copied from 5_randomized_waypoints.py) ───────────────────────────

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
    def forward(self, model, data, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
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


# ── Vision helpers (copied from 5_randomized_waypoints.py) ────────────────────

def isolate_green(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))


def analyze_sections(mask: np.ndarray) -> list[float]:
    sections = np.array_split(mask, 3, axis=1)
    return [cv2.countNonZero(s) / max(s.size, 1) for s in sections]


# ── Episode runner + fitness (copied from 5_randomized_waypoints.py) ──────────

def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: Network,
    waypoints: list[np.ndarray],
    duration: float,
    reach_radius: float,
    target_mocap_id: int,
    renderer: mujoco.Renderer,
    cam_name: Optional[str],
    control_step_freq: int = 50,
) -> dict[str, Any]:
    num_wps        = len(waypoints)
    current_wp_idx = 0
    waypoints_reached   = 0
    current_target = waypoints[0]
    data.mocap_pos[target_mocap_id] = current_target

    current_action      = np.zeros(model.nu)
    min_dist_to_current = float("inf")
    completion_time: Optional[float] = None
    step = 0

    while data.time < duration and current_wp_idx < num_wps:
        if step % control_step_freq == 0:
            renderer.update_scene(data, camera=cam_name)
            img    = renderer.render()
            vision = analyze_sections(isolate_green(img))

            robot_state = get_robot_state(data)
            phase = [
                2.0 * np.sin(data.time * 2.0 * np.pi),
                2.0 * np.cos(data.time * 2.0 * np.pi),
            ]
            progress = [current_wp_idx / max(num_wps - 1, 1)]

            state = np.concatenate([robot_state, vision, phase, progress]).astype(np.float32)
            current_action = network.forward(model, data, state)

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1

        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist_to_current = min(min_dist_to_current, dist)

        if dist <= reach_radius:
            waypoints_reached += 1
            current_wp_idx    += 1
            if current_wp_idx < num_wps:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist_to_current = float("inf")
            else:
                completion_time = data.time

    final_dist = 0.0 if current_wp_idx >= num_wps else min_dist_to_current

    return {
        "waypoints_reached":   waypoints_reached,
        "min_dist_to_current": final_dist,
        "completion_time":     completion_time,
    }


def compute_fitness(
    waypoints_reached: int,
    min_dist_to_current: float,
    num_waypoints: int,
    completion_time: Optional[float] = None,
    duration: float = DURATION,
) -> float:
    """Lower is better.

    fitness = -1 * waypoints_reached  +  closeness_to_next
    closeness_to_next = (d_min_to_next / RING_R_MAX) - 1   ∈ (-1, +∞), no clamp
                      = 0 when all waypoints reached (no "next").

    With RING_R_MAX = 1.0 m: 0 ≈ at the gate, +1 ≈ ring-radius away.
    Best possible: -num_waypoints (all reached, ended on the last gate).
    `completion_time` is unused here — kept in the signature so callers don't break.
    """
    del completion_time, duration  # no time bonus in this fitness
    if waypoints_reached >= num_waypoints:
        closeness = 0.0
    else:
        closeness = (min_dist_to_current / RING_R_MAX) - 1.0
    return -float(waypoints_reached) + closeness


# ── Body construction ────────────────────────────────────────────────────────

def _genome_to_spec(genome_dict: dict) -> Optional[mujoco.MjSpec]:
    try:
        graph = TreeGenome.from_dict(genome_dict).to_networkx()
        if graph.number_of_nodes() == 0:
            return None
        return construct_mjspec_from_graph(graph).spec
    except Exception:
        return None


def _build_world_for_body(genome_dict: dict) -> tuple[mujoco.MjModel, mujoco.MjData, int, Optional[str]]:
    """Build a SimpleFlatWorld + TreeGenome body + green mocap marker + cameras."""
    spec = _genome_to_spec(genome_dict)
    if spec is None:
        raise ValueError("Could not decode morphology")

    world = SimpleFlatWorld()
    try:
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=False)

    # Green pass-through goal: a vertical cylinder of radius REACH_RADIUS so
    # the visible footprint matches the (planar) reach test exactly. mocap=True
    # disables joint dynamics; contype/conaffinity=0 removes contact filtering
    # so the robot passes through. Cylinder sits on the floor (z=GATE_HALF_HEIGHT).
    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=[0.0, 0.0, GATE_HALF_HEIGHT]
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[REACH_RADIUS, GATE_HALF_HEIGHT],  # [radius, half-height], axis along z
        rgba=[0, 1, 0, 0.7],
        contype=0,
        conaffinity=0,
    )

    # Top-down overview camera (used for video recording).
    world.spec.worldbody.add_camera(
        name="overview_cam",
        pos=[0.0, 0.0, ARENA_RADIUS * 3.5],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )

    model = world.spec.compile()
    data  = mujoco.MjData(model)

    # Pick the onboard body camera (not the overview camera) for vision input.
    cam_name: Optional[str] = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "overview" not in name:
            cam_name = name
            break

    target_mocap_id = model.body("green_target").mocapid[0]
    return model, data, target_mocap_id, cam_name


def _genome_input_dim(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    robot_state_size = len(get_robot_state(data))
    # robot_state + 3 vision bins + 2 phase + 1 progress
    return robot_state_size + 3 + 2 + 1


def _genome_hash(genome_dict: dict) -> str:
    return hashlib.sha1(json.dumps(genome_dict, sort_keys=True).encode()).hexdigest()


# ── Lamarckian warm-start helpers ─────────────────────────────────────────────
#
# Goal: produce a flat weight vector for a child body's `Network` whose mean
# is "the parent's brain projected onto the child's joint layout". Joints that
# survived intact get their parent weights; new joints get random init.

# Number of non-body inputs: 3 quat_imag (head) + 3 vision + 2 phase + 1 progress (tail)
_HEAD_INPUTS = 3
_TAIL_INPUTS = 3 + 2 + 1


def _hinge_signatures(genome_dict: dict) -> list[tuple]:
    """Return one signature per HINGE node, in MuJoCo joint-index order.

    Joint order in MuJoCo follows the order modules are added in
    `construct_mjspec_from_graph`, which iterates `graph.nodes`. NetworkX
    preserves insertion order, which mirrors `TreeGenome.nodes` dict order.
    So iterating the `TreeGenome.nodes` dict and emitting a signature for
    every HINGE node gives the same ordering as `data.qpos[7:]` and
    `model.nu`.

    The signature is the tuple of (face, type, rotation) edges along the
    path from IDX_OF_CORE down to the node. Robust to ID reassignment by
    `mutate_subtree_replacement` and `crossover_subtree`.
    """
    genome = TreeGenome.from_dict(genome_dict)
    # parent map: child_id -> (parent_id, face)
    parent_of: dict[int, tuple[int, str]] = {}
    for e in genome.edges:
        parent_of[int(e["child"])] = (int(e["parent"]), e["face"])

    def path_sig(node_id: int) -> Optional[tuple]:
        # Walk to IDX_OF_CORE collecting (face, type, rotation) per edge.
        steps: list[tuple[str, str, str]] = []
        cur = node_id
        seen: set[int] = set()
        while cur != IDX_OF_CORE:
            if cur in seen or cur not in parent_of:
                # Disconnected from core (shouldn't happen post-prune) — bail.
                return None
            seen.add(cur)
            parent_id, face = parent_of[cur]
            attrs = genome.nodes[cur]
            steps.append((face, attrs["type"], attrs["rotation"]))
            cur = parent_id
        steps.reverse()
        return tuple(steps)

    sigs: list[tuple] = []
    for nid, attrs in genome.nodes.items():
        if attrs["type"] != ModuleType.HINGE.name:
            continue
        sig = path_sig(int(nid))
        # Distinct sentinel for orphan hinges so they never match.
        sigs.append(sig if sig is not None else ("__orphan__", int(nid)))
    return sigs


def _expected_param_count(input_dim: int, output_dim: int, hidden: int = HIDDEN_SIZE) -> int:
    return (
        hidden * input_dim + hidden                # fc1
        + hidden * hidden + hidden                 # fc2
        + output_dim * hidden + output_dim         # fc_out
    )


def _slice_parent_weights(
    parent_weights: np.ndarray, parent_input_dim: int, parent_output_dim: int, hidden: int = HIDDEN_SIZE,
) -> dict[str, np.ndarray]:
    """Slice flat parent weight vector into the six named tensors."""
    a = 0
    fc1_w = parent_weights[a : a + hidden * parent_input_dim].reshape(hidden, parent_input_dim); a += hidden * parent_input_dim
    fc1_b = parent_weights[a : a + hidden]; a += hidden
    fc2_w = parent_weights[a : a + hidden * hidden].reshape(hidden, hidden); a += hidden * hidden
    fc2_b = parent_weights[a : a + hidden]; a += hidden
    fco_w = parent_weights[a : a + parent_output_dim * hidden].reshape(parent_output_dim, hidden); a += parent_output_dim * hidden
    fco_b = parent_weights[a : a + parent_output_dim]; a += parent_output_dim
    return {"fc1_w": fc1_w, "fc1_b": fc1_b, "fc2_w": fc2_w, "fc2_b": fc2_b, "fco_w": fco_w, "fco_b": fco_b}


def _lamarckian_warm_start(
    parent_morph_dict: Optional[dict],
    parent_weights: Optional[np.ndarray],
    child_morph_dict: dict,
    child_input_dim: int,
    child_output_dim: int,
    rng: np.random.Generator,
    hidden: int = HIDDEN_SIZE,
) -> tuple[np.ndarray, dict]:
    """Build the warm-start initial vector for the child's CMA-ES.

    Returns (vector, info). `info` is `{matched, total, fallback}`.
    On any inconsistency, falls back to `rng.uniform(-0.5, 0.5, num_params)`.
    """
    num_params = _expected_param_count(child_input_dim, child_output_dim, hidden)
    fallback = rng.uniform(-0.5, 0.5, size=num_params)

    if parent_morph_dict is None or parent_weights is None or len(parent_weights) == 0:
        return fallback, {"matched": 0, "total": child_output_dim, "fallback": True}

    parent_w = np.asarray(parent_weights, dtype=np.float32)

    # Parent network shape: child_output_dim = num_joints; parent_output_dim = parent num_joints.
    # parent_input_dim = parent_output_dim + _HEAD_INPUTS + _TAIL_INPUTS.
    # Recover parent_output_dim from parent_w length.
    # len = h*(p_in) + h + h*h + h + p_out*h + p_out
    #     = h*(p_out + 9) + h + h*h + h + p_out*h + p_out
    #     = (2*h + 1) * p_out + h*9 + h + h*h + h
    # => p_out = (len - h*9 - h - h*h - h) / (2*h + 1)
    fixed = hidden * (_HEAD_INPUTS + _TAIL_INPUTS) + hidden + hidden * hidden + hidden
    denom = 2 * hidden + 1
    diff = len(parent_w) - fixed
    if diff < 0 or diff % denom != 0:
        return fallback, {"matched": 0, "total": child_output_dim, "fallback": True}
    parent_output_dim = diff // denom
    parent_input_dim = parent_output_dim + _HEAD_INPUTS + _TAIL_INPUTS
    if _expected_param_count(parent_input_dim, parent_output_dim, hidden) != len(parent_w):
        return fallback, {"matched": 0, "total": child_output_dim, "fallback": True}

    try:
        p = _slice_parent_weights(parent_w, parent_input_dim, parent_output_dim, hidden)
        parent_sigs = _hinge_signatures(parent_morph_dict)
        child_sigs  = _hinge_signatures(child_morph_dict)
    except Exception:
        return fallback, {"matched": 0, "total": child_output_dim, "fallback": True}

    if len(parent_sigs) != parent_output_dim or len(child_sigs) != child_output_dim:
        # Mismatch between recovered shape and signature count means the parent
        # weights vector doesn't correspond to the parent morphology. Bail.
        return fallback, {"matched": 0, "total": child_output_dim, "fallback": True}

    # Initialise child tensors with random defaults.
    c_fc1_w = rng.uniform(-0.5, 0.5, size=(hidden, child_input_dim)).astype(np.float32)
    c_fc1_b = rng.uniform(-0.5, 0.5, size=(hidden,)).astype(np.float32)
    c_fc2_w = rng.uniform(-0.5, 0.5, size=(hidden, hidden)).astype(np.float32)
    c_fc2_b = rng.uniform(-0.5, 0.5, size=(hidden,)).astype(np.float32)
    c_fco_w = rng.uniform(-0.5, 0.5, size=(child_output_dim, hidden)).astype(np.float32)
    c_fco_b = rng.uniform(-0.5, 0.5, size=(child_output_dim,)).astype(np.float32)

    # Body-invariant parts: copy verbatim.
    c_fc2_w[:] = p["fc2_w"]
    c_fc2_b[:] = p["fc2_b"]
    c_fc1_b[:] = p["fc1_b"]

    # fc1.weight head (quat_imag) — first 3 cols are body-invariant.
    c_fc1_w[:, :_HEAD_INPUTS] = p["fc1_w"][:, :_HEAD_INPUTS]

    # fc1.weight tail (vision/phase/progress) — last 6 cols are body-invariant.
    p_tail_start = _HEAD_INPUTS + parent_output_dim
    c_tail_start = _HEAD_INPUTS + child_output_dim
    c_fc1_w[:, c_tail_start : c_tail_start + _TAIL_INPUTS] = (
        p["fc1_w"][:, p_tail_start : p_tail_start + _TAIL_INPUTS]
    )

    # Per-joint adaptation: match by structural signature.
    parent_sig_to_idx: dict[tuple, int] = {}
    for i, s in enumerate(parent_sigs):
        parent_sig_to_idx.setdefault(s, i)  # ignore (impossible) duplicates

    matched = 0
    for j_c, sig_c in enumerate(child_sigs):
        j_p = parent_sig_to_idx.get(sig_c)
        if j_p is None:
            continue
        # fc1 column for joint j_c (in the joint_pos block, cols [3 : 3+njoints]).
        c_fc1_w[:, _HEAD_INPUTS + j_c] = p["fc1_w"][:, _HEAD_INPUTS + j_p]
        # fc_out row + bias for joint j_c.
        c_fco_w[j_c, :] = p["fco_w"][j_p, :]
        c_fco_b[j_c]    = p["fco_b"][j_p]
        matched += 1

    out = np.concatenate([
        c_fc1_w.ravel(), c_fc1_b.ravel(),
        c_fc2_w.ravel(), c_fc2_b.ravel(),
        c_fco_w.ravel(), c_fco_b.ravel(),
    ]).astype(np.float64)

    if len(out) != num_params:
        return fallback, {"matched": 0, "total": child_output_dim, "fallback": True}

    return out, {"matched": matched, "total": child_output_dim, "fallback": False}


# ── Worker context (per-process; rebuilt when body changes) ───────────────────

_RENDER_INIT_LOCK = threading.Lock()
_process_local_ctx: Optional[dict[str, Any]] = None


def _ensure_ctx_for_body(body_hash: str, genome_dict: dict) -> dict[str, Any]:
    global _process_local_ctx
    if _process_local_ctx is not None and _process_local_ctx.get("body_hash") == body_hash:
        return cast(dict[str, Any], _process_local_ctx)

    if _process_local_ctx is not None:
        # Release the previous renderer to free GPU/CPU resources.
        with contextlib.suppress(Exception):
            _process_local_ctx["renderer"].close()
        _process_local_ctx = None

    model, data, target_mocap_id, cam_name = _build_world_for_body(genome_dict)
    input_dim = _genome_input_dim(model, data)
    network = Network(input_size=input_dim, output_size=model.nu)
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
    }
    return cast(dict[str, Any], _process_local_ctx)


def _init_worker(base_seed: int) -> None:
    torch.set_num_threads(1)
    seed = (base_seed + os.getpid()) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate_brain(task: dict[str, Any]) -> float:
    body_hash    = task["body_hash"]
    genome_dict  = task["genome_dict"]
    weights      = task["weights"]
    waypoints    = task["waypoints"]

    ctx = _ensure_ctx_for_body(body_hash, genome_dict)
    model           = cast(mujoco.MjModel,  ctx["model"])
    data            = cast(mujoco.MjData,   ctx["data"])
    network         = cast(Network,         ctx["network"])
    renderer        = cast(mujoco.Renderer, ctx["renderer"])
    cam_name        = cast(Optional[str],   ctx["cam_name"])
    target_mocap_id = cast(int,             ctx["target_mocap_id"])

    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    result = run_episode(
        model=model,
        data=data,
        network=network,
        waypoints=waypoints,
        duration=DURATION,
        reach_radius=REACH_RADIUS,
        target_mocap_id=target_mocap_id,
        renderer=renderer,
        cam_name=cam_name,
    )

    return compute_fitness(
        waypoints_reached=result["waypoints_reached"],
        min_dist_to_current=result["min_dist_to_current"],
        num_waypoints=len(waypoints),
        completion_time=result["completion_time"],
        duration=DURATION,
    )


# ── Strategy B: train one whole body in a single worker process ──────────────

def _train_body_serial(task: dict[str, Any]) -> dict[str, Any]:
    """
    Run the inner CMA-ES brain search serially in this worker process.

    Used by --strategy across: each call trains one body to completion.
    Returns {"fitness": float, "weights": list[float] or None,
             "warm_start": {"matched", "total", "fallback"}}.
    """
    body_hash      = task["body_hash"]
    genome_dict    = task["genome_dict"]
    waypoints      = task["waypoints"]
    rng_seed       = task["rng_seed"]
    parent_morph   = task.get("parent_morph")
    parent_weights = task.get("parent_weights") or None

    try:
        ctx = _ensure_ctx_for_body(body_hash, genome_dict)
    except Exception:
        return {"fitness": float("inf"), "weights": None, "warm_start": {"matched": 0, "total": 0, "fallback": True}}

    model           = cast(mujoco.MjModel,  ctx["model"])
    data            = cast(mujoco.MjData,   ctx["data"])
    network         = cast(Network,         ctx["network"])
    renderer        = cast(mujoco.Renderer, ctx["renderer"])
    cam_name        = cast(Optional[str],   ctx["cam_name"])
    target_mocap_id = cast(int,             ctx["target_mocap_id"])
    num_params: int = ctx["num_params"]
    input_dim:  int = ctx["input_dim"]

    if model.nu == 0:
        return {"fitness": float("inf"), "weights": None, "warm_start": {"matched": 0, "total": 0, "fallback": True}}

    min_lambda = 4 + int(3 * np.log(max(num_params, 2)))
    pop_size   = max(BRAIN_POP, min_lambda)
    if pop_size % 2 != 0:
        pop_size += 1

    rng = np.random.default_rng(rng_seed)
    try:
        initial_guess, warm_info = _lamarckian_warm_start(
            parent_morph_dict=parent_morph,
            parent_weights=np.asarray(parent_weights, dtype=np.float32) if parent_weights else None,
            child_morph_dict=genome_dict,
            child_input_dim=input_dim,
            child_output_dim=int(model.nu),
            rng=rng,
        )
    except Exception:
        initial_guess = rng.uniform(-0.5, 0.5, size=num_params)
        warm_info = {"matched": 0, "total": int(model.nu), "fallback": True}

    if len(initial_guess) != num_params:
        initial_guess = rng.uniform(-0.5, 0.5, size=num_params)
        warm_info = {"matched": 0, "total": int(model.nu), "fallback": True}

    param = ng.p.Array(init=initial_guess).set_mutation(sigma=0.3)
    cma_config = ng.optimizers.ParametrizedCMA(popsize=pop_size)
    optimizer = cma_config(
        parametrization=param,
        budget=BRAIN_BUDGET * pop_size,
        num_workers=1,
    )

    best_fit: float = float("inf")
    best_w:   Optional[np.ndarray] = None

    for _ in range(BRAIN_BUDGET):
        candidates = [optimizer.ask() for _ in range(pop_size)]
        for cand in candidates:
            fill_parameters(network, cand.value)
            mujoco.mj_resetData(model, data)
            result = run_episode(
                model=model,
                data=data,
                network=network,
                waypoints=waypoints,
                duration=DURATION,
                reach_radius=REACH_RADIUS,
                target_mocap_id=target_mocap_id,
                renderer=renderer,
                cam_name=cam_name,
            )
            fit = compute_fitness(
                waypoints_reached=result["waypoints_reached"],
                min_dist_to_current=result["min_dist_to_current"],
                num_waypoints=len(waypoints),
                completion_time=result["completion_time"],
                duration=DURATION,
            )
            optimizer.tell(cand, fit)
            if fit < best_fit:
                best_fit = fit
                best_w   = cand.value.copy()

    return {
        "fitness": best_fit,
        "weights": best_w.tolist() if best_w is not None else None,
        "warm_start": warm_info,
    }


# ── Inner CMA-ES brain search ────────────────────────────────────────────────

def learn_brain(
    genome_dict: dict,
    waypoints: list[np.ndarray],
    executor: ProcessPoolExecutor,
    parent_morph: Optional[dict] = None,
    parent_weights: Optional[list[float]] = None,
    rng_seed: Optional[int] = None,
) -> tuple[float, Optional[np.ndarray], dict]:
    """Run CMA-ES on the brain for one body.

    Returns (best_fitness, best_weights, warm_info).
    """
    # Build the body once on the master to discover input/output dims and
    # parameter count. Worker processes will rebuild on first use.
    try:
        model, data, _, _ = _build_world_for_body(genome_dict)
    except Exception:
        return float("inf"), None, {"matched": 0, "total": 0, "fallback": True}

    if model.nu == 0:
        return float("inf"), None, {"matched": 0, "total": 0, "fallback": True}

    input_dim = _genome_input_dim(model, data)
    dummy_net = Network(input_size=input_dim, output_size=model.nu)
    num_params = sum(p.numel() for p in dummy_net.parameters())

    min_lambda = 4 + int(3 * np.log(max(num_params, 2)))
    pop_size   = max(BRAIN_POP, min_lambda)
    if pop_size % 2 != 0:
        pop_size += 1

    rng = np.random.default_rng(rng_seed)
    try:
        initial_guess, warm_info = _lamarckian_warm_start(
            parent_morph_dict=parent_morph,
            parent_weights=np.asarray(parent_weights, dtype=np.float32) if parent_weights else None,
            child_morph_dict=genome_dict,
            child_input_dim=input_dim,
            child_output_dim=int(model.nu),
            rng=rng,
        )
    except Exception:
        initial_guess = rng.uniform(-0.5, 0.5, size=num_params)
        warm_info = {"matched": 0, "total": int(model.nu), "fallback": True}

    if len(initial_guess) != num_params:
        initial_guess = rng.uniform(-0.5, 0.5, size=num_params)
        warm_info = {"matched": 0, "total": int(model.nu), "fallback": True}

    param = ng.p.Array(init=initial_guess).set_mutation(sigma=0.3)
    cma_config = ng.optimizers.ParametrizedCMA(popsize=pop_size)
    optimizer = cma_config(
        parametrization=param,
        budget=BRAIN_BUDGET * pop_size,
        num_workers=pop_size,
    )

    body_hash = _genome_hash(genome_dict)

    best_fit: float = float("inf")
    best_w:   Optional[np.ndarray] = None

    for _ in range(BRAIN_BUDGET):
        candidates = [optimizer.ask() for _ in range(pop_size)]
        tasks = [
            {
                "body_hash":   body_hash,
                "genome_dict": genome_dict,
                "weights":     c.value,
                "waypoints":   waypoints,
            }
            for c in candidates
        ]
        fitnesses = list(executor.map(_evaluate_brain, tasks))

        for cand, fit in zip(candidates, fitnesses):
            optimizer.tell(cand, fit)

        gen_best_idx = int(np.argmin(fitnesses))
        gen_best_fit = float(fitnesses[gen_best_idx])
        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_w   = candidates[gen_best_idx].value.copy()

    return best_fit, best_w, warm_info


# ── Body-evolution helpers ────────────────────────────────────────────────────

def _joint_count(genome: TreeGenome) -> int:
    spec = _genome_to_spec(genome.to_dict())
    if spec is None:
        return 0
    try:
        return spec.compile().nu
    except Exception:
        return 0


RNG = np.random.default_rng(BASE_SEED)


def _mutate_morph(genome: TreeGenome) -> TreeGenome:
    new = copy.deepcopy(genome)
    mutation_type = RNG.choice(["point", "subtree", "shrink", "hoist"], p=[0.45, 0.35, 0.1, 0.1])

    if mutation_type == "point":
        mutate_replace_node(new)
    elif mutation_type == "subtree":
        mutate_subtree_replacement(new, max_modules=NUM_MODULES)
    elif mutation_type == "shrink":
        mutate_shrink(new)
    else:
        mutate_hoist(new)

    if RNG.random() < 0.25:
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


def _create_individual() -> Individual:
    while True:
        genome = random_tree(NUM_MODULES)
        if _joint_count(genome) >= MIN_HINGES and validate_tree_depth(genome, MAX_DEPTH):
            break
    ind = Individual()
    ind.genotype = {"morph": genome.to_dict()}
    ind.tags = {"ps": False, "valid": True, "best_brain": []}
    return ind


# ── (mu+lambda) outer loop ────────────────────────────────────────────────────

class BodyBrainEvolution:
    """Co-evolves bodies (TreeGenome, mu+lambda) and brains (NN, CMA-ES inner loop)."""

    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,
            num_steps=BUDGET,
            target_population_size=MU,
            output_folder=DATA,
            db_file_name=f"database_{int(time.time())}.db",
            db_handling="delete",
        )
        self.executor: Optional[ProcessPoolExecutor] = None
        self.outer_gen: int = 0
        self.gen_waypoints: list[np.ndarray] = []
        # Track best-seen across the whole run (for final video).
        self.best_seen_fitness: float = float("inf")
        self.best_seen_genotype: Optional[dict] = None
        self.best_seen_weights:  Optional[np.ndarray] = None
        self.best_seen_waypoints: Optional[list[np.ndarray]] = None

    # -- ec.EAOperation pipeline ----------------------------------------------

    def parent_selection(self, population: Population) -> Population:
        population = population.sort(sort="min", attribute="fitness_")
        for i, ind in enumerate(population):
            ind.tags = {"ps": i < MU}
        return population

    def reproduction(self, population: Population) -> Population:
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        if not parents:
            parents = list(population)

        offspring: list[Individual] = []
        while len(offspring) < LAM:
            use_sexual = len(parents) >= 2 and RNG.random() < 0.6
            if use_sexual:
                p1, p2 = random.sample(parents, 2)
                t1 = TreeGenome.from_dict(p1.genotype["morph"])
                t2 = TreeGenome.from_dict(p2.genotype["morph"])
                c1, c2 = crossover_subtree(t1, t2)
                pick_first = RNG.random() < 0.5
                child_morph = c1 if pick_first else c2
                # The kept-subtree-side parent provides Lamarckian weights:
                # c1 retains p1's structure (its subtree was swapped for p2's),
                # c2 retains p2's structure. Pick the matching parent so the
                # signature map shares the most edges with the child.
                chosen_parent = p1 if pick_first else p2
            else:
                p = random.choice(parents)
                child_morph = TreeGenome.from_dict(p.genotype["morph"])
                chosen_parent = p

            child_morph = _mutate_morph(child_morph)

            attempts = 0
            valid = False
            while attempts < 50:
                if _joint_count(child_morph) >= MIN_HINGES and validate_tree_depth(child_morph, MAX_DEPTH):
                    valid = True
                    break
                child_morph = _mutate_morph(child_morph)
                attempts += 1

            parent_brain = chosen_parent.tags.get("best_brain") if chosen_parent is not None else None
            parent_morph = (
                copy.deepcopy(chosen_parent.genotype["morph"])
                if chosen_parent is not None and parent_brain
                else None
            )

            child = Individual()
            child.genotype = {"morph": child_morph.to_dict()}
            child.tags = {
                "ps": False,
                "valid": valid,
                "best_brain": [],
                "parent_morph": parent_morph,
                "parent_weights": list(parent_brain) if parent_brain else [],
            }
            child.requires_eval = True
            if not valid:
                child.fitness = float("inf")
                child.requires_eval = False
            offspring.append(child)

        population.extend(offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        # Sample one waypoint layout per outer generation; reused for every
        # body and every brain candidate of this generation.
        gen_rng = np.random.default_rng(BASE_SEED + self.outer_gen)
        self.gen_waypoints = sample_waypoints(gen_rng, n=NUM_WAYPOINTS)

        wp_str = "  ".join(f"({w[0]:.1f},{w[1]:.1f})" for w in self.gen_waypoints)
        console.rule(f"[bold magenta]Outer gen {self.outer_gen} — waypoints {wp_str}")

        to_eval = [
            ind for ind in population
            if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]

        assert self.executor is not None, "evolve() must create the executor before evaluate()"

        if STRATEGY == "across":
            # Strategy B: each worker trains one body serially. Bodies are
            # processed in waves of size BRAIN_WORKERS; each call to
            # executor.map fans (μ+λ) bodies across the pool.
            tasks = [
                {
                    "body_hash":     _genome_hash(ind.genotype["morph"]),
                    "genome_dict":   ind.genotype["morph"],
                    "waypoints":     self.gen_waypoints,
                    "rng_seed":      BASE_SEED + 1000 * self.outer_gen + idx,
                    "parent_morph":  ind.tags.get("parent_morph"),
                    "parent_weights": ind.tags.get("parent_weights") or [],
                }
                for idx, ind in enumerate(to_eval)
            ]
            t0 = time.time()
            results = list(self.executor.map(_train_body_serial, tasks))
            wave_elapsed = time.time() - t0

            warm_summary: list[str] = []
            for idx, (ind, res) in enumerate(zip(to_eval, results)):
                best_fit = float(res["fitness"])
                best_w_list = res["weights"]
                best_w = np.array(best_w_list) if best_w_list is not None else None

                ind.fitness = best_fit if np.isfinite(best_fit) else float("inf")
                ind.tags = {"best_brain": best_w_list if best_w_list is not None else []}
                ind.requires_eval = False

                ws = res.get("warm_start") or {}
                if not ws.get("fallback", True) and ws.get("total", 0) > 0:
                    warm_summary.append(f"{idx}:{ws['matched']}/{ws['total']}")

                if best_fit < self.best_seen_fitness and best_w is not None:
                    self.best_seen_fitness = best_fit
                    self.best_seen_genotype = copy.deepcopy(ind.genotype["morph"])
                    self.best_seen_weights = best_w.copy()
                    self.best_seen_waypoints = list(self.gen_waypoints)
                    self._save_checkpoint(tag=f"gen{self.outer_gen:03d}_body{idx:02d}")

            finite = [r["fitness"] for r in results if np.isfinite(r["fitness"])]
            stats = (
                f"min={np.min(finite):.3f}  avg={np.mean(finite):.3f}  max={np.max(finite):.3f}"
                if finite else "all-infinite"
            )
            console.log(
                f"  wave: {len(to_eval)} bodies in {wave_elapsed:.1f}s  ({wave_elapsed / max(len(to_eval), 1):.1f}s/body)  {stats}"
            )
            if warm_summary:
                console.log(f"  warm-start matches (idx:matched/total): {' '.join(warm_summary)}")

        else:  # STRATEGY == "within"
            # Strategy A: parallelise inner-CMA candidates of one body across
            # workers; bodies trained one at a time.
            for idx, ind in enumerate(to_eval):
                t0 = time.time()
                best_fit, best_w, warm_info = learn_brain(
                    ind.genotype["morph"],
                    self.gen_waypoints,
                    self.executor,
                    parent_morph=ind.tags.get("parent_morph"),
                    parent_weights=ind.tags.get("parent_weights") or None,
                    rng_seed=BASE_SEED + 1000 * self.outer_gen + idx,
                )
                elapsed = time.time() - t0

                if not np.isfinite(best_fit):
                    ind.fitness = float("inf")
                else:
                    ind.fitness = best_fit
                ind.tags = {"best_brain": best_w.tolist() if best_w is not None else []}
                ind.requires_eval = False

                warm_str = (
                    f"  warm={warm_info['matched']}/{warm_info['total']}"
                    if not warm_info.get("fallback", True) and warm_info.get("total", 0) > 0
                    else ""
                )
                console.log(
                    f"  body {idx + 1}/{len(to_eval)}  fitness={best_fit:.3f}  "
                    f"params={len(best_w) if best_w is not None else 0}  ({elapsed:.1f}s){warm_str}"
                )

                if best_fit < self.best_seen_fitness and best_w is not None:
                    self.best_seen_fitness = best_fit
                    self.best_seen_genotype = copy.deepcopy(ind.genotype["morph"])
                    self.best_seen_weights = best_w.copy()
                    self.best_seen_waypoints = list(self.gen_waypoints)
                    self._save_checkpoint(tag=f"gen{self.outer_gen:03d}_body{idx:02d}")

        self.outer_gen += 1
        return population

    def survivor_selection(self, population: Population) -> Population:
        # (mu+lambda): keep best mu of (mu+lambda).
        population = population.sort(sort="min", attribute="fitness_")
        survivors = population[:MU]
        for ind in population:
            if ind not in survivors:
                ind.alive = False

        finite = [ind.fitness_ for ind in survivors if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
        if finite:
            console.log(
                "[green]Survivors:[/green] "
                f"avg={np.mean(finite):.3f}  min={np.min(finite):.3f}  max={np.max(finite):.3f}"
            )
        return population

    # -- Checkpointing --------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        if self.best_seen_weights is None or self.best_seen_genotype is None:
            return
        sub = CHECKPOINTS / tag
        sub.mkdir(exist_ok=True, parents=True)
        np.save(sub / "best_weights.npy", self.best_seen_weights)
        if self.best_seen_waypoints is not None:
            np.save(sub / "best_waypoints.npy", np.array(self.best_seen_waypoints))
        with (sub / "best_genome.json").open("w") as fh:
            json.dump(self.best_seen_genotype, fh, indent=2)
        console.log(f"  [cyan]checkpoint → {sub}  fitness={self.best_seen_fitness:.3f}[/cyan]")

    # -- Main entry -----------------------------------------------------------

    def evolve(self) -> Optional[Individual]:
        console.log("[yellow]Initialising population...[/yellow]")
        population = Population([_create_individual() for _ in range(MU)])

        with ProcessPoolExecutor(
            max_workers=BRAIN_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(BASE_SEED,),
        ) as executor:
            self.executor = executor

            # Evaluate initial population.
            population = self.evaluate(population)

            ops = [
                EAOperation(self.parent_selection),
                EAOperation(self.reproduction),
                EAOperation(self.evaluate),
                EAOperation(self.survivor_selection),
            ]
            ea = EA(
                population,
                operations=ops,
                num_steps=BUDGET,
                db_file_path=self.config.db_file_path,
                db_handling=self.config.db_handling,
                quiet=self.config.quiet,
            )
            ea.run()

            self.executor = None
            return ea.get_solution("best", only_alive=False)


# ── Best-of-run video (single worker, master process) ─────────────────────────

def record_best_video(evo: BodyBrainEvolution) -> None:
    if evo.best_seen_genotype is None or evo.best_seen_weights is None or evo.best_seen_waypoints is None:
        console.log("[red]No best-seen individual — skipping video.[/red]")
        return

    model, data, target_mocap_id, cam_name = _build_world_for_body(evo.best_seen_genotype)
    input_dim = _genome_input_dim(model, data)
    net = Network(input_size=input_dim, output_size=model.nu)
    fill_parameters(net, evo.best_seen_weights)

    waypoints = list(evo.best_seen_waypoints)
    num_wps = len(waypoints)
    wp_idx = 0
    current_target = waypoints[0]
    mujoco.mj_resetData(model, data)
    data.mocap_pos[target_mocap_id] = current_target

    videos_dir = DATA / "videos"
    videos_dir.mkdir(exist_ok=True)
    video_recorder = VideoRecorder(file_name="best_body_brain", output_folder=str(videos_dir))

    fps = 30
    dt = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
    control_step_freq = 50
    current_ctrl = np.zeros(model.nu)
    render_step = 0

    control_renderer = mujoco.Renderer(model, height=24 * 4, width=32 * 4)

    def get_ctrl(d: mujoco.MjData) -> np.ndarray:
        control_renderer.update_scene(d, camera=cam_name)
        img    = control_renderer.render()
        vision = analyze_sections(isolate_green(img))
        rs     = get_robot_state(d)
        phase  = [2.0 * np.sin(d.time * 2.0 * np.pi), 2.0 * np.cos(d.time * 2.0 * np.pi)]
        prog   = [wp_idx / max(num_wps - 1, 1)]
        state  = np.concatenate([rs, vision, phase, prog]).astype(np.float32)
        return net.forward(model, d, state)

    try:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview_cam")
    except Exception:
        camera_id = -1

    with mujoco.Renderer(model, height=480, width=640) as renderer:
        while data.time < DURATION and wp_idx < num_wps:
            for _ in range(steps_per_frame):
                if render_step % control_step_freq == 0:
                    current_ctrl = get_ctrl(data)
                np.copyto(data.ctrl, current_ctrl)
                mujoco.mj_step(model, data)
                render_step += 1

                if wp_idx < num_wps:
                    dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
                    if dist <= REACH_RADIUS:
                        wp_idx += 1
                        if wp_idx < num_wps:
                            current_target = waypoints[wp_idx]
                            data.mocap_pos[target_mocap_id] = current_target

            renderer.update_scene(data, camera=camera_id)
            video_recorder.write(renderer.render())

    video_recorder.release()
    control_renderer.close()
    console.log(f"[green]Video → {videos_dir}/best_body_brain.mp4[/green]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    console.rule("[bold magenta]Body+Brain Co-evolution — Randomised Waypoints[/bold magenta]")
    console.log(
        f"outer: (mu+lambda) = ({MU}+{LAM}) over {BUDGET} gens   "
        f"inner: CMA-ES popsize={BRAIN_POP} budget={BRAIN_BUDGET} workers={BRAIN_WORKERS}   "
        f"strategy={STRATEGY}"
    )
    console.log(
        f"task: waypoints/ep={NUM_WAYPOINTS}  arena={ARENA_RADIUS}m  "
        f"reach={REACH_RADIUS}m  duration={DURATION}s"
    )

    start = time.time()
    evo = BodyBrainEvolution()
    best = evo.evolve()
    elapsed = time.time() - start

    console.rule("[bold green]Final[/bold green]")
    if best is not None and best.fitness_ is not None:
        console.log(f"DB best fitness: {best.fitness:.3f}")
    console.log(f"Best-seen fitness across run: {evo.best_seen_fitness:.3f}")
    console.log(f"Elapsed: {elapsed / 60:.1f} min")

    if not args.no_video:
        record_best_video(evo)


if __name__ == "__main__":
    main()
    gc.disable()
