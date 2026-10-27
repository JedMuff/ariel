"""
Shared helpers for the symmetry-pressure investigation.

Adapted from examples/re_book/6_body_brain_randomized_waypoints.py with
additions for extended data logging, NSGA-II selection, and repeat evaluations.
"""

import contextlib
import copy
import hashlib
import itertools
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
from collections.abc import Sequence
from typing import Any, Callable, Literal, Optional, cast

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
from ariel.ec.genotypes.tree.collision import (
    prune_colliding_subtrees,
    prune_colliding_subtrees_symmetric,
)
from ariel.ec.genotypes.tree.symmetry import MirrorAxis, symmetrize_genome
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.tying import crossover_tied, mutate_morph_tied, random_tree_tied
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


# ── Brains ────────────────────────────────────────────────────────────────────
#
# Every training/replay loop drives a brain through the same two calls:
# set_params(flat CMA vector) once, then act(data, state) every control step.
# The returned raw action is then alpha-blended and clipped by the caller, so
# the post-processing (and jerk penalty) is identical across brain types.

BrainKind = Literal["ann", "cpg", "revolve_cpg", "dann"]
BRAIN_KINDS: tuple[str, ...] = ("ann", "cpg", "revolve_cpg", "dann")
DANN_VARIANT = "dann_pos_hinge_h16"  # zoo_benchmark/brains.py DANN_VARIANTS: the default distributed ANN

CPG_F_MIN = 0.5  # Hz
CPG_F_MAX = 3.0  # Hz


class AnnBrain:
    """Closed-loop MLP on proprioceptive state (the original controller)."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.network    = Network(input_size=input_dim, output_size=output_dim)
        self.num_params = sum(p.numel() for p in self.network.parameters())

    def set_params(self, vec: np.ndarray) -> None:
        fill_parameters(self.network, np.asarray(vec, dtype=np.float32))

    def act(self, data: Any, state: np.ndarray) -> np.ndarray:
        return self.network.forward(None, data, state)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CpgBrain:
    """Open-loop CPG: one sine oscillator per joint sharing a single frequency.

    Parameter layout (3n+1): [freq, amp_0..n-1, phase_0..n-1, offset_0..n-1],
    each squashed into range so CMA's unbounded search space stays valid:
      f        = CPG_F_MIN + (CPG_F_MAX - CPG_F_MIN) * sigmoid(p_freq)
      amp_i    = (pi/2) * sigmoid(p_amp_i)
      phase_i  = pi * p_phase_i
      offset_i = (pi/2) * tanh(p_offset_i)
    Time is measured from the end of the settling phase, so every episode
    starts at phase 0 and the brain is stateless/deterministic. State input
    is ignored.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:  # noqa: ARG002
        self.n          = output_dim
        self.num_params = 3 * output_dim + 1
        self.set_params(np.zeros(self.num_params))

    def set_params(self, vec: np.ndarray) -> None:
        p = np.asarray(vec, dtype=np.float64)
        if len(p) != self.num_params:
            raise IndexError("Parameter vector length mismatch")
        n = self.n
        self.freq   = CPG_F_MIN + (CPG_F_MAX - CPG_F_MIN) * float(_sigmoid(p[0]))
        self.amp    = (math.pi / 2) * _sigmoid(p[1 : 1 + n])
        self.phase  = math.pi * p[1 + n : 1 + 2 * n]
        self.offset = (math.pi / 2) * np.tanh(p[1 + 2 * n : 1 + 3 * n])

    def output_at(self, t: float) -> np.ndarray:
        out = self.offset + self.amp * np.sin(2 * math.pi * self.freq * t + self.phase)
        return np.clip(out, -math.pi / 2, math.pi / 2)

    def act(self, data: Any, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self.output_at(float(data.time) - SETTLE_DURATION)


class RevolveCpgBrain:
    """Port of apets-ariel's RevolveCPG ("copied from revolve but fully connected").

    One (x, y) oscillator pair per joint. Weights (n + n(n-1)/2): n
    self-couplings x_i<->y_i followed by the couplings x_i<->x_j for i < j,
    placed antisymmetrically in a 2n x 2n matrix A. The state follows
    dS/dt = A S, advanced by one RK4 step over the time elapsed since the
    previous call and clipped to [-1, 1]; the x half, times pi/2, is the
    joint targets. So unlike CpgBrain the stride changes the integration
    step, as it does in revolve. State input is ignored.

    The oscillator is stateful: set_params and reset restore the initial
    state, with time measured from the end of the settling phase. Every
    caller decodes a fresh candidate (set_params / make_brain_for_model with
    params) before each episode, so episodes always start from that state.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:  # noqa: ARG002
        self.n          = output_dim
        self.num_params = output_dim + output_dim * (output_dim - 1) // 2
        self._pairs     = list(itertools.combinations(range(output_dim), 2))
        self._initial_state = (
            np.hstack([np.full(output_dim, 1.0), np.full(output_dim, -1.0)]) * 0.5 * np.sqrt(2)
        )
        self.set_params(np.zeros(self.num_params))

    def set_params(self, vec: np.ndarray) -> None:
        w = np.asarray(vec, dtype=np.float64)
        if len(w) != self.num_params:
            raise IndexError("Parameter vector length mismatch")
        n = self.n
        m = np.zeros((2 * n, 2 * n))
        for i in range(n):
            m[i, n + i] = +w[i]
            m[n + i, i] = -w[i]
        for (i, j), wij in zip(self._pairs, w[n:], strict=True):
            m[i, j] = +wij
            m[j, i] = -wij
        self._a = m
        self.reset()

    def reset(self) -> None:
        self._state = self._initial_state.copy()
        self._time  = 0.0

    def output_at(self, t: float) -> np.ndarray:
        """Advance the oscillator to time t (one RK4 step) and return the targets."""
        dt = t - self._time
        a, s = self._a, self._state
        a1 = a @ s
        a2 = a @ (s + dt / 2 * a1)
        a3 = a @ (s + dt / 2 * a2)
        a4 = a @ (s + dt * a3)
        self._state = np.clip(s + dt / 6 * (a1 + 2 * (a2 + a3) + a4), -1, 1)
        self._time  = t
        return self._state[: self.n] * (math.pi / 2)

    def act(self, data: Any, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self.output_at(float(data.time) - SETTLE_DURATION)


class DannBrain:
    """The distributed ANN (zoo_benchmark/brains.py ZooDistributedAnnV2, variant
    DANN_VARIANT): one 42-16-1 MLP (705 params) shared by every hinge, whose
    inputs are the hinge's own vel/angle, its nearest hinges slotted by
    direction, its rest position, the core orientation and a sine clock.

    The body topology is read from the compiled model's body names
    (brains.model_topology), so it works for any robogen body. The clock is a
    control-step counter, so set_params and reset restart it; callers decode a
    fresh candidate before every episode. Joint velocities come from the
    MjData the brain was built with, so act() must be called on that data.
    extra_inputs > 0 adds that many inputs broadcast to every hinge, given as
    act(..., extra=...) (gecko_food_dann.py's camera sections). topology
    overrides brains.model_topology(model), for bodies not named the robogen
    way (zoo canonical bodies: brains.body_topology(body, parent_slot="back")).
    tie_choice resolves hinges tied for a neighbour slot the way an older run
    did (brains._hinge_neighbour_slots); neighbour_ties lists those ties.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, extra_inputs: int = 0,
                 topology: Optional[dict[str, Any]] = None,
                 tie_choice: Optional[Sequence[int]] = None) -> None:
        import sys
        zoo_dir = str(Path(__file__).resolve().parent.parent / "zoo_benchmark")
        if zoo_dir not in sys.path:
            sys.path.insert(0, zoo_dir)
        import brains  # lazy: brains.py imports this module

        self._net = brains.ZooDistributedAnnV2(
            topology or brains.model_topology(model), model, data, **brains.DANN_VARIANTS[DANN_VARIANT],
            extra_inputs=extra_inputs, tie_choice=tie_choice,
        )
        self.num_params = self._net.num_params
        self.neighbour_ties = self._net.neighbour_ties

    def set_params(self, vec: np.ndarray) -> None:
        self._net.set_params(vec)
        self._net.reset()

    def reset(self) -> None:
        self._net.reset()

    def act(self, data: Any, state: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
        return self._net.act(float(data.time) - SETTLE_DURATION, state, extra)


Brain = "AnnBrain | CpgBrain | RevolveCpgBrain | DannBrain"


def make_brain(kind: str, input_dim: int, output_dim: int) -> Brain:
    if kind == "ann":
        return AnnBrain(input_dim, output_dim)
    if kind == "cpg":
        return CpgBrain(input_dim, output_dim)
    if kind == "revolve_cpg":
        return RevolveCpgBrain(input_dim, output_dim)
    if kind == "dann":
        raise ValueError("dann needs the body: use make_brain_for_model")
    raise ValueError(f"Unknown brain kind: {kind!r} (expected one of {BRAIN_KINDS})")


def make_brain_for_model(kind: str, model: mujoco.MjModel, data: mujoco.MjData,
                         params: Optional[np.ndarray] = None,
                         tie_choice: Optional[Sequence[int]] = None) -> Brain:
    """Build a brain sized for this body (proprioceptive input, one output per
    actuator). tie_choice: see DannBrain."""
    if kind == "dann":
        brain = DannBrain(model, data, tie_choice=tie_choice)
    else:
        brain = make_brain(kind, len(get_robot_state(data)), model.nu)
    if params is not None:
        brain.set_params(params)
    return brain


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


# kgd's zoo modules (examples/zoo_benchmark/zoo_modules.py; test_zoo_modules.py
# keeps these in sync). Only the values differ: the local module classes, and
# so their face layout (incl. the brick TOP/BOTTOM faces kgd's lack), are kept,
# and every site position scales with the dimensions.
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
    """Build modules with kgd's zoo sizes and masses inside this block; the
    previous values (local ones, or zoo ones after use_zoo_modules_globally)
    are restored on exit."""
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


def apply_zoo_modules(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Give a robot spec kgd's zoo hinge actuator (kp, kv, armature, a +-90 deg
    joint limit), in place. Hard-coded in hinge.py, so set after construction."""
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


def zoo_modules_spec(genome_dict: dict, to_spec_fn=genome_to_spec) -> Optional[mujoco.MjSpec]:
    """to_spec_fn(genome_dict) built with kgd's zoo modules: sizes and masses
    (zoo_module_constants) and hinge actuator (apply_zoo_modules). Picklable
    via with_zoo_modules."""
    with zoo_module_constants():
        spec = to_spec_fn(genome_dict)
    return apply_zoo_modules(spec) if spec is not None else None


def with_zoo_modules(to_spec_fn=genome_to_spec):
    """A to_spec_fn that builds bodies with kgd's zoo modules (--zoo-modules)."""
    import functools  # noqa: PLC0415
    return functools.partial(zoo_modules_spec, to_spec_fn=to_spec_fn)


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
    # Point/rotation changes and grafted subtrees can make parts overlap;
    # drop any subtree that now collides (FCL, real module geometry).
    prune_colliding_subtrees(new)
    with contextlib.suppress(ValueError):
        validate_genome_dict(new.to_dict())
    return new


def mutate_morph_symmetric(
    genome: TreeGenome, rng: np.random.Generator, num_modules: int, axis: MirrorAxis
) -> TreeGenome:
    """Like `mutate_morph`, but guarantees the result stays symmetric about ``axis``.

    Applies the ordinary (potentially symmetry-breaking) mutation, then
    re-symmetrizes — reusing all four existing mutation operators unchanged.
    Assumes ``genome`` is already symmetric about ``axis``. Mirroring can
    create new overlaps (an arm hitting its own mirror image across the
    plane), so colliding subtrees are pruned again, pairwise with their
    mirror images to keep the genome symmetric.
    """
    mutated = mutate_morph(genome, rng, num_modules)
    symmetric = symmetrize_genome(mutated, axis)
    prune_colliding_subtrees_symmetric(symmetric, axis)
    return symmetric


def joint_count(genome: TreeGenome) -> int:
    """Count hinge joints by walking the genome graph — no MuJoCo compilation needed."""
    return sum(1 for d in genome.nodes.values() if d.get("type") == "HINGE")


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
HEIGHT_PENALTY_THRESHOLD  = 0.21
JERK_PENALTY_WEIGHT       = 3.0   # penalty per unit of mean absolute ctrl delta, above JERK_THRESHOLD
JERK_THRESHOLD            = 0.15  # hurdle: mean_jerk below this is free
ROWING_RECORD_EVERY_N     = 10    # rowing penalty: body poses sampled every N sim steps (50Hz, as behavior_50hz)


@dataclass(frozen=True)
class EvalProfile:
    """Episode, penalty and inner CMA-ES settings for skill training.

    SWEEP_PROFILE is this module's constants (the original symmetry-pressure
    sweep). DSOCIAL_ZOO matches /home/jed/workspaces/ariel's
    examples/d_social_learning: sim_ariel/evaluator.py's "zoo" preset (zoo
    modules, 15 s translate episodes at 20 Hz control (turns: 5 s), alpha and jerk constants rescaled
    from the 9-step to the 25-step control period, spawn-height penalty above
    0.5 m) and its inner CMA-ES from x0 = 0 with sigma 0.5 (experiment.py),
    10 generations of popsize 20 (run_social_zoo_budget.sh's ig10 budget). It also normalises every skill
    (translate_score to [0, 2], turn_score to [0, 1]) before the penalties are subtracted.
    """

    loco_duration: float = LOCO_DURATION
    turn_duration: float = TURN_DURATION
    control_step_freq: int = CONTROL_STEP_FREQ
    skill_budget: int = 100
    zoo_modules: bool = False
    ctrl_alpha: float = CTRL_ALPHA
    jerk_weight: float = JERK_PENALTY_WEIGHT
    jerk_threshold: float = JERK_THRESHOLD
    height_threshold: float = HEIGHT_PENALTY_THRESHOLD
    height_at_spawn: bool = False   # core height at reset (d_social) instead of after settling
    cma_sigma: float = CMA_SIGMA
    cma_popsize: int = CMA_POPSIZE
    cma_init_scale: float = CMA_INIT_SCALE  # x0 ~ U(-s, s); 0 = start from zeros
    normalised: bool = False


_DSOCIAL_STEP_RATIO = 25 / 9  # zoo control period / legacy control period
SWEEP_PROFILE = EvalProfile()
DSOCIAL_ZOO = EvalProfile(
    loco_duration=15.0, turn_duration=5.0, control_step_freq=25, skill_budget=10, zoo_modules=True,
    ctrl_alpha=1.0 - (1.0 - CTRL_ALPHA) ** _DSOCIAL_STEP_RATIO,
    jerk_weight=JERK_PENALTY_WEIGHT / _DSOCIAL_STEP_RATIO,
    jerk_threshold=JERK_THRESHOLD * _DSOCIAL_STEP_RATIO,
    height_threshold=0.5, height_at_spawn=True,
    cma_sigma=0.5, cma_popsize=20, cma_init_scale=0.0,
    normalised=True,
)
EVAL_PROFILES = {"sweep": SWEEP_PROFILE, "dsocial_zoo": DSOCIAL_ZOO}

TRANSLATE_REF_SPEED = 0.5        # m/s; a normalised translate skill scores TRANSLATE_MAX at this mean speed
TRANSLATE_MAX       = 2.0        # normalised translate score cap
TURN_TARGET         = 3 * math.pi  # rad (540 deg) of net yaw for a normalised turn skill to score 1 (ends the episode)


def translate_score(disp: float, duration: float) -> float:
    """Normalised translate skill: TRANSLATE_MAX x displacement along the
    reward axis over TRANSLATE_REF_SPEED * duration, clipped to
    [0, TRANSLATE_MAX] (0.5 m/s or faster scores 2)."""
    return float(np.clip(TRANSLATE_MAX * disp / (TRANSLATE_REF_SPEED * duration), 0.0, TRANSLATE_MAX))


def turn_score(net_yaw: float) -> float:
    """Normalised turn skill: net signed yaw in the target direction over
    TURN_TARGET (540 degrees), clipped to [0, 1]."""
    return float(np.clip(net_yaw / TURN_TARGET, 0.0, 1.0))

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


def robot_bodies(model: mujoco.MjModel) -> list[int]:
    """Robot bodies that own at least one geom (core, stators, rotors, bricks)."""
    has_geom = set(int(b) for b in model.geom_bodyid)
    return [
        b for b in range(model.nbody)
        if model.body(b).name.startswith("robot1_") and b in has_geom
    ]


def rowing_scaled_disp(disp: float, rowing_frac: float) -> float:
    """Displacement after the rowing penalty: a forward displacement loses the
    rowing share of the stroke (rowing_detect's rowing_frac, in [0, 1]); a
    backward one is left alone, so scaling it never helps."""
    return disp * (1.0 - rowing_frac) if disp > 0 else disp


def rowing_info(poses: list[np.ndarray], quats: list[np.ndarray], dt: float) -> dict[str, Any]:
    """rowing_detect's (rowing_frac, rowing_label) of post-settle body poses
    sampled every dt s, core first. A detector failure counts as no rowing."""
    from rowing_detect import analyse_episode  # lazy: pulls in zoo_benchmark/gait_similarity

    try:
        ind, _ = analyse_episode(np.asarray(poses), np.asarray(quats), dt)
        return {"rowing_frac": float(ind["rowing_frac"]), "rowing_label": str(ind["rowing_label"])}
    except Exception as exc:  # noqa: BLE001
        print(f"rowing_detect failed ({exc!r}); no rowing penalty", flush=True)
        return {"rowing_frac": 0.0, "rowing_label": "error"}


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


def _skill_worker_init(
    seed: int, reward: SkillReward, genome_dict: dict, to_spec_fn, control_step_freq: int,
    brain_kind: str = "ann",
    rowing_penalty: bool = False,
    profile: EvalProfile = SWEEP_PROFILE,
) -> None:
    global _skill_worker_ctx  # noqa: PLW0603
    torch.set_num_threads(1)
    np.random.seed((seed + os.getpid()) % (2**32 - 1))
    model, data = build_loco_world_for_body(genome_dict, to_spec_fn)
    brain       = make_brain_for_model(brain_kind, model, data)
    _skill_worker_ctx = {
        "model": model, "data": data, "brain": brain, "reward": reward,
        "control_step_freq": control_step_freq,
        "rowing_bodies": robot_bodies(model) if rowing_penalty else None,
        "profile": profile,
    }


def _skill_worker_eval(weights_list: list[float]) -> tuple[float, dict[str, Any]]:
    """run_skill_episode for one candidate in a skill-training worker."""
    assert _skill_worker_ctx is not None
    ctx = _skill_worker_ctx
    ctx["brain"].set_params(np.array(weights_list, dtype=np.float32))
    return run_skill_episode(ctx["model"], ctx["data"], ctx["brain"], ctx["reward"],
                             ctx["control_step_freq"], ctx["profile"], ctx["rowing_bodies"])


def run_skill_episode(
    model: mujoco.MjModel, data: mujoco.MjData, brain: Any, reward: SkillReward,
    control_step_freq: int, profile: EvalProfile = SWEEP_PROFILE,
    rowing_bodies: Optional[list[int]] = None,
    recorder: Optional[Callable[[mujoco.MjData, float], None]] = None,
    on_settle_step: Optional[Callable[[mujoco.MjData], None]] = None,
    on_control: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], None]] = None,
) -> tuple[float, dict[str, Any]]:
    """(fitness, info) of one skill episode of a brain whose params are set.
    info holds raw_score (displacement or accumulated yaw), score (the
    normalised skill score, with a normalised profile) and, with
    rowing_bodies given (the rowing penalty), rowing_frac and rowing_label.

    With a normalised profile a turn accumulates net signed yaw (turning back
    cancels) and the episode ends once it reaches TURN_TARGET; otherwise only
    yaw in the target direction counts and the episode always runs its full
    duration. recorder, if given, is called as recorder(data, raw_so_far)
    after every post-settle physics step (videos); on_settle_step(data) after
    every settle step, and on_control(state, raw_action, applied_action) at
    every control update (behavior_common's recording replay). None of them
    may change data.
    """
    if hasattr(brain, "reset"):
        brain.reset()
    mujoco.mj_resetData(model, data)
    spawn_height = float(data.qpos[2])

    sim_step = 0
    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)
        if on_settle_step is not None:
            on_settle_step(data)
        sim_step += 1

    core_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    initial_height = spawn_height if profile.height_at_spawn else float(data.xpos[core_id, 2])
    step           = 0
    current_action = np.zeros(model.nu)
    prev_ctrl: Optional[np.ndarray] = None
    jerk_sum       = 0.0
    ctrl_step      = 0
    poses: list[np.ndarray] = []
    quats: list[np.ndarray] = []

    if reward.kind == "translate":
        xy0           = np.array([data.qpos[0], data.qpos[1]])
        reward_axis   = _rotate_2d(FORWARD_AXIS, reward.angle_deg)
    else:
        r_prev      = np.array(data.xmat[core_id]).reshape(3, 3).copy()
        accumulated = 0.0

    def raw_score() -> float:
        if reward.kind == "translate":
            return float(np.dot(np.array([data.qpos[0], data.qpos[1]]) - xy0, reward_axis))
        return accumulated

    episode_end = SETTLE_DURATION + reward.duration
    while data.time < episode_end:
        if step % control_step_freq == 0:
            state      = get_robot_state(data).astype(np.float32)
            raw_action = brain.act(data, state)
            current_action = np.clip(
                current_action * (1.0 - profile.ctrl_alpha) + raw_action * profile.ctrl_alpha,
                -math.pi / 2, math.pi / 2,
            )
            if prev_ctrl is not None:
                jerk_sum += float(np.mean(np.abs(current_action - prev_ctrl)))
            prev_ctrl = current_action.copy()
            if on_control is not None:
                on_control(state, raw_action, current_action)
            ctrl_step += 1
        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

        if reward.kind == "rotate":
            r_curr = np.array(data.xmat[core_id]).reshape(3, 3)
            delta  = signed_vertical_yaw_delta(r_prev, r_curr) * reward.turn_sign
            accumulated += delta if profile.normalised else max(0.0, delta)
            r_prev = r_curr.copy()

        # Same sampling as behavior_common.replay_skill_episode's pose rows,
        # whose 50Hz replays the rowing detector was validated on.
        if rowing_bodies is not None and sim_step % ROWING_RECORD_EVERY_N == 0:
            poses.append(data.xpos[rowing_bodies].copy())
            quats.append(data.xquat[rowing_bodies].copy())
        if recorder is not None:
            recorder(data, raw_score())

        step += 1
        sim_step += 1
        if reward.kind == "rotate" and profile.normalised and accumulated >= TURN_TARGET:
            break  # 540 deg: the score is capped at 1, so stop early

    mean_jerk    = jerk_sum / max(ctrl_step - 1, 1)
    jerk_penalty = profile.jerk_weight * mean_jerk if mean_jerk >= profile.jerk_threshold else 0.0

    height_penalty = initial_height if initial_height > profile.height_threshold else 0.0
    raw = raw_score()
    info: dict[str, Any] = {"raw_score": raw, "height_penalty": height_penalty, "jerk_penalty": jerk_penalty}
    score = raw
    if rowing_bodies is not None:
        info.update(rowing_info(poses, quats, ROWING_RECORD_EVERY_N * float(model.opt.timestep)))
        score = rowing_scaled_disp(raw, info["rowing_frac"])
    if profile.normalised:
        score = translate_score(score, reward.duration) if reward.kind == "translate" else turn_score(score)
        info["score"] = score
    return -(score - height_penalty - jerk_penalty), info


def train_skill_for_body(
    reward: SkillReward,
    genome_dict: dict,
    budget: int,
    workers: int,
    seed: int,
    to_spec_fn=genome_to_spec,
    control_step_freq: int = CONTROL_STEP_FREQ,
    brain_kind: str = "ann",
    rowing_penalty: bool = False,
    profile: EvalProfile = SWEEP_PROFILE,
) -> tuple[np.ndarray, list[float], float, dict[str, Any]]:
    """Train one skill via CMA-ES for a specific body.

    control_step_freq is the number of physics steps between controller
    updates (see CONTROL_STEP_FREQ). brain_kind selects the controller
    (see make_brain); the returned vector is that brain's flat parameters.
    rowing_penalty scales a forward displacement by 1 - rowing_frac (see
    rowing_scaled_disp), the share of the stroke rowing_detect calls rowing.
    profile sets the penalties, CMA-ES settings and score normalisation (see
    EvalProfile); durations, control_step_freq and budget are still taken
    from the arguments.

    Returns (best_weights, learning_curve, eval_time_s, best_info), best_info
    being _skill_worker_eval's info for best_weights.
    """
    t_start = time.perf_counter()

    model, data = build_loco_world_for_body(genome_dict, to_spec_fn)
    num_params  = make_brain_for_model(brain_kind, model, data).num_params

    rng = np.random.default_rng(seed)
    if profile.cma_init_scale > 0:
        x0 = rng.uniform(-profile.cma_init_scale, profile.cma_init_scale, size=num_params).tolist()
    else:
        x0 = [0.0] * num_params

    es = cma.CMAEvolutionStrategy(
        x0, profile.cma_sigma,
        {"popsize": profile.cma_popsize, "seed": seed, "verbose": -9, "maxiter": 10**9},
    )

    best_fit = float("inf")
    best_w   = np.array(x0, dtype=np.float32)
    best_info: dict[str, Any] = {}
    learning_curve: list[float] = []

    effective_workers = min(profile.cma_popsize, workers)
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=_skill_worker_init,
        initargs=(seed, reward, genome_dict, to_spec_fn, control_step_freq, brain_kind, rowing_penalty, profile),
    ) as pool:
        gen = 0
        while gen < budget:
            if es.stop():
                break

            solutions = es.ask()
            results   = list(pool.map(_skill_worker_eval, [s.tolist() for s in solutions]))
            fits      = [f for f, _ in results]
            es.tell(solutions, fits)

            gen_best = min(fits)
            if gen_best < best_fit:
                best_fit  = gen_best
                i_best    = fits.index(gen_best)
                best_w    = np.array(solutions[i_best], dtype=np.float32)
                best_info = results[i_best][1]
                learning_curve.append(best_fit)

            gen += 1

    return best_w, learning_curve, time.perf_counter() - t_start, best_info
