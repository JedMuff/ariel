"""Controllers compared in the zoo benchmark.

Every brain exposes the same interface so one CMA-ES setup drives them all:
  num_params          length of the flat (unbounded) parameter vector
  set_params(vec)     decode a CMA-ES candidate
  reset()             restore the initial internal state (call before each episode)
  act(t, state)       joint targets in ctrl units ([-pi/2, pi/2]), called at the
                      control frequency; t is sim time, state is
                      get_state_from_data(data) (only the ANN reads it)

Brains:
  ann          closed-loop MLP (shared.AnnBrain), the `ann` of run_control_stride_sweep.sh
  sine         open-loop sine CPG (shared.CpgBrain), the `cpg` of run_control_stride_sweep.sh
  revolve_cpg  kgd's fully connected RevolveCPG (shared.RevolveCpgBrain), the `revolve_cpg` of run_control_stride_sweep.sh
  matsuoka     Matsuoka oscillator network (morphlib@hyperneat matsuoka_brain/), bugs fixed
  square       eased square wave per hinge (square_wave.py), 12 genes per hinge
  square_sync  as square, but one frequency ModulatedValue shared by every hinge
  bang_bang    as square_sync, but fixed +-90 deg and instant switches (5 genes per hinge)
  distributed_ann  one MLP shared by every hinge on local observations (DistributedMLP
                   from the d_social_learning experiments); needs the body's topology,
                   so it is built by zoo_benchmark.make_brain_for(..., body=...)
  dann_*       improved distributed ANN variants (DANN_VARIANTS), built the same way;
               dann_pos_hinge_h16 is the default distributed ANN (labelled "distributed ANN"
               in figures), distributed_ann the original port it improves on
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Sequence
from typing import Any
from pathlib import Path

import mujoco
import numpy as np

import canonical_bodies
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.simulation.controllers.distributed_mlp import DistributedMLP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "symmetry_pressure"))
from shared import AnnBrain, CpgBrain, RevolveCpgBrain  # noqa: E402
from square_wave import ModulatedValue, PatternState, ServoPattern, step  # noqa: E402

BRAIN_KINDS: tuple[str, ...] = ("ann", "sine", "revolve_cpg", "matsuoka", "square", "square_sync", "bang_bang",
                                "distributed_ann", "dann_fixed", "dann_pos", "dann_phase", "dann_hinge",
                                "dann_pos_hinge", "dann_pos_hinge_h16")

HALF_PI = math.pi / 2


class ZooAnn:
    """shared.AnnBrain: tanh output already scaled to +-pi/2."""

    def __init__(self, n_inputs: int, n_hinges: int) -> None:
        self._brain = AnnBrain(n_inputs, n_hinges)
        self.num_params = self._brain.num_params

    def set_params(self, vec: np.ndarray) -> None:
        self._brain.set_params(vec)

    def reset(self) -> None:
        pass

    def act(self, t: float, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._brain.act(None, state.astype(np.float32))


class ZooSine:
    """shared.CpgBrain, timed from episode start (this protocol has no settle phase)."""

    def __init__(self, n_inputs: int, n_hinges: int) -> None:
        self._brain = CpgBrain(n_inputs, n_hinges)
        self.num_params = self._brain.num_params

    def set_params(self, vec: np.ndarray) -> None:
        self._brain.set_params(vec)

    def reset(self) -> None:
        pass

    def act(self, t: float, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._brain.output_at(t)


class RevolveCpg:
    """shared.RevolveCpgBrain, timed from episode start (this protocol has no settle phase)."""

    def __init__(self, n_inputs: int, n_hinges: int) -> None:
        self._brain = RevolveCpgBrain(n_inputs, n_hinges)
        self.num_params = self._brain.num_params

    def set_params(self, vec: np.ndarray) -> None:
        self._brain.set_params(vec)

    def reset(self) -> None:
        self._brain.reset()

    def act(self, t: float, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._brain.output_at(t)


# Gene ranges from morphlib matsuoka_brain/optimize.py (convert_cma_weights_to_params).
MATSUOKA_LIMITS = np.array([
    [1.0, 5.0],    # A     mutual inhibition between the two neurons
    [1.0, 10.0],   # b     self-inhibition (adaptation) weight
    [0.01, 0.2],   # tau_r neuron time constant
    [0.1, 2.0],    # tau_a adaptation time constant
    [0.1, 5.0],    # c     tonic input
])
MATSUOKA_COUPLING_LIMITS = (-1.0, 1.0)
MATSUOKA_INITIAL_STATE = (0.01, 0.008, 0.002, 0.006)  # xe, ye, xf, yf


def _to_range(g: np.ndarray, lo: float | np.ndarray, hi: float | np.ndarray) -> np.ndarray:
    return lo + (hi - lo) * (np.tanh(g) + 1) / 2


class Matsuoka:
    """Network of Matsuoka half-centre oscillators, one per hinge.

    Port of morphlib@hyperneat matsuoka_brain/matsuoka_oscillator.py, vectorised
    and generalised to n oscillators. Fixed relative to the original:
      - the output gain Oi was receiving xe (9 params into a 10-arg ctor), which
        scaled every output by 0.01; here Oi = 1;
      - StackedOscillatorNetwork.step integrated each oscillator twice per call;
        here each Euler step integrates once;
      - the hard-coded 6 oscillators.

    Genes: 5 per oscillator (A, b, tau_r, tau_a, c, in MATSUOKA_LIMITS) followed
    by n(n-1) off-diagonal coupling weights in [-1, 1]. Each gene is squashed
    from R with tanh, replacing the original's [-1, 1] CMA box bounds.

    The ODE is Euler-integrated at physics_dt (as the original did at the env
    timestep); act() advances it over the elapsed control period. Output is
    clip(ze - zf, -1, 1) * pi/2.
    """

    def __init__(self, n_inputs: int, n_hinges: int, physics_dt: float = 0.002) -> None:  # noqa: ARG002
        self.n = n_hinges
        self.physics_dt = physics_dt
        self.num_params = 5 * n_hinges + n_hinges * (n_hinges - 1)
        self._offdiag = ~np.eye(n_hinges, dtype=bool)
        self.set_params(np.zeros(self.num_params))

    def set_params(self, vec: np.ndarray) -> None:
        g = np.asarray(vec, dtype=np.float64)
        if len(g) != self.num_params:
            raise IndexError("Parameter vector length mismatch")
        n = self.n
        osc = _to_range(g[: 5 * n].reshape(n, 5), MATSUOKA_LIMITS[:, 0], MATSUOKA_LIMITS[:, 1])
        self.A, self.b, self.tau_r, self.tau_a, self.c = osc.T
        w = np.zeros((n, n))
        w[self._offdiag] = _to_range(g[5 * n :], *MATSUOKA_COUPLING_LIMITS)
        self.w = w
        self.reset()

    def reset(self) -> None:
        xe, ye, xf, yf = MATSUOKA_INITIAL_STATE
        self.xe = np.full(self.n, xe)
        self.ye = np.full(self.n, ye)
        self.xf = np.full(self.n, xf)
        self.yf = np.full(self.n, yf)
        self._time = 0.0

    def _output(self) -> np.ndarray:
        out = np.maximum(self.xe, 0) - np.maximum(self.xf, 0)
        out = np.where(np.isfinite(out), out, 0.0)
        return np.clip(out, -1, 1) * HALF_PI

    def _euler_step(self, dt: float) -> None:
        ze, zf = np.maximum(self.xe, 0), np.maximum(self.xf, 0)
        d_xe = (-self.xe - self.A * zf - self.b * self.ye - self.w @ self.ye + self.c) / self.tau_r
        d_ye = (ze - self.ye) / self.tau_a
        d_xf = (-self.xf - self.A * ze - self.b * self.yf - self.w @ self.yf + self.c) / self.tau_r
        d_yf = (zf - self.yf) / self.tau_a
        self.xe += d_xe * dt
        self.ye += d_ye * dt
        self.xf += d_xf * dt
        self.yf += d_yf * dt

    def act(self, t: float, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        steps = round((t - self._time) / self.physics_dt)
        for _ in range(steps):
            self._euler_step(self.physics_dt)
        self._time += steps * self.physics_dt
        return self._output()


# Gene ranges for the square-wave brains (each gene squashed from R). The
# ranges are centred so that CMA-ES's x0 = 0.5 starts near what the ANN
# champions converged to on every body: saturated bang-bang outputs at
# 0.7-1 Hz. A constant square wave fitted to those outputs and replayed
# open-loop reaches 82-100% of the ANN's speed. At x0 this gives f = 1.3 Hz,
# angles -68/+68 deg, a 0.16 s transition and wobble amplitudes about 8% of max.
# Phase is scaled down because the gait is far more sensitive to it than to any
# other gene: around the fitted gecko gait, gene noise of sd 0.05 on the phases
# alone (18 deg at scale 1) more than halves the speed, while every other group
# tolerates sd 0.2. At scale 1 CMA-ES's shared sigma is dominated by the
# phases; at 0.1 square_sync reached the ANN's 3k-eval gecko speed (pilot).
SQUARE_F_RANGE = (0.2, 2.0)             # Hz, frequency start/end
SQUARE_F_WOBBLE_AMP = 0.5               # Hz
SQUARE_DUTY_RANGE = (0.02, 0.98)        # duty start/end
SQUARE_DUTY_WOBBLE_AMP = 0.3
SQUARE_WOBBLE_RATE = 1.0                # Hz, both wobbles
SQUARE_WOBBLE_BIAS = 3.0                # wobble amp = max * sigmoid(g - bias): near off at x0
SQUARE_MAX_ANGLE = 90.0                 # deg, high = 90 tanh(gain g), low = -90 tanh(gain g)
SQUARE_ANGLE_GAIN = 2.0
SQUARE_MAX_TRANSITION = 0.25            # s
SQUARE_PHASE_SCALE = 0.1                # phase_offset = 0.1 g mod 1 (cycles), see above


def _sigmoid_range(g: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + (hi - lo) / (1 + np.exp(-g))


def _square_rhythm(g: np.ndarray, value_range: tuple[float, float], wobble_amp: float) -> ModulatedValue:
    """4 genes (start, end, wobble amp, wobble rate), each a scalar or a per-hinge array."""
    return ModulatedValue(
        start=_sigmoid_range(g[0], *value_range), end=_sigmoid_range(g[1], *value_range),
        wobble_amp=_sigmoid_range(g[2] - SQUARE_WOBBLE_BIAS, 0.0, wobble_amp),
        wobble_rate=_sigmoid_range(g[3], 0.0, SQUARE_WOBBLE_RATE),
    )


class _ZooSquareBase:
    """Eased square wave per hinge, driven by square_wave.step (freq_duty mode).

    The timeline is the episode (duration s, hold). Every gene is squashed into
    range (SQUARE_* constants); phase_offset is 0.1 x gene mod 1, and low_angle's
    gene is mirrored so equal genes give a full swing rather than no motion. The
    phase is integrated over the elapsed control period at each act() call. No
    slew limit.
    """

    def __init__(self, n_inputs: int, n_hinges: int, duration: float = 15.0) -> None:  # noqa: ARG002
        self.n = n_hinges
        self.duration = duration
        self.set_params(np.zeros(self.num_params))

    def _decode(self, g: np.ndarray) -> dict[str, Any]:
        """Return the ServoPattern fields a, b, low_angle, high_angle, phase_offset, transition_time."""
        raise NotImplementedError

    @staticmethod
    def _hinge_fields(freq: ModulatedValue, h: np.ndarray) -> dict[str, Any]:
        """Fields from a frequency and per-hinge genes (8, n): duty x4, low, high, phase, transition."""
        return {
            "a": freq, "b": _square_rhythm(h[0:4], SQUARE_DUTY_RANGE, SQUARE_DUTY_WOBBLE_AMP),
            "low_angle": -SQUARE_MAX_ANGLE * np.tanh(SQUARE_ANGLE_GAIN * h[4]),
            "high_angle": SQUARE_MAX_ANGLE * np.tanh(SQUARE_ANGLE_GAIN * h[5]),
            "phase_offset": np.mod(SQUARE_PHASE_SCALE * h[6], 1.0),
            "transition_time": _sigmoid_range(h[7], 0.0, SQUARE_MAX_TRANSITION),
        }

    def set_params(self, vec: np.ndarray) -> None:
        g = np.asarray(vec, dtype=np.float64)
        if len(g) != self.num_params:
            raise IndexError("Parameter vector length mismatch")
        self.pattern = ServoPattern(servo_id=tuple(range(self.n)), mode="freq_duty", **self._decode(g))
        self.reset()

    def reset(self) -> None:
        self._state = PatternState.initial(self.pattern)
        self._time = 0.0

    def act(self, t: float, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        self._state, angle = step(self.pattern, self._state, t, t - self._time, duration=self.duration)
        self._time = t
        return np.radians(angle)


class ZooSquare(_ZooSquareBase):
    """Full spec: 12 genes per hinge in square_wave.VECTOR_FIELDS order (own frequency per hinge)."""

    @property
    def num_params(self) -> int:
        return 12 * self.n

    def _decode(self, g: np.ndarray) -> dict[str, Any]:
        g = g.reshape(self.n, 12).T
        return self._hinge_fields(_square_rhythm(g[0:4], SQUARE_F_RANGE, SQUARE_F_WOBBLE_AMP), g[4:12])


class ZooSquareSync(_ZooSquareBase):
    """One frequency ModulatedValue shared by every hinge (4 genes), then 8 genes per
    hinge (duty x4, low, high, phase_offset, transition), so phase relations stay fixed."""

    @property
    def num_params(self) -> int:
        return 4 + 8 * self.n

    def _decode(self, g: np.ndarray) -> dict[str, Any]:
        freq = _square_rhythm(g[:4], SQUARE_F_RANGE, SQUARE_F_WOBBLE_AMP)
        return self._hinge_fields(freq, g[4:].reshape(self.n, 8).T)


class ZooBangBang(_ZooSquareBase):
    """square_sync restricted to bang-bang: every hinge switches instantly between
    -90 and +90 deg (transition_time 0), as the ANN champions do. One shared
    frequency ModulatedValue (4 genes), then 5 genes per hinge (duty x4, phase_offset)."""

    @property
    def num_params(self) -> int:
        return 4 + 5 * self.n

    def _decode(self, g: np.ndarray) -> dict[str, Any]:
        h = g[4:].reshape(self.n, 5).T
        return {
            "a": _square_rhythm(g[:4], SQUARE_F_RANGE, SQUARE_F_WOBBLE_AMP),
            "b": _square_rhythm(h[0:4], SQUARE_DUTY_RANGE, SQUARE_DUTY_WOBBLE_AMP),
            "low_angle": np.full(self.n, -SQUARE_MAX_ANGLE),
            "high_angle": np.full(self.n, SQUARE_MAX_ANGLE),
            "phase_offset": np.mod(SQUARE_PHASE_SCALE * h[4], 1.0),
            "transition_time": np.zeros(self.n),
        }


# One-hot index per module type, as ariel's MorphologyAdapter (0 = empty, 4 unused).
_DIST_TYPE_ONEHOT = {ModuleType.BRICK: 1, ModuleType.HINGE: 2, ModuleType.CORE: 3}
DIST_N_FACES = len(ModuleFaces)          # 6 neighbour slots per node
DIST_FEATURES = 8                        # vel, 0, angle / (pi/2), 5-slot type one-hot
DIST_HIDDEN = 32
DIST_TIME_PERIOD = 25                    # control steps per period of the sine time input


def body_topology(body: str, parent_slot: str = "attach_face") -> dict[str, Any]:
    """Module types and face-neighbour slots of a canonical body.

    The zoo bodies are built by canonical_bodies.Attacher, not from a robogen
    graph, so the parent/face/child edges are recorded by wrapping Attacher
    while the body is built. Slots follow MorphologyAdapter.from_graph exactly:
    a node's slot f holds its parent if the node is attached at the parent's
    face f, then each child attached at the node's own face f overwrites slot f
    (so a hinge on a parent's FRONT with a child on its own FRONT does not see
    its parent; kept for fidelity with the original brain).

    parent_slot="back" fixes that: the parent goes in the BACK slot, the face
    through which every hinge and brick touches its parent. Children never
    attach to a hinge's or brick's BACK (asserted), so nothing is overwritten.

    Returns {"type": {name: one-hot index}, "slots": {name: [name | None] * 6},
    "edges": [(parent, child, face)]}.
    """
    edges: list[tuple[str, str, ModuleFaces]] = []
    types: dict[str, int] = {}
    attach = canonical_bodies.Attacher.__call__

    def recording_attach(self, parent, face, module_t, name=None, rotation=0):  # noqa: ANN001, ANN202
        module = attach(self, parent, face, module_t, name, rotation)
        edges.append((parent.name, module.name, face))
        types[parent.name] = _DIST_TYPE_ONEHOT[parent.module_type]
        types[module.name] = _DIST_TYPE_ONEHOT[module.module_type]
        return module

    canonical_bodies.Attacher.__call__ = recording_attach
    try:
        canonical_bodies.get(body)
    finally:
        canonical_bodies.Attacher.__call__ = attach

    return {"type": types, "slots": _face_slots(body, types, edges, parent_slot), "edges": edges}


def _face_slots(body: str, types: dict[str, int], edges: list[tuple[str, str, ModuleFaces]],
                parent_slot: str) -> dict[str, list[str | None]]:
    slots: dict[str, list[str | None]] = {}
    for node in types:
        s: list[str | None] = [None] * DIST_N_FACES
        for parent, child, face in edges:
            if child == node:
                s[ModuleFaces.BACK.value if parent_slot == "back" else face.value] = parent
        for parent, child, face in edges:
            if parent == node:
                if parent_slot == "back" and s[face.value] is not None:
                    raise ValueError(f"{body}: child {child} collides with {node}'s parent slot")
                s[face.value] = child
        slots[node] = s
    return slots


# robogen_lite's constructor names every attached module's bodies
# "<prefix><parent node>-<child node>-<face value>-<part>" (constructor.py);
# a module on a hinge also inherits the hinge's "<p>-<c>-<f>-" prefix, so
# names can nest ("13-14-0-14-15-0-brick"), and the last triple is the edge.
_ROBOGEN_BODY_RE = re.compile(r"^robot1_((?:\d+-\d+-\d+-)*(\d+)-(\d+)-(\d+))-(brick|hinge)$")
_ROBOGEN_TYPES = {"brick": ModuleType.BRICK, "hinge": ModuleType.HINGE}


def model_topology(model: mujoco.MjModel, parent_slot: str = "back") -> dict[str, Any]:
    """body_topology for a compiled robogen body (tree, cppn, local or zoo
    modules), read from its body names, so no genome is needed. Module names
    are the body names' "<parent>-<child>-<face>" part, nested prefixes and
    all, as in the joint names ("robot1_<name>-servo"); the core is "core". Same return value and parent_slot rule as
    body_topology; with "back", a child on a hinge's or brick's BACK raises
    (robogen_lite gives those modules no BACK site, so it never happens).
    """
    found = []
    for b in range(model.nbody):
        m = _ROBOGEN_BODY_RE.match(model.body(b).name)
        if m:
            found.append((int(m[2]), int(m[3]), ModuleFaces(int(m[4])), _ROBOGEN_TYPES[m[5]], m[1]))
    children = {c for _, c, _, _, _ in found}
    roots = {p for p, _, _, _, _ in found} - children
    if len(roots) > 1:
        raise ValueError(f"robogen body has several roots: {sorted(roots)}")
    key = {c: name for _, c, _, _, name in found} | dict.fromkeys(roots, "core")
    types = {"core": _DIST_TYPE_ONEHOT[ModuleType.CORE]}
    types |= {key[c]: _DIST_TYPE_ONEHOT[t] for _, c, _, t, _ in found}
    edges = [(key[p], key[c], face) for p, c, face, _, _ in found]
    return {"type": types, "slots": _face_slots("robogen body", types, edges, parent_slot), "edges": edges}


class ZooDistributedAnn:
    """DistributedMLP (one 57-32-1 MLP shared by every hinge), as used with
    ariel's MorphologyAdapter in the d_social_learning experiments.

    Each hinge's input is its own node features, those of its 6 face slots
    (zeros when empty) and sin(2 pi k / 25) for control step k. Node features
    are [joint velocity, 0, joint angle / (pi/2) clipped to +-1, type one-hot];
    bricks and the core have no joint, so only their one-hot is non-zero. The
    output tanh is scaled to +-pi/2 (MorphologyAdapter's evaluator also
    alpha-blends it, which this protocol does not). The parameter count is
    fixed (1889 at hidden 32), whatever the body.

    Joint velocities are not in the protocol's state vector, so the brain
    reads them from the MjData it was built with; act() must be called once
    per control step on that data.
    """

    def __init__(self, topology: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData,
                 hidden: int = DIST_HIDDEN) -> None:
        self._net = DistributedMLP(n_neighbors=DIST_N_FACES, features_per_node=DIST_FEATURES, hidden=hidden)
        self.num_params = self._net.n_params
        self._data = data

        names = list(topology["type"])
        row = {n: i for i, n in enumerate(names)}
        empty = len(names)                # all-zero feature row for empty slots
        self._features = np.zeros((empty + 1, DIST_FEATURES), dtype=np.float32)
        for n, oh in topology["type"].items():
            self._features[row[n], 3 + oh] = 1.0

        hinges, qpos, dof = [], [], []
        for a in range(model.nu):
            joint = model.actuator_trnid[a, 0]
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            module = jname.removeprefix("robot1_").removesuffix("-servo")
            if topology["type"].get(module) != _DIST_TYPE_ONEHOT[ModuleType.HINGE]:
                raise ValueError(f"actuator {a} ({jname}) does not drive a recorded hinge module")
            hinges.append(row[module])
            qpos.append(model.jnt_qposadr[joint])
            dof.append(model.jnt_dofadr[joint])
        self._hinge_rows = np.array(hinges)
        self._qpos, self._dof = np.array(qpos), np.array(dof)
        # (nu, 7) feature-row indices: self, then the 6 face slots
        self._gather = np.array([
            [row[names[h]]] + [row[nb] if nb is not None else empty for nb in topology["slots"][names[h]]]
            for h in hinges
        ])
        self._step = 0
        self.set_params(np.zeros(self.num_params))

    def set_params(self, vec: np.ndarray) -> None:
        if len(vec) != self.num_params:
            raise IndexError("Parameter vector length mismatch")
        self._net.set_theta(vec)

    def reset(self) -> None:
        self._step = 0

    def node_inputs(self) -> np.ndarray:
        """(nu, 57) network inputs at the current data and control step."""
        f = self._features
        f[self._hinge_rows, 0] = self._data.qvel[self._dof]
        f[self._hinge_rows, 2] = np.clip(self._data.qpos[self._qpos] / HALF_PI, -1.0, 1.0)
        x = f[self._gather].reshape(len(self._gather), -1)
        t = np.full((len(x), 1), math.sin(2 * math.pi * self._step / DIST_TIME_PERIOD), dtype=np.float32)
        return np.hstack([x, t])

    def act(self, t: float, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        net = self._net
        h = np.maximum(0.0, self.node_inputs() @ net._w1.T + net._b1)  # noqa: SLF001
        out = np.tanh(h @ net._w2.T + net._b2)[:, 0]  # noqa: SLF001
        self._step += 1
        return out.astype(np.float64) * HALF_PI


# ── Improved distributed ANN variants ─────────────────────────────────────────
#
# Every variant keeps DistributedMLP's shape (one MLP, ReLU hidden layer, tanh
# output, shared by every hinge) and fixes three things relative to the
# original:
#   - the parent sits in the BACK slot (body_topology(parent_slot="back")), so
#     no hinge loses its parent to the child-overwrite bug;
#   - node features are [joint velocity / 10 clipped to +-1, joint angle / (pi/2),
#     brick, hinge, core]: velocity is scaled like the angle, and the always-zero
#     vel_y and the two unused one-hot slots are dropped (8 -> 5 features), which
#     cuts ~220 parameters and pays for the extra inputs below;
#   - the core's orientation (the state vector's quaternion x, y, z, as the ann
#     sees it) is appended, the same for every hinge.
# On top of that, each variant adds at most one way to tell hinges apart:
#   position   the hinge's rest-pose joint position in the core frame (/0.5 m):
#              transferable across bodies, lets mirror-image legs differ
#   phase      learned per-hinge clock phase + one shared clock frequency (0.2-2 Hz),
#              clock as sin and cos; body-specific genes (n + 1)
#   hinge      neighbours are the nearest hinges (through bricks and the core),
#              slotted by the direction to them in the core frame (+x, -x, +y,
#              -y, +z, -z; nearest wins a slot), so hinges see moving joints
#              rather than bricks

DANN_VARIANTS: dict[str, dict[str, Any]] = {
    "dann_fixed": {},
    "dann_pos": {"position": True},
    "dann_phase": {"phase": True},
    "dann_hinge": {"neighbours": "hinge"},
    "dann_pos_hinge": {"position": True, "neighbours": "hinge"},
    "dann_pos_hinge_h16": {"position": True, "neighbours": "hinge", "hidden": 16},
}
DANN_FEATURES = 5               # vel, angle, brick, hinge, core
DANN_VEL_SCALE = 10.0           # rad/s; champions' |qvel| 99th percentile is ~7
DANN_POS_SCALE = 0.5            # m; hinges sit up to ~1.2 m from the core
DANN_F_RANGE = (0.2, 2.0)       # Hz, learned clock frequency (phase variant)


def hinge_neighbour_ties(topology: dict[str, Any], positions: dict[str, np.ndarray]) -> list[tuple[str, int, list[str]]]:
    """Per hinge and direction slot, the hinges tied for nearest, as
    (hinge, slot, sorted candidates) for every slot with more than one."""
    return [(h, k, c) for h, cands in _hinge_neighbour_candidates(topology, positions).items()
            for k, c in enumerate(cands) if len(c) > 1]


def _hinge_neighbour_candidates(topology: dict[str, Any], positions: dict[str, np.ndarray]) -> dict[str, list]:
    """Per hinge and slot, the sorted names of the nearest hinges in that slot
    (one, or several at exactly the same distance)."""
    adj: dict[str, set[str]] = {n: set() for n in topology["type"]}
    for parent, child, _ in topology["edges"]:
        adj[parent].add(child)
        adj[child].add(parent)
    is_hinge = {n: oh == _DIST_TYPE_ONEHOT[ModuleType.HINGE] for n, oh in topology["type"].items()}

    out = {}
    for h in positions:
        found, seen, frontier = [], {h}, [h]
        while frontier:  # flood through non-hinge modules, stopping at hinges
            nxt = []
            for n in frontier:
                for m in sorted(adj[n] - seen):
                    seen.add(m)
                    (found if is_hinge[m] else nxt).append(m)
            frontier = nxt
        cands: list[list[str]] = [[] for _ in range(DIST_N_FACES)]
        best = [np.inf] * DIST_N_FACES
        for m in found:
            d = positions[m] - positions[h]
            axis = int(np.argmax(np.abs(d)))
            k = 2 * axis + (d[axis] < 0)
            dist = float(np.linalg.norm(d))
            if dist < best[k]:
                cands[k], best[k] = [m], dist
            elif dist == best[k]:
                cands[k].append(m)
        out[h] = [sorted(c) for c in cands]
    return out


def _hinge_neighbour_slots(topology: dict[str, Any], positions: dict[str, np.ndarray],
                           tie_choice: Sequence[int] | None = None) -> dict[str, list]:
    """Per hinge, its nearest hinges in the body tree (paths through bricks and
    the core only), slotted by the dominant axis of the direction to them.

    Hinges at exactly the same distance used to be picked in set-iteration
    order, which follows Python's per-process string hashing, so runs trained
    before this was fixed resolved those ties differently in each job. Ties
    now go to the first name; tie_choice instead picks, for each entry of
    hinge_neighbour_ties in order, the index of its candidate (to replay an
    older run's controller)."""
    cands = _hinge_neighbour_candidates(topology, positions)
    pick = {}
    if tie_choice is not None:
        ties = hinge_neighbour_ties(topology, positions)
        if len(tie_choice) != len(ties):
            raise ValueError(f"tie_choice has {len(tie_choice)} entries, the body has {len(ties)} ties")
        pick = {(h, k): c[i] for (h, k, c), i in zip(ties, tie_choice)}
    return {h: [pick.get((h, k), c[0] if c else None) for k, c in enumerate(cs)] for h, cs in cands.items()}


class ZooDistributedAnnV2:
    """Improved distributed ANN (see DANN_VARIANTS above).

    Per hinge input: [self features, 6 neighbour slots x 5 features (zeros when
    empty), clock, core orientation (3), (extra (extra_inputs)), (position (3))].
    extra is a vector broadcast to every hinge, passed to act (e.g. camera
    sections for the food task). The clock is
    sin(2 pi k / 25) at control step k as in the original, or for the phase
    variant sin and cos of (2 pi f t + phase_i). topology must use the "back"
    parent slot: body_topology(body, parent_slot="back") for a zoo body,
    model_topology(model) for a robogen one. Parameters: the MLP's
    (W1, b1, W2, b2), then (phase variant) the frequency gene and n phase genes,
    phase_i = pi * gene.
    """

    def __init__(self, topology: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData, *,
                 position: bool = False, phase: bool = False, neighbours: str = "module",
                 hidden: int = DIST_HIDDEN, extra_inputs: int = 0,
                 tie_choice: Sequence[int] | None = None) -> None:
        self._data = data
        self.position, self.phase, self.hidden = position, phase, hidden
        self.extra_inputs = extra_inputs
        n = model.nu

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
        rot = data.xmat[core].reshape(3, 3)

        hinges, qpos, dof, pos = [], [], [], {}
        for a in range(n):
            joint = model.actuator_trnid[a, 0]
            module = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint).removeprefix("robot1_").removesuffix("-servo")
            if topology["type"].get(module) != _DIST_TYPE_ONEHOT[ModuleType.HINGE]:
                raise ValueError(f"actuator {a} does not drive a recorded hinge module ({module})")
            hinges.append(module)
            qpos.append(model.jnt_qposadr[joint])
            dof.append(model.jnt_dofadr[joint])
            pos[module] = rot.T @ (data.xanchor[joint] - data.xpos[core])
        self._qpos, self._dof = np.array(qpos), np.array(dof)

        # Exact ties for the hinge-neighbour slots (see _hinge_neighbour_slots).
        self.neighbour_ties = hinge_neighbour_ties(topology, pos) if neighbours == "hinge" else []
        slots = _hinge_neighbour_slots(topology, pos, tie_choice) if neighbours == "hinge" else topology["slots"]
        names = list(topology["type"])
        row = {m: i for i, m in enumerate(names)}
        empty = len(names)
        self._features = np.zeros((empty + 1, DANN_FEATURES), dtype=np.float64)
        for m, oh in topology["type"].items():
            self._features[row[m], 1 + oh] = 1.0  # one-hot 1..3 -> columns 2..4
        self._hinge_rows = np.array([row[h] for h in hinges])
        self._gather = np.array([[row[h]] + [row[m] if m is not None else empty for m in slots[h]]
                                 for h in hinges])
        self._pos = np.array([pos[h] for h in hinges]) / DANN_POS_SCALE

        self.n_inputs = (1 + DIST_N_FACES) * DANN_FEATURES + (2 if phase else 1) + 3 + extra_inputs + (3 if position else 0)
        self._shapes = [(hidden, self.n_inputs), (hidden,), (1, hidden), (1,)]
        self._n_mlp = sum(int(np.prod(s)) for s in self._shapes)
        self.num_params = self._n_mlp + (1 + n if phase else 0)
        self._step = 0
        self.set_params(np.zeros(self.num_params))

    def set_params(self, vec: np.ndarray) -> None:
        g = np.asarray(vec, dtype=np.float64)
        if len(g) != self.num_params:
            raise IndexError("Parameter vector length mismatch")
        out, i = [], 0
        for shape in self._shapes:
            size = int(np.prod(shape))
            out.append(g[i:i + size].reshape(shape))
            i += size
        self._w1, self._b1, self._w2, self._b2 = out
        if self.phase:
            self._freq = _sigmoid_range(g[i], *DANN_F_RANGE)
            self._phases = math.pi * g[i + 1:]

    def reset(self) -> None:
        self._step = 0

    def node_inputs(self, t: float, state: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        f = self._features
        f[self._hinge_rows, 0] = np.clip(self._data.qvel[self._dof] / DANN_VEL_SCALE, -1.0, 1.0)
        f[self._hinge_rows, 1] = np.clip(self._data.qpos[self._qpos] / HALF_PI, -1.0, 1.0)
        n = len(self._gather)
        parts = [f[self._gather].reshape(n, -1)]
        if self.phase:
            arg = 2 * math.pi * self._freq * t + self._phases
            parts += [np.sin(arg)[:, None], np.cos(arg)[:, None]]
        else:
            parts.append(np.full((n, 1), math.sin(2 * math.pi * self._step / DIST_TIME_PERIOD)))
        parts.append(np.broadcast_to(state[:3], (n, 3)))
        if self.extra_inputs:
            parts.append(np.broadcast_to(np.asarray(extra, dtype=np.float64), (n, self.extra_inputs)))
        if self.position:
            parts.append(self._pos)
        return np.hstack(parts)

    def act(self, t: float, state: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        h = np.maximum(0.0, self.node_inputs(t, state, extra) @ self._w1.T + self._b1)
        self._step += 1
        return np.tanh(h @ self._w2.T + self._b2)[:, 0] * HALF_PI


def make_brain(
    kind: str, n_inputs: int, n_hinges: int, physics_dt: float = 0.002, duration: float = 15.0,
) -> ZooAnn | ZooSine | RevolveCpg | Matsuoka | ZooSquare | ZooSquareSync | ZooBangBang:
    if kind == "ann":
        return ZooAnn(n_inputs, n_hinges)
    if kind == "sine":
        return ZooSine(n_inputs, n_hinges)
    if kind == "revolve_cpg":
        return RevolveCpg(n_inputs, n_hinges)
    if kind == "matsuoka":
        return Matsuoka(n_inputs, n_hinges, physics_dt)
    if kind == "square":
        return ZooSquare(n_inputs, n_hinges, duration)
    if kind == "square_sync":
        return ZooSquareSync(n_inputs, n_hinges, duration)
    if kind == "bang_bang":
        return ZooBangBang(n_inputs, n_hinges, duration)
    if kind == "distributed_ann" or kind in DANN_VARIANTS:
        raise ValueError(f"{kind} needs the body topology: use zoo_benchmark.make_brain_for(..., body=...)")
    raise ValueError(f"Unknown brain kind: {kind!r} (expected one of {BRAIN_KINDS})")
