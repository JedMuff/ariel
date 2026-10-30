"""Controllers of the paper: bang_bang, bang_bang_free and the distributed ANN.

Every brain is driven the same way by sim.run_skill_episode: set_params(flat
CMA-ES vector) once, reset() before each episode, then act(data, state) at
every control update. act returns hinge targets in ctrl units ([-pi/2, pi/2]).

  bang_bang       open loop: every hinge switches instantly between -90 and
                  +90 deg. One frequency (4 genes, shared by every hinge), then
                  per hinge a duty cycle (4 genes) and a phase (1 gene), so
                  relative phases stay fixed once found. 4 + 5 n params.
  bang_bang_free  bang_bang with the shared frequency removed: each hinge has
                  its own frequency genes, 9 n params (per hinge: frequency x4,
                  duty x4, phase). From x0 = 0.5 every hinge starts at the same
                  frequency; phases drift only if CMA-ES moves them apart.
  dann            distributed ANN (dann_pos_hinge_h16 of the zoo benchmark):
                  one 42-16-1 MLP (705 params) shared by every hinge; inputs are
                  the hinge's own velocity and angle, its nearest hinges slotted
                  by direction, the core orientation, a sine clock and the
                  hinge's rest position.

The square-wave rhythm is the zoo benchmark's square_wave.py engine (phase
integrated per update, "hold" after the timeline ends) restricted to what
bang-bang patterns use: frequency/duty mode and instant switches.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import mujoco
import numpy as np

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType

BRAIN_KINDS: tuple[str, ...] = ("bang_bang", "bang_bang_free", "dann")
# CMA-ES mean (the profile starts from a fixed point): the square-wave genes
# are centred for x0 = 0.5, the DANN's MLP for 0.
BRAIN_X0: dict[str, float] = {"bang_bang": 0.5, "bang_bang_free": 0.5}

SETTLE_DURATION = 3.0  # s before the controller starts; brains see t - SETTLE_DURATION
HALF_PI = math.pi / 2


# ── Square-wave engine (zoo_benchmark/square_wave.py, freq_duty, w = 0) ──────

F_MIN, F_MAX = 0.01, 10.0     # Hz, engine clamp
D_MIN, D_MAX = 0.02, 0.98     # duty cycle, engine clamp

SQUARE_F_RANGE = (0.2, 2.0)             # Hz, frequency start/end
SQUARE_F_WOBBLE_AMP = 0.5               # Hz
SQUARE_DUTY_RANGE = (0.02, 0.98)        # duty start/end
SQUARE_DUTY_WOBBLE_AMP = 0.3
SQUARE_WOBBLE_RATE = 1.0                # Hz, both wobbles
SQUARE_WOBBLE_BIAS = 3.0                # wobble amp = max * sigmoid(g - bias): near off at x0
SQUARE_MAX_ANGLE = 90.0                 # deg
SQUARE_PHASE_SCALE = 0.1                # phase_offset = 0.1 g mod 1 (cycles)


@dataclass
class ModulatedValue:
    """start -> end linear ramp over the episode, plus a sinusoidal wobble."""

    start: Any
    end: Any
    wobble_amp: Any = 0.0
    wobble_rate: Any = 0.0  # Hz

    def value(self, t: float, duration: float) -> Any:
        ramp = min(max(t / duration, 0.0), 1.0)
        return (self.start + (np.asarray(self.end) - self.start) * ramp
                + self.wobble_amp * np.sin(2 * math.pi * np.asarray(self.wobble_rate) * t))


def _sigmoid_range(g: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + (hi - lo) / (1 + np.exp(-g))


def _square_rhythm(g: np.ndarray, value_range: tuple[float, float], wobble_amp: float) -> ModulatedValue:
    """4 genes (start, end, wobble amp, wobble rate), each a scalar or a per-hinge array."""
    return ModulatedValue(
        start=_sigmoid_range(g[0], *value_range), end=_sigmoid_range(g[1], *value_range),
        wobble_amp=_sigmoid_range(g[2] - SQUARE_WOBBLE_BIAS, 0.0, wobble_amp),
        wobble_rate=_sigmoid_range(g[3], 0.0, SQUARE_WOBBLE_RATE),
    )


class _BangBangBase:
    """Per hinge: high (+90 deg) while the cycle position p < duty, else low
    (-90 deg). The phase is integrated over the elapsed control period at each
    act() call: phase += f(t) dt. The frequency and duty modulations run over
    duration s (the episode) and hold after it."""

    def __init__(self, n_hinges: int, duration: float = 15.0) -> None:
        self.n = n_hinges
        self.duration = duration
        self.set_params(np.zeros(self.num_params))

    @property
    def num_params(self) -> int:
        raise NotImplementedError

    def _decode(self, g: np.ndarray) -> tuple[ModulatedValue, ModulatedValue, np.ndarray]:
        """(frequency, duty, phase offset per hinge) from the gene vector."""
        raise NotImplementedError

    def set_params(self, vec: np.ndarray) -> None:
        g = np.asarray(vec, dtype=np.float64)
        if len(g) != self.num_params:
            raise IndexError("Parameter vector length mismatch")
        self.freq, self.duty, self.phase_offset = self._decode(g)
        self.reset()

    def reset(self) -> None:
        self._phase = np.mod(np.asarray(self.phase_offset, dtype=np.float64), 1.0)
        self._time = 0.0

    def output_at(self, t: float) -> np.ndarray:
        tp = t if t < self.duration else self.duration
        f = np.clip(np.asarray(self.freq.value(tp, self.duration), dtype=np.float64), F_MIN, F_MAX)
        duty = np.clip(np.asarray(self.duty.value(tp, self.duration), dtype=np.float64), D_MIN, D_MAX)
        phase = self._phase + f * (t - self._time)
        self._phase = phase - np.floor(phase)  # keep the accumulator in [0, 1)
        self._time = t
        angle = np.where(self._phase < duty, SQUARE_MAX_ANGLE, -SQUARE_MAX_ANGLE)
        return np.radians(angle)

    def act(self, data: Any, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self.output_at(float(data.time) - SETTLE_DURATION)


class BangBangBrain(_BangBangBase):
    """One shared frequency (4 genes), then 5 genes per hinge (duty x4, phase)."""

    @property
    def num_params(self) -> int:
        return 4 + 5 * self.n

    def _decode(self, g: np.ndarray) -> tuple[ModulatedValue, ModulatedValue, np.ndarray]:
        h = g[4:].reshape(self.n, 5).T
        return (_square_rhythm(g[:4], SQUARE_F_RANGE, SQUARE_F_WOBBLE_AMP),
                _square_rhythm(h[0:4], SQUARE_DUTY_RANGE, SQUARE_DUTY_WOBBLE_AMP),
                np.mod(SQUARE_PHASE_SCALE * h[4], 1.0))


class BangBangFreeBrain(_BangBangBase):
    """9 genes per hinge (frequency x4, duty x4, phase): no shared frequency."""

    @property
    def num_params(self) -> int:
        return 9 * self.n

    def _decode(self, g: np.ndarray) -> tuple[ModulatedValue, ModulatedValue, np.ndarray]:
        h = g.reshape(self.n, 9).T
        return (_square_rhythm(h[0:4], SQUARE_F_RANGE, SQUARE_F_WOBBLE_AMP),
                _square_rhythm(h[4:8], SQUARE_DUTY_RANGE, SQUARE_DUTY_WOBBLE_AMP),
                np.mod(SQUARE_PHASE_SCALE * h[8], 1.0))


# ── Distributed ANN (zoo_benchmark/brains.py ZooDistributedAnnV2) ────────────

# One-hot index per module type, as ariel's MorphologyAdapter (0 = empty, 4 unused).
_TYPE_ONEHOT = {ModuleType.BRICK: 1, ModuleType.HINGE: 2, ModuleType.CORE: 3}
N_SLOTS = len(ModuleFaces)      # 6 neighbour slots per hinge
DANN_HIDDEN = 16
DANN_FEATURES = 5               # vel, angle, brick, hinge, core
DANN_VEL_SCALE = 10.0           # rad/s; champions' |qvel| 99th percentile is ~7
DANN_POS_SCALE = 0.5            # m; hinges sit up to ~1.2 m from the core
DANN_TIME_PERIOD = 25           # control steps per period of the sine clock input

# robogen_lite's constructor names every attached module's bodies
# "<prefix><parent node>-<child node>-<face value>-<part>" (constructor.py);
# a module on a hinge also inherits the hinge's "<p>-<c>-<f>-" prefix, so
# names can nest ("13-14-0-14-15-0-brick"), and the last triple is the edge.
_ROBOGEN_BODY_RE = re.compile(r"^robot1_((?:\d+-\d+-\d+-)*(\d+)-(\d+)-(\d+))-(brick|hinge)$")
_ROBOGEN_TYPES = {"brick": ModuleType.BRICK, "hinge": ModuleType.HINGE}


def model_topology(model: mujoco.MjModel) -> dict[str, Any]:
    """Module types and (parent, child, face) edges of a compiled robogen body,
    read from its body names. Module names are the body names'
    "<parent>-<child>-<face>" part, as in the joint names
    ("robot1_<name>-servo"); the core is "core".

    Returns {"type": {name: one-hot index}, "edges": [(parent, child, face)]}.
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
    types = {"core": _TYPE_ONEHOT[ModuleType.CORE]}
    types |= {key[c]: _TYPE_ONEHOT[t] for _, c, _, t, _ in found}
    edges = [(key[p], key[c], face) for p, c, face, _, _ in found]
    return {"type": types, "edges": edges}


def _hinge_neighbour_candidates(topology: dict[str, Any], positions: dict[str, np.ndarray]) -> dict[str, list]:
    """Per hinge and slot, the sorted names of the nearest hinges in that slot
    (one, or several at exactly the same distance)."""
    adj: dict[str, set[str]] = {n: set() for n in topology["type"]}
    for parent, child, _ in topology["edges"]:
        adj[parent].add(child)
        adj[child].add(parent)
    is_hinge = {n: oh == _TYPE_ONEHOT[ModuleType.HINGE] for n, oh in topology["type"].items()}

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
        cands: list[list[str]] = [[] for _ in range(N_SLOTS)]
        best = [np.inf] * N_SLOTS
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


def hinge_neighbour_ties(topology: dict[str, Any], positions: dict[str, np.ndarray]) -> list[tuple[str, int, list[str]]]:
    """Per hinge and direction slot, the hinges tied for nearest, as
    (hinge, slot, sorted candidates) for every slot with more than one."""
    return [(h, k, c) for h, cands in _hinge_neighbour_candidates(topology, positions).items()
            for k, c in enumerate(cands) if len(c) > 1]


def _hinge_neighbour_slots(topology: dict[str, Any], positions: dict[str, np.ndarray],
                           tie_choice: Optional[Sequence[int]] = None) -> dict[str, list]:
    """Per hinge, its nearest hinges in the body tree (paths through bricks and
    the core only), slotted by the dominant axis of the direction to them
    (+x, -x, +y, -y, +z, -z in the core frame).

    Exact ties go to the first name. tie_choice instead picks, for each entry
    of hinge_neighbour_ties in order, the index of its candidate: DANN runs
    trained before the ties were made deterministic resolved them by their
    job's hash seed, and replaying them means finding that resolution."""
    cands = _hinge_neighbour_candidates(topology, positions)
    pick = {}
    if tie_choice is not None:
        ties = hinge_neighbour_ties(topology, positions)
        if len(tie_choice) != len(ties):
            raise ValueError(f"tie_choice has {len(tie_choice)} entries, the body has {len(ties)} ties")
        pick = {(h, k): c[i] for (h, k, c), i in zip(ties, tie_choice)}
    return {h: [pick.get((h, k), c[0] if c else None) for k, c in enumerate(cs)] for h, cs in cands.items()}


class DannBrain:
    """Per hinge input: [self features, 6 neighbour-hinge slots x 5 features
    (zeros when empty), sin(2 pi k / 25) at control step k, core orientation
    (state[:3]), rest position in the core frame (3)]. Parameters: the MLP's
    (W1, b1, W2, b2). The topology is read from the compiled model; joint
    velocities come from the MjData the brain was built with, so act() must be
    called on that data. The clock restarts on set_params and reset."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 tie_choice: Optional[Sequence[int]] = None, hidden: int = DANN_HIDDEN) -> None:
        topology = model_topology(model)
        self._data = data
        n = model.nu

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
        rot = data.xmat[core].reshape(3, 3)

        hinges, qpos, dof, pos = [], [], [], {}
        for a in range(n):
            joint = model.actuator_trnid[a, 0]
            module = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint).removeprefix("robot1_").removesuffix("-servo")
            if topology["type"].get(module) != _TYPE_ONEHOT[ModuleType.HINGE]:
                raise ValueError(f"actuator {a} does not drive a recorded hinge module ({module})")
            hinges.append(module)
            qpos.append(model.jnt_qposadr[joint])
            dof.append(model.jnt_dofadr[joint])
            pos[module] = rot.T @ (data.xanchor[joint] - data.xpos[core])
        self._qpos, self._dof = np.array(qpos), np.array(dof)

        self.neighbour_ties = hinge_neighbour_ties(topology, pos)
        slots = _hinge_neighbour_slots(topology, pos, tie_choice)
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

        self.n_inputs = (1 + N_SLOTS) * DANN_FEATURES + 1 + 3 + 3
        self._shapes = [(hidden, self.n_inputs), (hidden,), (1, hidden), (1,)]
        self.num_params = sum(int(np.prod(s)) for s in self._shapes)
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
        self.reset()

    def reset(self) -> None:
        self._step = 0

    def node_inputs(self, state: np.ndarray) -> np.ndarray:
        f = self._features
        f[self._hinge_rows, 0] = np.clip(self._data.qvel[self._dof] / DANN_VEL_SCALE, -1.0, 1.0)
        f[self._hinge_rows, 1] = np.clip(self._data.qpos[self._qpos] / HALF_PI, -1.0, 1.0)
        n = len(self._gather)
        return np.hstack([
            f[self._gather].reshape(n, -1),
            np.full((n, 1), math.sin(2 * math.pi * self._step / DANN_TIME_PERIOD)),
            np.broadcast_to(state[:3], (n, 3)),
            self._pos,
        ])

    def act(self, data: Any, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        h = np.maximum(0.0, self.node_inputs(state) @ self._w1.T + self._b1)
        self._step += 1
        return np.tanh(h @ self._w2.T + self._b2)[:, 0] * HALF_PI


Brain = "BangBangBrain | BangBangFreeBrain | DannBrain"


def make_brain_for_model(kind: str, model: mujoco.MjModel, data: mujoco.MjData,
                         params: Optional[np.ndarray] = None,
                         tie_choice: Optional[Sequence[int]] = None,
                         duration: float = 15.0) -> Brain:
    """A brain sized for this body (one output per actuator). duration is the
    episode length (the bang-bang modulations' timeline); tie_choice: see
    _hinge_neighbour_slots (dann only)."""
    if kind == "bang_bang":
        brain = BangBangBrain(model.nu, duration)
    elif kind == "bang_bang_free":
        brain = BangBangFreeBrain(model.nu, duration)
    elif kind == "dann":
        brain = DannBrain(model, data, tie_choice=tie_choice)
    else:
        raise ValueError(f"Unknown brain kind: {kind!r} (expected one of {BRAIN_KINDS})")
    if params is not None:
        brain.set_params(params)
    return brain
