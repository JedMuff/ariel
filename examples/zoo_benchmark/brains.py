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
  revolve_cpg  kgd's fully connected RevolveCPG (apets-ariel common/controllers/cpg.py)
  matsuoka     Matsuoka oscillator network (morphlib@hyperneat matsuoka_brain/), bugs fixed
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "symmetry_pressure"))
from shared import AnnBrain, CpgBrain  # noqa: E402

BRAIN_KINDS: tuple[str, ...] = ("ann", "sine", "revolve_cpg", "matsuoka")

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
    """Port of apets-ariel's RevolveCPG ("copied from revolve but fully connected").

    One (x, y) oscillator pair per hinge. Weights: n self-couplings x_i<->y_i
    followed by n(n-1)/2 couplings x_i<->x_j, placed antisymmetrically in a
    2n x 2n matrix A. The state follows dS/dt = A S (RK4 over the elapsed
    control period, clipped to [-1, 1]) and the first n entries, times the
    actuator ctrlrange (pi/2), are the joint targets.
    """

    def __init__(self, n_inputs: int, n_hinges: int) -> None:  # noqa: ARG002
        self.n = n_hinges
        self.num_params = n_hinges + n_hinges * (n_hinges - 1) // 2
        self._pairs = list(itertools.combinations(range(n_hinges), 2))
        self._initial_state = (
            np.hstack([np.full(n_hinges, 1.0), np.full(n_hinges, -1.0)]) * 0.5 * np.sqrt(2)
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
        self._time = 0.0

    def act(self, t: float, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        dt = t - self._time
        a, s = self._a, self._state
        a1 = a @ s
        a2 = a @ (s + dt / 2 * a1)
        a3 = a @ (s + dt / 2 * a2)
        a4 = a @ (s + dt * a3)
        self._state = np.clip(s + dt / 6 * (a1 + 2 * (a2 + a3) + a4), -1, 1)
        self._time = t
        return self._state[: self.n] * HALF_PI


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


def make_brain(
    kind: str, n_inputs: int, n_hinges: int, physics_dt: float = 0.002,
) -> ZooAnn | ZooSine | RevolveCpg | Matsuoka:
    if kind == "ann":
        return ZooAnn(n_inputs, n_hinges)
    if kind == "sine":
        return ZooSine(n_inputs, n_hinges)
    if kind == "revolve_cpg":
        return RevolveCpg(n_inputs, n_hinges)
    if kind == "matsuoka":
        return Matsuoka(n_inputs, n_hinges, physics_dt)
    raise ValueError(f"Unknown brain kind: {kind!r} (expected one of {BRAIN_KINDS})")
