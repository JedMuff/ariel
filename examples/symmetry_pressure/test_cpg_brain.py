"""
Tests for the brain abstraction in shared.py (make_brain / AnnBrain / CpgBrain).

Covers:
  - CPG parameter count (3n+1) and output bounds
  - CPG periodicity at its decoded frequency, and determinism
  - CPG clock starts at the end of the settling phase
  - AnnBrain reproduces the raw Network + fill_parameters output exactly
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from shared import (
    CPG_F_MAX,
    CPG_F_MIN,
    SETTLE_DURATION,
    AnnBrain,
    CpgBrain,
    Network,
    RevolveCpgBrain,
    fill_parameters,
    make_brain,
)

N_IN, N_OUT = 11, 8


def _data(t: float) -> SimpleNamespace:
    return SimpleNamespace(time=t)


def test_cpg_param_count():
    assert make_brain("cpg", N_IN, N_OUT).num_params == 3 * N_OUT + 1


def test_cpg_rejects_wrong_length():
    with pytest.raises(IndexError):
        CpgBrain(N_IN, N_OUT).set_params(np.zeros(3 * N_OUT))


def test_cpg_output_bounded_and_ignores_state():
    rng = np.random.default_rng(0)
    brain = CpgBrain(N_IN, N_OUT)
    for _ in range(20):
        brain.set_params(rng.normal(0, 5, brain.num_params))
        for t in np.linspace(0, 10, 50):
            a = brain.act(_data(SETTLE_DURATION + t), rng.normal(size=N_IN))
            b = brain.act(_data(SETTLE_DURATION + t), np.zeros(N_IN))
            assert a.shape == (N_OUT,)
            assert np.all(np.abs(a) <= math.pi / 2 + 1e-12)
            np.testing.assert_array_equal(a, b)


def test_cpg_frequency_range_and_periodicity():
    rng = np.random.default_rng(1)
    brain = CpgBrain(N_IN, N_OUT)
    for _ in range(10):
        p = rng.normal(0, 2, brain.num_params)
        brain.set_params(p)
        assert CPG_F_MIN <= brain.freq <= CPG_F_MAX
        period = 1.0 / brain.freq
        for t in (0.0, 0.37, 2.9):
            np.testing.assert_allclose(brain.output_at(t), brain.output_at(t + period), atol=1e-9)


def test_cpg_deterministic_and_clock_starts_after_settle():
    p = np.random.default_rng(2).normal(size=3 * N_OUT + 1)
    a, b = CpgBrain(N_IN, N_OUT), CpgBrain(N_IN, N_OUT)
    a.set_params(p)
    b.set_params(p)
    np.testing.assert_array_equal(a.act(_data(SETTLE_DURATION), None), b.output_at(0.0))
    # At t=0 the output is offset + amp * sin(phase).
    expected = np.clip(a.offset + a.amp * np.sin(a.phase), -math.pi / 2, math.pi / 2)
    np.testing.assert_allclose(a.output_at(0.0), expected)


def test_ann_brain_matches_raw_network():
    brain = make_brain("ann", N_IN, N_OUT)
    assert isinstance(brain, AnnBrain)
    rng = np.random.default_rng(3)
    w = rng.normal(size=brain.num_params).astype(np.float32)
    brain.set_params(w)

    net = Network(input_size=N_IN, output_size=N_OUT)
    fill_parameters(net, w)
    state = rng.normal(size=N_IN).astype(np.float32)
    np.testing.assert_array_equal(brain.act(_data(0.0), state), net.forward(None, None, state))


def test_revolve_cpg_param_count_and_coupling_matrix():
    brain = make_brain("revolve_cpg", N_IN, N_OUT)
    assert isinstance(brain, RevolveCpgBrain)
    assert brain.num_params == N_OUT + N_OUT * (N_OUT - 1) // 2
    brain.set_params(np.random.default_rng(4).normal(size=brain.num_params))
    np.testing.assert_array_equal(brain._a, -brain._a.T)
    with pytest.raises(IndexError):
        brain.set_params(np.zeros(brain.num_params + 1))


def _rollout(brain: RevolveCpgBrain, dt: float, t_end: float) -> list[np.ndarray]:
    return [brain.act(_data(SETTLE_DURATION + t), None).copy()
            for t in np.arange(0.0, t_end, dt)]


def test_revolve_cpg_bounded_and_set_params_resets():
    rng = np.random.default_rng(5)
    brain = RevolveCpgBrain(N_IN, N_OUT)
    p = rng.normal(0, 2, brain.num_params)
    brain.set_params(p)
    first = _rollout(brain, 0.04, 5.0)
    for a in first:
        assert a.shape == (N_OUT,)
        assert np.all(np.abs(a) <= math.pi / 2 + 1e-12)
    # A second episode after re-decoding the candidate replays exactly.
    brain.set_params(p)
    for a, b in zip(first, _rollout(brain, 0.04, 5.0), strict=True):
        np.testing.assert_array_equal(a, b)


def test_revolve_cpg_oscillates_from_initial_state():
    brain = RevolveCpgBrain(N_IN, N_OUT)
    brain.set_params(np.r_[np.full(N_OUT, 2.0), np.zeros(brain.num_params - N_OUT)])
    # Uncoupled pairs with self-coupling w are x = r cos(w t + phi): at t=0
    # x = sqrt(2)/2 (scaled by pi/2), and the sign of x flips within a period.
    np.testing.assert_allclose(brain.act(_data(SETTLE_DURATION), None),
                               np.full(N_OUT, math.sqrt(2) / 2 * math.pi / 2))
    xs = np.array(_rollout(brain, 0.01, 2 * math.pi / 2.0))[:, 0]
    assert xs.min() < -0.5 and xs.max() > 0.5


def test_unknown_brain_kind():
    with pytest.raises(ValueError):
        make_brain("snn", N_IN, N_OUT)
