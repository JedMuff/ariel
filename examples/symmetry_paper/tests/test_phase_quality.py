"""Phase quality (gait.phase_quality and its episode summary) on synthetic pairs."""

import numpy as np
import pytest
from gait import LOCK_MIN, analyse_episode, phase_quality

DT = 0.02
T = np.arange(round(15.0 / DT)) * DT


def limb(freq, phase=0.0, amp=0.04):
    w = 2 * np.pi * freq * T[:, None] + phase
    return amp * np.hstack([np.sin(w), 0.1 * np.sin(w), 0.5 * np.cos(w)])


def test_locked_anti_phase_pair() -> None:
    q = phase_quality(limb(1.0), limb(1.0, np.pi), 1.0, DT)
    assert q["phase_quality"] == pytest.approx(1.0, abs=1e-6)
    assert q["mean_lag"] == pytest.approx(0.5, abs=1e-3)
    assert q["lag_within_frac"] == 1.0
    assert q["phase_relation"] == "anti_phase"


def test_locked_in_phase_pair() -> None:
    q = phase_quality(limb(1.0), limb(1.0, 0.1), 1.0, DT)
    assert q["phase_quality"] > 0.99
    assert q["phase_relation"] == "in_phase"


def test_detuned_pair_is_unlocked() -> None:
    # 1.0 vs 1.15 Hz: the lag drifts through 2 full cycles in 15 s.
    q = phase_quality(limb(1.0), limb(1.15), 1.075, DT)
    assert q["phase_quality"] < 0.3
    assert q["phase_relation"] == "unlocked"
    assert q["phase_quality"] < LOCK_MIN


def test_too_few_cycles() -> None:
    q = phase_quality(limb(0.1)[:200], limb(0.1)[:200], 0.1, DT)
    assert np.isnan(q["phase_quality"])
    assert q["n_cycles"] < 3


def test_episode_summary_on_mirrored_body() -> None:
    """A walking body with one mirrored, anti-phase pair is fully phase-locked."""
    n = len(T)
    core = np.stack([0.05 * T, np.zeros(n), np.zeros(n)], axis=1)
    left = np.array([0.1, 0.15, 0.0]) + limb(1.0)
    right = np.array([0.1, -0.15, 0.0]) + limb(1.0, np.pi) * np.array([1.0, -1.0, 1.0])
    P = np.stack([core, core + left, core + right], axis=1)
    Q = np.broadcast_to([1.0, 0.0, 0.0, 0.0], (n, 3, 4)).copy()
    ind, _pairs = analyse_episode(P, Q, DT)
    assert ind["rowing_label"] == "alternating"
    assert ind["phase_quality"] == pytest.approx(1.0, abs=1e-3)
    assert ind["phase_locked_frac"] == pytest.approx(1.0)
    assert ind["anti_phase_frac"] == pytest.approx(1.0)
