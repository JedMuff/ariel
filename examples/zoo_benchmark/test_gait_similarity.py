"""Checks for gait_similarity (run: pytest test_gait_similarity.py)."""

from pathlib import Path

import numpy as np
import pytest

from gait_similarity import (
    _charged_mean,
    average_cycle,
    core_frame,
    cycle_distance,
    estimate_period,
    shape_cycle,
    record_module_trajectories,
    trajectory_distance,
    world_frame,
)

GECKO_ANN = Path(__file__).parent / "__data__" / "local_gecko" / "gecko_ann"


def _random_traj(rng, m=6, t=50):
    return rng.normal(size=(m, t, 3)).cumsum(axis=1) * 0.01


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_identity_and_symmetry():
    rng = np.random.default_rng(0)
    a, b = _random_traj(rng), _random_traj(rng, m=8)
    assert trajectory_distance(a, a) == pytest.approx(0.0, abs=1e-12)
    assert trajectory_distance(a, b) == pytest.approx(trajectory_distance(b, a))
    assert trajectory_distance(a, b) > 0


def test_module_order_invariant():
    rng = np.random.default_rng(1)
    a, b = _random_traj(rng), _random_traj(rng, m=7)
    assert trajectory_distance(a, b) == pytest.approx(trajectory_distance(a[rng.permutation(6)], b))


def test_extra_duplicate_module_is_free():
    rng = np.random.default_rng(2)
    a = _random_traj(rng)
    a_extra = np.concatenate([a, a[:1]])
    assert trajectory_distance(a, a_extra) == pytest.approx(0.0, abs=1e-12)


def test_ignored_leftover_is_free_and_symmetric():
    rng = np.random.default_rng(8)
    a = _random_traj(rng)
    a_extra = np.concatenate([a, a[:1] + 0.5])  # an extra module unlike any other
    assert trajectory_distance(a, a_extra) > 0.1
    assert trajectory_distance(a, a_extra, leftover="ignore") == pytest.approx(0.0, abs=1e-12)
    b = _random_traj(rng, m=9)
    assert trajectory_distance(a, b, leftover="ignore") == pytest.approx(trajectory_distance(b, a, leftover="ignore"))


def test_known_offset():
    rng = np.random.default_rng(3)
    a = _random_traj(rng)
    assert trajectory_distance(a, a + np.array([0.3, 0.0, 0.4])) == pytest.approx(0.5)


def test_lag_recovers_shift():
    rng = np.random.default_rng(4)
    long = _random_traj(rng, t=60)
    a, b = long[:, :50], long[:, 3:53]
    assert trajectory_distance(a, b) > 0
    assert trajectory_distance(a, b, max_lag=3) == pytest.approx(0.0, abs=1e-12)


def _recording(rng, T=40, M=5):
    core_pos = rng.normal(size=(T, 3)).cumsum(axis=0) * 0.01
    core_R = np.array([_rot_z(th) for th in np.linspace(0, 1.5, T)])
    local = rng.normal(size=(T, M, 3)) * 0.05
    pos = core_pos[:, None, :] + np.einsum("tij,tmj->tmi", core_R, local)
    return pos, core_pos, core_R


def test_world_frame_ignores_start_position():
    pos, core_pos, _ = _recording(np.random.default_rng(5))
    shift = np.array([2.0, -1.0, 0.0])
    np.testing.assert_allclose(world_frame(pos, core_pos), world_frame(pos + shift, core_pos + shift))


def test_core_frame_ignores_rigid_motion():
    pos, core_pos, core_R = _recording(np.random.default_rng(6))
    R, shift = _rot_z(0.7), np.array([1.0, 2.0, 0.5])
    moved = (pos @ R.T + shift, core_pos @ R.T + shift, np.einsum("ij,tjk->tik", R, core_R))
    np.testing.assert_allclose(core_frame(pos, core_pos, core_R), core_frame(*moved), atol=1e-12)


DT = 0.05
LEG_PHASES = np.array([0.0, 0.5, 0.25, 0.75])  # trot-like
LEG_DIRS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])


def _gait(freq, phase0=0.0, leg_phases=LEG_PHASES, duration=15.0):
    """(M, T, 3) legs tracing a tilted ellipse along their own direction."""
    t = np.arange(round(duration / DT) + 1) * DT
    legs = []
    for p, (dx, dy) in zip(leg_phases, LEG_DIRS):
        ang = 2 * np.pi * (freq * t + p + phase0)
        fwd, up = 0.1 * np.cos(ang) + 0.02 * np.cos(2 * ang), 0.04 * np.sin(ang)
        legs.append(np.stack([fwd * dx, fwd * dy, up], axis=-1))
    return np.array(legs)


def _transform(traj, yaw_deg, mirror):
    M = np.diag([1.0, -1.0 if mirror else 1.0, 1.0])
    return traj @ (_rot_z(np.deg2rad(yaw_deg)) @ M).T


def _cycle(traj):
    return average_cycle(traj, DT, estimate_period(traj, DT)[0])


@pytest.mark.parametrize("freq", [0.4, 0.8, 1.7])
def test_estimate_period(freq):
    period, autocorr = estimate_period(_gait(freq), DT)
    assert period == pytest.approx(1 / freq, rel=0.02)
    assert autocorr > 0.95


def test_average_cycle_keeps_shape():
    traj = _gait(0.8)
    cycle = _cycle(traj)
    steady = traj[:, round(2.0 / DT):]
    steady = steady - steady.mean(axis=1, keepdims=True)
    assert cycle.shape == (4, 64, 3)
    np.testing.assert_allclose(np.sqrt((cycle ** 2).sum(axis=2).mean(axis=1)),
                               np.sqrt((steady ** 2).sum(axis=2).mean(axis=1)), rtol=0.03)


def test_cycle_distance_ignores_speed_phase_orientation_and_order():
    a = _cycle(_gait(0.7))
    b_traj = _transform(_gait(1.3, phase0=0.37), yaw_deg=70, mirror=True)[[2, 0, 3, 1]]
    d, align = cycle_distance(a, _cycle(b_traj))
    rms = np.sqrt((a ** 2).sum(axis=2).mean())
    assert d < 0.05 * rms
    assert align["mirrored"]


def test_cycle_distance_sees_coordination():
    trot, pace = _cycle(_gait(0.8)), _cycle(_gait(0.8, leg_phases=np.array([0.0, 0.0, 0.5, 0.5])))
    rms = np.sqrt((trot ** 2).sum(axis=2).mean())
    assert cycle_distance(trot, pace)[0] > 0.3 * rms


def test_cycle_shape_ignores_amplitude_but_sees_coordination():
    trot = _cycle(_gait(0.8))
    pace = _cycle(_gait(0.8, leg_phases=np.array([0.0, 0.0, 0.5, 0.5])))
    assert cycle_distance(shape_cycle(trot), shape_cycle(3.0 * trot))[0] == pytest.approx(0.0, abs=1e-6)
    assert cycle_distance(trot, 3.0 * trot)[0] > 1.0 * np.sqrt((trot ** 2).sum(axis=2).mean())
    assert cycle_distance(shape_cycle(trot), shape_cycle(pace))[0] > 0.3


@pytest.mark.parametrize("leftover", ["charge", "ignore"])
def test_cycle_distance_is_exact_minimum(leftover):
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(4, 16, 3)), rng.normal(size=(5, 16, 3))
    d, _ = cycle_distance(a, b, n_yaw=8, leftover=leftover)
    brute = min(
        _charged_mean(np.array([[np.mean(((ai - np.roll(bj, -s, axis=0)) ** 2).sum(axis=1)) for bj in bt]
                                for ai in a]), leftover)
        for mirror in (False, True) for k in range(8) for s in range(16)
        for bt in [_transform(b, 45 * k, mirror)]
    )
    assert d == pytest.approx(np.sqrt(brute))
    assert d == pytest.approx(cycle_distance(b, a, n_yaw=8, leftover=leftover)[0])


@pytest.mark.skipif(not GECKO_ANN.exists(), reason="no local gecko champion")
def test_record_gecko_champion():
    rec = record_module_trajectories(GECKO_ANN)
    T, M = rec["pos"].shape[:2]
    assert T == 301 and M == len(rec["body_names"]) == 19
    assert rec["core_R"].shape == (T, 3, 3)
    assert world_frame(rec["pos"], rec["core_pos"]).shape == (M, T, 3)
    assert np.isfinite(rec["xspeed"])


def test_rank_similar_sorted_without_self_and_alignment_both_ways():
    import pandas as pd

    from gait_similarity_page import rank_similar

    labels = ["a", "b", "c"]
    dist = pd.DataFrame([[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]], index=labels, columns=labels)
    align = pd.DataFrame([{"a": "a", "b": "c", "mirrored": True, "yaw_deg": 90.0, "phase_shift": 0.25}])
    ranked = rank_similar(dist, align)
    assert [s["label"] for s in ranked["a"]] == ["c", "b"]
    assert [s["label"] for s in ranked["b"]] == ["c", "a"]
    assert ranked["c"][0] == {"label": "a", "distance": 0.1, "mirrored": True, "yaw_deg": 90.0, "phase_shift": 0.25}
    assert "mirrored" not in ranked["b"][0]
