"""Tests for stride_gait_similarity.py's pose -> cycle conversion and medoid stats."""

import numpy as np
from scipy.spatial.transform import Rotation

from stride_gait_similarity import core_frame, episode_cycle, medoid_stats, quat_to_mat


def test_quat_to_mat_matches_scipy():
    rot = Rotation.random(20, random_state=0)
    xyzw = rot.as_quat()
    wxyz = np.roll(xyzw, 1, axis=1) * 3.0  # unnormalised on purpose
    np.testing.assert_allclose(quat_to_mat(wxyz), rot.as_matrix(), atol=1e-12)


def test_core_frame_removes_rigid_motion():
    """A body that only translates and spins as a whole has no core-frame motion."""
    rng = np.random.default_rng(0)
    local = rng.normal(size=(6, 3))
    local[0] = 0.0                                              # core at the origin
    t = np.linspace(0, 10, 200)
    rot = Rotation.from_euler("z", 0.7 * t)
    shift = np.stack([0.3 * t, np.sin(t), 0.1 * t], axis=1)
    pos = np.einsum("tij,bj->tbi", rot.as_matrix(), local) + shift[:, None]
    quat = np.roll(rot.as_quat(), 1, axis=1)
    traj = core_frame(pos, pos[:, 0], quat_to_mat(quat))
    np.testing.assert_allclose(traj, 0.0, atol=1e-10)


def test_episode_cycle_periodic_and_aperiodic_fallback():
    dt, T = 0.02, 1501
    t = np.arange(T) * dt
    wave = np.stack([np.sin(2 * np.pi * t / 1.5), np.cos(2 * np.pi * t / 1.5), 0 * t], axis=-1)
    cycle, stats = episode_cycle(np.stack([wave, 0.5 * wave]), dt)
    assert not stats["aperiodic"]
    assert abs(stats["period_s"] - 1.5) < 0.02
    assert cycle.shape == (2, 64, 3)

    drift = np.stack([t, 0 * t, 0 * t], axis=-1)[None]       # no autocorrelation peak
    cycle, stats = episode_cycle(drift - drift.mean(axis=1, keepdims=True), dt)
    assert stats["aperiodic"]
    assert np.isclose(stats["period_s"], (T - 1) * dt - 2.0)
    assert np.isfinite(cycle).all()


def test_medoid_stats_picks_central_point():
    # A plus sign with one far point: the centre (index 2) is unambiguous.
    xy = np.array([[1, 0], [-1, 0], [0, 0], [0, 1], [0, -1], [0, 3]], dtype=float)
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    st = medoid_stats(dist)
    assert st["medoid"] == 2
    others = np.array([1.0, 1.0, 1.0, 1.0, 3.0])
    assert st["medoid_median"] == 1.0
    assert np.isclose(st["medoid_iqr"], np.subtract(*np.percentile(others, [75, 25])))
    assert st["n"] == 6
