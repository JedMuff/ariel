"""Tests for rowing_detect.py on synthetic bodies with analytic limb trajectories."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rowing_detect import analyse_episode, morph_plane, review_detail

DT, DURATION = 0.02, 20.0
T = np.arange(round(DURATION / DT)) * DT


def episode(limbs, heading_deg=37.0, axis_deg=None, yaw=0.6, speed=0.05):
    """World poses (P, Q) of a body travelling at heading_deg in its core frame,
    with the core yawed by `yaw` in the world.

    limbs: list of (fore, lat, motion) in the body's own axis frame, which is at
    axis_deg in the core frame (default: the heading). motion(t) -> (T, 3)
    displacement (fore, lat, up)."""
    axis = np.deg2rad(heading_deg if axis_deg is None else axis_deg)
    to_core = Rotation.from_euler("z", axis).as_matrix()
    local = [np.zeros((len(T), 3))]
    for fore, lat, motion in limbs:
        local.append((np.array([fore, lat, 0.0]) + motion(T)) @ to_core.T)
    local = np.stack(local, axis=1)                                     # (T, B, 3)
    h = np.deg2rad(heading_deg)
    world_R = Rotation.from_euler("z", yaw).as_matrix()
    core = (speed * T)[:, None] * (world_R @ np.array([np.cos(h), np.sin(h), 0.0]))
    P = local @ world_R.T + core[:, None]
    q = np.roll(Rotation.from_euler("z", yaw).as_quat(), 1)            # MuJoCo w, x, y, z
    Q = np.broadcast_to(q, (len(T), P.shape[1], 4)).copy()
    return P, Q


def stroke(freq=0.6, phase=0.0, amp=0.04):
    """A rowing-like limb stroke: fore-aft sweep with a lift a quarter cycle ahead."""
    def f(t):
        w = 2 * np.pi * freq * t + phase
        return np.stack([amp * np.sin(w), 0.1 * amp * np.sin(w), 0.5 * amp * np.cos(w)], axis=-1)
    return f


def mirrored(motion):
    return lambda t: motion(t) * np.array([1.0, -1.0, 1.0])


def static(t):
    return np.zeros((len(t), 3))


def pair(fore, lat, left, right):
    return [(fore, lat, left), (fore, -lat, mirrored(right))]


def test_rowing_detected_under_heading_and_yaw():
    P, Q = episode(pair(0.1, 0.15, stroke(), stroke()), heading_deg=37.0, yaw=0.6)
    ind, pairs = analyse_episode(P, Q, DT, snap_deg=0)
    assert abs(ind["heading_deg"] - 37.0) < 1.0
    assert ind["heading_consistency"] > 0.95
    assert ind["rowing_label"] == "rowing"
    assert ind["rowing_score"] > 0.95
    (p,) = pairs
    assert p["pair_kind"] == "rowing" and p["mirror_sync"] > 0.99 and p["phase_lag"] < 0.02
    assert abs(p["freq_l"] - 0.6) < 0.02 and p["freq_match"]


def test_alternating_detected():
    P, Q = episode(pair(0.1, 0.15, stroke(), stroke(phase=np.pi)))
    ind, pairs = analyse_episode(P, Q, DT)
    assert ind["rowing_label"] == "alternating"
    assert ind["rowing_score"] < -0.9
    assert abs(pairs[0]["phase_lag"] - 0.5) < 0.02


def test_frequency_mismatch_is_not_rowing():
    P, Q = episode(pair(0.1, 0.15, stroke(freq=0.5), stroke(freq=0.8)))
    ind, pairs = analyse_episode(P, Q, DT)
    assert ind["rowing_label"] != "rowing"
    assert pairs[0]["pair_kind"] == "freq_mismatch"


def test_one_sided_stroke_is_not_rowing():
    P, Q = episode(pair(0.1, 0.15, stroke(), static))
    ind, pairs = analyse_episode(P, Q, DT)
    assert ind["rowing_label"] != "rowing"
    assert pairs[0]["pair_kind"] == "amp_mismatch"


def test_noise_is_aperiodic():
    rng = np.random.default_rng(0)
    kernel = np.hanning(25) / np.hanning(25).sum()  # ~0.5 s smoothing: wobble, not a cycle
    noise = [np.stack([np.convolve(rng.normal(scale=0.05, size=len(T)), kernel, "same") for _ in range(3)], -1)
             for _ in range(2)]
    P, Q = episode(pair(0.1, 0.15, lambda t: noise[0], lambda t: noise[1]))
    ind, _ = analyse_episode(P, Q, DT)
    assert ind["rowing_label"] == "aperiodic"


def test_limb_on_one_side_only_is_unpaired():
    P, Q = episode([(0.1, 0.15, stroke()), (-0.1, 0.15, stroke())])
    ind, pairs = analyse_episode(P, Q, DT)
    assert ind["rowing_label"] == "unpaired" and not pairs


def test_rowing_alternating_and_out_of_sync_pairs_are_mixed():
    limbs = (pair(0.2, 0.15, stroke(), stroke()) + pair(0.0, 0.2, stroke(), stroke(phase=np.pi))
             + pair(-0.2, 0.15, stroke(), stroke(phase=2 * np.pi / 3))
             + pair(-0.4, 0.15, stroke(), stroke(phase=-2 * np.pi / 3))
             + pair(-0.6, 0.15, stroke(), stroke(phase=np.pi / 1.6)))
    P, Q = episode(limbs)
    ind, pairs = analyse_episode(P, Q, DT)
    assert sorted(p["pair_kind"] for p in pairs) == ["alternating", "out_of_sync", "out_of_sync", "out_of_sync",
                                                     "rowing"]
    assert ind["rowing_label"] == "mixed"
    assert abs(ind["rowing_frac"] - 1 / 5) < 0.05


def test_slightly_lagged_stroke_is_rowing():
    P, Q = episode(pair(0.1, 0.15, stroke(), stroke(phase=2 * np.pi * 0.15)))
    ind, pairs = analyse_episode(P, Q, DT)
    assert pairs[0]["pair_kind"] == "rowing" and ind["rowing_label"] == "rowing"


def test_sideways_sway_is_not_alternating():
    def sway(t):  # both limbs swing to the same side: anti-phase once the right side is mirrored
        return np.stack([np.zeros_like(t), 0.04 * np.sin(2 * np.pi * 0.6 * t), np.zeros_like(t)], axis=-1)
    P, Q = episode([(0.1, 0.15, sway), (0.1, -0.15, sway)])
    ind, pairs = analyse_episode(P, Q, DT)
    assert pairs[0]["mirror_sync"] < -0.9 and pairs[0]["pair_kind"] == "lateral"
    assert ind["rowing_label"] == "mixed"


def test_in_phase_bobbing_is_not_rowing():
    def bob(t):  # both limbs lift and drop together, little fore-aft sweep
        w = 2 * np.pi * 0.6 * t
        return np.stack([0.008 * np.sin(w), np.zeros_like(t), 0.04 * np.sin(w)], axis=-1)
    P, Q = episode(pair(0.1, 0.15, bob, bob))
    ind, pairs = analyse_episode(P, Q, DT)
    assert pairs[0]["phase_lag"] < 0.05 and pairs[0]["pair_kind"] == "no_sweep"
    assert ind["rowing_label"] == "mixed"


def test_static_body():
    P, Q = episode(pair(0.1, 0.15, static, static))
    ind, _ = analyse_episode(P, Q, DT)
    assert ind["rowing_label"] == "static"


def test_snap_splits_along_body_axis_when_travelling_off_axis():
    limbs = pair(0.15, 0.15, stroke(), stroke()) + pair(-0.15, 0.2, stroke(), stroke())
    P, Q = episode(limbs, heading_deg=25.0, axis_deg=0.0)
    ind, _ = analyse_episode(P, Q, DT)
    assert ind["rowing_label"] == "rowing"
    assert abs(ind["split_offset_deg"] + 25.0) < 0.1
    assert ind["morph_plane_offset_deg"] == pytest.approx(25.0)
    strict, _ = analyse_episode(P, Q, DT, snap_deg=0)
    assert strict["rowing_label"] != "rowing"


def test_morph_plane_finds_mirror_axis():
    a = np.deg2rad(30)
    u, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    pts = np.array([0.2 * u + 0.1 * n, 0.2 * u - 0.1 * n, -0.1 * u + 0.15 * n, -0.1 * u - 0.15 * n])
    angle, mismatch = morph_plane(pts)
    assert np.rad2deg(angle) == pytest.approx(30.0) and mismatch < 1e-9


def test_review_detail_is_json_serialisable():
    P, Q = episode(pair(0.1, 0.15, stroke(), stroke()))
    _, _, d = analyse_episode(P, Q, DT, detail=True)
    rd = review_detail(d)
    json.dumps(rd)
    assert np.array(rd["cycle"]).shape == (3, 64, 3)
    assert rd["pairs"] == [[1, 2]] and len(rd["traces"]) == 1
    assert rd["side"] == [0, 1, -1]
