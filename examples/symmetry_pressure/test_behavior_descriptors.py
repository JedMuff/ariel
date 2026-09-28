"""
Sanity checks for behavior_common.compute_behavior_descriptors on synthetic
traces: known gait frequency, mirrored in-phase / anti-phase joint pairs,
straight vs. circular core paths, contact duty factor.

Usage:
    cd /path/to/ariel
    python examples/symmetry_pressure/test_behavior_descriptors.py
    # or: pytest examples/symmetry_pressure/test_behavior_descriptors.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from behavior_common import FORWARD_AXIS, compute_behavior_descriptors

DT = 0.002
T = 15000  # 30 s
t = np.arange(T) * DT

# Four hinges: a mirrored pair across core-frame x=0, plus two on the plane.
ANCHORS = np.array([
    [0.11, 0.35, -0.05],
    [-0.11, 0.35, -0.05],
    [0.0, 0.25, -0.16],
    [0.0, 0.25, -0.26],
])


def _descriptors(core_xy=None, joint_angles=None, body_contact=None, anchors=ANCHORS, **kw):
    if core_xy is None:
        core_xy = np.outer(t, FORWARD_AXIS * 0.1)
    if joint_angles is None:
        joint_angles = np.stack([np.sin(2 * math.pi * 1.0 * t)] * len(anchors), axis=1)
    if body_contact is None:
        body_contact = np.zeros((T, 3), dtype=bool)
    return compute_behavior_descriptors(
        core_xy=core_xy, core_z=np.full(T, 0.15), core_tilt=np.zeros(T),
        yaw_delta=np.zeros(T), joint_angles=joint_angles, body_contact=body_contact,
        ctrl_applied=joint_angles[::9], joint_anchors_core=anchors, dt=DT, **kw,
    )


def test_straight_path() -> None:
    d = _descriptors()
    assert abs(d["straightness"] - 1.0) < 1e-6
    assert abs(d["axis_displacement"] - 0.1 * t[-1]) < 1e-6
    assert d["lateral_drift"] < 1e-9
    assert abs(d["mean_speed"] - 0.1) < 1e-3


def test_circular_path_is_not_straight() -> None:
    theta = 2 * math.pi * t / t[-1]
    d = _descriptors(core_xy=np.stack([np.cos(theta), np.sin(theta)], axis=1))
    assert d["straightness"] < 0.01


def test_gait_frequency() -> None:
    for f in (0.5, 1.5, 3.0):
        angles = np.stack([np.sin(2 * math.pi * f * t + k) for k in range(4)], axis=1)
        d = _descriptors(joint_angles=angles)
        assert abs(d["gait_frequency_hz"] - f) < 1 / t[-1] + 1e-9, (f, d["gait_frequency_hz"])


def test_mirror_pairs_anti_phase() -> None:
    s = np.sin(2 * math.pi * t)
    angles = np.stack([s, -s, 0.3 * s, 0.3 * s], axis=1)
    d = _descriptors(joint_angles=angles)
    assert d["mirror_pair_frac"] == 1.0
    assert d["mirror_plane"] == 0.0
    assert d["mirror_pair_corr_mean"] < -0.99
    assert abs(d["mirror_pair_phase_lag"] - math.pi) < 0.05
    assert d["mirror_pair_amp_asym"] < 1e-6


def test_mirror_pairs_in_phase_unequal_amplitude() -> None:
    s = np.sin(2 * math.pi * t)
    angles = np.stack([s, 0.5 * s, s, s], axis=1)
    d = _descriptors(joint_angles=angles)
    assert d["mirror_pair_corr_mean"] > 0.99
    assert d["mirror_pair_phase_lag"] < 0.05
    assert abs(d["mirror_pair_amp_asym"] - 1 / 3) < 1e-6


def test_asymmetric_body_has_no_pairs() -> None:
    anchors = np.array([[0.11, 0.35, -0.05], [0.2, 0.1, 0.0], [0.0, 0.25, -0.16]])
    angles = np.stack([np.sin(2 * math.pi * t)] * 3, axis=1)
    d = _descriptors(joint_angles=angles, anchors=anchors)
    assert d["mirror_pair_frac"] == 0.0
    assert math.isnan(d["mirror_pair_corr_mean"])


def test_contact_duty_factor_and_entropy() -> None:
    contact = np.zeros((T, 2), dtype=bool)
    contact[:, 0] = True                 # always down
    contact[: T // 2, 1] = True          # down half the time
    d = _descriptors(body_contact=contact)
    assert abs(d["duty_factor_mean"] - 0.75) < 1e-9
    assert abs(d["n_contact_bodies_mean"] - 1.5) < 1e-9
    assert d["frac_bodies_ever_contact"] == 1.0
    assert abs(d["contact_entropy_bits"] - 1.0) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} passed")
