"""Tests for the rowing penalty of gecko_skill_tasks.py --rowing-penalty (shared.py)."""

import numpy as np
import pytest

from shared import rowing_info, rowing_scaled_disp
from test_rowing_detect import DT, episode, pair, stroke


def test_scaled_disp_scales_forward_only():
    assert rowing_scaled_disp(2.0, 0.0) == 2.0
    assert rowing_scaled_disp(2.0, 0.25) == pytest.approx(1.5)
    assert rowing_scaled_disp(2.0, 1.0) == 0.0
    assert rowing_scaled_disp(-1.0, 0.8) == -1.0  # going backwards is never helped


def test_rowing_info_full_rower():
    P, Q = episode(pair(0.1, 0.15, stroke(), stroke()))
    info = rowing_info(list(P), list(Q), DT)
    assert info["rowing_label"] == "rowing"
    assert info["rowing_frac"] > 0.95


def test_rowing_info_alternating_is_free():
    P, Q = episode(pair(0.1, 0.15, stroke(), stroke(phase=np.pi)))
    assert rowing_info(list(P), list(Q), DT)["rowing_frac"] == 0.0


def test_rowing_info_failure_means_no_penalty():
    info = rowing_info([], [], DT)  # no poses: the detector raises
    assert info == {"rowing_frac": 0.0, "rowing_label": "error"}
