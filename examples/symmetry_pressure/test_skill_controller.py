"""
Tests for the skill-based controller plan (SKILL_CONTROLLER_PLAN.md).

Covers:
  - Turn reward accumulation (signed vertical-axis rotation, left/right directions)
  - State-based controller decision logic
  - Network compatibility (11-dim input, 8-dim output)
  - Robust rotation-about-vertical extraction from consecutive rotation matrices

Note: the turn-skill fitness used to track yaw via
atan2(xmat[3], xmat[0]) -- i.e. it assumed the core's local X/Z axes stay
aligned with a fixed "standard" spawn orientation. That assumption breaks for
a body whose core ends up rotated relative to that orientation, or that
tumbles: the same small amount of true rotation can then read as a huge fake
yaw swing, and because the accumulation only ever adds the "correct
direction" delta (never subtracts), noise from that mismeasurement can
accumulate into a large fake reward with no real turning behind it. Some
evolved bodies exploited exactly this. signed_vertical_yaw_delta replaces the
atan2 proxy with the actual geodesic rotation between frames, projected onto
the world-vertical axis -- convention-free, so it can't be fooled by how the
core happens to be oriented.
"""

import math

import numpy as np
import pytest
import torch

from shared import Network, fill_parameters, signed_vertical_yaw_delta


# ── Yaw math helpers ──────────────────────────────────────────────────────────


def rotation_matrix_z(angle: float) -> np.ndarray:
    """3x3 rotation matrix for a rotation of `angle` radians about world Z."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def accumulate_turn_reward(yaw_sequence: list[float], direction: int) -> float:
    """Simulate the per-step accumulation used in run_turn_episode, given a
    sequence of pure-yaw-about-Z angles (as produced by rotation_matrix_z)."""
    accumulated = 0.0
    r_prev = rotation_matrix_z(yaw_sequence[0])
    for yaw_curr in yaw_sequence[1:]:
        r_curr = rotation_matrix_z(yaw_curr)
        delta = signed_vertical_yaw_delta(r_prev, r_curr)
        accumulated += max(0.0, delta * direction)
        r_prev = r_curr
    return accumulated


# ── State-based controller decision logic ────────────────────────────────────

CENTRE_FWD_THRESH = 0.6


class SkillController:
    def __init__(self, initial_dir: int = 3, commit_steps: int = 1) -> None:
        self.last_turn_dir   = initial_dir
        self.commit_steps    = commit_steps
        self._current_skill  = initial_dir
        self._steps_held     = commit_steps  # treat start as already committed

    def _decide(self, left: float, centre: float, right: float) -> int:
        any_visible = (left + centre + right) > 0.0
        if any_visible:
            if centre >= CENTRE_FWD_THRESH or (centre >= left and centre >= right):
                return 1
            elif left >= right:
                self.last_turn_dir = 2
                return 2
            else:
                self.last_turn_dir = 3
                return 3
        return self.last_turn_dir

    def select(self, left: float, centre: float, right: float) -> int:
        self._steps_held += 1
        if self._steps_held < self.commit_steps:
            return self._current_skill
        candidate = self._decide(left, centre, right)
        if candidate != self._current_skill:
            self._current_skill = candidate
            self._steps_held    = 0
        return self._current_skill


# ── Vertical-yaw-delta extraction tests ──────────────────────────────────────


class TestSignedVerticalYawDelta:
    def test_zero_rotation(self):
        """No rotation between frames → delta = 0."""
        R = np.eye(3)
        assert signed_vertical_yaw_delta(R, R) == pytest.approx(0.0, abs=1e-6)

    def test_90_deg_left(self):
        """90° CCW rotation around Z → delta = +π/2."""
        r_prev = np.eye(3)
        r_curr = rotation_matrix_z(math.pi / 2)
        assert signed_vertical_yaw_delta(r_prev, r_curr) == pytest.approx(math.pi / 2, abs=1e-6)

    def test_90_deg_right(self):
        """90° CW rotation around Z → delta = -π/2."""
        r_prev = np.eye(3)
        r_curr = rotation_matrix_z(-math.pi / 2)
        assert signed_vertical_yaw_delta(r_prev, r_curr) == pytest.approx(-math.pi / 2, abs=1e-6)

    def test_near_180_deg(self):
        """~180° rotation about Z → |delta| ~= pi.

        Exactly pi is a singularity for the antisymmetric-part axis
        extraction (sin(theta) = 0, axis undefined) -- signed_vertical_yaw_delta
        returns 0.0 there rather than crashing, since at control-loop step
        sizes an exact pi rotation between consecutive frames never occurs
        in practice. This test checks a value close to, but not exactly, pi.
        """
        r_prev = np.eye(3)
        r_curr = rotation_matrix_z(math.pi - 0.01)
        assert abs(signed_vertical_yaw_delta(r_prev, r_curr)) == pytest.approx(math.pi - 0.01, abs=1e-6)

    def test_exact_180_deg_singularity_returns_zero(self):
        """At exactly pi, the axis is undefined (sin(theta)=0) -- verify this
        known edge case degrades to 0.0 rather than raising or returning NaN."""
        r_prev = np.eye(3)
        r_curr = rotation_matrix_z(math.pi)
        assert signed_vertical_yaw_delta(r_prev, r_curr) == 0.0

    def test_pure_roll_or_pitch_contributes_near_zero(self):
        """Rotation entirely about a horizontal axis (roll/pitch, e.g. a body
        tumbling or tipping over) should not register as vertical-axis
        turning -- this is the failure mode atan2(xmat[3], xmat[0]) was
        vulnerable to."""
        angle = 0.4
        c, s = math.cos(angle), math.sin(angle)
        # Rotation about world X axis (pure roll) -- no yaw component at all.
        r_curr = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c,  -s],
            [0.0, s,   c],
        ])
        delta = signed_vertical_yaw_delta(np.eye(3), r_curr)
        assert delta == pytest.approx(0.0, abs=1e-6)

    def test_robust_to_rotated_core_frame(self):
        """The metric must not depend on which local axis is 'forward'/'up'.
        A body whose core frame is permanently rotated relative to the
        'standard' spawn orientation (e.g. local X aligned with world Z
        instead of local Z) should measure the same vertical-axis turning as
        an identically-turning body in the standard orientation. This is
        exactly the case observed for evolved bodies with unconventional
        core orientations, where atan2(xmat[3], xmat[0]) read tens of
        thousands of fake 'yaw' degrees despite negligible real turning."""
        # A fixed frame swap (local X <-> local Z) applied consistently to
        # both prev and curr should not change the measured vertical delta,
        # since the world-Z axis in the *world* frame is unaffected by how
        # the body's own local axes are labelled.
        angle = 0.5
        r_prev_standard = np.eye(3)
        r_curr_standard = rotation_matrix_z(angle)

        # Swap local X and local Z axes (columns) for both frames -- this
        # simulates a core whose "forward" axis is actually what atan2 would
        # have treated as "up".
        swap = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        r_prev_swapped = r_prev_standard @ swap
        r_curr_swapped = r_curr_standard @ swap

        delta_standard = signed_vertical_yaw_delta(r_prev_standard, r_curr_standard)
        delta_swapped = signed_vertical_yaw_delta(r_prev_swapped, r_curr_swapped)
        assert delta_standard == pytest.approx(delta_swapped, abs=1e-6)
        assert delta_standard == pytest.approx(angle, abs=1e-6)


# ── Turn reward accumulation tests ───────────────────────────────────────────


class TestTurnRewardAccumulation:
    def test_left_turn_positive_reward(self):
        """CCW yaw sequence gives positive reward for direction=+1."""
        yaws = [0.0, 0.1, 0.2, 0.3]  # monotonically increasing
        reward = accumulate_turn_reward(yaws, direction=+1)
        assert reward == pytest.approx(0.3, abs=1e-6)

    def test_right_turn_positive_reward(self):
        """CW yaw sequence gives positive reward for direction=-1."""
        yaws = [0.0, -0.1, -0.2, -0.3]  # monotonically decreasing
        reward = accumulate_turn_reward(yaws, direction=-1)
        assert reward == pytest.approx(0.3, abs=1e-6)

    def test_left_skill_ignores_right_turns(self):
        """CCW skill (direction=+1) doesn't reward CW motion."""
        yaws = [0.0, -0.1, -0.2]  # turning right
        reward = accumulate_turn_reward(yaws, direction=+1)
        assert reward == 0.0

    def test_right_skill_ignores_left_turns(self):
        """CW skill (direction=-1) doesn't reward CCW motion."""
        yaws = [0.0, 0.1, 0.2]  # turning left
        reward = accumulate_turn_reward(yaws, direction=-1)
        assert reward == 0.0

    def test_wrap_around_positive_pi(self):
        """Yaw wrapping near +π is handled correctly."""
        yaws = [math.pi - 0.05, -(math.pi - 0.05)]  # ~0.1 CCW through wrap
        reward = accumulate_turn_reward(yaws, direction=+1)
        assert reward == pytest.approx(0.1, abs=1e-4)

    def test_wrap_around_negative_pi(self):
        """Yaw wrapping near -π is handled correctly for CW direction."""
        yaws = [-(math.pi - 0.05), math.pi - 0.05]  # ~0.1 CW through wrap
        reward = accumulate_turn_reward(yaws, direction=-1)
        assert reward == pytest.approx(0.1, abs=1e-4)

    def test_stationary_no_reward(self):
        """No rotation gives zero reward."""
        yaws = [0.5, 0.5, 0.5, 0.5]
        assert accumulate_turn_reward(yaws, direction=+1) == 0.0
        assert accumulate_turn_reward(yaws, direction=-1) == 0.0

    def test_fitness_sign(self):
        """fitness = -accumulated, so turning more → lower (better) fitness."""
        yaws_fast  = [0.0, 0.3, 0.6, 0.9]
        yaws_slow  = [0.0, 0.1, 0.2, 0.3]
        fit_fast  = -accumulate_turn_reward(yaws_fast, direction=+1)
        fit_slow  = -accumulate_turn_reward(yaws_slow, direction=+1)
        assert fit_fast < fit_slow  # more turning is better (lower fitness value)

    def test_left_right_rewards_are_symmetric(self):
        """Mirrored sequences give equal rewards for mirrored directions."""
        yaws_left  = [0.0, 0.15, 0.30]
        yaws_right = [0.0, -0.15, -0.30]
        r_left  = accumulate_turn_reward(yaws_left,  direction=+1)
        r_right = accumulate_turn_reward(yaws_right, direction=-1)
        assert r_left == pytest.approx(r_right, abs=1e-6)


# ── Controller decision logic tests ──────────────────────────────────────────


class TestSelectSkill:
    # ── forward conditions ────────────────────────────────────────────────────

    def test_centre_above_thresh_goes_forward(self):
        c = SkillController()
        assert c.select(left=0.0, centre=0.6, right=0.0) == 1

    def test_centre_exactly_at_thresh_goes_forward(self):
        c = SkillController()
        assert c.select(left=0.0, centre=0.6, right=0.0) == 1

    def test_centre_below_thresh_but_highest_goes_forward(self):
        """Centre < 0.6 but larger than both sides → forward."""
        c = SkillController()
        assert c.select(left=0.1, centre=0.3, right=0.2) == 1

    def test_centre_equal_to_sides_goes_forward(self):
        """Centre ties with one side and beats the other → forward."""
        c = SkillController()
        assert c.select(left=0.3, centre=0.3, right=0.1) == 1

    def test_centre_below_thresh_and_not_highest_turns(self):
        """Centre < 0.6 and not the max → should turn, not go forward."""
        c = SkillController()
        assert c.select(left=0.5, centre=0.3, right=0.1) == 2

    # ── turning when food is visible ──────────────────────────────────────────

    def test_left_more_than_right_turns_left(self):
        c = SkillController()
        assert c.select(left=0.4, centre=0.0, right=0.1) == 2

    def test_right_more_than_left_turns_right(self):
        c = SkillController()
        assert c.select(left=0.1, centre=0.0, right=0.4) == 3

    def test_left_equal_right_turns_left(self):
        """Tie between left and right → left (left >= right)."""
        c = SkillController()
        assert c.select(left=0.3, centre=0.0, right=0.3) == 2

    # ── memory-based search when nothing visible ──────────────────────────────

    def test_nothing_visible_uses_initial_dir(self):
        c = SkillController(initial_dir=3)
        assert c.select(0.0, 0.0, 0.0) == 3

    def test_nothing_visible_uses_initial_dir_left(self):
        c = SkillController(initial_dir=2)
        assert c.select(0.0, 0.0, 0.0) == 2

    def test_last_dir_updated_when_food_seen_on_right(self):
        c = SkillController(initial_dir=2)   # starts remembering left
        c.select(left=0.0, centre=0.0, right=0.5)   # food on right → remember right
        assert c.select(0.0, 0.0, 0.0) == 3          # blind → turn right

    def test_last_dir_updated_when_food_seen_on_left(self):
        c = SkillController(initial_dir=3)
        c.select(left=0.5, centre=0.0, right=0.0)
        assert c.select(0.0, 0.0, 0.0) == 2

    def test_forward_does_not_update_last_dir(self):
        """Going forward (centre dominant) should not change the search direction."""
        c = SkillController(initial_dir=3)
        c.select(left=0.0, centre=0.8, right=0.0)   # forward
        assert c.select(0.0, 0.0, 0.0) == 3          # memory unchanged

    def test_memory_persists_across_multiple_blind_steps(self):
        c = SkillController(initial_dir=2)
        c.select(left=0.0, centre=0.0, right=0.4)   # see food right → update
        for _ in range(5):
            assert c.select(0.0, 0.0, 0.0) == 3


# ── Commitment tests ──────────────────────────────────────────────────────────


class TestCommitSteps:
    def test_new_skill_locked_for_commit_steps(self):
        """After switching to a new skill it runs for at least commit_steps steps."""
        c = SkillController(initial_dir=3, commit_steps=3)
        # Start already committed as RIGHT; food on left → switch to LEFT, counter=0
        assert c.select(0.5, 0.0, 0.0) == 2   # switch, steps_held=0
        # LEFT must hold for 2 more calls even though vision says RIGHT
        assert c.select(0.0, 0.0, 0.5) == 2   # held (steps_held=1)
        assert c.select(0.0, 0.0, 0.5) == 2   # held (steps_held=2)
        # Now commitment window is satisfied → can switch back to RIGHT
        assert c.select(0.0, 0.0, 0.5) == 3

    def test_same_skill_does_not_reset_counter(self):
        """If vision agrees with the current skill the counter keeps incrementing."""
        c = SkillController(initial_dir=3, commit_steps=3)
        # Start committed as RIGHT; food on left → switch to LEFT
        c.select(0.5, 0.0, 0.0)               # switch to LEFT, steps_held=0
        c.select(0.5, 0.0, 0.0)               # held (steps_held=1, vision still LEFT)
        c.select(0.5, 0.0, 0.0)               # steps_held=2, vision still LEFT — same skill, no reset
        # After 3 steps (0,1,2) minimum is met — next call with RIGHT vision switches
        assert c.select(0.0, 0.0, 0.5) == 3

    def test_commit_steps_1_switches_immediately(self):
        """commit_steps=1: a single step is enough, so vision can switch every call."""
        c = SkillController(initial_dir=3, commit_steps=1)
        assert c.select(0.5, 0.0, 0.0) == 2   # switch to LEFT immediately
        assert c.select(0.0, 0.0, 0.5) == 3   # switch back to RIGHT immediately

    def test_initial_skill_also_locked(self):
        """The initial skill is treated as already having run commit_steps steps,
        so the very first call can switch if vision disagrees."""
        c = SkillController(initial_dir=3, commit_steps=5)
        # Start as RIGHT, food on left → should switch on first call
        assert c.select(0.5, 0.0, 0.0) == 2

    def test_new_skill_locked_full_window_after_switch(self):
        """After switching, the full commit_steps window applies to the new skill."""
        c = SkillController(initial_dir=3, commit_steps=5)
        c.select(0.5, 0.0, 0.0)               # switch to LEFT, steps_held=0
        for _ in range(4):
            assert c.select(0.0, 0.0, 0.5) == 2   # held (steps 1-4)
        assert c.select(0.0, 0.0, 0.5) == 3       # step 5: window done, switch to RIGHT

    def test_memory_updated_on_switch(self):
        """last_turn_dir is updated when a turn skill is selected."""
        c = SkillController(initial_dir=2, commit_steps=3)
        c.select(0.0, 0.0, 0.5)   # switch to RIGHT, memory=right, steps_held=0
        # hold for 2 more steps (commitment)
        c.select(0.0, 0.0, 0.0)
        c.select(0.0, 0.0, 0.0)
        # commitment done; nothing visible → use memory (right)
        assert c.select(0.0, 0.0, 0.0) == 3


# ── Network shape tests ───────────────────────────────────────────────────────


class TestNetworkShape:
    """Verify all three skill networks have the expected input/output sizes."""

    @pytest.fixture(params=[
        ("loco",  11, 8),
        ("left",  11, 8),
        ("right", 11, 8),
    ])
    def skill_net(self, request):
        name, input_size, output_size = request.param
        return name, Network(input_size=input_size, output_size=output_size)

    def test_output_shape(self, skill_net):
        name, net = skill_net
        state = np.zeros(11, dtype=np.float32)
        out = net.forward(None, None, state)
        assert out.shape == (8,), f"{name}: expected (8,), got {out.shape}"

    def test_output_range(self, skill_net):
        """Output should be in (-π/2, π/2) due to Tanh * π/2."""
        name, net = skill_net
        rng = np.random.default_rng(0)
        for _ in range(10):
            state = rng.uniform(-5.0, 5.0, size=11).astype(np.float32)
            out = net.forward(None, None, state)
            assert np.all(out > -math.pi / 2 - 1e-6), f"{name}: output below -π/2"
            assert np.all(out <  math.pi / 2 + 1e-6), f"{name}: output above +π/2"

    def test_fill_parameters_roundtrip(self, skill_net):
        """fill_parameters correctly loads a weight vector into the network."""
        name, net = skill_net
        num_params = sum(p.numel() for p in net.parameters())
        rng = np.random.default_rng(1)
        weights = rng.uniform(-1.0, 1.0, size=num_params).astype(np.float32)
        fill_parameters(net, weights)

        recovered = np.concatenate([p.data.cpu().numpy().ravel() for p in net.parameters()])
        np.testing.assert_allclose(recovered, weights, atol=1e-6,
                                   err_msg=f"{name}: weights not recovered after fill_parameters")

    def test_fill_parameters_length_mismatch_raises(self, skill_net):
        name, net = skill_net
        num_params = sum(p.numel() for p in net.parameters())
        with pytest.raises(IndexError):
            fill_parameters(net, np.zeros(num_params + 1))

    def test_different_weights_give_different_outputs(self, skill_net):
        """Two random weight vectors should produce different actions."""
        name, net = skill_net
        num_params = sum(p.numel() for p in net.parameters())
        rng = np.random.default_rng(2)
        w1 = rng.uniform(-1.0, 1.0, size=num_params).astype(np.float32)
        w2 = rng.uniform(-1.0, 1.0, size=num_params).astype(np.float32)
        state = np.ones(11, dtype=np.float32)

        fill_parameters(net, w1)
        out1 = net.forward(None, None, state).copy()

        fill_parameters(net, w2)
        out2 = net.forward(None, None, state).copy()

        assert not np.allclose(out1, out2), f"{name}: different weights gave identical outputs"


# ── Action blending tests ─────────────────────────────────────────────────────


class TestActionBlending:
    """Verify the CTRL_ALPHA blending rule used in deployment."""

    CTRL_ALPHA = 0.5

    def test_blending_formula(self):
        current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        raw     = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        blended = current * (1.0 - self.CTRL_ALPHA) + raw * self.CTRL_ALPHA
        np.testing.assert_allclose(blended, np.full(8, 0.5))

    def test_blending_clamp(self):
        """Clipped blended action stays within [-π/2, π/2]."""
        current = np.full(8, math.pi / 2)
        raw     = np.full(8, math.pi / 2)
        blended = np.clip(
            current * (1.0 - self.CTRL_ALPHA) + raw * self.CTRL_ALPHA,
            -math.pi / 2, math.pi / 2,
        )
        assert np.all(blended <= math.pi / 2 + 1e-9)
        assert np.all(blended >= -math.pi / 2 - 1e-9)

    def test_blending_is_convex(self):
        """Blended action lies strictly between current and raw."""
        rng = np.random.default_rng(3)
        current = rng.uniform(-1.0, 1.0, 8).astype(np.float32)
        raw     = rng.uniform(-1.0, 1.0, 8).astype(np.float32)
        blended = current * (1.0 - self.CTRL_ALPHA) + raw * self.CTRL_ALPHA
        lo = np.minimum(current, raw)
        hi = np.maximum(current, raw)
        assert np.all(blended >= lo - 1e-6)
        assert np.all(blended <= hi + 1e-6)
