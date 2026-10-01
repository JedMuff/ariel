"""Sanity checks for the zoo benchmark (run: pytest test_zoo_benchmark.py)."""

import math

import numpy as np
import pytest

import canonical_bodies
from brains import BRAIN_KINDS, Matsuoka, make_brain
from zoo_benchmark import build_world, make_brain_for, run_episode

EXPECTED_HINGES = {"spider": 8, "gecko": 6, "snake": 8, "salamander": 14}


@pytest.mark.parametrize("body", sorted(canonical_bodies.get_all()))
def test_body_builds(body):
    model, _ = build_world(body)
    assert model.nu > 0
    assert model.jnt_limited[1:].all()  # every hinge limited to +-90 deg (kgd hinge)
    np.testing.assert_allclose(np.abs(model.actuator_ctrlrange), math.pi / 2)
    if body in EXPECTED_HINGES:
        assert model.nu == EXPECTED_HINGES[body]


def test_body_count():
    assert len(canonical_bodies.get_all()) == 23


@pytest.mark.parametrize("n", [1, 6, 14])
def test_num_params(n):
    expected = {
        "sine": 3 * n + 1,
        "revolve_cpg": n + n * (n - 1) // 2,
        "matsuoka": 5 * n + n * (n - 1),
        "square": 12 * n,
        "square_sync": 8 * n + 4,
        "bang_bang": 5 * n + 4,
    }
    for kind, count in expected.items():
        assert make_brain(kind, 3 + n, n).num_params == count


@pytest.mark.parametrize("kind", BRAIN_KINDS)
def test_outputs_bounded_and_deterministic(kind):
    model, data = build_world("gecko")
    brain = make_brain_for(kind, model, data)
    rng = np.random.default_rng(0)
    brain.set_params(rng.normal(0, 2, brain.num_params))
    state = np.zeros(3 + model.nu)

    def rollout():
        brain.reset()
        return np.array([brain.act(k / 20, state) for k in range(300)])

    a, b = rollout(), rollout()
    assert np.all(np.abs(a) <= math.pi / 2 + 1e-9)
    np.testing.assert_array_equal(a, b)


def test_matsuoka_oscillates():
    brain = Matsuoka(0, 4)
    brain.set_params(np.zeros(brain.num_params))  # mid-range genes
    out = np.array([brain.act(k / 20, None) for k in range(300)])
    assert np.all(out[100:].std(axis=0) > 0.1)


def test_zero_action_does_not_move():
    model, data = build_world("spider")

    class Zero:
        def reset(self):
            pass

        def act(self, t, state):
            return np.zeros(model.nu)

    assert abs(run_episode(model, data, Zero(), 15, 20)) < 1e-3


def test_bang_bang_is_bang_bang():
    model, data = build_world("gecko")
    brain = make_brain_for("bang_bang", model, data)
    brain.set_params(np.random.default_rng(0).normal(0, 2, brain.num_params))
    out = np.array([brain.act(k / 20, None) for k in range(300)])
    assert set(np.unique(np.abs(out))) == {math.pi / 2}
    assert np.all(out.std(axis=0) > 0)  # every hinge switches
