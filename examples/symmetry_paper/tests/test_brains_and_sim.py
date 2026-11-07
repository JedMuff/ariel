"""Brains, the zoo body, the episode and controller training."""

import importlib.util
import random
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

import genomes
from brains import AnnBrain, BangBangBrain, DannBrain, make_brain_for_model, model_topology
from config import BRAIN_KINDS, PROTOCOL
from sim import (
    ZOO_HINGE_ARMATURE,
    ZOO_HINGE_KP,
    ZOO_HINGE_KV,
    build_world,
    evaluate,
    make_pool,
    run_episode,
    train_controller,
    train_controllers,
)

LEGACY = Path(__file__).resolve().parents[1] / "legacy"


def body(genome_type: str = "tree_symmetric", seed: int = 0) -> dict:
    random.seed(seed)
    return genomes.random_body(genomes.genome_ops(genome_type))[0]


# ── Brains ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", BRAIN_KINDS)
def test_param_counts(kind):
    model, data = build_world(body())
    n = model.nu
    expected = {"bang_bang": 4 + 5 * n, "dann": 16 * 42 + 16 + 16 + 1, "ann": 1184 + 65 * n,
                "cpg": n + n * (n - 1) // 2}[kind]
    assert make_brain_for_model(kind, model, data).num_params == expected
    assert BangBangBrain(4).num_params == 24


@pytest.mark.parametrize("genome_type", ["tree", "tree_symmetric"])
def test_dann_builds_and_resets(genome_type):
    model, data = build_world(body(genome_type))
    brain = DannBrain(model, data)
    brain.set_params(np.random.default_rng(0).normal(size=brain.num_params))
    state = np.zeros(3 + model.nu, dtype=np.float32)
    a1 = brain.act(data, state)
    assert a1.shape == (model.nu,)
    assert np.all(np.abs(a1) <= np.pi / 2)
    brain.act(data, state)
    brain.reset()
    assert np.array_equal(brain.act(data, state), a1)


@pytest.mark.parametrize("kind", BRAIN_KINDS)
def test_wrong_length_is_rejected(kind):
    model, data = build_world(body())
    with pytest.raises(IndexError):
        make_brain_for_model(kind, model, data).set_params(np.zeros(3))


def test_model_topology_has_every_hinge():
    model, _ = build_world(body())
    topo = model_topology(model)
    assert sum(t == 2 for t in topo["type"].values()) == model.nu


def test_ann_matches_the_torch_version():
    """The numpy ANN gives the zoo / v2 torch ANN's outputs for the same weights."""
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("legacy_brains", LEGACY / "brains.py")
    legacy = importlib.util.module_from_spec(spec)
    sys.modules["legacy_brains"] = legacy  # dataclasses look their module up
    spec.loader.exec_module(legacy)
    rng = np.random.default_rng(0)
    for n in (4, 9, 17):
        new, old = AnnBrain(3 + n, n), legacy.AnnBrain(3 + n, n)
        assert new.num_params == old.num_params == 1184 + 65 * n
        w = rng.normal(0, 0.5, new.num_params).astype(np.float32)
        new.set_params(w)
        old.set_params(w)
        for _ in range(5):
            x = rng.normal(0, 1, 3 + n).astype(np.float32)
            np.testing.assert_allclose(new.act(None, x), old.act(None, x), atol=1e-5)  # float32 sum order


# ── Body and episode ──────────────────────────────────────────────────────────


def test_zoo_hinge():
    model, _ = build_world(body())
    assert model.actuator_gainprm[0, 0] == pytest.approx(ZOO_HINGE_KP)
    assert model.actuator_biasprm[0, 2] == pytest.approx(-ZOO_HINGE_KV)
    hinge = int(model.actuator_trnid[0, 0])
    assert model.dof_armature[model.jnt_dofadr[hinge]] == pytest.approx(ZOO_HINGE_ARMATURE)
    assert model.jnt_limited[hinge]


def test_episode_is_deterministic():
    model, data = build_world(body("tree"))
    brain = make_brain_for_model("bang_bang", model, data, np.full(4 + 5 * model.nu, 0.5, dtype=np.float32))
    a = run_episode(model, data, brain)
    mujoco.mj_resetData(model, data)
    assert run_episode(model, data, brain) == a
    assert evaluate(body("tree"), "bang_bang", np.full(4 + 5 * model.nu, 0.5)) == a


def test_episode_length_and_control_rate():
    model, data = build_world(body())
    brain = make_brain_for_model("cpg", model, data)
    calls = []
    act = brain.act
    brain.act = lambda d, s: calls.append(d.time) or act(d, s)
    steps = []
    run_episode(model, data, brain, recorder=lambda d: steps.append(d.time))
    assert len(steps) == pytest.approx(PROTOCOL.episode_s / model.opt.timestep, abs=1)
    assert len(calls) == pytest.approx(len(steps) / PROTOCOL.control_every, abs=1)


# ── Training ──────────────────────────────────────────────────────────────────


def test_training_curve_and_budget():
    with make_pool(4) as pool:
        t = train_controller(body(), "bang_bang", 60, seed=1, pool=pool)
    assert t.evals == 60 and len(t.curve["best"]) == 3
    assert t.score == max(t.curve["best"])
    assert all(b >= m for b, m in zip(t.curve["best"], t.curve["median"]))


def test_concurrent_training_is_deterministic():
    """A training gives the same result alone or alongside others."""
    jobs = [(body("tree", s), kind, 40, 7 + s) for s, kind in enumerate(["bang_bang", "cpg", "dann"])]
    with make_pool(6) as pool:
        alone = [train_controller(*job, pool=pool) for job in jobs]
        together = train_controllers(jobs, pool, concurrent=3)
    for a, b in zip(alone, together):
        assert a.score == b.score
        np.testing.assert_array_equal(a.weights, b.weights)
    # and the stored weights reproduce the score
    assert evaluate(jobs[0][0], "bang_bang", alone[0].weights) == alone[0].score
