"""Brains, the hinge actuator and the episode loop."""

import random
from dataclasses import asdict

import mujoco
import numpy as np
import pytest

import replay
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from brains import BangBangBrain, BangBangFreeBrain, DannBrain, make_brain_for_model, model_topology
from sim import (
    PROFILE,
    TRANSLATE_MAX,
    ZOO_HINGE,
    HingeActuator,
    build_world,
    create_individual,
    run_skill_episode,
    translate_score,
    zoo_spec,
    zoo_spec_fn,
)


def body(genome_type: str = "tree_symmetric", seed: int = 0) -> dict:
    random.seed(seed)
    axis = MirrorAxis.Y_ZERO if genome_type == "tree_symmetric" else None
    return create_individual(25, 25, symmetry_axis=axis).genotype["morph"]


# ── bang_bang ─────────────────────────────────────────────────────────────────


def test_bang_bang_free_with_equal_frequencies_is_bang_bang():
    n, rng = 5, np.random.default_rng(0)
    shared = BangBangBrain(n)
    g = rng.normal(size=shared.num_params)
    shared.set_params(g)
    per_hinge = g[4:].reshape(n, 5)
    free = BangBangFreeBrain(n)
    free.set_params(np.hstack([np.tile(g[:4], (n, 1)), per_hinge]).ravel())
    for t in np.arange(0.0, 15.0, 0.05):
        assert np.array_equal(shared.output_at(t), free.output_at(t))


def test_bang_bang_free_frequencies_drift_apart():
    n = 2
    free = BangBangFreeBrain(n)
    g = np.full((n, 9), 0.5)
    g[1, 0:2] = 2.0  # hinge 1: higher start and end frequency
    free.set_params(g.ravel())
    out = np.array([free.output_at(t) for t in np.arange(0.0, 15.0, 0.05)])
    assert set(np.unique(out)) <= {-np.pi / 2, np.pi / 2}
    assert (out[:, 0] != out[:, 1]).mean() > 0.2


def test_bang_bang_param_counts():
    assert BangBangBrain(4).num_params == 24
    assert BangBangFreeBrain(4).num_params == 36


# ── DANN ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("genome_type", ["tree", "tree_symmetric"])
def test_dann_builds_and_resets(genome_type):
    model, data = build_world(body(genome_type))
    brain = DannBrain(model, data)
    assert brain.num_params == 16 * 42 + 16 + 16 + 1
    brain.set_params(np.random.default_rng(0).normal(size=brain.num_params))
    state = np.zeros(3 + model.nu, dtype=np.float32)
    a1 = brain.act(data, state)
    assert a1.shape == (model.nu,)
    assert np.all(np.abs(a1) <= np.pi / 2)
    brain.act(data, state)
    brain.reset()
    assert np.array_equal(brain.act(data, state), a1)


def test_dann_rejects_wrong_length():
    model, data = build_world(body())
    with pytest.raises(IndexError):
        DannBrain(model, data).set_params(np.zeros(3))


def test_model_topology_has_every_hinge():
    model, _ = build_world(body())
    topo = model_topology(model)
    assert sum(t == 2 for t in topo["type"].values()) == model.nu


# ── Actuator and episode ──────────────────────────────────────────────────────


def test_default_actuator_is_zoo_hinge():
    model, _ = build_world(body(), zoo_spec)
    assert model.actuator_gainprm[0, 0] == pytest.approx(ZOO_HINGE.kp)
    assert model.actuator_biasprm[0, 2] == pytest.approx(-ZOO_HINGE.kv)
    hinge = int(model.actuator_trnid[0, 0])
    assert model.dof_armature[model.jnt_dofadr[hinge]] == pytest.approx(ZOO_HINGE.armature)
    assert model.jnt_limited[hinge]


def test_custom_actuator():
    act = HingeActuator(2.5, 1.0, 0.0, False)
    model, _ = build_world(body(), zoo_spec_fn(act))
    assert model.actuator_gainprm[0, 0] == pytest.approx(2.5)
    assert model.actuator_biasprm[0, 2] == pytest.approx(-1.0)
    assert not model.jnt_limited[int(model.actuator_trnid[0, 0])]
    assert act.label == "kp2.5_kv1_arm0_lim0"


def test_translate_score_clips_to_0_2():
    assert translate_score(-1.0, 15.0) == 0.0
    assert translate_score(100.0, 15.0) == TRANSLATE_MAX
    assert translate_score(0.5 * 15.0 / 4, 15.0) == pytest.approx(0.5)


@pytest.mark.parametrize("brain_kind", ["bang_bang", "bang_bang_free", "dann"])
def test_replay_reproduces_training_episode(brain_kind):
    genome = body()
    model, data = build_world(genome)
    brain = make_brain_for_model(brain_kind, model, data, duration=PROFILE.loco_duration)
    w = np.random.default_rng(1).normal(0.5, 0.5, size=brain.num_params).astype(np.float32)
    brain.set_params(w)
    fit, _ = run_skill_episode(model, data, brain)
    cfg = {"hinge_actuator": asdict(ZOO_HINGE)}
    fit2, _, P, Q, dt = replay.replay_episode(genome, w, brain_kind, cfg)
    assert fit2 == fit
    assert dt == pytest.approx(0.02, abs=1e-6)
    assert P.shape[0] == pytest.approx(PROFILE.loco_duration / dt, abs=2)
    assert Q.shape[-1] == 4


def test_episode_is_deterministic():
    genome = body("tree")
    model, data = build_world(genome)
    brain = make_brain_for_model("bang_bang", model, data, np.full(4 + 5 * model.nu, 0.5, dtype=np.float32))
    a = run_skill_episode(model, data, brain)
    mujoco.mj_resetData(model, data)
    assert run_skill_episode(model, data, brain) == a
