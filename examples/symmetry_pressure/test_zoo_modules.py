"""shared.with_zoo_modules (gecko_skill_tasks.py --zoo-modules) must match
kgd's zoo modules, which examples/zoo_benchmark vendors in zoo_modules.py."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

import shared

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "zoo_benchmark"))
import zoo_benchmark  # noqa: E402
import zoo_modules  # noqa: E402


SUFFIXES = ("core", "brick", "stator", "rotor")


def _body_with_hinges() -> dict:
    """A random tree body with a brick; create_individual guarantees >= MIN_HINGES hinges."""
    for _ in range(100):
        genome = shared.create_individual(12, 6).genotype["morph"]
        if "BRICK" in str(genome):
            return genome
    pytest.skip("no random body with a brick")


def _first_mass(model: mujoco.MjModel, suffix: str) -> float:
    return next(float(model.body_mass[b]) for b in range(model.nbody) if model.body(b).name.endswith(suffix))


def _first_size(model: mujoco.MjModel, suffix: str) -> np.ndarray:
    return next(model.geom_size[g] for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.endswith(suffix))


def test_constants_match_vendored_zoo_modules():
    assert shared.ZOO_HINGE_KP == zoo_modules.HINGE_KP
    assert shared.ZOO_HINGE_KV == zoo_modules.HINGE_KV
    assert shared.ZOO_HINGE_ARMATURE == zoo_modules.HINGE_ARMATURE
    assert shared.ZOO_HINGE_RANGE == pytest.approx(zoo_modules.CTRL_RANGE)
    assert shared.ZOO_CORE_MASS == zoo_modules.CORE_MASS
    assert shared.ZOO_STATOR_MASS == zoo_modules.STATOR_MASS
    assert shared.ZOO_ROTOR_MASS == zoo_modules.ROTOR_MASS
    assert shared.ZOO_BRICK_MASS == zoo_modules.BRICK_MASS
    assert shared.ZOO_CORE_DIMENSIONS == zoo_modules.CORE_DIMENSIONS
    assert shared.ZOO_BRICK_DIMENSIONS == zoo_modules.BRICK_DIMENSIONS
    assert shared.ZOO_STATOR_DIMENSIONS == zoo_modules.STATOR_DIMENSIONS
    assert shared.ZOO_ROTOR_DIMENSIONS == zoo_modules.ROTOR_DIMENSIONS


def test_compiled_model_matches_zoo_spider():
    genome = _body_with_hinges()
    ours, _ = shared.build_loco_world_for_body(genome, shared.with_zoo_modules())
    zoo, _ = zoo_benchmark.build_world("spider")
    hinge = lambda m: [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    for m in (ours, zoo):
        for j in hinge(m):
            assert m.jnt_limited[j]
            np.testing.assert_allclose(m.jnt_range[j], zoo_modules.CTRL_RANGE)
            assert m.dof_armature[m.jnt_dofadr[j]] == pytest.approx(zoo_modules.HINGE_ARMATURE)
    np.testing.assert_allclose(ours.actuator_gainprm[:, :3], np.broadcast_to(zoo.actuator_gainprm[0, :3], (ours.nu, 3)))
    np.testing.assert_allclose(ours.actuator_biasprm[:, :3], np.broadcast_to(zoo.actuator_biasprm[0, :3], (ours.nu, 3)))
    np.testing.assert_allclose(ours.actuator_ctrlrange, np.broadcast_to(zoo.actuator_ctrlrange[0], (ours.nu, 2)))
    for suffix in (*SUFFIXES, "hinge"):
        assert _first_mass(ours, suffix) == pytest.approx(_first_mass(zoo, suffix)), suffix
    for suffix in SUFFIXES:
        np.testing.assert_allclose(_first_size(ours, suffix), _first_size(zoo, suffix), err_msg=suffix)


def test_default_build_is_unchanged():
    genome = _body_with_hinges()
    shared.build_loco_world_for_body(genome, shared.with_zoo_modules())  # must restore the local constants
    model, _ = shared.build_loco_world_for_body(genome)
    np.testing.assert_allclose(_first_size(model, "core"), 0.1)
    np.testing.assert_allclose(_first_size(model, "brick"), 0.05)
    assert _first_mass(model, "core") == 1.0
    assert _first_mass(model, "stator") == pytest.approx(0.02)
    assert model.actuator_gainprm[0, 0] == 1.0 and not model.jnt_limited[model.actuator_trnid[0, 0]]


def _hinge_joints(m: mujoco.MjModel) -> list[int]:
    return [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def test_default_actuator_is_zoo_hinge():
    genome = _body_with_hinges()
    default, _ = shared.build_loco_world_for_body(genome, shared.with_zoo_modules())
    explicit, _ = shared.build_loco_world_for_body(genome, shared.with_zoo_modules(actuator=shared.ZOO_HINGE))
    for field in ("actuator_gainprm", "actuator_biasprm", "dof_armature", "jnt_limited", "jnt_range"):
        np.testing.assert_array_equal(getattr(default, field), getattr(explicit, field), err_msg=field)


@pytest.mark.parametrize("actuator", [shared.LOCAL_HINGE, shared.HingeActuator(2.5, 2.0, 0.081, True)])
def test_custom_actuator_on_zoo_bodies(actuator):
    genome = _body_with_hinges()
    model, _ = shared.build_loco_world_for_body(genome, shared.with_zoo_modules(actuator=actuator))
    zoo, _ = shared.build_loco_world_for_body(genome, shared.with_zoo_modules())
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], actuator.kp)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1:3], np.broadcast_to([-actuator.kp, -actuator.kv], (model.nu, 2)))
    np.testing.assert_allclose(model.actuator_ctrlrange, np.broadcast_to(zoo_modules.CTRL_RANGE, (model.nu, 2)))
    for j in _hinge_joints(model):
        assert bool(model.jnt_limited[j]) == actuator.limited
        if actuator.limited:
            np.testing.assert_allclose(model.jnt_range[j], zoo_modules.CTRL_RANGE)
        assert model.dof_armature[model.jnt_dofadr[j]] == pytest.approx(actuator.armature)
    np.testing.assert_allclose(model.body_mass, zoo.body_mass)  # zoo sizes and masses either way
    np.testing.assert_allclose(model.geom_size, zoo.geom_size)


def test_actuator_label():
    assert shared.LOCAL_HINGE.label == "kp1_kv1_arm0_lim0"
    assert shared.HingeActuator(1.36, 0.356, 0.081, True).label == "kp1.36_kv0.356_arm0.081_lim1"


def test_bang_bang_brain():
    genome = _body_with_hinges()
    model, data = shared.build_loco_world_for_body(genome, shared.with_zoo_modules())
    brain = shared.make_brain_for_model("bang_bang", model, data, duration=15.0)
    assert brain.num_params == 4 + 5 * model.nu
    brain.set_params(np.full(brain.num_params, shared.BRAIN_X0["bang_bang"]))
    state = shared.get_robot_state(data).astype(np.float32)
    outs = []
    for k in range(40):  # 2 s at 20 Hz
        data.time = shared.SETTLE_DURATION + k * 0.05
        outs.append(brain.act(data, state))
    outs = np.array(outs)
    assert outs.shape == (40, model.nu)
    assert np.all(np.abs(outs) <= np.pi / 2 + 1e-9)
    assert np.allclose(np.abs(outs), np.pi / 2)  # instant switches: always at +-90 deg
    assert len(np.unique(np.sign(outs[:, 0]))) == 2  # it switches within 2 s
