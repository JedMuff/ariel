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
