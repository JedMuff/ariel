"""Body physics, legacy profile and ANN port for the head-mass experiment."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from brains import make_brain_for_model
from sim import (
    LEGACY_PROFILE,
    LOCAL_CORE_MASS,
    LOCAL_HINGE,
    PROFILE,
    ZOO_CORE_MASS,
    ZOO_HINGE,
    BodyPhysics,
    build_world,
    genome_to_spec,
    physics_spec,
    physics_spec_fn,
    profile_for,
    run_skill_episode,
    zoo_spec,
)

REPO = Path(__file__).resolve().parents[3]
BODIES = REPO / "__data__/ariel_paper_v2/e2/bodies_random.jsonl"
OLD_ROOT = REPO / "__data__/ariel_control_stride_sweep"


def _genome() -> dict:
    from ariel.ec.genotypes.tree.tree_genome import TreeGenome  # noqa: PLC0415
    from sim import random_tree  # noqa: PLC0415

    np.random.seed(1)
    import random  # noqa: PLC0415

    random.seed(1)
    while True:
        g = random_tree(20).to_dict()
        s = genome_to_spec(g)
        if s is not None and s.compile().nu >= 4:
            assert TreeGenome.from_dict(g)
            return g


def _core_mass(model: mujoco.MjModel) -> float:
    return float(model.body("core").mass[0])


@pytest.fixture(scope="module")
def genome() -> dict:
    return _genome()


def test_default_physics_matches_zoo_spec(genome: dict) -> None:
    a, b = physics_spec(genome).compile(), zoo_spec(genome).compile()
    assert np.array_equal(a.body_mass, b.body_mass)
    assert np.array_equal(a.geom_size, b.geom_size)
    assert np.array_equal(a.actuator_gainprm, b.actuator_gainprm)
    assert np.array_equal(a.actuator_biasprm, b.actuator_biasprm)
    assert np.array_equal(a.dof_armature, b.dof_armature)
    assert np.array_equal(a.jnt_limited, b.jnt_limited)


_CLEAN_LOCAL_BODY = """
import json, sys
import numpy as np
from sim import genome_to_spec
m = genome_to_spec(json.loads(sys.stdin.read())).compile()
print(json.dumps({k: np.asarray(getattr(m, k)).tolist() for k in FIELDS}))
"""
_LOCAL_FIELDS = ("body_mass", "geom_size", "body_pos", "actuator_gainprm", "actuator_biasprm",
                 "dof_armature", "dof_damping", "jnt_limited")


def test_local_physics_matches_the_module_classes(genome: dict) -> None:
    """The old body, through the override path, equals the hard-coded classes.
    Those are read in a fresh process: other tests set the constants globally."""
    code = f"FIELDS = {_LOCAL_FIELDS!r}\n{_CLEAN_LOCAL_BODY}"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    out = subprocess.run([sys.executable, "-c", code], input=json.dumps(genome), capture_output=True,
                         text=True, check=True, env=env)
    clean = json.loads(out.stdout.strip().splitlines()[-1])
    a = physics_spec(genome, BodyPhysics("local", None, LOCAL_HINGE)).compile()
    for field in _LOCAL_FIELDS:
        assert np.array_equal(np.asarray(getattr(a, field)), np.asarray(clean[field])), field


@pytest.mark.parametrize("module_set", ["zoo", "local"])
@pytest.mark.parametrize("core_mass", [0.25, 2.0])
def test_core_mass_changes_only_the_core(genome: dict, module_set: str, core_mass: float) -> None:
    base = physics_spec(genome, BodyPhysics(module_set)).compile()
    heavy = physics_spec(genome, BodyPhysics(module_set, core_mass)).compile()
    assert _core_mass(base) == pytest.approx(ZOO_CORE_MASS if module_set == "zoo" else LOCAL_CORE_MASS)
    assert _core_mass(heavy) == pytest.approx(core_mass)
    i = heavy.body("core").id
    other = np.arange(heavy.nbody) != i
    assert np.array_equal(base.body_mass[other], heavy.body_mass[other])
    assert np.array_equal(base.geom_size, heavy.geom_size)


def test_physics_spec_restores_the_module_constants(genome: dict) -> None:
    before = genome_to_spec(genome).compile().body_mass.copy()
    physics_spec(genome, BodyPhysics("zoo", 2.0))
    assert np.array_equal(before, genome_to_spec(genome).compile().body_mass)


def test_ann_is_deterministic(genome: dict) -> None:
    profile = replace(PROFILE, fitness="displacement", loco_duration=1.0)
    model, data = build_world(genome, physics_spec_fn(BodyPhysics()))
    brain = make_brain_for_model("ann", model, data)
    params = np.random.default_rng(0).uniform(-1, 1, brain.num_params)
    brain.set_params(params)
    a, _ = run_skill_episode(model, data, brain, profile)
    b, _ = run_skill_episode(model, data, brain, profile)
    assert a == b


def test_profile_for_reads_the_legacy_profile() -> None:
    assert profile_for({"eval_profile": "legacy"}) == LEGACY_PROFILE
    assert profile_for({"fitness": "displacement"}).fitness == "displacement"
    assert ZOO_HINGE.limited and not LOCAL_HINGE.limited


@pytest.mark.skipif(not OLD_ROOT.exists(), reason="old sweep data not present")
@pytest.mark.parametrize("run", ["ctrlstride_forward_tree_symmetric_s20_rep0_43569_3"])
def test_ann_port_reproduces_old_checkpoints(run: str) -> None:
    """Old s20 ANN champions score their stored fitness under the port."""
    ckpts = sorted((OLD_ROOT / run / "__data__/gecko_skill_tasks/forward/checkpoints").glob("gen030_body*"))[:4]
    assert ckpts
    to_spec = physics_spec_fn(BodyPhysics("local", None, LOCAL_HINGE))
    for c in ckpts:
        genome = json.loads((c / "best_genome.json").read_text())
        stored = json.loads((c / "meta.json").read_text())
        model, data = build_world(genome, to_spec)
        brain = make_brain_for_model("ann", model, data, np.load(c / "fwd_weights.npy"))
        fit, _ = run_skill_episode(model, data, brain, LEGACY_PROFILE)
        assert fit == pytest.approx(stored["fitness"], abs=1e-9), c.name
