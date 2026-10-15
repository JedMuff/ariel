"""Tests for the zoo simulation preset (examples/d_social_learning
--sim-preset zoo): kgd's zoo module physics and timing.

Each test runs in a fresh subprocess, because the preset changes module-level
constants for the rest of the process.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

SOCIAL_DIR = Path(__file__).resolve().parents[3] / "examples" / "d_social_learning"

_PRELUDE = f"""
import json, random, sys
import mujoco
import numpy as np
sys.path.append({str(SOCIAL_DIR)!r})
random.seed(0)
np.random.seed(0)
import simulator_dependent_functions as sdf
sdf.simulator = sdf.SIMULATOR_ARIEL

def body_with(*types):
    for _ in range(200):
        g = sdf.random_individual()
        found = {{n["type"] for n in g["nodes"].values()}}
        if all(t in found for t in types):
            return g
    raise RuntimeError("no random body with " + str(types))

def first(model, suffix, attr):
    for g in range(model.ngeom):
        if model.body(model.geom_bodyid[g]).name.endswith(suffix):
            return np.asarray(getattr(model, attr)[g]).tolist()

def first_body_mass(model, suffix):
    return next(float(model.body_mass[b]) for b in range(model.nbody)
                if model.body(b).name.endswith(suffix))
"""


def _run(code: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(code)],
        capture_output=True, text=True, check=True, cwd=SOCIAL_DIR,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _model_summary_code(preset: str) -> str:
    return f"""
        sdf.set_sim_preset({preset!r})
        world = sdf.initialize_world(body_with("BRICK", "HINGE"))
        m = world["model"]
        hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        print(json.dumps({{
            "core_size": first(m, "core", "geom_size"),
            "brick_size": first(m, "brick", "geom_size"),
            "stator_size": first(m, "stator", "geom_size"),
            "rotor_size": first(m, "rotor", "geom_size"),
            "core_mass": first_body_mass(m, "core"),
            "brick_mass": first_body_mass(m, "brick"),
            "stator_mass": first_body_mass(m, "stator"),
            "rotor_mass": first_body_mass(m, "rotor"),
            "gainprm": m.actuator_gainprm[:, :3].tolist(),
            "biasprm": m.actuator_biasprm[:, :3].tolist(),
            "limited": [bool(m.jnt_limited[j]) for j in hinges],
            "range": [m.jnt_range[j].tolist() for j in hinges],
            "armature": [float(m.dof_armature[m.jnt_dofadr[j]]) for j in hinges],
        }}))
    """


def test_zoo_preset_builds_zoo_modules():
    import numpy as np
    import pytest

    s = _run(_model_summary_code("zoo"))
    np.testing.assert_allclose(s["core_size"], (0.075, 0.075, 0.075))
    np.testing.assert_allclose(s["brick_size"], (0.0375, 0.0375, 0.0375))
    np.testing.assert_allclose(s["stator_size"], np.array((0.026, 0.022, 0.026)) * 0.99)
    np.testing.assert_allclose(s["rotor_size"], (0.026, 0.0155, 0.026))
    assert s["core_mass"] == pytest.approx(1.363)
    assert s["brick_mass"] == pytest.approx(0.055)
    assert s["stator_mass"] == pytest.approx(0.065)
    assert s["rotor_mass"] == pytest.approx(0.040)
    kp, kv = 1.3605113362038874, 0.3555497840137177
    for gain, bias in zip(s["gainprm"], s["biasprm"], strict=True):
        np.testing.assert_allclose(gain, (kp, 0, 0))
        np.testing.assert_allclose(bias, (0, -kp, -kv))
    assert s["limited"] and all(s["limited"])
    np.testing.assert_allclose(s["range"], [(-np.pi / 2, np.pi / 2)] * len(s["range"]))
    np.testing.assert_allclose(s["armature"], 0.08117047736037042)


def test_legacy_preset_keeps_local_modules():
    import numpy as np
    import pytest

    s = _run(_model_summary_code("legacy"))
    np.testing.assert_allclose(s["core_size"], (0.1, 0.1, 0.1))
    np.testing.assert_allclose(s["brick_size"], (0.05, 0.05, 0.05))
    assert s["core_mass"] == pytest.approx(1.0)
    assert s["stator_mass"] == pytest.approx(0.02)
    for gain in s["gainprm"]:
        assert gain[0] == 1.0
    assert not any(s["limited"])


def test_zoo_preset_timing_and_displacement():
    s = _run("""
        sdf.set_sim_preset("zoo")
        from sim_ariel import evaluator as ev
        from ariel.simulation.controllers.distributed_mlp import DistributedMLP
        world = sdf.initialize_world(body_with("HINGE"))
        brain = DistributedMLP(n_neighbors=sdf.n_neighbours(), hidden=4)
        ep = sdf.run_episode(np.zeros(brain.n_params), brain, world)
        print(json.dumps({
            "duration": ev.DURATION, "ctrl_every": ev.CTRL_EVERY,
            "dt": world["model"].opt.timestep, "fitness": ep["fitness"],
            "time": float(world["data"].time),
        }))
    """)
    assert s["duration"] == 15.0
    assert s["ctrl_every"] * s["dt"] == 0.05  # 20 Hz control
    assert abs(s["time"] - 18.0) < 0.01  # 3 s settle + 15 s rollout
    # A zero controller holds still: displacement from the start is ~0, not
    # the spawn x of -0.8 m.
    assert abs(s["fitness"]) < 0.1


def test_zoo_preset_collision_checks_use_zoo_geometry():
    s = _run("""
        from ariel.body_phenotypes.robogen_lite.collision_utils import module_template
        from ariel.body_phenotypes.robogen_lite.modules import core
        sdf.set_sim_preset("zoo")
        print(json.dumps({"core_dims": list(core.CORE_DIMENSIONS),
                          "cache": module_template.cache_info().currsize}))
    """)
    assert s["core_dims"] == [0.075, 0.075, 0.075]
    assert s["cache"] == 0


def test_topdown_similarity_is_top_down_tree_distance():
    s = _run("""
        from sim_ariel.tree_edit_distance import top_down_tree_distance
        print(json.dumps({"same": sdf.similarity_function("TOPDOWN") is top_down_tree_distance}))
    """)
    assert s["same"]
