"""The GRN port against the original code (tests/data/grn_reference.json, made by
tools/grn_reference.py from karinemiras/revolve2@593bfc2), and its bodies in ARIEL."""

import json
import math
import random
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

import genomes
import sim
from ariel.ec.genotypes.grn import GRN_R2, crossover_grn, develop, develop_body, modules_of, mutate_grn, to_tree

REF = json.loads((Path(__file__).parent / "data" / "grn_reference.json").read_text())


def record(m) -> dict:
    return {"id": str(m.id), "parent": None if m.parent is None else str(m.parent.id), "slot": m.slot,
            "type": m.type, "rotation": m.rotation, "absolute_rotation": m.absolute_rotation,
            "coords": list(m.coords)}


@pytest.mark.parametrize("case", range(len(REF["develop"])))
def test_development_matches_the_original(case):
    ref = REF["develop"][case]
    core = develop(ref["genome"], GRN_R2, REF["max_modules"])
    if "error" in ref:
        assert core is None
    else:
        assert [record(m) for m in modules_of(core)] == ref["modules"]


def test_reference_covers_small_and_full_bodies():
    sizes = [len(r.get("modules", [])) for r in REF["develop"]]
    assert min(sizes) <= 2 and max(sizes) == REF["max_modules"]


def test_mutation_matches_the_original():
    for case in REF["mutate"]:
        before = list(case["genome"])
        assert mutate_grn(case["genome"], GRN_R2, random.Random(case["seed"])) == case["child"]
        assert case["genome"] == before  # unlike the original, the argument is not changed


def test_crossover_matches_the_original():
    for case in REF["crossover"]:
        assert crossover_grn(case["a"], case["b"], GRN_R2, random.Random(case["seed"])) == case["child"]


# ── ARIEL bodies ──────────────────────────────────────────────────────────────


def _rz(a):
    c, s = round(math.cos(a)), round(math.sin(a))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rx(a):
    c, s = round(math.cos(a)), round(math.sin(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


SLOT_ANGLE = {"CORE": {0: 0, 1: 3 * math.pi / 2, 2: math.pi / 2, 3: math.pi},
              "BRICK": {0: 0, 1: 3 * math.pi / 2, 2: math.pi / 2}, "HINGE": {0: 0}}


def revolve2_frames(core) -> dict[int, np.ndarray]:
    """World rotation of every module as revolve2's body builder composes them
    (x forward, y left, z up): parent, then the slot's yaw, then the module's roll."""
    frames = {core.id: np.eye(3, dtype=int)}
    for m in modules_of(core)[1:]:
        frames[m.id] = frames[m.parent.id] @ _rz(SLOT_ANGLE[m.parent.type][m.slot]) @ _rx(m.rotation * math.pi / 2)
    return frames


def ariel_geometry(tree: dict) -> tuple[dict, dict, np.ndarray]:
    """Per module id: world rotation of its body (brick / hinge stator), joint
    axes of hinges (world), and the core's (forward, left, up) as a matrix."""
    model = sim.body_spec(tree).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frames, axes = {}, {}
    for b in range(model.nbody):
        m = re.match(r"^(?:\d+-\d+-\d+-)*(\d+)-(\d+)-(\d+)-(brick|stator)$", model.body(b).name)
        if m:
            frames[int(m[2])] = data.xmat[b].reshape(3, 3)
    for j in range(model.njnt):
        m = re.match(r"^(?:\d+-\d+-\d+-)*(\d+)-(\d+)-(\d+)-servo$", model.joint(j).name)
        if m:
            axes[int(m[2])] = data.xaxis[j].copy()
    core = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "core")].reshape(3, 3)
    # the core's forward is the direction of its FRONT face: probe it with a child there
    return frames, axes, core


def core_basis() -> np.ndarray:
    """ARIEL world vectors of revolve2's core axes (x forward, y left, z up)."""
    probe = {"nodes": {0: {"type": "CORE", "rotation": "DEG_0"}, 1: {"type": "BRICK", "rotation": "DEG_0"}},
             "edges": [{"parent": 0, "child": 1, "face": "FRONT"}]}
    frames, _, _ = ariel_geometry(probe)
    fwd = frames[1][:, 1]                 # ARIEL modules point along local +y
    up = np.array([0.0, 0.0, 1.0])
    return np.column_stack([fwd, np.cross(up, fwd), up])


@pytest.mark.parametrize("case", [i for i, r in enumerate(REF["develop"]) if len(r.get("modules", [])) >= 8])
def test_ariel_bodies_have_revolve2_geometry(case):
    """Every module points where revolve2's does (so faces and slots match), every
    brick lies flat, and every hinge turns about revolve2's joint axis."""
    core = develop(REF["develop"][case]["genome"], GRN_R2, REF["max_modules"])
    tree = to_tree(core)
    r2 = revolve2_frames(core)
    frames, axes, _ = ariel_geometry(tree)
    g = core_basis()
    for mid, node in tree["nodes"].items():
        if mid == 0:
            continue
        want = g @ r2[mid]                   # revolve2's (forward, left, up) in ARIEL's world
        have = frames[mid]
        np.testing.assert_allclose(have[:, 1], want[:, 0], atol=1e-6)        # forward
        if node["type"] == "BRICK":
            # flat, as in revolve2 (upside down is allowed: the converter then swaps the
            # brick's left and right faces, which its children's forward check covers)
            assert abs(abs(have[:, 2] @ want[:, 2]) - 1) < 1e-6
        else:
            assert abs(abs(axes[mid] @ want[:, 1]) - 1) < 1e-6               # joint axis = revolve2's left axis


def test_leaf_hinges_become_bricks():
    for ref in REF["develop"]:
        tree = develop_body(ref["genome"])
        parents = {e["parent"] for e in tree["edges"]}
        assert all(n["type"] != "HINGE" or k in parents for k, n in tree["nodes"].items())


def test_grn_genome_ops():
    """Random bodies and children are valid, and the stored values regrow the stored body."""
    from ariel.ec.genotypes.tree.collision import prune_colliding_subtrees  # noqa: PLC0415
    from ariel.ec.genotypes.tree.tree_genome import TreeGenome  # noqa: PLC0415

    random.seed(0)
    ops = genomes.genome_ops("grn")
    parents = [genomes.random_body(ops)[0] for _ in range(4)]
    kids = [genomes.make_child(ops, parents, np.random.default_rng(k)).genome for k in range(6)]
    for g in parents + kids:
        assert genomes.is_valid(g)
        tree = TreeGenome.from_dict(develop_body(g["grn"], max_modules=26))
        prune_colliding_subtrees(tree)
        assert json.dumps({**tree.to_dict(), "grn": g["grn"]}, sort_keys=True) == json.dumps(g, sort_keys=True)
