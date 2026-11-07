"""Brick-only mutants of the added-blocks experiment (added_blocks.py)."""

import random

import numpy as np
import pytest
from added_blocks import ARMS, add_bricks
from analyse_lineages import body_diff, change_class, module_paths
from ariel.ec.genotypes.tree.tying import random_tree_tied
from sim import build_world, create_individual, use_zoo_modules_globally, zoo_spec


@pytest.fixture(scope="module")
def parents():
    use_zoo_modules_globally()
    random.seed(5)
    np.random.seed(5)
    bodies = [random_tree_tied(25, 0.47).to_dict() for _ in range(6)]
    bodies += [create_individual(25, 25).genotype["morph"] for _ in range(6)]
    return bodies


@pytest.mark.parametrize("arm", ARMS)
def test_arm_adds_only_bricks_and_keeps_hinges(parents, arm) -> None:
    made = 0
    for i, g in enumerate(parents):
        out = add_bricks(g, arm, random.Random(i))
        if out is None:
            continue
        made += 1
        child, sites = out
        d = body_diff(g, child)
        expected = {"identical": 0, "one": 1, "two_same": 2, "two_diff": 2, "chain": 2}[arm]
        assert d["added"] == expected and d["removed"] == 0 and d["changed"] == 0
        assert d["added_hinges"] == 0
        assert d["sites"] == {"identical": 0, "one": 1, "two_same": 2, "two_diff": 2, "chain": 1}[arm]
        assert len(sites) == expected
        assert build_world(child, zoo_spec)[0].nu == build_world(g, zoo_spec)[0].nu
    assert made > 0


def _added_rotations(parent, child):
    p, c = module_paths(parent), module_paths(child)
    return [c[k][1] for k in set(c) - set(p)]


def test_same_and_different_rotations(parents) -> None:
    for i, g in enumerate(parents):
        same = add_bricks(g, "two_same", random.Random(i))
        diff = add_bricks(g, "two_diff", random.Random(i))
        if same:
            assert len(set(_added_rotations(g, same[0]))) == 1
        if diff:
            assert len(set(_added_rotations(g, diff[0]))) == 2


def test_parent_is_not_changed(parents) -> None:
    g = parents[0]
    before = module_paths(g)
    add_bricks(g, "two_same", random.Random(0))
    assert module_paths(g) == before


def test_same_seed_makes_the_same_mutant(parents) -> None:
    a = add_bricks(parents[0], "one", random.Random(3))
    b = add_bricks(parents[0], "one", random.Random(3))
    assert a == b


def test_change_classes() -> None:
    base = {"added": 0, "removed": 0, "changed": 0}
    assert change_class(base) == "identical"
    assert change_class({**base, "added": 1}) == "add"
    assert change_class({**base, "removed": 1}) == "remove"
    assert change_class({**base, "changed": 1}) == "change"
    assert change_class({**base, "added": 1, "changed": 1}) == "mixed"
