"""Every genome yields only valid bodies: the module cap, hinge minimum and depth."""

import random

import numpy as np
import pytest

import genomes
from config import GENOME_TYPES, PROTOCOL

TREE_GENOMES = [g for g in GENOME_TYPES if g != "grn"]


def assert_valid(genome: dict) -> None:
    assert genomes.module_count(genome) <= PROTOCOL.max_modules
    assert genomes.hinge_count(genome) >= PROTOCOL.min_hinges
    assert genomes.is_valid(genome)


@pytest.mark.parametrize("name", GENOME_TYPES)
def test_offspring_respect_the_cap(name):
    """Evolve a population of random bodies for a few generations of
    unselected offspring: every body stays within the cap."""
    if name == "grn":
        pytest.importorskip("ariel.ec.genotypes.grn")
    random.seed(1)
    rng = np.random.default_rng(1)
    ops = genomes.genome_ops(name)
    pop = [genomes.random_body(ops)[0] for _ in range(10)]
    for g in pop:
        assert_valid(g)
    for _ in range(8):
        kids = [genomes.make_child(ops, pop, rng) for _ in range(20)]
        for k in kids:
            assert_valid(k.genome)
            assert k.op in {"crossover", "mutation", "fallback"}
        # keep the largest bodies, pushing against the cap
        pop = sorted(pop + [k.genome for k in kids], key=genomes.module_count)[-10:]


def test_cap_counts_modules_besides_the_core():
    random.seed(0)
    g = genomes.random_body(genomes.genome_ops("tree"))[0]
    assert genomes.module_count(g) == len(g["nodes"]) - 1 == PROTOCOL.max_modules  # random trees fill the cap


def test_oversized_body_is_invalid():
    random.seed(0)
    g = genomes.random_body(genomes.genome_ops("tree"))[0]
    assert genomes.is_valid(g)
    from dataclasses import replace  # noqa: PLC0415
    assert not genomes.is_valid(g, replace(PROTOCOL, max_modules=PROTOCOL.max_modules - 1))


def test_fallback_when_no_child_is_valid():
    """A parent that cannot yield a valid child is copied, flagged fallback."""
    random.seed(0)
    ops = genomes.genome_ops("tree")
    parent = genomes.random_body(ops)[0]
    from dataclasses import replace  # noqa: PLC0415
    strict = replace(PROTOCOL, min_hinges=100, offspring_attempts=3)
    child = genomes.make_child(ops, [parent], np.random.default_rng(0), strict)
    assert child.op == "fallback" and child.genome == parent and child.attempts == 6
