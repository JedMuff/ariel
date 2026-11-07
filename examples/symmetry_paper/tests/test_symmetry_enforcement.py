"""tree_symmetric generation, mutation and crossover always give bilaterally
symmetric bodies (genomes.bilateral_symmetry == 1).
"""

import random

import numpy as np
import pytest
from genomes import bilateral_symmetry, genome_ops

from ariel.ec.genotypes.tree.operators import (
    crossover_subtree_symmetric,
    random_tree_symmetric,
)
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from ariel.ec.genotypes.tree.validation import validate_genome_dict

AXIS = MirrorAxis.Y_ZERO
N_TRIALS = 20
MAX_MODULES = 25


def assert_symmetric(genome_dict: dict) -> None:
    validate_genome_dict(genome_dict)
    assert bilateral_symmetry(genome_dict) == 1.0


@pytest.fixture(autouse=True)
def seeded() -> None:
    random.seed(0)


def test_random_generation_is_symmetric() -> None:
    for _ in range(N_TRIALS):
        assert_symmetric(random_tree_symmetric(MAX_MODULES, AXIS).to_dict())


def test_mutation_keeps_symmetry() -> None:
    rng, ops = np.random.default_rng(0), genome_ops("tree_symmetric")
    for _ in range(N_TRIALS // 2):
        g = random_tree_symmetric(MAX_MODULES, AXIS).to_dict()
        for _ in range(5):
            g = ops.mutate(g, rng)
            assert_symmetric(g)


def test_crossover_keeps_symmetry() -> None:
    for _ in range(N_TRIALS // 2):
        a, b = random_tree_symmetric(MAX_MODULES, AXIS), random_tree_symmetric(MAX_MODULES, AXIS)
        for child in crossover_subtree_symmetric(a, b, AXIS):
            assert_symmetric(child.to_dict())
