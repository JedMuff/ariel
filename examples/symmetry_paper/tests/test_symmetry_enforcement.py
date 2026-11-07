"""tree_symmetric generation, mutation and crossover always give bilaterally
symmetric bodies (genomes.bilateral_symmetry == 1, or no off-midline modules).
"""

import random

import numpy as np
import pytest
from genomes import bilateral_symmetry, genome_ops

from ariel.ec.genotypes.tree.operators import (
    crossover_subtree_symmetric,
    random_tree_symmetric,
)
from ariel.ec.genotypes.tree.symmetry import MirrorAxis, mirror_face
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

AXIS = MirrorAxis.Y_ZERO
N_TRIALS = 20
MAX_MODULES = 25


def has_off_midline_modules(genome_dict: dict) -> bool:
    graph = TreeGenome.from_dict(genome_dict).to_networkx()
    root = next(n for n in graph.nodes() if graph.in_degree(n) == 0)
    paths: dict = {root: ()}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in graph.successors(node):
            if child not in paths:
                paths[child] = (*paths[node], (graph.get_edge_data(node, child) or {}).get("face", "FRONT"))
                queue.append(child)
    mirror = lambda p: tuple(mirror_face(f, AXIS, is_outer=(i == 0)) for i, f in enumerate(p))  # noqa: E731
    return any(p != mirror(p) for p in paths.values())


def assert_symmetric(genome_dict: dict) -> None:
    validate_genome_dict(genome_dict)
    score = bilateral_symmetry(genome_dict)
    assert score == 1.0 or (score == 0.0 and not has_off_midline_modules(genome_dict))


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
