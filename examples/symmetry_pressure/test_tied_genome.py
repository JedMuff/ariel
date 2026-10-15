"""Tests for the random-tied tree genome (ariel.ec.genotypes.tree.tying), the
dimensionality control for tree_symmetric (gecko_skill_tasks.py --genome-type tree_tied).
"""

import random

import numpy as np
import pytest
import shared
from genome_adapter import DEFAULT_TIE_PROB

from ariel.ec.genotypes.tree.collision import first_collision, mirror_node
from ariel.ec.genotypes.tree.operators import (
    random_tree_symmetric,
    remove_subtree,
)
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from ariel.ec.genotypes.tree.tying import (
    Tie,
    TiedGenome,
    _subtree,
    crossover_tied,
    mutate_morph_tied,
    random_tree_tied,
    remove_tied,
    settle,
    tied_fraction,
)
from ariel.ec.genotypes.tree.validation import validate_genome_dict

P = DEFAULT_TIE_PROB
N_MODULES = 25


def check_invariant(data: dict) -> None:
    """Valid, collision-free, each module in at most one tie, tied modules share
    the tie's gene (a leaf HINGE may be expressed as a BRICK).
    """
    g = TiedGenome.from_dict(data)
    validate_genome_dict(g.tree.to_dict())
    assert first_collision(g.tree) is None
    seen = set()
    for t in g.ties:
        assert t.a in g.tree.nodes
        assert t.b in g.tree.nodes
        assert t.a != t.b
        assert not {t.a, t.b} & seen
        seen |= {t.a, t.b}
        for n in (t.a, t.b):
            node = g.tree.nodes[n]
            assert node["rotation"] == t.rotation
            assert node["type"] == t.type or (t.type == "HINGE" and node["type"] == "BRICK")


def hinges(data: dict) -> int:
    return sum(1 for n in data["nodes"].values() if n["type"] == "HINGE")


@pytest.fixture(autouse=True)
def seeded() -> None:
    random.seed(0)


def test_random_genomes_match_symmetric() -> None:
    tied = [random_tree_tied(N_MODULES, P).to_dict() for _ in range(300)]
    sym = [random_tree_symmetric(N_MODULES, MirrorAxis.Y_ZERO).to_dict() for _ in range(300)]
    for d in tied:
        check_invariant(d)
        assert shared.genome_to_spec(d) is not None
    assert np.mean([tied_fraction(d) for d in tied]) == pytest.approx(2 * P / (1 + P), abs=0.05)
    n_tied, n_sym = np.mean([len(d["nodes"]) for d in tied]), np.mean([len(d["nodes"]) for d in sym])
    assert n_tied == pytest.approx(n_sym, rel=0.1)
    assert np.mean([hinges(d) for d in tied]) == pytest.approx(np.mean([hinges(d) for d in sym]), rel=0.1)


def test_no_twin_at_mirror_position() -> None:
    for _ in range(200):
        g = random_tree_tied(N_MODULES, P)
        for t in g.ties:
            assert mirror_node(g.tree, t.a, MirrorAxis.Y_ZERO) != t.b


def test_invariant_under_evolution() -> None:
    rng = np.random.default_rng(0)
    pop = [random_tree_tied(N_MODULES, P).to_dict() for _ in range(10)]
    for _ in range(25):
        new = []
        for _ in range(20):
            child = (random.choice(crossover_tied(*random.sample(pop, 2))) if rng.random() < 0.6
                     else random.choice(pop))
            child = mutate_morph_tied(child, rng, N_MODULES, P)
            check_invariant(child)
            new.append(child)
        pop = random.sample(pop + new, 10)
    assert np.mean([tied_fraction(d) for d in pop]) > 0.3  # ties are not eroded away


def tied_pair() -> tuple[TiedGenome, Tie]:
    while True:
        g = random_tree_tied(N_MODULES, P)
        if g.ties:
            return g, g.ties[0]


def test_point_mutation_changes_both_ends() -> None:
    g, t = tied_pair()
    g.set_gene(t.a, "BRICK" if t.type == "HINGE" else "HINGE", "DEG_90")
    settle(g)
    for n in (t.a, t.b):
        if n in g.tree.nodes:
            assert g.gene(n) == g.gene(t.a)
    if t.a in g.tree.nodes and t.b in g.tree.nodes:
        assert g.twin(t.a) == t.b


def test_removing_a_module_removes_its_twin() -> None:
    g, t = tied_pair()
    gone = remove_tied(g, t.a)
    assert t.a in gone
    assert t.b in gone


def test_lost_twin_is_replaced() -> None:
    g, t = tied_pair()
    # cut the twin's subtree away without the tie-aware removal, as a crossover cut would
    if t.a in _subtree(g.tree, t.b):
        pytest.skip("twin is an ancestor of the module")
    remove_subtree(g.tree, t.b)
    settle(g)
    if t.a in g.tree.nodes:
        twin = g.twin(t.a)
        assert twin is not None
        assert twin != t.b
        check_invariant(g.to_dict())


def test_shared_offspring_path() -> None:
    parents = [shared.create_individual(N_MODULES, 25, tie_prob=P) for _ in range(4)]
    kids = shared.make_offspring(parents, 8, np.random.default_rng(0), N_MODULES, 25, tie_prob=P)
    for ind in parents + kids:
        assert "ties" in ind.genotype["morph"]
        check_invariant(ind.genotype["morph"])
