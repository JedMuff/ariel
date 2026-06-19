"""Random tree initialisation and mutation wrappers for ARIEL domain."""

import copy
import random

from ariel.ec.genotypes.tree.operators import (
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

_MAX_MODULES = 12
_ROTATION_PROB = 0.25


def random_individual() -> dict:
    """Return a new random genome dict (``TreeGenome.to_dict()`` format)."""
    genome = random_tree(_MAX_MODULES)
    return genome.to_dict()


def mutate(genome_dict: dict) -> dict:
    """Return a mutated copy of *genome_dict*.

    Applies exactly one structural mutation (point/subtree/shrink/hoist)
    with optional 25% chance of additional rotation mutation.
    """
    genome = TreeGenome.from_dict(copy.deepcopy(genome_dict))

    mutation_type = random.choices(
        ["point", "subtree", "shrink", "hoist"],
        weights=[0.45, 0.35, 0.10, 0.10],
    )[0]

    if mutation_type == "point":
        mutate_replace_node(genome)
    elif mutation_type == "subtree":
        mutate_subtree_replacement(genome)
    elif mutation_type == "shrink":
        mutate_shrink(genome)
    else:
        mutate_hoist(genome)

    if random.random() < _ROTATION_PROB:
        _mutate_rotation(genome)

    return genome.to_dict()


def _mutate_rotation(genome: TreeGenome) -> None:
    from ariel.body_phenotypes.robogen_lite.config import ALLOWED_ROTATIONS, IDX_OF_CORE, ModuleType
    candidates = [nid for nid in genome.nodes if nid != IDX_OF_CORE]
    if not candidates:
        return
    nid = random.choice(candidates)
    mtype = ModuleType[genome.nodes[nid]["type"]]
    rotations = [r.name for r in ALLOWED_ROTATIONS[mtype]]
    if rotations:
        genome.nodes[nid]["rotation"] = random.choice(rotations)
