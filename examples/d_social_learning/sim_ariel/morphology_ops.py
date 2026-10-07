"""Random tree initialisation and mutation wrappers for ARIEL domain."""

import copy
import random

from ariel.ec.genotypes.tree.collision import prune_colliding_subtrees
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
    """Return a mutated copy of *genome_dict* (see `mutate_with_info`)."""
    return mutate_with_info(genome_dict)[0]


def mutate_with_info(genome_dict: dict) -> tuple[dict, dict]:
    """Return a mutated copy of *genome_dict* and a record of what was done.

    Applies exactly one structural mutation (point/subtree/shrink/hoist)
    with optional 25% chance of additional rotation mutation. The record is
    ``{"op": <structural mutation>, "rotation": bool, "n_pruned": int}``,
    where ``n_pruned`` counts subtrees removed for self-collision.
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

    rotation = random.random() < _ROTATION_PROB
    if rotation:
        _mutate_rotation(genome)

    # Point/rotation changes and grafted subtrees can make parts overlap;
    # drop any subtree that now collides (FCL, real module geometry).
    n_pruned = prune_colliding_subtrees(genome)

    info = {"op": mutation_type, "rotation": rotation, "n_pruned": n_pruned}
    return genome.to_dict(), info


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
