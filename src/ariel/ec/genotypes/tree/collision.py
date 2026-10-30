"""FCL self-collision checks and repair for tree genomes.

A tree genome is built into a body exactly as written (`TreeGenome.to_networkx`
-> `construct_mjspec_from_graph`), so collision-free bodies have to come from
collision-free genotypes. These helpers place a genome's modules breadth-first
from the core with `BodyCollisionChecker` (real module geometry, see
`ariel.body_phenotypes.robogen_lite.collision_utils`) and remove subtrees whose
root module would overlap one placed before it.
"""
from __future__ import annotations

import networkx as nx

from ariel.body_phenotypes.robogen_lite.collision_utils import (
    IDENTITY,
    BodyCollisionChecker,
)
from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE

from .symmetry import MirrorAxis, mirror_face
from .tree_genome import TreeGenome


def core_checker(genome: TreeGenome) -> BodyCollisionChecker:
    checker = BodyCollisionChecker()
    core = genome.nodes[IDX_OF_CORE]
    checker.add_module(IDX_OF_CORE, IDENTITY, core["type"], core["rotation"])
    return checker


def try_place(
    checker: BodyCollisionChecker,
    node_id: int,
    parent_id: int,
    face: str,
    module_type: str,
    rotation: str,
) -> bool:
    """Add the module to `checker` unless it would collide; return whether it
    was added."""
    frame = checker.child_frame(parent_id, face)
    if checker.collides(frame, module_type, rotation, ignore={parent_id}):
        return False
    checker.add_module(node_id, frame, module_type, rotation)
    return True


def first_collision(genome: TreeGenome) -> int | None:
    """First module, in breadth-first order from the core, that overlaps a
    module placed before it; None if the body is collision-free. Children of
    a colliding module are not placed."""
    checker = core_checker(genome)
    graph = genome.to_networkx()
    for parent_id, child_id in nx.bfs_edges(graph, IDX_OF_CORE):
        if parent_id not in checker.frames:
            continue
        node = genome.nodes[child_id]
        face = graph.edges[parent_id, child_id]["face"]
        if not try_place(checker, child_id, parent_id, face, node["type"], node["rotation"]):
            return child_id
    return None


def _remove_subtree(genome: TreeGenome, node_id: int) -> None:
    from .operators import remove_subtree  # local import: avoid cycle

    remove_subtree(genome, node_id)


def prune_colliding_subtrees(genome: TreeGenome) -> int:
    """Remove, in place, every subtree whose root module overlaps a module
    placed before it (breadth-first from the core). Returns the number of
    subtrees removed.

    One pass is enough: parents are placed before children, and removing a
    subtree only removes geometry, so it can't create a new collision.
    """
    removed = 0
    checker = core_checker(genome)
    graph = genome.to_networkx()
    for parent_id, child_id in nx.bfs_edges(graph, IDX_OF_CORE):
        if child_id not in genome.nodes or parent_id not in checker.frames:
            continue
        node = genome.nodes[child_id]
        face = graph.edges[parent_id, child_id]["face"]
        if not try_place(checker, child_id, parent_id, face, node["type"], node["rotation"]):
            _remove_subtree(genome, child_id)
            removed += 1
    if removed:
        _fix_terminal_hinges(genome)
    return removed


def mirror_node(genome: TreeGenome, node_id: int, axis: MirrorAxis) -> int | None:
    """The node at the mirrored position of `node_id` in a symmetric genome
    (itself for midline nodes), found by mirroring the path of faces from the
    core the same way `symmetrize_genome` does. None if there is no such node.
    """
    graph = genome.to_networkx()
    path = nx.shortest_path(graph, IDX_OF_CORE, node_id)
    children = {(e["parent"], e["face"]): e["child"] for e in genome.edges}
    current = IDX_OF_CORE
    for depth, (parent_id, child_id) in enumerate(zip(path, path[1:])):
        face = graph.edges[parent_id, child_id]["face"]
        current = children.get((current, mirror_face(face, axis, is_outer=depth == 0)))
        if current is None:
            return None
    return current


def prune_colliding_subtrees_symmetric(genome: TreeGenome, axis: MirrorAxis) -> int:
    """Like `prune_colliding_subtrees` for a genome symmetric about `axis`,
    but each colliding subtree is removed together with its mirror image, so
    the genome stays symmetric. Returns the number of subtrees removed.

    Re-running `symmetrize_genome` after an ordinary prune would not do: when
    an arm crosses the mirror plane and hits its own mirror image, the prune
    removes one of the two and symmetrizing copies it straight back.
    """
    removed = 0
    while (node_id := first_collision(genome)) is not None:
        mirror_id = mirror_node(genome, node_id, axis)
        _remove_subtree(genome, node_id)
        removed += 1
        if mirror_id is not None and mirror_id != node_id and mirror_id in genome.nodes:
            _remove_subtree(genome, mirror_id)
            removed += 1
    if removed:
        _fix_terminal_hinges(genome)
    return removed


def _fix_terminal_hinges(genome: TreeGenome) -> None:
    # Removing a subtree can leave its parent hinge as a leaf. Hinges are
    # checked with a brick's volume (collision_volume_type), so converting
    # them can't create an overlap.
    from .operators import _fix_terminal_hinges as fix  # local import: avoid cycle

    fix(genome)
