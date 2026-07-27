"""Bilateral-symmetry enforcement for tree genomes.

Provides mirror-axis definitions and a `symmetrize_genome` function that takes
an arbitrary (possibly asymmetric) `TreeGenome` and returns a new genome that
is structurally symmetric about a chosen axis: every paired face (e.g.
RIGHT/LEFT for the y=0 axis) has matching subtrees, with rotations mirrored to
preserve handedness.

Two axes are supported, both expressed in the same abstract face-grid used by
`bilateral_symmetry_score` (FRONT/BACK=+-x, RIGHT/LEFT=+-y, TOP/BOTTOM=+-z):

* ``Y_ZERO``:      mirror plane y=0.  Swaps RIGHT<->LEFT at every depth.
  FRONT/BACK/TOP/BOTTOM are the "middle stem" and are left as-is (but still
  recursed into, since a midline node can itself branch RIGHT/LEFT).

* ``X_EQUALS_Y``:  mirror plane x=y (45 degree diagonal).  Only the edge
  directly off the CORE is relabeled, swapping FRONT<->RIGHT and BACK<->LEFT —
  this is what picks which diagonal "arm" a subtree belongs to. Every edge
  *within* an arm (i.e. everywhere below that first hop) uses the same rule as
  ``Y_ZERO``: RIGHT<->LEFT swap, FRONT/BACK/TOP/BOTTOM unchanged. This keeps
  face labels that are only valid on a fixed socket (e.g. HINGE only has a
  FRONT socket) intact deeper in a limb, while still producing a true mirror
  image of each arm's internal structure. Concretely: ``Core-FRONT->Brick
  -RIGHT->Brick`` mirrors to ``Core-RIGHT->Brick-LEFT->Brick``. Only the core
  itself sits exactly on the x=y line, so there is no non-trivial "middle
  stem" beyond it (TOP/BOTTOM children of the core are the only midline
  content, matching ``Y_ZERO``'s TOP/BOTTOM handling).
"""
from __future__ import annotations

import copy
from enum import Enum
from typing import Any

import networkx as nx

from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE

from .tree_genome import TreeGenome
from .validation import validate_genome_dict


class MirrorAxis(Enum):
    """Which plane to mirror a genome across."""

    Y_ZERO = "y_zero"
    X_EQUALS_Y = "x_equals_y"


# Face-swap table used for the edge directly off the CORE. For Y_ZERO this is
# also the table used at every deeper level (there is no "outer vs inner"
# distinction for a left-right mirror). For X_EQUALS_Y this table applies only
# to the core's own direct children — it picks which diagonal arm a subtree
# belongs to. Faces not listed map to themselves.
OUTER_FACE_MIRROR_TABLES: dict[MirrorAxis, dict[str, str]] = {
    MirrorAxis.Y_ZERO: {
        "RIGHT": "LEFT",
        "LEFT": "RIGHT",
    },
    MirrorAxis.X_EQUALS_Y: {
        "FRONT": "RIGHT",
        "RIGHT": "FRONT",
        "BACK": "LEFT",
        "LEFT": "BACK",
    },
}

# Face-swap table used everywhere below the first hop off the core, for both
# axes: a plain left-right (RIGHT<->LEFT) swap. This keeps single-socket faces
# like HINGE's FRONT-only attachment point intact while still producing a true
# mirror image of branches inside an arm.
INNER_FACE_MIRROR_TABLE: dict[str, str] = {
    "RIGHT": "LEFT",
    "LEFT": "RIGHT",
}

# Faces hanging directly off the core that are *not* mirrored to a different
# face (they lie in the mirror plane itself), per axis.
MIDLINE_FACES: dict[MirrorAxis, set[str]] = {
    MirrorAxis.Y_ZERO: {"FRONT", "BACK", "TOP", "BOTTOM"},
    MirrorAxis.X_EQUALS_Y: {"TOP", "BOTTOM"},
}

# The canonical "source of truth" face in each mirrored pair: when a node has
# children on both faces of a pair, the primary side's subtree is copied onto
# the other side. Kept separate per axis since X_EQUALS_Y has two independent
# pairs (FRONT/RIGHT and BACK/LEFT) at the core; inner (non-core) levels always
# use the Y_ZERO-style RIGHT/LEFT pairing regardless of axis.
PRIMARY_FACES: dict[MirrorAxis, set[str]] = {
    MirrorAxis.Y_ZERO: {"RIGHT"},
    MirrorAxis.X_EQUALS_Y: {"RIGHT", "LEFT"},
}
INNER_PRIMARY_FACES: set[str] = {"RIGHT"}

# Rotation mirror table: a reflection flips the handedness of the roll around
# a module's local FRONT axis, so +45/+90 <-> -45/-90. DEG_0 is self-mirroring.
# Axis-independent: any mirror reflection flips roll sign the same way.
ROTATION_MIRROR_TABLE: dict[str, str] = {
    "DEG_0": "DEG_0",
    "DEG_45": "DEG_NEG_45",
    "DEG_NEG_45": "DEG_45",
    "DEG_90": "DEG_NEG_90",
    "DEG_NEG_90": "DEG_90",
}


def mirror_face(face: str, axis: MirrorAxis, *, is_outer: bool) -> str:
    """Return the mirrored counterpart of ``face`` under ``axis``.

    ``is_outer`` selects the table: True for the edge directly off the core
    (which picks the diagonal arm for X_EQUALS_Y), False for every edge deeper
    inside an arm (always a plain RIGHT<->LEFT swap, both axes).
    """
    table = OUTER_FACE_MIRROR_TABLES[axis] if is_outer else INNER_FACE_MIRROR_TABLE
    return table.get(face, face)


def mirror_rotation(rotation: str) -> str:
    """Return the mirrored counterpart of ``rotation``."""
    return ROTATION_MIRROR_TABLE.get(rotation, rotation)


def is_on_midline(face: str, axis: MirrorAxis, *, is_outer: bool) -> bool:
    """Whether a child attached on ``face`` lies in the mirror plane itself."""
    if is_outer:
        return face in MIDLINE_FACES[axis]
    return face not in INNER_FACE_MIRROR_TABLE


def is_primary_face(face: str, axis: MirrorAxis, *, is_outer: bool) -> bool:
    """Whether ``face`` is the canonical/source-of-truth side of its pair."""
    if is_outer:
        return face in PRIMARY_FACES[axis]
    return face in INNER_PRIMARY_FACES


def _reassign_ids(
    nodes: dict[int, dict[str, str]],
    edges: list[dict[str, Any]],
    existing_ids: set[int],
) -> tuple[dict[int, dict[str, str]], list[dict[str, Any]], dict[int, int]]:
    """Copy *nodes*/*edges* with fresh ids greater than any in *existing_ids*.

    Mirrors the id-remapping helper duplicated across `operators.py`.
    """
    if not nodes:
        return {}, [], {}
    mapping: dict[int, int] = {}
    next_id = max(existing_ids, default=-1) + 1
    for old in sorted(nodes):
        mapping[old] = next_id
        next_id += 1
    new_nodes = {mapping[old]: copy.deepcopy(attr) for old, attr in nodes.items()}
    new_edges: list[dict[str, Any]] = []
    for e in edges:
        edge_copy = {k: copy.deepcopy(v) for k, v in e.items()}
        edge_copy["parent"] = mapping[e["parent"]]
        edge_copy["child"] = mapping[e["child"]]
        new_edges.append(edge_copy)
    return new_nodes, new_edges, mapping


def _extract_subtree(
    genome: TreeGenome, root: int
) -> tuple[dict[int, dict[str, str]], list[dict[str, Any]]]:
    g = genome.to_networkx()
    subnodes = {root, *nx.descendants(g, root)}
    internal_edges = [
        e for e in genome.edges if e["parent"] in subnodes and e["child"] in subnodes
    ]
    nodes = {n: genome.nodes[n] for n in subnodes}
    return nodes, internal_edges


def _remove_subtree(genome: TreeGenome, node_id: int) -> None:
    g = genome.to_networkx()
    if node_id not in g:
        return
    descendants = {node_id, *nx.descendants(g, node_id)}
    for d in descendants:
        genome.nodes.pop(d, None)
    genome.edges = [
        e for e in genome.edges if e["child"] not in descendants and e["parent"] not in descendants
    ]


def mirror_subtree_nodes_edges(
    nodes: dict[int, dict[str, str]],
    edges: list[dict[str, Any]],
    axis: MirrorAxis,
) -> tuple[dict[int, dict[str, str]], list[dict[str, Any]]]:
    """Return a deep copy of *nodes*/*edges* with rotations and faces mirrored.

    The subtree root's own attachment face is not part of *edges* (it is the
    edge connecting the subtree to its parent, handled by the caller) — only
    faces *within* the subtree (edges between two nodes both in *nodes*) are
    mirrored, always using the inner (RIGHT<->LEFT) table since everything
    here is below the first hop off the core.
    """
    mirrored_nodes = {
        nid: {"type": attr["type"], "rotation": mirror_rotation(attr["rotation"])}
        for nid, attr in nodes.items()
    }
    mirrored_edges = [
        {**e, "face": mirror_face(e["face"], axis, is_outer=False)} for e in edges
    ]
    return mirrored_nodes, mirrored_edges


def symmetrize_subtree(
    genome: TreeGenome, node_id: int, axis: MirrorAxis
) -> tuple[dict[int, dict[str, str]], list[dict[str, Any]], int]:
    """Build a mirrored, freshly-id'd copy of the subtree rooted at ``node_id``.

    Returns ``(nodes, edges, new_root_id)`` with brand-new node ids (disjoint
    from every id currently in ``genome``), ready to be grafted onto a
    mirrored attachment point by the caller. Does not mutate ``genome``.
    """
    src_nodes, src_edges = _extract_subtree(genome, node_id)
    mirrored_nodes, mirrored_edges = mirror_subtree_nodes_edges(src_nodes, src_edges, axis)
    existing_ids = set(genome.nodes.keys())
    new_nodes, new_edges, mapping = _reassign_ids(mirrored_nodes, mirrored_edges, existing_ids)
    return new_nodes, new_edges, mapping[node_id]


def _symmetrize_children(
    genome: TreeGenome, parent_id: int, axis: MirrorAxis, *, is_outer: bool
) -> None:
    """Enforce symmetry among the direct children of ``parent_id``, then recurse.

    ``is_outer`` is True only when ``parent_id`` is the core: that is the one
    level where the axis-specific outer face table applies (picking which
    diagonal arm a subtree belongs to for X_EQUALS_Y); every deeper level
    always mirrors with the inner RIGHT<->LEFT table.

    For each pair of faces: if only the primary side is present, mirror it
    onto the other side. If both are present, the primary side's subtree wins
    (mirrored copy overwrites whatever was on the other side) so the result is
    deterministic/idempotent. Midline children are left in place but still
    recursed into, since they may themselves branch left/right.
    """
    children_by_face = {
        e["face"]: e["child"] for e in genome.edges if e["parent"] == parent_id
    }

    handled_faces: set[str] = set()
    for face, child_id in list(children_by_face.items()):
        if face in handled_faces:
            continue
        if is_on_midline(face, axis, is_outer=is_outer):
            handled_faces.add(face)
            # Midline nodes sit exactly on the mirror plane: only DEG_0 is a
            # self-mirroring rotation, so force it to guarantee true geometric
            # symmetry rather than just matching module types.
            genome.nodes[child_id]["rotation"] = "DEG_0"
            _symmetrize_children(genome, child_id, axis, is_outer=False)
            continue

        mirror = mirror_face(face, axis, is_outer=is_outer)
        handled_faces.add(face)
        handled_faces.add(mirror)

        # Prefer the primary face as the source of truth when both sides are
        # populated (deterministic/idempotent); otherwise use whichever side
        # actually has content.
        canonical_face = face if is_primary_face(face, axis, is_outer=is_outer) else mirror
        source_face = (
            canonical_face
            if children_by_face.get(canonical_face) is not None
            else mirror_face(canonical_face, axis, is_outer=is_outer)
        )
        source_child = children_by_face.get(source_face)
        if source_child is None:
            continue  # neither side actually present (shouldn't happen)

        target_face = mirror_face(source_face, axis, is_outer=is_outer)

        # Drop whatever currently sits on the target side, then graft a fresh
        # mirrored copy of the source side's subtree onto it.
        target_child = children_by_face.get(target_face)
        if target_child is not None:
            _remove_subtree(genome, target_child)

        new_nodes, new_edges, mirrored_root = symmetrize_subtree(genome, source_child, axis)
        genome.nodes.update(new_nodes)
        genome.edges.extend(new_edges)
        genome.edges.append({"parent": parent_id, "child": mirrored_root, "face": target_face})

        # Note: the source side's own interior is copied as-is into the
        # mirror (with faces/rotations relabeled by symmetrize_subtree) and is
        # NOT itself recursively re-symmetrized — an arm's internal structure
        # (e.g. a nested RIGHT-only branch) is preserved verbatim in both the
        # original and its mirror image, not force-balanced further.


def symmetrize_genome(genome: TreeGenome, axis: MirrorAxis) -> TreeGenome:
    """Return a new genome that is structurally symmetric about ``axis``.

    Does not mutate ``genome``. Raises ``ValueError`` if the result fails
    genome validation (should not normally happen given a valid input).
    """
    from .operators import _prune_invalid_edges  # local import: avoid cycle

    result = copy.deepcopy(genome)
    _symmetrize_children(result, IDX_OF_CORE, axis, is_outer=True)
    _prune_invalid_edges(result)
    validate_genome_dict(result.to_dict())
    return result
