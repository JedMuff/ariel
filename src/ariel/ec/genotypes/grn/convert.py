"""A GRN body (develop.Module tree) as an ARIEL tree genome (nodes, edges).

revolve2 and ARIEL build bodies the same way: a child's frame is its parent's,
turned about the vertical axis to the face it sits on (left = +90 degrees), then
rolled about its own forward axis by its rotation; both use right-handed
frames, so face names and rotation angles carry over. The one difference is
the hinge: revolve2's joint turns about the module's left axis (an unrotated
hinge pitches), ARIEL's about its up axis (an unrotated hinge yaws).

So every ARIEL module is given the revolve2 module's frame rolled by an offset
about its forward axis: 0 or 180 degrees for the core and bricks (flat, as in
revolve2; at 180 the brick is upside down, so its left and right faces swap),
and 90 or 270 for hinges (joint axis along revolve2's). The offset is chosen
so that each module's rotation relative to its parent is one ARIEL allows
(0 or +-90 degrees, never 180). The bodies then match revolve2's geometry, up
to the different module sizes. tests/test_grn.py checks this on compiled
bodies.

As for every ARIEL tree genome, a hinge without a child becomes a brick.
"""

from __future__ import annotations

from .develop import BACK, FRONT, LEFT, RIGHT, Module, modules_of

FACE = {FRONT: "FRONT", RIGHT: "RIGHT", LEFT: "LEFT", BACK: "BACK"}
MIRRORED = {"LEFT": "RIGHT", "RIGHT": "LEFT", "FRONT": "FRONT", "BACK": "BACK"}
ROTATION = {0: "DEG_0", 90: "DEG_90", 270: "DEG_NEG_90"}
OFFSETS = {"CORE": (0, 180), "BRICK": (0, 180), "HINGE": (90, 270)}


def to_tree(core: Module | None) -> dict:
    """{"nodes": {id: {type, rotation}}, "edges": [{parent, child, face}]} with
    integer ids (the core is 0)."""
    if core is None:
        return {"nodes": {0: {"type": "CORE", "rotation": "DEG_0"}}, "edges": []}
    modules = modules_of(core)
    mtype = {m.id: ("BRICK" if m.type == "HINGE" and not any(m.children) else m.type) for m in modules}
    offset = {core.id: 0}
    nodes = {core.id: {"type": "CORE", "rotation": "DEG_0"}}
    edges = []
    for m in modules[1:]:  # parents are created before their children
        p = m.parent
        candidates = []
        for o in OFFSETS[mtype[m.id]]:
            rel = (90 * m.rotation + o - offset[p.id]) % 360
            if rel in ROTATION:
                candidates.append((rel != 0, o, rel))  # prefer no relative rotation, then the smaller offset
        _, offset[m.id], rel = min(candidates)
        face = FACE[m.slot]
        if offset[p.id] == 180:  # an upside-down parent's left and right are swapped
            face = MIRRORED[face]
        nodes[m.id] = {"type": mtype[m.id], "rotation": ROTATION[rel]}
        edges.append({"parent": p.id, "child": m.id, "face": face})
    return {"nodes": nodes, "edges": edges}
