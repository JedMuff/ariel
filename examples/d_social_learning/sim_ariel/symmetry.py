"""Bilateral symmetry of a TreeGenome body, two ways.

`tree_symmetry` is ported from ariel_symmetry's
examples/symmetry_pressure/shared.py::bilateral_symmetry_score: modules are
matched by their path of faces from the core, so it is cheap and needs no
model, but it ignores module rotations (a 45 deg module's mirror image,
-45 deg, is not an allowed rotation, yet it counts as matched).

`geometric_symmetry` builds the body (sim_ariel.evaluator.initialize_world,
so it follows the active sim preset's module sizes), reads every part's
position at rest in the core's frame, and reflects them through the vertical
planes through the core centre. A part is matched if a part of the same kind
sits at its mirror position; `strict` also requires a matched hinge's
rotation axis to mirror onto its partner's. This is what the body looks like,
so it is the reference the cheaper scores are checked against.

MorphologicalMeasures' xy/xz/yz_symmetry (src/ariel/utils/morphological_descriptor.py)
is not used: it places each child one cell along its face's global direction,
ignoring the parent's orientation, and its loops stop one cell short of the
bounding box, so a core with one brick on each side scores 0.
"""

from __future__ import annotations

import numpy as np

# Vertical mirror planes through the core centre, by unit normal in the core
# frame: x=0, y=0 and the two diagonals.
PLANES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "xy": np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
    "x-y": np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
}
POS_TOL = 0.005  # m; zoo bricks are 0.075 m across
AXIS_TOL = 0.99  # |cos| between a mirrored hinge axis and its partner's

_INNER_MIRROR = {"RIGHT": "LEFT", "LEFT": "RIGHT"}
_OUTER_MIRROR = {
    "y_zero": _INNER_MIRROR,
    "x_equals_y": {"FRONT": "RIGHT", "RIGHT": "FRONT", "BACK": "LEFT", "LEFT": "BACK"},
}


def tree_symmetry(genome_dict: dict, axis: str = "y_zero") -> float:
    """Fraction of off-midline modules with a same-type module at the mirrored
    root path, about ``axis`` ("y_zero": RIGHT<->LEFT at every hop;
    "x_equals_y": FRONT<->RIGHT, BACK<->LEFT at the first hop off the core,
    RIGHT<->LEFT below it). 0.0 if no module is off the midline."""
    nodes = {int(k): v["type"] for k, v in genome_dict["nodes"].items()}
    children: dict[int, list[tuple[int, str]]] = {}
    for e in genome_dict["edges"]:
        children.setdefault(int(e["parent"]), []).append((int(e["child"]), e["face"]))
    paths: dict[tuple[str, ...], str] = {(): nodes[0]}
    stack = [(0, ())]
    while stack:
        node, path = stack.pop()
        for child, face in children.get(node, []):
            paths[(*path, face)] = nodes[child]
            stack.append((child, (*path, face)))

    def mirror(path):
        return tuple((_OUTER_MIRROR[axis] if i == 0 else _INNER_MIRROR).get(f, f) for i, f in enumerate(path))

    seen: set[tuple[str, ...]] = set()
    n_off = n_sym = 0
    for path, t in paths.items():
        m = mirror(path)
        if m == path or path in seen:
            continue
        seen.update((path, m))
        n_off += 1
        n_sym += paths.get(m) == t
    return n_sym / n_off if n_off else 0.0


def body_parts(genome_dict: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(kind, position, hinge axis) of every non-core part at rest, in the core
    frame with the core centre at the origin. kind is "brick", "stator" or
    "rotor"; axis is the hinge's rotation axis for stator and rotor parts and
    zero for bricks."""
    import mujoco

    from sim_ariel.evaluator import initialize_world

    spec = initialize_world(genome_dict)
    model, data = spec["model"], spec["data"]
    mujoco.mj_kinematics(model, data)
    core = next(i for i in range(model.ngeom) if model.geom(i).name.endswith("core"))
    origin, rot = data.geom_xpos[core], data.geom_xmat[core].reshape(3, 3)
    # Each hinge's stator and rotor bodies are children of the hinge body,
    # which the servo joint's axis is defined in.
    axis_by_body = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_body = model.body_parentid[model.jnt_bodyid[j]]
            axis_by_body[hinge_body] = rot.T @ data.xaxis[j]
    kinds, pos, axes = [], [], []
    for i in range(model.ngeom):
        name = model.geom(i).name
        kind = name.rsplit("-", 1)[-1]
        if kind not in ("brick", "stator", "rotor"):
            continue
        kinds.append(kind)
        pos.append(rot.T @ (data.geom_xpos[i] - origin))
        body = model.geom_bodyid[i]
        axes.append(axis_by_body.get(model.body_parentid[body], np.zeros(3)) if kind != "brick" else np.zeros(3))
    return kinds, np.array(pos).reshape(-1, 3), np.array(axes).reshape(-1, 3)


def geometric_symmetry(genome_dict: dict) -> dict[str, float | str]:
    """Bilateral symmetry about the best vertical plane in PLANES.

    ``sym``: fraction of parts with a same-kind part at their mirror position,
    best plane. ``strict``: the same, with matched hinge axes mirrored too.
    ``paired``: fraction of parts that lie off the plane and have a strict
    mirror partner, best plane; a chain or a single arm lies in one plane
    and is its own mirror image (``strict`` 1) but has ``paired`` 0.
    ``plane``: the plane of ``paired`` (ties broken by ``strict``).
    A core-only body scores 0 throughout."""
    kinds, pos, axes = body_parts(genome_dict)
    kinds = np.array(kinds)
    if not len(kinds):
        return {"sym": 0.0, "strict": 0.0, "paired": 0.0, "plane": ""}
    per_plane = {}
    for name, n in PLANES.items():
        reflect = np.eye(3) - 2.0 * np.outer(n, n)
        dist = np.linalg.norm((pos @ reflect)[:, None, :] - pos[None, :, :], axis=2)
        same = (dist < POS_TOL) & (kinds[:, None] == kinds[None, :])
        axis_ok = (np.abs((axes @ reflect) @ axes.T) > AXIS_TOL) | (kinds == "brick")[:, None]
        off = np.abs(pos @ n) > POS_TOL
        strict = (same & axis_ok).any(axis=1)
        per_plane[name] = (same.any(axis=1).mean(), strict.mean(), (strict & off).mean())
    best = max(PLANES, key=lambda p: (per_plane[p][2], per_plane[p][1]))
    return {
        "sym": max(v[0] for v in per_plane.values()),
        "strict": max(v[1] for v in per_plane.values()),
        "paired": per_plane[best][2],
        "plane": best,
    }
