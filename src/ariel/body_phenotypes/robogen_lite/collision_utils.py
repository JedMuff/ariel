"""Self-collision checks for robogen-lite bodies using FCL.

Module geometry and attachment sites are read from the MuJoCo module classes
themselves (each module type/rotation is compiled once, standalone, and
cached), so the collision model follows any change to `modules/*.py`.

Frames are (R, p) pairs: a 3x3 rotation matrix and a position. A module's
"attach frame" is where its root body sits: the parent's attachment site, or
the robot origin for the core. Everything is in the robot's own frame at rest
pose (all hinge joints at 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import fcl
import mujoco
import numpy as np
import numpy.typing as npt

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsTheta,
    ModuleType,
)

type Frame = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]

IDENTITY: Frame = (np.eye(3), np.zeros(3))

# Boxes are shrunk by this much per half-extent so that faces that merely
# touch (e.g. two bricks side by side) are not reported as colliding.
COLLISION_MARGIN = 1e-3


def compose(a: Frame, b: Frame) -> Frame:
    """Frame b expressed in frame a, returned in a's parent frame."""
    return a[0] @ b[0], a[0] @ b[1] + a[1]


@dataclass(frozen=True)
class ModuleTemplate:
    """A module's box geoms and attachment sites, relative to its attach frame."""

    boxes: tuple[tuple[Frame, npt.NDArray[np.float64]], ...]  # (frame, half-extents)
    sites: dict[str, Frame]  # face name -> site frame


def _make_module(module_type: str):
    # Imported here: the module classes pull in pydantic settings, which the
    # rest of this file doesn't need.
    from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
    from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
    from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

    match module_type:
        case ModuleType.CORE.name:
            return CoreModule(index=0)
        case ModuleType.BRICK.name:
            return BrickModule(index=1)
        case ModuleType.HINGE.name:
            return HingeModule(index=1)
    msg = f"No geometry for module type: {module_type}"
    raise ValueError(msg)


@cache
def module_template(module_type: str, rotation: str) -> ModuleTemplate:
    """Compile one module standalone and read its geoms and sites.

    The standalone module's world frame plays the role of the attach frame:
    `attach_body` places the child body in the parent site's frame, keeping
    the body's own (rotation) quat, exactly as it sits in worldbody here.
    """
    module = _make_module(module_type)
    if module_type != ModuleType.CORE.name:
        module.rotate(ModuleRotationsTheta[rotation].value)

    model = module.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    boxes = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        frame = (data.geom_xmat[g].reshape(3, 3).copy(), data.geom_xpos[g].copy())
        boxes.append((frame, model.geom_size[g].copy()))

    sites = {}
    for face, site in module.sites.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site.name)
        sites[face.name] = (data.site_xmat[sid].reshape(3, 3).copy(), data.site_xpos[sid].copy())

    return ModuleTemplate(boxes=tuple(boxes), sites=sites)


def collision_volume_type(module_type: str) -> str:
    """Module type whose geometry stands in for `module_type` when checking
    collisions.

    Hinges are checked as a brick at the same rotation: the brick box contains
    the hinge's stator and rotor, and decoders turn leaf hinges into bricks
    after growth (`_fix_terminal_hinges`), so a hinge that was placed with
    only its own, thinner geometry could overlap once converted.
    """
    if module_type == ModuleType.HINGE.name:
        return ModuleType.BRICK.name
    return module_type


@dataclass
class _Box:
    module_id: int
    obj: fcl.CollisionObject
    centre: npt.NDArray[np.float64]
    radius: float


class BodyCollisionChecker:
    """Incremental self-collision checking while a body is grown module by
    module.

    Usage: `add_module` the core, then for each candidate child compute its
    `child_frame`, test it with `collides`, and `add_module` it if accepted.
    """

    def __init__(self, margin: float = COLLISION_MARGIN) -> None:
        self.margin = margin
        self.frames: dict[int, Frame] = {}
        self._types: dict[int, tuple[str, str]] = {}
        self._boxes: list[_Box] = []

    def child_frame(self, parent_id: int, face: str) -> Frame:
        parent_type, parent_rot = self._types[parent_id]
        site = module_template(parent_type, parent_rot).sites[face]
        return compose(self.frames[parent_id], site)

    def module_centre(self, module_id: int) -> npt.NDArray[np.float64]:
        """Mean of the module's (real, not stand-in) geom centres."""
        module_type, rotation = self._types[module_id]
        frame = self.frames[module_id]
        centres = [compose(frame, f)[1] for f, _ in module_template(module_type, rotation).boxes]
        return np.mean(centres, axis=0)

    def _make_boxes(self, module_id: int, frame: Frame, module_type: str, rotation: str) -> list[_Box]:
        template = module_template(collision_volume_type(module_type), rotation)
        boxes = []
        for local, half in template.boxes:
            R, p = compose(frame, local)
            size = 2 * np.maximum(half - self.margin, 1e-6)
            obj = fcl.CollisionObject(fcl.Box(*size), fcl.Transform(R, p))
            boxes.append(_Box(module_id, obj, p, float(np.linalg.norm(size) / 2)))
        return boxes

    def collides(
        self,
        frame: Frame,
        module_type: str,
        rotation: str,
        ignore: set[int] = frozenset(),
    ) -> bool:
        """Would a module placed at `frame` overlap any placed module not in
        `ignore` (pass the parent: it shares a face with its child)?"""
        request = fcl.CollisionRequest()
        for cand in self._make_boxes(-1, frame, module_type, rotation):
            for box in self._boxes:
                if box.module_id in ignore:
                    continue
                if np.linalg.norm(cand.centre - box.centre) > cand.radius + box.radius:
                    continue
                if fcl.collide(cand.obj, box.obj, request, fcl.CollisionResult()) > 0:
                    return True
        return False

    def add_module(self, module_id: int, frame: Frame, module_type: str, rotation: str) -> None:
        self.frames[module_id] = frame
        self._types[module_id] = (module_type, rotation)
        self._boxes.extend(self._make_boxes(module_id, frame, module_type, rotation))


def spec_self_collisions(
    spec: mujoco.MjSpec,
    margin: float = COLLISION_MARGIN,
) -> list[tuple[str, str]]:
    """Overlapping body pairs in a built robot at rest pose, for checking a
    decoder's output. Only explicit excludes (a hinge's stator/rotor) are
    skipped; faces that just touch are let through by the margin. MuJoCo's own
    contact filter is deliberately not used: it ignores geoms on bodies welded
    together, which hides overlaps within rigid clusters of bricks.
    """
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    exclude_pairs = {
        (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, e.bodyname1),
         mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, e.bodyname2))
        for e in spec.excludes
    }

    geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    objs = {
        g: fcl.CollisionObject(
            fcl.Box(*(2 * np.maximum(model.geom_size[g] - margin, 1e-6))),
            fcl.Transform(data.geom_xmat[g].reshape(3, 3).copy(), data.geom_xpos[g].copy()),
        )
        for g in geoms
    }

    hits = []
    request = fcl.CollisionRequest()
    for i, g1 in enumerate(geoms):
        for g2 in geoms[i + 1:]:
            b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
            if b1 == b2 or (b1, b2) in exclude_pairs or (b2, b1) in exclude_pairs:
                continue
            if fcl.collide(objs[g1], objs[g2], request, fcl.CollisionResult()) > 0:
                hits.append((model.body(b1).name, model.body(b2).name))
    return hits
