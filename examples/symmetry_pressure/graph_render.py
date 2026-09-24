"""Single spawn-pose MuJoCo render of a decoded body graph (no settling).

Entry-point scripts must set MUJOCO_GL before importing this module (or
anything else that imports mujoco); see phenotype_thumbnail.py.
"""

from __future__ import annotations

import mujoco
import networkx as nx
import numpy as np

import shared
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

TOP = (90.0, -90.0)  # (azimuth, elevation)
ISO = (45.0, -35.0)


def render_graph(graph: nx.DiGraph, azimuth: float, elevation: float, size_px: int = 200) -> np.ndarray:
    spec = construct_mjspec_from_graph(graph).spec
    model, data = shared.build_loco_world_for_body({}, lambda _: spec)
    mujoco.mj_forward(model, data)
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    ids = [i for i in range(1, model.nbody) if model.body_rootid[i] == model.body_rootid[core_id]]
    pts = data.xpos[ids]
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
    extent = float(np.max(np.linalg.norm(pts - centre, axis=1)))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = centre
    cam.distance = max(0.6, extent * 3.0 + 0.25)
    cam.azimuth, cam.elevation = azimuth, elevation
    renderer = mujoco.Renderer(model, height=size_px, width=size_px)
    try:
        renderer.update_scene(data, camera=cam)
        return renderer.render().copy()
    finally:
        renderer.close()
