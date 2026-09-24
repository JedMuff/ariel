"""Single rest-pose MuJoCo render of a genome's phenotype body.

Unlike the episode-replay renderers (`render_skill_task_checkpoint.py`,
`render_food_skills_checkpoint.py`), this only needs the genome dict -- no
brain weights, no reward rollout. The body is spawned, allowed to settle
under gravity for `shared.SETTLE_DURATION` with zero control input (matching
the settle phase every skill-training episode already uses), and a single
frame is captured from directly overhead (a true top-down camera, distinct
from the 3/4 "hero" angle `render_skills.py::_make_tracking_camera` uses for
episode videos elsewhere in this directory).

IMPORTANT: mujoco's headless/GL rendering backend is resolved once, the
*first* time `mujoco` is imported anywhere in the process (verified: setting
`MUJOCO_GL` after an earlier bare `import mujoco` has no effect -- the
backend is already locked in). This module sets `MUJOCO_GL` before its own
`import mujoco`, but that only helps if this module is the *first* thing
imported. Any entry-point script that also imports `shared` or
`genome_adapter` (which import `mujoco` themselves) must set `MUJOCO_GL`
as literally its first lines, before those imports -- mirroring
`examples/re_book/6_replay_best.py:33-34`.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import genome_adapter
import shared

CORE_BODY_NAME = "robot1_core"


def _make_camera(lookat: np.ndarray, distance: float) -> mujoco.MjvCamera:
    """True top-down camera (per the user's "top down image" requirement),
    unlike the 3/4 "hero" angle (`render_skills.py::_make_tracking_camera`)
    used by the episode-replay video renderers elsewhere in this directory."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = 0.0
    cam.elevation = -90.0
    return cam


def render_phenotype_thumbnail(
    genome_dict: dict,
    genome_type: str,
    out_path: Path,
    size_px: int = 220,
    max_modules: int = 25,
) -> bool:
    """Render one rest-pose frame of the decoded body to `out_path`.

    Returns False (without raising) on any decode/spawn failure, so a batch
    prepare run can mark the individual as having no image instead of
    aborting.
    """
    to_spec_fn = (
        partial(genome_adapter.cppn_genome_to_spec, max_modules=max_modules)
        if genome_type == "cppn"
        else shared.genome_to_spec
    )
    try:
        model, data = shared.build_loco_world_for_body(genome_dict, to_spec_fn)
    except Exception:
        return False

    try:
        while data.time < shared.SETTLE_DURATION:
            mujoco.mj_step(model, data)

        core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CORE_BODY_NAME)
        core_pos = data.xpos[core_id].copy() if core_id >= 0 else np.array([0.0, 0.0, 0.1])

        # Frame the whole body: distance sized to the furthest body part from
        # the core, in the horizontal plane (elevation=-90 makes vertical
        # extent irrelevant to what's visible from directly above).
        body_ids = [i for i in range(1, model.nbody)]
        if body_ids:
            xy = data.xpos[body_ids, :2] - core_pos[:2]
            horizontal_extent = float(np.max(np.linalg.norm(xy, axis=1))) if len(xy) else 0.0
        else:
            horizontal_extent = 0.0
        distance = max(0.9, horizontal_extent * 2.6 + 0.3)

        renderer = mujoco.Renderer(model, height=size_px, width=size_px)
        try:
            cam = _make_camera(core_pos, distance)
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
        finally:
            renderer.close()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(out_path)
        return True
    except Exception:
        return False
