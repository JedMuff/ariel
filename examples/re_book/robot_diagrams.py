"""
Top-view diagram images of simple robot bodies for presentation slides.

Produces 5 square PNG images in __data__/robot_diagrams/:

  robot_a.png        core → hinge → brick → hinge
  robot_b.png        core → hinge → brick
  robot_c.png        hinge ← core → hinge → brick → hinge
  robot_a_minus.png  robot_a with the tail hinge highlighted orange (removed)
  robot_c_plus.png   robot_c with the back hinge highlighted green (added)

Usage
-----
    uv run python examples/re_book/robot_diagrams.py
    uv run python examples/re_book/robot_diagrams.py --size 800 --out __data__/robot_diagrams
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")

import mujoco
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Highlight colours (RGBA 0-1)
ORANGE = np.array([1.0, 0.40, 0.0, 1.0])
GREEN  = np.array([0.0, 0.85, 0.20, 1.0])


# ---------------------------------------------------------------------------
# Body builders
# ---------------------------------------------------------------------------

def _build_robot_a():
    """core → hinge → brick → hinge"""
    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
    from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
    from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
    from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

    core  = CoreModule(index=0)
    h1    = HingeModule(index=1)
    brick = BrickModule(index=2)
    h2    = HingeModule(index=3)

    core.sites[ModuleFaces.FRONT].attach_body(h1.body,    prefix="h1")
    h1.sites[ModuleFaces.FRONT].attach_body(brick.body,   prefix="brick")
    brick.sites[ModuleFaces.FRONT].attach_body(h2.body,   prefix="h2")

    return core, {"tail_hinge_prefix": "robot1_h2"}


def _build_robot_b():
    """core → hinge → brick"""
    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
    from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
    from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
    from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

    core  = CoreModule(index=0)
    h1    = HingeModule(index=1)
    brick = BrickModule(index=2)

    core.sites[ModuleFaces.FRONT].attach_body(h1.body,    prefix="h1")
    h1.sites[ModuleFaces.FRONT].attach_body(brick.body,   prefix="brick")

    return core, {}


def _build_robot_c():
    """hinge ← core → hinge → brick → hinge"""
    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
    from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
    from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
    from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

    core   = CoreModule(index=0)
    h_back = HingeModule(index=4)
    h1     = HingeModule(index=1)
    brick  = BrickModule(index=2)
    h2     = HingeModule(index=3)

    core.sites[ModuleFaces.BACK].attach_body(h_back.body,  prefix="hback")
    core.sites[ModuleFaces.FRONT].attach_body(h1.body,     prefix="h1")
    h1.sites[ModuleFaces.FRONT].attach_body(brick.body,    prefix="brick")
    brick.sites[ModuleFaces.FRONT].attach_body(h2.body,    prefix="h2")

    return core, {"back_hinge_prefix": "robot1_hback"}


# ---------------------------------------------------------------------------
# World / compile helpers
# ---------------------------------------------------------------------------

def _build_world(core_module, highlight_colours: dict[str, np.ndarray] | None = None):
    """Build, compile and step a world.

    highlight_colours maps geom-name-prefix → RGBA array.
    """
    from ariel.simulation.environments import SimpleFlatWorld

    world = SimpleFlatWorld(floor_size=(10, 10, 1))
    world.spawn(core_module.spec, position=[0.0, 0.0, 0.05])
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if highlight_colours:
        for prefix, rgba in highlight_colours.items():
            for i in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                if name and name.startswith(prefix):
                    model.geom_rgba[i] = rgba

    return model, data


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def _top_camera(model: mujoco.MjModel, data: mujoco.MjData,
                padding: float = 0.12) -> mujoco.MjvCamera:
    """Orthographic-style top-down camera centred on all geoms."""
    # Gather geom positions in world frame after mj_forward
    positions = data.geom_xpos  # (ngeom, 3)
    xs, ys = positions[:, 0], positions[:, 1]
    cx, cy = xs.mean(), ys.mean()
    span   = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0 + padding

    cam           = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat    = np.array([cx, cy, 0.05], dtype=np.float64)
    cam.distance  = span * 4.0   # pull back enough for a clear top view
    cam.azimuth   = 90.0
    cam.elevation = -90.0
    return cam


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _render(model: mujoco.MjModel, data: mujoco.MjData,
            size: int, out_path: Path, *, flat_light: bool = False) -> None:
    cam = _top_camera(model, data)

    scene_opt = mujoco.MjvOption()
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]        = False
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR]     = False
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

    if flat_light:
        # Disable scene lights and use pure white ambient headlight so
        # geom_rgba shows exactly as set — no shading washout.
        for i in range(model.nlight):
            model.light_active[i] = 0
        model.vis.headlight.ambient[:] = [1.0, 1.0, 1.0]
        model.vis.headlight.diffuse[:] = [0.0, 0.0, 0.0]
        model.vis.headlight.specular[:] = [0.0, 0.0, 0.0]

    with mujoco.Renderer(model, height=size, width=size) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        frame = renderer.render()

    Image.fromarray(frame).save(out_path)
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Render robot diagram images.")
    ap.add_argument("--size", type=int, default=600,
                    help="Image side length in pixels (default 600)")
    ap.add_argument("--out", type=Path,
                    default=Path("__data__/robot_diagrams"),
                    help="Output directory")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Robot A ---
    print("Building robot A (core → hinge → brick → hinge)…")
    core_a, meta_a = _build_robot_a()
    model_a, data_a = _build_world(core_a)
    _render(model_a, data_a, args.size, args.out / "robot_a.png", flat_light=True)

    # --- Robot A minus (tail hinge highlighted orange) ---
    print("Building robot A-minus (tail hinge orange)…")
    core_am, meta_am = _build_robot_a()
    model_am, data_am = _build_world(core_am, {meta_am["tail_hinge_prefix"]: ORANGE})
    _render(model_am, data_am, args.size, args.out / "robot_a_minus.png", flat_light=True)

    # --- Robot B ---
    print("Building robot B (core → hinge → brick)…")
    core_b, _ = _build_robot_b()
    model_b, data_b = _build_world(core_b)
    _render(model_b, data_b, args.size, args.out / "robot_b.png", flat_light=True)

    # --- Robot C ---
    print("Building robot C (hinge ← core → hinge → brick → hinge)…")
    core_c, meta_c = _build_robot_c()
    model_c, data_c = _build_world(core_c)
    _render(model_c, data_c, args.size, args.out / "robot_c.png", flat_light=True)

    # --- Robot C plus (back hinge highlighted green) ---
    print("Building robot C-plus (back hinge green)…")
    core_cp, meta_cp = _build_robot_c()
    model_cp, data_cp = _build_world(core_cp, {meta_cp["back_hinge_prefix"]: GREEN})
    _render(model_cp, data_cp, args.size, args.out / "robot_c_plus.png", flat_light=True)

    print(f"\nDone. Five images written to {args.out}/")


if __name__ == "__main__":
    main()
