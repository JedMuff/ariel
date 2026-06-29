"""
Side-view and top-view GIF + MP4 video of the amputee gecko (front-right arm removed)
using the original gecko_forward weights with the fr-arm nodes/edges pruned out.

The front-right leg consists of two hinges (fr_legservo, fr_flipperservo).  When
building the amputee body those actuators and state inputs simply do not exist, so we
surgically slice them out of the original weight matrix before loading.

Weight surgery
--------------
Original network: input_size=13, output_size=8, hidden=32
  State layout:  [quat(3), neck, spine, bl_leg, br_leg, fl_leg, fl_flipper, fr_leg*, fr_flipper*, sin, cos]
  Actuator layout: [neck, spine, fl_leg, fl_flipper, fr_leg*, fr_flipper*, bl_leg, br_leg]

Pruned network:  input_size=11, output_size=6, hidden=32
  Remove input  columns 9 & 10 (fr_leg, fr_flipper state dims)
  Remove output rows   4 & 5   (fr_leg, fr_flipper actuator dims)

Usage
-----
    uv run python examples/re_book/gecko_amputee_video.py \\
        --weights __data__/gecko_forward/best_weights.npy

    uv run python examples/re_book/gecko_amputee_video.py \\
        --weights __data__/gecko_forward/best_weights.npy --no-mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")

import mujoco
import numpy as np
import torch
from PIL import Image
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

CONTROL_HZ = 20
SPAWN_POS  = [0.0, 0.0, 0.1]

SIDE_W, SIDE_H = 640, 360
TOP_W,  TOP_H  = 640, 360

# Indices in the original 13-dim state vector that belong to the fr arm
_FR_STATE_INDICES  = [9, 10]   # fr_leg, fr_flipper joint readings
# Indices in the original 8-dim actuator vector that belong to the fr arm
_FR_CTRL_INDICES   = [4, 5]    # fr_legservo, fr_flipperservo


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden: int = 32) -> None:
        super().__init__()
        self.fc1     = nn.Linear(input_size, hidden)
        self.fc2     = nn.Linear(hidden, hidden)
        self.fc_out  = nn.Linear(hidden, output_size)
        self.act     = nn.ELU()
        self.out_act = nn.Tanh()
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, state: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(state, dtype=torch.float32)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return (self.out_act(self.fc_out(x)) * (np.pi / 2)).numpy()


def _fill(net: nn.Module, vector: np.ndarray) -> None:
    offset = 0
    for p in net.parameters():
        n = p.numel()
        p.data.view(-1)[:] = torch.as_tensor(vector[offset: offset + n])
        offset += n


def _extract_flat_params(net: nn.Module) -> np.ndarray:
    return np.concatenate([p.data.view(-1).numpy() for p in net.parameters()])


# ---------------------------------------------------------------------------
# Weight surgery: prune fr-arm rows/cols from original weight vector
# ---------------------------------------------------------------------------

def prune_weights(
    orig_weights: np.ndarray,
    input_size_orig: int  = 13,
    output_size_orig: int = 8,
    hidden: int           = 32,
) -> np.ndarray:
    """Remove fr-arm input columns and output rows from the flat weight vector."""
    orig_net = Network(input_size_orig, output_size_orig, hidden)
    _fill(orig_net, orig_weights)

    # fc1: shape (hidden, input_size) — drop columns for fr inputs
    keep_in = [i for i in range(input_size_orig) if i not in _FR_STATE_INDICES]
    fc1_w = orig_net.fc1.weight.data[:, keep_in]  # (hidden, input_size-2)
    fc1_b = orig_net.fc1.bias.data                 # (hidden,)  unchanged

    # fc2: shape (hidden, hidden) — unchanged
    fc2_w = orig_net.fc2.weight.data
    fc2_b = orig_net.fc2.bias.data

    # fc_out: shape (output_size, hidden) — drop rows for fr outputs
    keep_out = [i for i in range(output_size_orig) if i not in _FR_CTRL_INDICES]
    fco_w = orig_net.fc_out.weight.data[keep_out, :]  # (output_size-2, hidden)
    fco_b = orig_net.fc_out.bias.data[keep_out]       # (output_size-2,)

    return np.concatenate([
        fc1_w.view(-1).numpy(), fc1_b.numpy(),
        fc2_w.view(-1).numpy(), fc2_b.numpy(),
        fco_w.view(-1).numpy(), fco_b.numpy(),
    ])


# ---------------------------------------------------------------------------
# Amputee gecko (no front-right arm)
# ---------------------------------------------------------------------------

def _build_amputee_world():
    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
    from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
    from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
    from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
    from ariel.simulation.controllers.utils.data_get import get_state_from_data
    from ariel.simulation.environments import SimpleFlatWorld

    core = CoreModule(index=0)
    neck = HingeModule(index=1)
    abdomen = BrickModule(index=2)
    spine = HingeModule(index=3)
    butt = BrickModule(index=4)

    fl_leg = HingeModule(index=5)
    fl_leg.rotate(90)
    fl_leg2 = HingeModule(index=15)
    fl_leg2.rotate(90)
    fl_flipper = BrickModule(index=6)

    # fr arm omitted

    bl_leg = HingeModule(index=9)
    bl_leg.rotate(45)
    bl_flipper = BrickModule(index=10)
    br_leg = HingeModule(index=11)
    br_leg.rotate(-45)
    br_flipper = BrickModule(index=12)

    core.sites[ModuleFaces.FRONT].attach_body(body=neck.body,    prefix="neck")
    neck.sites[ModuleFaces.FRONT].attach_body(body=abdomen.body, prefix="abdomen")
    abdomen.sites[ModuleFaces.FRONT].attach_body(body=spine.body, prefix="spine")
    spine.sites[ModuleFaces.FRONT].attach_body(body=butt.body,   prefix="butt")

    core.sites[ModuleFaces.LEFT].attach_body(body=fl_leg.body,  prefix="fl_leg")
    fl_leg.sites[ModuleFaces.FRONT].attach_body(body=fl_leg2.body, prefix="fl_flipper")
    fl_leg2.sites[ModuleFaces.FRONT].attach_body(body=fl_flipper.body, prefix="fl_flipper2")

    # fr arm omitted

    butt.sites[ModuleFaces.LEFT].attach_body(body=bl_leg.body,  prefix="bl_leg")
    bl_leg.sites[ModuleFaces.FRONT].attach_body(body=bl_flipper.body, prefix="bl_flipper")
    butt.sites[ModuleFaces.RIGHT].attach_body(body=br_leg.body, prefix="br_leg")
    br_leg.sites[ModuleFaces.FRONT].attach_body(body=br_flipper.body, prefix="br_flipper")

    world = SimpleFlatWorld(floor_size=(100, 100, 1))
    world.spawn(core.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data  = mujoco.MjData(model)

    state_dim = len(get_state_from_data(data)) + 2
    return model, data, state_dim


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _side_camera(x, y, z, distance=1.8):
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat    = np.array([x, y, z], dtype=np.float64)
    cam.distance  = distance
    cam.azimuth   = 0.0
    cam.elevation = -10.0
    return cam


def _top_camera(x, y, z, height_m=1.2):
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat    = np.array([x, y, z], dtype=np.float64)
    cam.distance  = height_m
    cam.azimuth   = 0.0
    cam.elevation = -90.0
    return cam


# ---------------------------------------------------------------------------
# GIF / MP4 writers
# ---------------------------------------------------------------------------

def _save_gif(frames, path, fps):
    duration_ms = int(1000 / fps)
    pil_frames  = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        loop=0, duration=duration_ms, optimize=False,
    )
    print(f"Saved GIF → {path}  ({len(frames)} frames @ {fps} fps)")


def _save_mp4(frames, path, fps):
    import cv2
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        for codec in ("avc1", "XVID", "MJPG"):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
            if writer.isOpened():
                break
    if not writer.isOpened():
        print(f"Warning: could not open MP4 writer for {path} — skipping.")
        return
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved MP4 → {path}  ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Simulation + frame capture
# ---------------------------------------------------------------------------

def _run(args):
    from ariel.simulation.controllers.utils.data_get import get_state_from_data

    model, data, state_dim = _build_amputee_world()
    orig_weights = np.load(args.weights)
    pruned_w     = prune_weights(orig_weights,
                                  input_size_orig=args.input_size_orig,
                                  output_size_orig=args.output_size_orig,
                                  hidden=args.hidden)

    net = Network(input_size=state_dim, output_size=model.nu, hidden=args.hidden)
    _fill(net, pruned_w)

    print(f"Amputee model: state_dim={state_dim}, nu={model.nu}")

    mujoco.mj_resetData(model, data)
    dt             = model.opt.timestep
    control_period = int(round(1.0 / (CONTROL_HZ * dt)))
    total_steps    = int(args.duration / dt)
    steps_per_frame = max(1, int(round(1.0 / (args.fps * dt))))
    current_ctrl   = np.zeros(model.nu)

    side_frames: list[np.ndarray] = []
    top_frames:  list[np.ndarray] = []

    side_renderer = mujoco.Renderer(model, height=SIDE_H, width=SIDE_W)
    top_renderer  = mujoco.Renderer(model, height=TOP_H,  width=TOP_W)
    scene_opt     = mujoco.MjvOption()

    for step in range(total_steps):
        if step % control_period == 0:
            rs    = get_state_from_data(data)
            phase = [np.sin(data.time * 2.0 * np.pi), np.cos(data.time * 2.0 * np.pi)]
            state = np.concatenate([rs, phase]).astype(np.float32)
            current_ctrl = net.forward(state)
        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)

        if step % steps_per_frame == 0:
            cx = args.side_x if args.side_x is not None else float(data.qpos[0])
            cy = args.side_y if args.side_y is not None else float(data.qpos[1])
            tx = args.top_x  if args.top_x  is not None else float(data.qpos[0])
            ty = args.top_y  if args.top_y  is not None else float(data.qpos[1])
            side_cam = _side_camera(cx, cy, args.side_z, distance=args.side_distance)
            top_cam  = _top_camera(tx, ty, args.top_z,  height_m=args.top_height)

            side_renderer.update_scene(data, camera=side_cam, scene_option=scene_opt)
            top_renderer.update_scene(data,  camera=top_cam,  scene_option=scene_opt)
            side_frames.append(side_renderer.render().copy())
            top_frames.append(top_renderer.render().copy())

    side_renderer.close()
    top_renderer.close()
    return side_frames, top_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights",          required=True,   help="Path to best_weights.npy from gecko_forward.py")
    parser.add_argument("--input-size-orig",  type=int, default=13, help="Original network input size")
    parser.add_argument("--output-size-orig", type=int, default=8,  help="Original network output size")
    parser.add_argument("--hidden",           type=int, default=32, help="Network hidden size")
    parser.add_argument("--duration",         type=float, default=16.0)
    parser.add_argument("--side-duration",    type=float, default=None)
    parser.add_argument("--top-duration",     type=float, default=18.0)
    parser.add_argument("--fps",              type=int,   default=30)
    parser.add_argument("--out-dir",          default="__data__")
    parser.add_argument("--no-gif",           action="store_true")
    parser.add_argument("--no-mp4",           action="store_true")
    parser.add_argument("--side-distance",    type=float, default=1.8)
    parser.add_argument("--side-x",           type=float, default=0.0)
    parser.add_argument("--side-y",           type=float, default=-1.6)
    parser.add_argument("--side-z",           type=float, default=0.12)
    parser.add_argument("--top-height",       type=float, default=1.2)
    parser.add_argument("--top-x",            type=float, default=0.0)
    parser.add_argument("--top-y",            type=float, default=-1.6)
    parser.add_argument("--top-z",            type=float, default=1.0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _run_view(duration):
        args.duration = duration
        return _run(args)

    side_dur = args.side_duration or args.duration
    top_dur  = args.top_duration  or args.duration

    print(f"Recording side view ({side_dur}s) @ {args.fps} fps …")
    side_frames, _ = _run_view(side_dur)
    print(f"Recording top view ({top_dur}s) @ {args.fps} fps …")
    _, top_frames  = _run_view(top_dur)

    if not args.no_gif:
        _save_gif(side_frames, out / "gecko_amputee_side.gif", args.fps)
        _save_gif(top_frames,  out / "gecko_amputee_top.gif",  args.fps)

    if not args.no_mp4:
        _save_mp4(side_frames, out / "gecko_amputee_side.mp4", args.fps)
        _save_mp4(top_frames,  out / "gecko_amputee_top.mp4",  args.fps)


if __name__ == "__main__":
    main()
