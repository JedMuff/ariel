"""
Side-view and top-view GIF + MP4 video of the gecko's gait.

Runs the simulation in real-time, captures frames from a side camera and a
top-down camera, then writes:
  - __data__/gecko_trail_side.gif  / gecko_trail_side.mp4
  - __data__/gecko_trail_top.gif   / gecko_trail_top.mp4

Usage
-----
    # From gecko_forward.py weights:
    uv run python examples/re_book/gecko_trail_video.py \\
        --weights __data__/gecko_forward/best_weights.npy

    # From a script-6 co-evolution checkpoint:
    uv run python examples/re_book/gecko_trail_video.py \\
        --checkpoint __data__/6_body_brain_randomized_waypoints/checkpoints/gen002_body00

    # No MP4, just GIF:
    uv run python examples/re_book/gecko_trail_video.py \\
        --weights __data__/gecko_forward/best_weights.npy --no-mp4
"""
from __future__ import annotations

import argparse
import json
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


# ---------------------------------------------------------------------------
# Network (matches gecko_forward.py)
# ---------------------------------------------------------------------------

class _Network(nn.Module):
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


# ---------------------------------------------------------------------------
# World builders
# ---------------------------------------------------------------------------

def _build_gecko_world():
    from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
    from ariel.simulation.controllers.utils.data_get import get_state_from_data
    from ariel.simulation.environments import SimpleFlatWorld

    world = SimpleFlatWorld(floor_size=(100, 100, 1))
    world.spawn(gecko().spec, position=SPAWN_POS)
    model = world.spec.compile()
    data  = mujoco.MjData(model)

    state_dim = len(get_state_from_data(data)) + 2
    return model, data, state_dim


def _build_checkpoint_world(checkpoint: Path, reach_radius: float):
    _orig_argv = sys.argv
    sys.argv = [str(SCRIPT_DIR / "6_body_brain_randomized_waypoints.py"),
                "--no-video", "--reach-radius", str(reach_radius)]
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_bb_exp", SCRIPT_DIR / "6_body_brain_randomized_waypoints.py"
        )
        mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                  # type: ignore[union-attr]
    finally:
        sys.argv = _orig_argv

    genome    = json.loads((checkpoint / "best_genome.json").read_text())
    weights   = np.load(checkpoint / "best_weights.npy")
    waypoints = [np.asarray(w) for w in np.load(checkpoint / "best_waypoints.npy")]

    model, data, target_mocap_id, cam_name = mod._build_world_for_body(genome)
    return mod, model, data, weights, waypoints, target_mocap_id, cam_name


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _side_camera(x: float, y: float, z: float, distance: float = 1.8) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat    = np.array([x, y, z], dtype=np.float64)
    cam.distance  = distance
    cam.azimuth   = 0.0
    cam.elevation = -10.0
    return cam


def _top_camera(x: float, y: float, z: float, height_m: float = 1.2) -> mujoco.MjvCamera:
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

def _save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    duration_ms = int(1000 / fps)
    pil_frames  = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    print(f"Saved GIF → {path}  ({len(frames)} frames @ {fps} fps)")


def _save_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import cv2
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
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

def _run_gecko(args) -> tuple[list[np.ndarray], list[np.ndarray]]:
    from ariel.simulation.controllers.utils.data_get import get_state_from_data

    model, data, state_dim = _build_gecko_world()
    net = _Network(input_size=state_dim, output_size=model.nu, hidden=args.hidden)
    _fill(net, np.load(args.weights))

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


def _run_checkpoint(args) -> tuple[list[np.ndarray], list[np.ndarray]]:
    checkpoint = Path(args.checkpoint)
    mod, model, data, weights, waypoints, target_mocap_id, cam_name = \
        _build_checkpoint_world(checkpoint, args.reach_radius)

    input_dim = mod._genome_input_dim(model, data)
    net = mod.Network(input_size=input_dim, output_size=model.nu)
    mod.fill_parameters(net, weights)

    num_wps        = len(waypoints)
    wp_idx         = 0
    current_target = waypoints[0]
    mujoco.mj_resetData(model, data)
    data.mocap_pos[target_mocap_id] = current_target

    dt                = model.opt.timestep
    total_steps       = int(args.duration / dt)
    steps_per_frame   = max(1, int(round(1.0 / (args.fps * dt))))
    control_step_freq = 50
    current_ctrl      = np.zeros(model.nu)

    POV_H, POV_W = 96, 128
    ctrl_renderer = mujoco.Renderer(model, height=POV_H, width=POV_W)
    side_renderer = mujoco.Renderer(model, height=SIDE_H, width=SIDE_W)
    top_renderer  = mujoco.Renderer(model, height=TOP_H,  width=TOP_W)
    scene_opt     = mujoco.MjvOption()

    side_frames: list[np.ndarray] = []
    top_frames:  list[np.ndarray] = []

    for step in range(total_steps):
        if step % control_step_freq == 0:
            ctrl_renderer.update_scene(data, camera=cam_name)
            img    = ctrl_renderer.render()
            mask   = mod.isolate_green(img)
            vision = mod.analyze_sections(mask)
            rs     = mod.get_robot_state(data)
            phase  = [2.0 * np.sin(data.time * 2.0 * np.pi),
                      2.0 * np.cos(data.time * 2.0 * np.pi)]
            prog   = [wp_idx / max(num_wps - 1, 1)]
            state  = np.concatenate([rs, vision, phase, prog]).astype(np.float32)
            current_ctrl = net.forward(model, data, state)
        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)

        if wp_idx < num_wps:
            dist = float(np.linalg.norm(data.qpos[:2] - current_target[:2]))
            if dist <= args.reach_radius:
                wp_idx += 1
                if wp_idx < num_wps:
                    current_target = waypoints[wp_idx]
                    data.mocap_pos[target_mocap_id] = current_target

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

    ctrl_renderer.close()
    side_renderer.close()
    top_renderer.close()
    return side_frames, top_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights",       help="Path to best_weights.npy from gecko_forward.py")
    src.add_argument("--checkpoint",    help="Path to script-6 checkpoint dir")
    parser.add_argument("--duration",      type=float, default=16.0,  help="Simulation seconds to record (both views)")
    parser.add_argument("--side-duration", type=float, default=None,  help="Override duration for side view")
    parser.add_argument("--top-duration",  type=float, default=18.0, help="Override duration for top view")
    parser.add_argument("--fps",           type=int,   default=30,   help="Output frame rate")
    parser.add_argument("--out-dir",       default="__data__",       help="Output directory")
    parser.add_argument("--no-gif",        action="store_true",      help="Skip GIF output")
    parser.add_argument("--no-mp4",        action="store_true",      help="Skip MP4 output")
    parser.add_argument("--reach-radius",  type=float, default=0.20)
    parser.add_argument("--side-distance", type=float, default=1.8,  help="Side-cam distance (m)")
    parser.add_argument("--side-x",        type=float, default=0.0,  help="Fixed side-cam lookat X (default: track gecko)")
    parser.add_argument("--side-y",        type=float, default=-1.6, help="Fixed side-cam lookat Y (default: track gecko)")
    parser.add_argument("--side-z",        type=float, default=0.12, help="Side-cam lookat Z (m)")
    parser.add_argument("--top-height",    type=float, default=1.2,  help="Top-cam height (m)")
    parser.add_argument("--top-x",         type=float, default=0.0,  help="Fixed top-cam lookat X (default: track gecko)")
    parser.add_argument("--top-y",         type=float, default=-1.6, help="Fixed top-cam lookat Y (default: track gecko)")
    parser.add_argument("--top-z",         type=float, default=1.0,  help="Top-cam lookat Z (m)")
    parser.add_argument("--hidden",        type=int,   default=32,   help="Network hidden size")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    run = _run_gecko if args.weights else _run_checkpoint

    def _run_view(duration: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        args.duration = duration
        return run(args)

    side_dur = args.side_duration or args.duration
    top_dur  = args.top_duration  or args.duration

    print(f"Recording side view ({side_dur}s) @ {args.fps} fps …")
    side_frames, _ = _run_view(side_dur)
    print(f"Recording top view ({top_dur}s) @ {args.fps} fps …")
    _, top_frames  = _run_view(top_dur)

    if not args.no_gif:
        _save_gif(side_frames, out / "gecko_trail_side.gif", args.fps)
        _save_gif(top_frames,  out / "gecko_trail_top.gif",  args.fps)

    if not args.no_mp4:
        _save_mp4(side_frames, out / "gecko_trail_side.mp4", args.fps)
        _save_mp4(top_frames,  out / "gecko_trail_top.mp4",  args.fps)


if __name__ == "__main__":
    main()
