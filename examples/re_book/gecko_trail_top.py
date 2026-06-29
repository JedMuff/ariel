"""
Ghost-trail bird's-eye-view image of the gecko's gait.

Samples --num-poses evenly-spaced snapshots, renders each with MuJoCo offscreen
rendering, then alpha-composites them onto a single dark-background PNG.
Early poses are opaque, later poses fade out.

Usage
-----
    # From gecko_forward.py weights:
    uv run python examples/re_book/gecko_trail_top.py \\
        --weights __data__/gecko_forward/best_weights.npy

    # From a script-6 co-evolution checkpoint:
    uv run python examples/re_book/gecko_trail_top.py \\
        --checkpoint __data__/6_body_brain_randomized_waypoints/checkpoints/gen002_body00
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

BG_COLOR   = (30, 30, 30)
WIDTH      = 960
HEIGHT     = 5200
CONTROL_HZ = 20
SPAWN_POS  = [0.0, 0.0, 0.1]


# ---------------------------------------------------------------------------
# Gecko-forward network (must match gecko_forward.py)
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

    state_dim = len(get_state_from_data(data)) + 2  # robot state + phase
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
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)                 # type: ignore[union-attr]
    finally:
        sys.argv = _orig_argv

    genome    = json.loads((checkpoint / "best_genome.json").read_text())
    weights   = np.load(checkpoint / "best_weights.npy")
    waypoints = [np.asarray(w) for w in np.load(checkpoint / "best_waypoints.npy")]

    model, data, target_mocap_id, cam_name = mod._build_world_for_body(genome)
    return mod, model, data, weights, waypoints, target_mocap_id, cam_name


# ---------------------------------------------------------------------------
# Camera / composite helpers
# ---------------------------------------------------------------------------

def _make_top_camera(median_x: float, median_y: float,
                     height_m: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat    = np.array([median_x, median_y, 0.0], dtype=np.float64)
    cam.distance  = height_m
    cam.azimuth   = 90.0
    cam.elevation = -90.0
    return cam


def _composite(canvas: np.ndarray, frame: np.ndarray, alpha: float) -> np.ndarray:
    out = alpha * frame.astype(np.float32) + (1.0 - alpha) * canvas.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Snapshot collection
# ---------------------------------------------------------------------------

def _collect_snapshots_gecko(model, data, weights, num_poses, duration, hidden):
    from ariel.simulation.controllers.utils.data_get import get_state_from_data

    state_dim = len(get_state_from_data(data)) + 2
    net = _Network(input_size=state_dim, output_size=model.nu, hidden=hidden)
    _fill(net, weights)

    mujoco.mj_resetData(model, data)
    dt             = model.opt.timestep
    control_period = int(round(1.0 / (CONTROL_HZ * dt)))
    total_steps    = int(duration / dt)
    current_ctrl   = np.zeros(model.nu)

    sample_indices = {
        int(round(i * (total_steps - 1) / (num_poses - 1)))
        for i in range(num_poses)
    } if num_poses > 1 else {total_steps - 1}

    snapshots, times = [], []
    for step in range(total_steps):
        if step % control_period == 0:
            rs    = get_state_from_data(data)
            phase = [np.sin(data.time * 2.0 * np.pi), np.cos(data.time * 2.0 * np.pi)]
            state = np.concatenate([rs, phase]).astype(np.float32)
            current_ctrl = net.forward(state)
        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)
        if step in sample_indices:
            snapshots.append(data.qpos.copy())
            times.append(float(data.time))

    return snapshots, times


def _collect_snapshots_checkpoint(mod, model, data, weights, waypoints,
                                  target_mocap_id, cam_name, num_poses,
                                  duration, reach_radius):
    input_dim = mod._genome_input_dim(model, data)
    net = mod.Network(input_size=input_dim, output_size=model.nu)
    mod.fill_parameters(net, weights)

    num_wps        = len(waypoints)
    wp_idx         = 0
    current_target = waypoints[0]
    mujoco.mj_resetData(model, data)
    data.mocap_pos[target_mocap_id] = current_target

    dt                = model.opt.timestep
    total_steps       = int(duration / dt)
    control_step_freq = 50
    current_ctrl      = np.zeros(model.nu)

    POV_H, POV_W = 96, 128
    ctrl_renderer = mujoco.Renderer(model, height=POV_H, width=POV_W)

    sample_indices = {
        int(round(i * (total_steps - 1) / (num_poses - 1)))
        for i in range(num_poses)
    } if num_poses > 1 else {total_steps - 1}

    snapshots, times = [], []
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
            if dist <= reach_radius:
                wp_idx += 1
                if wp_idx < num_wps:
                    current_target = waypoints[wp_idx]
                    data.mocap_pos[target_mocap_id] = current_target

        if step in sample_indices:
            snapshots.append(data.qpos.copy())
            times.append(float(data.time))

    ctrl_renderer.close()
    return snapshots, times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_trail(args) -> None:
    from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
    from ariel.simulation.environments import SimpleFlatWorld

    # --- collect snapshots using a single-gecko simulation ---
    if args.weights:
        sim_model, sim_data, _ = _build_gecko_world()
        snapshots, times = _collect_snapshots_gecko(
            sim_model, sim_data, np.load(args.weights),
            args.num_poses, args.duration, args.hidden,
        )
    else:
        checkpoint = Path(args.checkpoint)
        mod, sim_model, sim_data, weights, waypoints, target_mocap_id, cam_name = \
            _build_checkpoint_world(checkpoint, args.reach_radius)
        snapshots, times = _collect_snapshots_checkpoint(
            mod, sim_model, sim_data, weights, waypoints,
            target_mocap_id, cam_name,
            args.num_poses, args.duration, args.reach_radius,
        )

    if not snapshots:
        raise SystemExit("No snapshots collected.")
    print(f"Collected {len(snapshots)} poses at t={[f'{t:.2f}' for t in times]}")

    n         = len(snapshots)
    MIN_ALPHA = 0.15

    # --- build a fresh world with N gecko copies, one per snapshot ---
    world = SimpleFlatWorld(floor_size=(200, 200, 1))
    for qpos in snapshots:
        world.spawn(gecko().spec, position=[float(qpos[0]), float(qpos[1]), float(qpos[2])])

    model = world.spec.compile()
    data  = mujoco.MjData(model)

    # Set each copy's geom alphas. Spawned geoms are prefixed "robot{k+1}_".
    for i in range(n):
        prefix = f"robot{i + 1}_"
        t      = i / max(n - 1, 1)
        alpha  = MIN_ALPHA + (1.0 - MIN_ALPHA) * t if i < n - 1 else 1.0
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name.startswith(prefix):
                model.geom_rgba[g, 3] = alpha
        print(f"  copy {i+1}/{n}  alpha={alpha:.2f}")

    # Apply each snapshot's full qpos to the corresponding robot's dofs.
    mujoco.mj_resetData(model, data)
    single_nq = sim_model.nq
    for i, qpos in enumerate(snapshots):
        offset = i * single_nq
        data.qpos[offset: offset + single_nq] = qpos
    mujoco.mj_forward(model, data)

    median_x = float(np.median([s[0] for s in snapshots]))
    median_y = float(np.median([s[1] for s in snapshots]))

    model.vis.headlight.ambient  = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse  = [0.8, 0.8, 0.8]
    model.vis.headlight.specular = [0.4, 0.4, 0.4]
    model.vis.rgba.fog           = [0.0, 0.0, 0.0, 0.0]
    model.vis.global_.offwidth   = args.width
    model.vis.global_.offheight  = args.height

    scene_opt = mujoco.MjvOption()
    cam = _make_top_camera(median_x, median_y, args.camera_height)

    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG]    = False
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        frame = renderer.render()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output)
    print(f"Saved → {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights",    help="Path to best_weights.npy from gecko_forward.py")
    src.add_argument("--checkpoint", help="Path to script-6 checkpoint dir")
    parser.add_argument("--num-poses",     type=int,   default=4)
    parser.add_argument("--duration",      type=float, default=4.0)
    parser.add_argument("--output",        default="__data__/gecko_trail_top.png")
    parser.add_argument("--reach-radius",  type=float, default=0.20)
    parser.add_argument("--width",         type=int,   default=WIDTH)
    parser.add_argument("--height",        type=int,   default=HEIGHT)
    parser.add_argument("--camera-height", type=float, default=1.2,
                        help="Distance of top camera above gecko (metres)")
    parser.add_argument("--hidden",        type=int,   default=32,
                        help="Hidden size used in gecko_forward.py (ignored for --checkpoint)")
    args = parser.parse_args()
    render_trail(args)


if __name__ == "__main__":
    main()
