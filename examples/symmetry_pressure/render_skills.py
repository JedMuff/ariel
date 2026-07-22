"""
Render videos of the trained gecko skill networks for visual inspection.

Loads weights from __data__/gecko_skills/ and renders one MP4 per skill.
An overhead tracking camera follows the gecko. Overlays show the current skill,
elapsed time, and cumulative yaw or X-displacement so learning quality is
immediately visible.

Usage:
    # After training:
    python render_skills.py

    # Quick smoke-test with untrained random weights:
    python render_skills.py --random-weights

    # Render only specific skills:
    python render_skills.py --skills loco left

    # Override weights directory:
    python render_skills.py --weights-dir /path/to/weights
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import cv2
import mujoco
import numpy as np
import torch
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

from shared import (
    SPAWN_POSITION,
    Network,
    fill_parameters,
    signed_vertical_yaw_delta,
)

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Render gecko skill networks")
parser.add_argument("--weights-dir",   type=Path,
                    default=Path("__data__/gecko_skills"),
                    help="Directory containing {loco,left,right}_weights.npy")
parser.add_argument("--out-dir",       type=Path,
                    default=Path("__data__/gecko_skills/videos"))
parser.add_argument("--skills",        nargs="+",
                    choices=["loco", "left", "right"],
                    default=["loco", "left", "right"])
parser.add_argument("--random-weights", action="store_true",
                    help="Use random weights instead of loading from disk (sanity check)")
parser.add_argument("--render-fps",    type=int,   default=30)
parser.add_argument("--render-height", type=int,   default=480)
parser.add_argument("--render-width",  type=int,   default=640)
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)

# ── Episode constants (must match gecko_skills.py) ────────────────────────────

SETTLE_DURATION   = 3.0
CTRL_ALPHA        = 0.5
CONTROL_STEP_FREQ = 50
LOCO_DURATION     = 30.0
TURN_DURATION     = 15.0

# ── World builder ─────────────────────────────────────────────────────────────


def build_world() -> tuple[mujoco.MjModel, mujoco.MjData]:
    core  = gecko()
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                correct_collision_with_floor=True)
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    return model, data


# ── Renderer helpers ──────────────────────────────────────────────────────────


def _make_tracking_camera(core_pos: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = core_pos
    cam.distance  = 2.2
    cam.azimuth   = 225.0
    cam.elevation = -35.0
    return cam


def _overlay(frame_bgr: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        cv2.putText(
            frame_bgr, text,
            (10, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (255, 255, 255), 1, cv2.LINE_AA,
        )


# ── Per-skill render ──────────────────────────────────────────────────────────


def render_skill(
    skill: str,
    weights: np.ndarray,
    out_path: Path,
) -> None:
    direction = {"loco": 0, "left": +1, "right": -1}[skill]
    duration  = LOCO_DURATION if skill == "loco" else TURN_DURATION

    model, data = build_world()
    input_dim   = len(get_robot_state(data))
    network     = Network(input_size=input_dim, output_size=model.nu)
    fill_parameters(network, weights)

    renderer    = mujoco.Renderer(model, height=args.render_height, width=args.render_width)
    core_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    frame_every = max(1, round(1.0 / (model.opt.timestep * args.render_fps)))

    mujoco.mj_resetData(model, data)

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    # Reference values measured after settling
    x0        = float(data.qpos[0])
    r_prev    = np.array(data.xmat[core_id]).reshape(3, 3).copy()
    yaw_total = 0.0

    step           = 0
    current_action = np.zeros(model.nu)
    frames: list[np.ndarray] = []

    episode_end = SETTLE_DURATION + duration
    while data.time < episode_end:
        if step % CONTROL_STEP_FREQ == 0:
            state      = get_robot_state(data).astype(np.float32)
            raw_action = network.forward(model, data, state)
            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -math.pi / 2, math.pi / 2,
            )

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

        # Track yaw every physics step for accuracy
        r_curr = np.array(data.xmat[core_id]).reshape(3, 3)
        delta  = signed_vertical_yaw_delta(r_prev, r_curr)
        if direction == +1:
            yaw_total += max(0.0, delta)
        elif direction == -1:
            yaw_total += max(0.0, -delta)
        r_prev = r_curr.copy()

        if step % frame_every == 0:
            core_pos = data.xpos[core_id].copy()
            cam      = _make_tracking_camera(core_pos)
            renderer.update_scene(data, camera=cam)
            frame     = renderer.render().copy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            active_time = max(0.0, data.time - SETTLE_DURATION)
            if skill == "loco":
                x_disp = float(data.qpos[0]) - x0
                metric_line = f"X displacement: {x_disp:+.3f} m"
            else:
                direction_label = "CCW (left)" if direction == +1 else "CW (right)"
                yaw_deg = math.degrees(yaw_total)
                metric_line = f"Yaw accumulated ({direction_label}): {yaw_deg:.1f} deg"

            _overlay(frame_bgr, [
                f"Skill: {skill}",
                f"t = {active_time:.1f} / {duration:.0f} s",
                metric_line,
            ])
            frames.append(frame_bgr)

        step += 1

    renderer.close()

    if not frames:
        console.log(f"  [red]No frames captured for {skill}[/red]")
        return

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.render_fps,
        (w, h),
    )
    for fr in frames:
        writer.write(fr)
    writer.release()
    console.log(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    torch.set_num_threads(1)

    console.rule("[bold magenta]Gecko Skill Renderer[/bold magenta]")

    # Derive num_params from a throw-away world
    model, data = build_world()
    input_dim   = len(get_robot_state(data))
    output_dim  = model.nu
    network     = Network(input_size=input_dim, output_size=output_dim)
    num_params  = sum(p.numel() for p in network.parameters())
    console.log(f"Network: input={input_dim}  output={output_dim}  params={num_params}")

    rng = np.random.default_rng(0)

    for skill in args.skills:
        weights_path = args.weights_dir / f"{skill}_weights.npy"

        if args.random_weights:
            weights = rng.uniform(-1.3, 1.3, size=num_params).astype(np.float32)
            console.log(f"  {skill}: using random weights")
        elif weights_path.exists():
            weights = np.load(weights_path).astype(np.float32)
            console.log(f"  {skill}: loaded weights from {weights_path}")
        else:
            console.log(f"  [yellow]{skill}: weights not found at {weights_path} — skipping[/yellow]")
            continue

        out_path = args.out_dir / f"{skill}.mp4"
        console.log(f"  Rendering {skill}…")
        render_skill(skill, weights, out_path)

    console.rule("[bold green]Done[/bold green]")
    console.log(f"Videos → {args.out_dir}")


if __name__ == "__main__":
    main()
