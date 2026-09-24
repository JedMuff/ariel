"""
Render videos of a saved gecko_skill_tasks.py checkpoint: rebuilds the body
from best_genome.json and replays each of the task's trained skills (as
defined by gecko_skill_tasks.py's _skills_for_task) using the matching
{skill_name}_weights.npy, with an overhead tracking camera.

Usage:
    python render_skill_task_checkpoint.py \\
        --checkpoint-dir ../../__data__/sympress_forward_tree_rep0_37568_0/__data__/gecko_skill_tasks/forward/checkpoints/gen030_body18 \\
        --out-dir ../../__data__/sympress_forward_tree_rep0_37568_0/analysis/videos
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from rich.console import Console

import genome_adapter
from shared import (
    CONTROL_STEP_FREQ,
    CTRL_ALPHA,
    HEIGHT_PENALTY_THRESHOLD,
    HINGE_CONTACT_LIMIT,
    HINGE_GLITCH_FITNESS,
    JERK_PENALTY_WEIGHT,
    JERK_THRESHOLD,
    LOCO_DURATION,
    SETTLE_DURATION,
    TURN_DURATION,
    FORWARD_AXIS,
    Network,
    build_loco_world_for_body,
    fill_parameters,
    floor_id,
    genome_to_spec,
    rotor_geom_ids,
    signed_vertical_yaw_delta,
)

console = Console()

N_DIRECTIONS = 5  # must match gecko_skill_tasks.py


def _skills_for_task(task: str) -> list[tuple[str, dict]]:
    """Duplicated from gecko_skill_tasks.py:_skills_for_task (importing that
    module would trigger its own module-level argparse.parse_args() and
    fail). Each entry is (skill_name, reward_spec) where reward_spec is a
    dict describing the SkillReward used at training time.
    """
    if task == "forward":
        return [("fwd", {"kind": "translate", "angle_deg": 0.0, "duration": LOCO_DURATION})]
    if task == "multidirection":
        step = 360.0 / N_DIRECTIONS
        return [
            (f"dir{i}", {"kind": "translate", "angle_deg": i * step, "duration": LOCO_DURATION})
            for i in range(N_DIRECTIONS)
        ]
    if task == "turn_avg":
        return [
            ("fwd", {"kind": "translate", "angle_deg": 0.0, "duration": LOCO_DURATION}),
            ("left", {"kind": "rotate", "turn_sign": +1, "duration": TURN_DURATION}),
            ("right", {"kind": "rotate", "turn_sign": -1, "duration": TURN_DURATION}),
        ]
    raise ValueError(f"Unknown task: {task!r}")


def _rotate_2d(v: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _make_tracking_camera(core_pos: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = core_pos
    cam.distance = 2.2
    cam.azimuth = 225.0
    cam.elevation = -35.0
    return cam


def _overlay(frame_bgr: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        cv2.putText(
            frame_bgr, text, (10, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )


def render_skill_episode(
    genome: dict,
    weights: np.ndarray,
    reward_spec: dict,
    skill_name: str,
    meta: dict,
    out_path: Path,
    render_fps: int = 30,
    render_height: int = 480,
    render_width: int = 640,
    to_spec_fn=genome_to_spec,
    control_step_freq: int = CONTROL_STEP_FREQ,
) -> dict:
    model, data = build_loco_world_for_body(genome, to_spec_fn=to_spec_fn)
    from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state

    input_dim = len(get_robot_state(data))
    output_dim = model.nu
    network = Network(input_size=input_dim, output_size=output_dim)
    fill_parameters(network, weights.astype(np.float32))

    renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    rotor_ids = rotor_geom_ids(model)
    floor = floor_id(model)
    frame_every = max(1, round(1.0 / (model.opt.timestep * render_fps)))

    mujoco.mj_resetData(model, data)
    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    initial_height = float(data.xpos[core_id, 2])

    kind = reward_spec["kind"]
    duration = reward_spec["duration"]
    if kind == "translate":
        xy0 = np.array([data.qpos[0], data.qpos[1]])
        reward_axis = _rotate_2d(FORWARD_AXIS, reward_spec["angle_deg"])
    else:
        r_prev = np.array(data.xmat[core_id]).reshape(3, 3).copy()
        turn_sign = reward_spec["turn_sign"]
        accumulated = 0.0

    step = 0
    action = np.zeros(model.nu)
    # (frame, metric_line, active_time) -- the fitness line is drawn in a
    # second pass below, once the full episode has run and the true
    # per-skill fitness is known.
    pending_frames: list[tuple[np.ndarray, str, float]] = []
    episode_end = SETTLE_DURATION + duration

    # Tracked purely to recompute this skill's own true (penalized) fitness
    # for the overlay -- see shared._skill_worker_eval, whose formula this
    # mirrors exactly. meta['fitness'] is the checkpoint's task-wide mean
    # across every trained skill (e.g. 5 directions for multidirection, 3 for
    # turn_avg), not this one episode's fitness, so it must not be displayed
    # as if it were.
    c_hinge = 0
    prev_contacts: set[int] = set()
    prev_ctrl: np.ndarray | None = None
    jerk_sum = 0.0
    ctrl_step = 0

    while data.time < episode_end:
        if step % control_step_freq == 0:
            state = get_robot_state(data).astype(np.float32)
            raw_action = network.forward(model, data, state)
            action = np.clip(
                action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -math.pi / 2, math.pi / 2,
            )
            if prev_ctrl is not None:
                jerk_sum += float(np.mean(np.abs(action - prev_ctrl)))
            prev_ctrl = action.copy()
            ctrl_step += 1

        data.ctrl[:] = action
        mujoco.mj_step(model, data)

        curr_contacts: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor and g2 in rotor_ids:
                curr_contacts.add(g2)
            elif g2 == floor and g1 in rotor_ids:
                curr_contacts.add(g1)
        c_hinge += len(curr_contacts - prev_contacts)
        prev_contacts = curr_contacts

        if kind == "rotate":
            r_curr = np.array(data.xmat[core_id]).reshape(3, 3)
            delta = signed_vertical_yaw_delta(r_prev, r_curr)
            accumulated += max(0.0, delta * turn_sign)
            r_prev = r_curr.copy()

        step += 1

        if step % frame_every == 0:
            core_pos = data.xpos[core_id].copy()
            cam = _make_tracking_camera(core_pos)
            renderer.update_scene(data, camera=cam)
            frame_bgr = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)

            active_time = max(0.0, data.time - SETTLE_DURATION)
            if kind == "translate":
                xy_now = np.array([data.qpos[0], data.qpos[1]])
                metric_line = f"Displacement (angle={reward_spec['angle_deg']:.0f} deg): " \
                              f"{float(np.dot(xy_now - xy0, reward_axis)):+.3f} m"
            else:
                direction_label = "CCW (left)" if turn_sign == +1 else "CW (right)"
                metric_line = f"Yaw accumulated ({direction_label}): {math.degrees(accumulated):.1f} deg"

            pending_frames.append((frame_bgr, metric_line, active_time))

    renderer.close()

    if not pending_frames:
        console.log(f"  [red]No frames captured for skill={skill_name}[/red]")
        return {}

    if c_hinge > HINGE_CONTACT_LIMIT:
        skill_fitness = HINGE_GLITCH_FITNESS
    else:
        mean_jerk = jerk_sum / max(ctrl_step - 1, 1)
        jerk_penalty = JERK_PENALTY_WEIGHT * mean_jerk if mean_jerk >= JERK_THRESHOLD else 0.0
        height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
        if kind == "translate":
            xy_now = np.array([data.qpos[0], data.qpos[1]])
            raw = float(np.dot(xy_now - xy0, reward_axis))
        else:
            raw = accumulated
        skill_fitness = -(raw - height_penalty - jerk_penalty)

    frames: list[np.ndarray] = []
    for frame_bgr, metric_line, active_time in pending_frames:
        _overlay(frame_bgr, [
            f"skill={skill_name}  fitness={skill_fitness:.3f}",
            f"t={active_time:.1f} / {duration:.0f}s",
            metric_line,
            f"task-mean fitness (all skills)={meta.get('fitness', float('nan')):.3f}",
        ])
        frames.append(frame_bgr)

    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), render_fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()

    console.log(f"  Saved -> {out_path}  skill_fitness={skill_fitness:.3f}")
    return {"skill": skill_name, "video": str(out_path), "skill_fitness": skill_fitness}


def render_checkpoint(
    ckpt_dir: Path,
    out_dir: Path,
    render_fps: int = 30,
    render_height: int = 480,
    render_width: int = 640,
    max_modules: int = 25,
    filename_prefix: str = "",
) -> list[dict]:
    """Render one video per skill trained for this checkpoint's task."""
    genome = json.loads((ckpt_dir / "best_genome.json").read_text())
    meta = json.loads((ckpt_dir / "meta.json").read_text()) if (ckpt_dir / "meta.json").exists() else {}
    task = meta.get("task")
    if task is None:
        raise ValueError(f"{ckpt_dir}/meta.json has no 'task' field")

    to_spec_fn = (
        functools.partial(genome_adapter.cppn_genome_to_spec, max_modules=max_modules)
        if meta.get("genome_type") == "cppn"
        else genome_to_spec
    )

    # Checkpoints live at <run>/checkpoints/<ckpt>; run_config.json sits in <run>.
    # Runs predating --control-step-freq have no such key and used the shared default.
    run_config_path = ckpt_dir.parent.parent / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    control_step_freq = run_config.get("control_step_freq", CONTROL_STEP_FREQ)

    results = []
    for skill_name, reward_spec in _skills_for_task(task):
        weights_path = ckpt_dir / f"{skill_name}_weights.npy"
        if not weights_path.exists():
            console.log(f"  [yellow]{skill_name}: weights not found at {weights_path} — skipping[/yellow]")
            continue
        weights = np.load(weights_path)
        out_path = out_dir / f"{filename_prefix}{ckpt_dir.name}_{skill_name}.mp4"
        res = render_skill_episode(
            genome, weights, reward_spec, skill_name, meta, out_path,
            render_fps=render_fps, render_height=render_height, render_width=render_width,
            to_spec_fn=to_spec_fn, control_step_freq=control_step_freq,
        )
        if res:
            results.append(res)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Render gecko_skill_tasks.py checkpoints")
    parser.add_argument("--checkpoint-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("__data__/gecko_skill_tasks_videos"))
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--render-width", type=int, default=640)
    args = parser.parse_args()

    console.rule("[bold magenta]Gecko Skill-Task Checkpoint Renderer[/bold magenta]")

    summary = []
    for ckpt_dir in args.checkpoint_dir:
        console.log(f"Rendering {ckpt_dir} -> {args.out_dir}")
        summary.extend(render_checkpoint(
            ckpt_dir, args.out_dir,
            render_fps=args.render_fps,
            render_height=args.render_height, render_width=args.render_width,
        ))

    console.rule("[bold green]Done[/bold green]")
    for r in summary:
        console.log(r)


if __name__ == "__main__":
    main()
