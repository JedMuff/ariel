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
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from rich.console import Console

from behavior_common import resolve_eval_profile, resolve_replay_config, skills_for_task
from shared import (
    CONTROL_STEP_FREQ,
    SETTLE_DURATION,
    SWEEP_PROFILE,
    EvalProfile,
    SkillReward,
    build_loco_world_for_body,
    genome_to_spec,
    make_brain_for_model,
    run_skill_episode,
)

console = Console()

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
    brain_kind: str = "ann",
    profile: EvalProfile = SWEEP_PROFILE,
) -> dict:
    """Replay one skill with shared.run_skill_episode -- the training episode
    itself, so the run's eval profile (durations, smoothing, penalties, score
    normalisation, a turn's early end) applies -- grabbing frames from its
    recorder hook."""
    model, data = build_loco_world_for_body(genome, to_spec_fn=to_spec_fn)
    brain = make_brain_for_model(brain_kind, model, data, weights.astype(np.float32))
    reward = SkillReward(**reward_spec)

    renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    frame_every = max(1, round(1.0 / (model.opt.timestep * render_fps)))
    # (frame, metric_line, active_time) -- the fitness line is drawn in a
    # second pass below, once the full episode has run and the true
    # per-skill fitness is known. meta['fitness'] is the checkpoint's
    # task-wide mean across every trained skill, not this episode's fitness.
    pending_frames: list[tuple[np.ndarray, str, float]] = []
    n_steps = 0

    def record(d: mujoco.MjData, raw: float) -> None:
        nonlocal n_steps
        n_steps += 1
        if n_steps % frame_every:
            return
        renderer.update_scene(d, camera=_make_tracking_camera(d.xpos[core_id].copy()))
        frame_bgr = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
        if reward.kind == "translate":
            metric_line = f"Displacement (angle={reward.angle_deg:.0f} deg): {raw:+.3f} m"
        else:
            direction_label = "CCW (left)" if reward.turn_sign == +1 else "CW (right)"
            metric_line = f"Yaw accumulated ({direction_label}): {math.degrees(raw):.1f} deg"
        pending_frames.append((frame_bgr, metric_line, max(0.0, d.time - SETTLE_DURATION)))

    skill_fitness, info = run_skill_episode(model, data, brain, reward, control_step_freq, profile,
                                            recorder=record)
    renderer.close()

    if not pending_frames:
        console.log(f"  [red]No frames captured for skill={skill_name}[/red]")
        return {}

    score_line = (f"score={info['score']:.3f}  height_pen={info['height_penalty']:.2f}  "
                  f"jerk_pen={info['jerk_penalty']:.2f}" if "score" in info else None)
    frames: list[np.ndarray] = []
    for frame_bgr, metric_line, active_time in pending_frames:
        _overlay(frame_bgr, [line for line in [
            f"skill={skill_name}  fitness={skill_fitness:.3f}",
            score_line,
            f"t={active_time:.1f} / {reward.duration:.0f}s",
            metric_line,
            f"task-mean fitness (all skills)={meta.get('fitness', float('nan')):.3f}",
        ] if line is not None])
        frames.append(frame_bgr)

    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), render_fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()

    console.log(f"  Saved -> {out_path}  skill_fitness={skill_fitness:.3f}  {info}")
    return {"skill": skill_name, "video": str(out_path), "skill_fitness": skill_fitness, **info}


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

    to_spec_fn, control_step_freq, brain_kind = resolve_replay_config(ckpt_dir, max_modules=max_modules)
    profile = resolve_eval_profile(ckpt_dir)

    results = []
    for skill_name, reward_spec in skills_for_task(task, profile):
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
            brain_kind=brain_kind, profile=profile,
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
