"""
Render a video of a saved gecko_food_skills.py checkpoint: rebuilds the body
from best_genome.json, loads the loco/left/right weights, and replays the
full food episode (SkillController-driven) with an overhead tracking camera.

Usage:
    python render_food_skills_checkpoint.py \\
        --checkpoint-dir data/food_skils/food_skills_.../__data__/gecko_food_skills/checkpoints/gen001_body22 \\
        --out out.mp4

    # Render several checkpoints in one call:
    python render_food_skills_checkpoint.py --checkpoint-dir DIR1 DIR2 DIR3 --out-dir data/food_skills_videos
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

from shared import (
    Network,
    analyze_sections,
    fill_parameters,
    genome_to_spec,
    isolate_green,
    signed_vertical_yaw_delta,
)

console = Console()

# ── Episode constants (must match gecko_food_skills.py) ──────────────────────

SPAWN_POSITION        = (0.0, 0.0, 0.1)
GATE_HALF_HEIGHT       = 0.15
# Must match gecko_food_skills.py's FORWARD_AXIS: the core-mounted camera
# looks down world -Y for every body (no spawn rotation is ever applied), so
# the loco skill's reward — and this overlay's displacement readout — are
# both measured along that axis, not world +X.
FORWARD_AXIS           = np.array([0.0, -1.0])
HIDDEN_SIZES           = [32]
CONTROL_STEP_FREQ      = 100
CTRL_ALPHA             = 0.5
SETTLE_DURATION        = 3.0
FOOD_EVAL_DURATION     = 120.0
LOCO_DURATION          = 30.0
TURN_DURATION          = 15.0
HINGE_CONTACT_LIMIT    = 200
COMMIT_STEPS           = 40
CENTRE_FWD_THRESH      = 0.4


class SkillController:
    def __init__(self, rng, commit_steps=COMMIT_STEPS, centre_fwd_thresh=CENTRE_FWD_THRESH):
        self.last_turn_dir = 2 if rng.random() < 0.5 else 3
        self.commit_steps = commit_steps
        self.centre_fwd_thresh = centre_fwd_thresh
        self._current_skill = self.last_turn_dir
        self._steps_held = commit_steps

    def _decide(self, left, centre, right):
        if (left + centre + right) > 0.0:
            if centre >= self.centre_fwd_thresh or (centre >= left and centre >= right):
                return 1
            elif left >= right:
                self.last_turn_dir = 2
                return 2
            else:
                self.last_turn_dir = 3
                return 3
        return self.last_turn_dir

    def select(self, left, centre, right):
        self._steps_held += 1
        if self._steps_held < self.commit_steps:
            return self._current_skill
        candidate = self._decide(left, centre, right)
        if candidate != self._current_skill:
            self._current_skill = candidate
            self._steps_held = 0
        return self._current_skill


def build_food_world(genome_dict: dict, reach_radius: float):
    spec = genome_to_spec(genome_dict)
    if spec is None:
        raise ValueError("Could not decode morphology")
    from ariel.simulation.environments import SimpleFlatWorld

    world = SimpleFlatWorld()
    try:
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=False)

    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=[0.0, 0.0, GATE_HALF_HEIGHT]
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[reach_radius, GATE_HALF_HEIGHT],
        rgba=[0, 1, 0, 0.7],
        contype=0, conaffinity=0,
    )

    model = world.spec.compile()
    data = mujoco.MjData(model)
    target_mocap_id = model.body("green_target").mocapid[0]

    cam_name = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "overview" not in name:
            cam_name = name
            break

    return model, data, target_mocap_id, cam_name


def _make_tracking_camera(core_pos: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = core_pos
    cam.distance = 4.5
    cam.azimuth = 225.0
    cam.elevation = -35.0
    return cam


def _overlay(frame_bgr: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        cv2.putText(
            frame_bgr, text, (10, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA,
        )


FPV_BORDER = 3  # white border thickness around the PiP inset


def _pip_overlay_bottom(base_bgr: np.ndarray, fpv_bgr: np.ndarray, margin: int = 10) -> np.ndarray:
    """Composite fpv_bgr as a picture-in-picture along the bottom of base_bgr."""
    out = base_bgr.copy()
    h, w = fpv_bgr.shape[:2]
    bh, bw = base_bgr.shape[:2]
    x0 = bw - w - margin - FPV_BORDER * 2
    y0 = bh - h - margin - FPV_BORDER * 2
    out[y0 : y0 + h + FPV_BORDER * 2, x0 : x0 + w + FPV_BORDER * 2] = 255
    out[y0 + FPV_BORDER : y0 + FPV_BORDER + h,
        x0 + FPV_BORDER : x0 + FPV_BORDER + w] = fpv_bgr
    return out


def render_checkpoint(
    ckpt_dir: Path,
    out_path: Path,
    reach_radius: float = 0.20,
    seed: int = 0,
    render_fps: int = 30,
    render_height: int = 480,
    render_width: int = 640,
) -> dict:
    genome = json.loads((ckpt_dir / "best_genome.json").read_text())
    meta = json.loads((ckpt_dir / "meta.json").read_text()) if (ckpt_dir / "meta.json").exists() else {}
    waypoints = np.load(ckpt_dir / "best_waypoints.npy")

    loco_w = np.load(ckpt_dir / "loco_weights.npy")
    left_w = np.load(ckpt_dir / "left_weights.npy")
    right_w = np.load(ckpt_dir / "right_weights.npy")

    model, data, target_mocap_id, cam_name = build_food_world(genome, reach_radius)
    input_dim = model.nq + model.nv  # placeholder overwritten below
    from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
    input_dim = len(get_robot_state(data))
    output_dim = model.nu

    def _load(w):
        net = Network(input_size=input_dim, output_size=output_dim, hidden_size=HIDDEN_SIZES[0])
        fill_parameters(net, w.astype(np.float32))
        return net

    loco_net = _load(loco_w)
    left_net = _load(left_w)
    right_net = _load(right_w)

    vis_renderer = mujoco.Renderer(model, height=96, width=128)
    ov_renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    rotor_geom_ids = {i for i in range(model.ngeom) if model.geom(i).name.endswith("-rotor")}
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    rng = np.random.default_rng(seed)

    num_wps = len(waypoints)
    cur_wp = 0
    reached = 0
    target = waypoints[0]
    data.mocap_pos[target_mocap_id] = target
    min_dist = float("inf")

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)
        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= reach_radius:
            reached += 1
            cur_wp += 1
            if cur_wp < num_wps:
                target = waypoints[cur_wp]
                data.mocap_pos[target_mocap_id] = target
                min_dist = float("inf")

    controller = SkillController(rng)
    step = 0
    action = np.zeros(model.nu)
    skill = 1
    c_hinge = 0
    prev_rotor_contacts: set[int] = set()
    frames: list[np.ndarray] = []
    frame_every = max(1, round(1.0 / (model.opt.timestep * render_fps)))
    skill_names = {1: "LOCO", 2: "LEFT", 3: "RIGHT"}
    skill_log: list[int] = []
    fpv_scale = 1.5
    fpv_w, fpv_h = round(128 * fpv_scale), round(96 * fpv_scale)
    last_fpv_bgr = np.zeros((fpv_h, fpv_w, 3), dtype=np.uint8)

    episode_end = SETTLE_DURATION + FOOD_EVAL_DURATION
    while data.time < episode_end and cur_wp < num_wps:
        if step % CONTROL_STEP_FREQ == 0:
            vis_renderer.update_scene(data, camera=cam_name)
            img = vis_renderer.render()
            vision = analyze_sections(isolate_green(img))
            left_frac, centre_frac, right_frac = vision
            skill = controller.select(left_frac, centre_frac, right_frac)
            skill_log.append(skill)

            fpv_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            last_fpv_bgr = cv2.resize(
                fpv_bgr, (fpv_w, fpv_h), interpolation=cv2.INTER_NEAREST
            )
            # Section dividers (matches analyze_sections' 3-way vertical split).
            third = fpv_w // 3
            cv2.line(last_fpv_bgr, (third, 0), (third, fpv_h), (0, 255, 255), 1)
            cv2.line(last_fpv_bgr, (2 * third, 0), (2 * third, fpv_h), (0, 255, 255), 1)
            for i, pct in enumerate((left_frac, centre_frac, right_frac)):
                cv2.putText(
                    last_fpv_bgr, f"{100 * pct:.0f}%", (i * third + 4, fpv_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA,
                )
            cv2.putText(last_fpv_bgr, "FPV", (4, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            state = get_robot_state(data).astype(np.float32)
            net = {1: loco_net, 2: left_net, 3: right_net}[skill]
            raw_action = net.forward(model, data, state)
            action = np.clip(action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                              -math.pi / 2, math.pi / 2)

        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        step += 1

        curr_rotor_contacts: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor_id and g2 in rotor_geom_ids:
                curr_rotor_contacts.add(g2)
            elif g2 == floor_id and g1 in rotor_geom_ids:
                curr_rotor_contacts.add(g1)
        c_hinge += len(curr_rotor_contacts - prev_rotor_contacts)
        prev_rotor_contacts = curr_rotor_contacts

        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= reach_radius:
            reached += 1
            cur_wp += 1
            if cur_wp < num_wps:
                target = waypoints[cur_wp]
                data.mocap_pos[target_mocap_id] = target
                min_dist = float("inf")

        if step % frame_every == 0:
            core_pos = data.xpos[core_id].copy()
            cam = _make_tracking_camera(core_pos)
            ov_renderer.update_scene(data, camera=cam)
            frame_bgr = cv2.cvtColor(ov_renderer.render().copy(), cv2.COLOR_RGB2BGR)
            _overlay(frame_bgr, [
                f"{ckpt_dir.name}  fitness={meta.get('fitness', float('nan')):.3f}",
                f"t={data.time:.1f}s  wp={reached}/{num_wps}  skill={skill_names.get(skill, '?')}",
                f"c_hinge={c_hinge}",
            ])
            frame_bgr = _pip_overlay_bottom(frame_bgr, last_fpv_bgr)
            frames.append(frame_bgr)

    vis_renderer.close()
    ov_renderer.close()

    if not frames:
        console.log(f"  [red]No frames captured for {ckpt_dir}[/red]")
        return {}

    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), render_fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()

    total_ctrl_steps = len(skill_log)
    skill_pct = {
        "loco": 100.0 * skill_log.count(1) / max(total_ctrl_steps, 1),
        "left": 100.0 * skill_log.count(2) / max(total_ctrl_steps, 1),
        "right": 100.0 * skill_log.count(3) / max(total_ctrl_steps, 1),
    }
    result = {
        "checkpoint": str(ckpt_dir),
        "video": str(out_path),
        "waypoints_reached": reached,
        "num_waypoints": int(num_wps),
        "c_hinge": c_hinge,
        "skill_pct": skill_pct,
        "meta_fitness": meta.get("fitness"),
    }
    console.log(f"  Saved -> {out_path}  wp={reached}/{num_wps}  c_hinge={c_hinge}")
    return result


def render_skill_only(
    ckpt_dir: Path,
    skill: str,
    out_path: Path,
    render_fps: int = 30,
    render_height: int = 480,
    render_width: int = 640,
) -> dict:
    """Replay just the isolated single-skill training episode (matches
    _skill_worker_eval in gecko_food_skills.py): plain flat world, no food
    marker, no vision/SkillController — the {skill} network drives the whole
    episode. Useful for inspecting whether a suspiciously extreme turn/loco
    learning-curve fitness corresponds to a real behaviour or a metric exploit.
    """
    genome = json.loads((ckpt_dir / "best_genome.json").read_text())
    meta = json.loads((ckpt_dir / "meta.json").read_text()) if (ckpt_dir / "meta.json").exists() else {}
    weights = np.load(ckpt_dir / f"{skill}_weights.npy")

    from ariel.simulation.environments import SimpleFlatWorld
    from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state

    spec = genome_to_spec(genome)
    if spec is None:
        raise ValueError("Could not decode morphology")
    world = SimpleFlatWorld()
    try:
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=False)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    input_dim = len(get_robot_state(data))
    output_dim = model.nu
    net = Network(input_size=input_dim, output_size=output_dim, hidden_size=HIDDEN_SIZES[0])
    fill_parameters(net, weights.astype(np.float32))

    renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    rotor_geom_ids = {i for i in range(model.ngeom) if model.geom(i).name.endswith("-rotor")}
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    frame_every = max(1, round(1.0 / (model.opt.timestep * render_fps)))

    mujoco.mj_resetData(model, data)
    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    xy0 = np.array([data.qpos[0], data.qpos[1]])
    r_prev = np.array(data.xmat[core_id]).reshape(3, 3).copy()
    yaw_total = 0.0
    direction = {"loco": 0, "left": +1, "right": -1}[skill]
    duration = LOCO_DURATION if skill == "loco" else TURN_DURATION

    step = 0
    action = np.zeros(model.nu)
    c_hinge = 0
    prev_rotor_contacts: set[int] = set()
    frames: list[np.ndarray] = []

    episode_end = SETTLE_DURATION + duration
    while data.time < episode_end:
        if step % CONTROL_STEP_FREQ == 0:
            state = get_robot_state(data).astype(np.float32)
            raw_action = net.forward(model, data, state)
            action = np.clip(action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                              -math.pi / 2, math.pi / 2)

        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        step += 1

        curr_rotor_contacts: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor_id and g2 in rotor_geom_ids:
                curr_rotor_contacts.add(g2)
            elif g2 == floor_id and g1 in rotor_geom_ids:
                curr_rotor_contacts.add(g1)
        c_hinge += len(curr_rotor_contacts - prev_rotor_contacts)
        prev_rotor_contacts = curr_rotor_contacts

        r_curr = np.array(data.xmat[core_id]).reshape(3, 3)
        delta = signed_vertical_yaw_delta(r_prev, r_curr)
        if direction == +1:
            yaw_total += max(0.0, delta)
        elif direction == -1:
            yaw_total += max(0.0, -delta)
        r_prev = r_curr.copy()

        if step % frame_every == 0:
            core_pos = data.xpos[core_id].copy()
            cam = _make_tracking_camera(core_pos)
            renderer.update_scene(data, camera=cam)
            frame_bgr = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)

            active_time = max(0.0, data.time - SETTLE_DURATION)
            if skill == "loco":
                xy_now = np.array([data.qpos[0], data.qpos[1]])
                fwd_disp = float(np.dot(xy_now - xy0, FORWARD_AXIS))
                metric_line = f"Forward-axis displacement: {fwd_disp:+.3f} m"
            else:
                direction_label = "CCW (left)" if direction == +1 else "CW (right)"
                metric_line = f"Yaw accumulated ({direction_label}): {math.degrees(yaw_total):.1f} deg"

            _overlay(frame_bgr, [
                f"{ckpt_dir.name}  skill={skill}  food_fitness={meta.get('fitness', float('nan')):.3f}",
                f"t={active_time:.1f} / {duration:.0f}s",
                metric_line,
                f"c_hinge={c_hinge}",
            ])
            frames.append(frame_bgr)

    renderer.close()

    if not frames:
        console.log(f"  [red]No frames captured for {ckpt_dir} ({skill})[/red]")
        return {}

    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), render_fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()

    result = {
        "checkpoint": str(ckpt_dir),
        "skill": skill,
        "video": str(out_path),
        "yaw_accumulated_deg": math.degrees(yaw_total) if skill != "loco" else None,
        "forward_displacement_m": (
            float(np.dot(np.array([data.qpos[0], data.qpos[1]]) - xy0, FORWARD_AXIS))
            if skill == "loco" else None
        ),
        "c_hinge": c_hinge,
        "meta_fitness": meta.get("fitness"),
    }
    console.log(f"  Saved -> {out_path}  c_hinge={c_hinge}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Render gecko_food_skills.py checkpoints")
    parser.add_argument("--checkpoint-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--skill-only", choices=["loco", "left", "right"], default=None,
                         help="Render only the isolated single-skill episode "
                              "(plain flat world, no food/vision) instead of the "
                              "full food-collection episode")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output video path (single checkpoint only)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/food_skills_videos"))
    parser.add_argument("--reach-radius", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--render-width", type=int, default=640)
    args = parser.parse_args()

    console.rule("[bold magenta]Gecko Food-Skills Checkpoint Renderer[/bold magenta]")

    summary = []
    for ckpt_dir in args.checkpoint_dir:
        run_tag = ckpt_dir.parents[3].name if len(ckpt_dir.parents) >= 4 else ""
        if args.skill_only is not None:
            if args.out is not None and len(args.checkpoint_dir) == 1:
                out_path = args.out
            else:
                out_path = args.out_dir / f"{run_tag}_{ckpt_dir.name}_{args.skill_only}.mp4"
            console.log(f"Rendering {ckpt_dir} [{args.skill_only} only] -> {out_path}")
            res = render_skill_only(
                ckpt_dir, args.skill_only, out_path,
                render_fps=args.render_fps,
                render_height=args.render_height, render_width=args.render_width,
            )
        else:
            if args.out is not None and len(args.checkpoint_dir) == 1:
                out_path = args.out
            else:
                out_path = args.out_dir / f"{run_tag}_{ckpt_dir.name}.mp4"
            console.log(f"Rendering {ckpt_dir} -> {out_path}")
            res = render_checkpoint(
                ckpt_dir, out_path,
                reach_radius=args.reach_radius, seed=args.seed,
                render_fps=args.render_fps,
                render_height=args.render_height, render_width=args.render_width,
            )
        if res:
            summary.append(res)

    console.rule("[bold green]Done[/bold green]")
    for r in summary:
        console.log(r)


if __name__ == "__main__":
    main()
