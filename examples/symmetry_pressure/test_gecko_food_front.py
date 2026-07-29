"""
End-to-end smoke test: fixed gecko morphology, food-in-front convention.

Verifies two things visually:
  1. The loco skill's training axis matches the direction the onboard camera
     actually faces (world -Y for the SimpleFlatWorld default spawn rotation
     used by gecko_food_skills.py -- confirmed via data.cam_xmat, not assumed).
  2. The first food waypoint, placed 1.5 m along that same forward axis
     (matching gecko_food_skills.py's BodyBrainEvolution.evaluate), is actually
     in front of the robot at spawn.

Trains loco/left/right skills from scratch via CMA-ES (same reward shapes as
gecko_skills.py, but loco rewards -Y displacement instead of +X -- see
FORWARD_AXIS below), then renders:
  - one isolated video per skill (tracking camera + displacement/yaw overlay)
  - one food-collection episode with the SkillController, tracking camera,
    and a picture-in-picture onboard-camera (FPV) view so you can see exactly
    what the vision pipeline sees.

Usage:
    python test_gecko_food_front.py
    python test_gecko_food_front.py --loco-budget 40 --turn-budget 40 --workers 8
    python test_gecko_food_front.py --skip-training  # reuse saved weights, just render
"""

import argparse
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cma
import cv2
import mujoco
import numpy as np
import torch
from rich.console import Console
from rich.traceback import install

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

from shared import (
    GATE_HALF_HEIGHT,
    RING_R_MAX,
    SPAWN_POSITION,
    Network,
    analyze_sections,
    fill_parameters,
    isolate_green,
    sample_waypoints,
    signed_vertical_yaw_delta,
)

install()
warnings.filterwarnings("ignore", message="TPA: apparent inconsistency",
                        category=UserWarning, module="cma")

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Gecko food-in-front smoke test")
parser.add_argument("--loco-budget",  type=int,   default=100)
parser.add_argument("--turn-budget",  type=int,   default=100)
parser.add_argument("--sigma",        type=float, default=0.7)
parser.add_argument("--init-scale",   type=float, default=1.3)
parser.add_argument("--workers",      type=int,   default=max(1, os.cpu_count() or 1))
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--skip-training", action="store_true",
                    help="Skip CMA-ES; load weights already saved under --out-dir")
parser.add_argument("--out-dir",      type=Path, default=Path("__data__/test_gecko_food_front"))
parser.add_argument("--num-waypoints", type=int,  default=3,
                    help="Total food waypoints for the final eval (first one is always in front)")
parser.add_argument("--food-duration", type=float, default=120.0)
parser.add_argument("--render-fps",    type=int,   default=30)
parser.add_argument("--render-height", type=int,   default=480)
parser.add_argument("--render-width",  type=int,   default=640)
args = parser.parse_args()

LOCO_BUDGET  = args.loco_budget
TURN_BUDGET  = args.turn_budget
SIGMA        = args.sigma
INIT_SCALE   = args.init_scale
WORKERS      = args.workers
SEED         = args.seed
DATA         = args.out_dir
DATA.mkdir(parents=True, exist_ok=True)

# ── Episode constants (match gecko_food_skills.py) ────────────────────────────

SETTLE_DURATION          = 3.0
CTRL_ALPHA               = 0.5
CONTROL_STEP_FREQ        = 100
HINGE_CONTACT_LIMIT      = 200
HINGE_CONTACT_PENALTY    = 0.005
HINGE_GLITCH_FITNESS     = 1.0
HEIGHT_PENALTY_THRESHOLD = 0.21

LOCO_DURATION      = 30.0
TURN_DURATION      = 15.0
FOOD_EVAL_DURATION = args.food_duration
COMMIT_STEPS       = 40
CENTRE_FWD_THRESH  = 0.4
REACH_RADIUS       = 0.20
COLLECT_RADIUS     = REACH_RADIUS

FPV_H, FPV_W = 180, 240


# ── World builders — NO spawn rotation, matching gecko_food_skills.py ────────
# (gecko_skills.py / run_skill_controller.py instead spawn with
#  rotation=(0, 0, 90), which happens to point the onboard camera down +X.
#  gecko_food_skills.py never applies that rotation, so its camera faces a
#  different world axis -- confirmed empirically below the first time this
#  module is imported into a running interpreter, see `FORWARD_AXIS`.)


def build_loco_world() -> tuple[mujoco.MjModel, mujoco.MjData]:
    core  = gecko()
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POSITION, correct_collision_with_floor=True)
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    return model, data


def build_food_world() -> tuple[mujoco.MjModel, mujoco.MjData, int, str]:
    core  = gecko()
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POSITION, correct_collision_with_floor=True)

    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=[0.0, 0.0, GATE_HALF_HEIGHT]
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[REACH_RADIUS, GATE_HALF_HEIGHT],
        rgba=[0, 1, 0, 0.7],
        contype=0, conaffinity=0,
    )

    model = world.spec.compile()
    data  = mujoco.MjData(model)
    target_mocap_id = model.body("green_target").mocapid[0]

    cam_name = ""
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "overview" not in name:
            cam_name = name
            break

    return model, data, target_mocap_id, cam_name


def _measure_forward_axis() -> np.ndarray:
    """Query MuJoCo for the onboard camera's actual world-frame forward vector
    at spawn, rather than assuming it. Returns a unit 2D (x, y) vector."""
    model, data, _, cam_name = build_food_world()
    mujoco.mj_forward(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_xmat = np.array(data.cam_xmat[cam_id]).reshape(3, 3)
    forward3 = cam_xmat @ np.array([0.0, 0.0, -1.0])
    forward2 = forward3[:2]
    return forward2 / np.linalg.norm(forward2)


FORWARD_AXIS = _measure_forward_axis()  # e.g. (0, -1) -- measured, not assumed
if __name__ == "__main__":
    console.log(
        f"Measured onboard-camera forward direction (world xy): "
        f"({FORWARD_AXIS[0]:+.3f}, {FORWARD_AXIS[1]:+.3f})"
    )


def _forward_displacement(data: mujoco.MjData, xy0: np.ndarray) -> float:
    """Displacement along the camera's measured forward axis (not assumed +X)."""
    xy_now = np.array([data.qpos[0], data.qpos[1]])
    return float(np.dot(xy_now - xy0, FORWARD_AXIS))


def _rotor_geom_ids(model: mujoco.MjModel) -> set[int]:
    return {i for i in range(model.ngeom) if model.geom(i).name.endswith("-rotor")}


def _floor_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


# ── Episode runners ────────────────────────────────────────────────────────────


def run_loco_episode(model, data, network, weights: np.ndarray) -> float:
    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    rotor_geom_ids = _rotor_geom_ids(model)
    floor_id       = _floor_id(model)

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    core_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    xy0            = np.array([data.qpos[0], data.qpos[1]])
    initial_height = float(data.xpos[core_id, 2])

    step, c_hinge = 0, 0
    current_action = np.zeros(model.nu)
    prev_rotor_contacts: set[int] = set()

    episode_end = SETTLE_DURATION + LOCO_DURATION
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

        curr: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor_id and g2 in rotor_geom_ids:
                curr.add(g2)
            elif g2 == floor_id and g1 in rotor_geom_ids:
                curr.add(g1)
        c_hinge += len(curr - prev_rotor_contacts)
        prev_rotor_contacts = curr
        step += 1

    if c_hinge > HINGE_CONTACT_LIMIT:
        return HINGE_GLITCH_FITNESS

    fwd_disp       = _forward_displacement(data, xy0)
    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    return -(fwd_disp - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)


def run_turn_episode(model, data, network, weights: np.ndarray, direction: int) -> float:
    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    rotor_geom_ids = _rotor_geom_ids(model)
    floor_id       = _floor_id(model)

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    core_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    initial_height = float(data.xpos[core_id, 2])
    r_prev         = np.array(data.xmat[core_id]).reshape(3, 3).copy()

    accumulated, step, c_hinge = 0.0, 0, 0
    current_action = np.zeros(model.nu)
    prev_rotor_contacts: set[int] = set()

    episode_end = SETTLE_DURATION + TURN_DURATION
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

        r_curr = np.array(data.xmat[core_id]).reshape(3, 3)
        delta  = signed_vertical_yaw_delta(r_prev, r_curr)
        accumulated += max(0.0, delta * direction)
        r_prev = r_curr.copy()

        curr: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor_id and g2 in rotor_geom_ids:
                curr.add(g2)
            elif g2 == floor_id and g1 in rotor_geom_ids:
                curr.add(g1)
        c_hinge += len(curr - prev_rotor_contacts)
        prev_rotor_contacts = curr
        step += 1

    if c_hinge > HINGE_CONTACT_LIMIT:
        return HINGE_GLITCH_FITNESS

    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    return -(accumulated - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)


# ── CMA-ES workers (module-level so ProcessPoolExecutor can pickle them) ─────

_loco_ctx:  Optional[dict[str, Any]] = None
_left_ctx:  Optional[dict[str, Any]] = None
_right_ctx: Optional[dict[str, Any]] = None


def _worker_init_loco(seed: int) -> None:
    global _loco_ctx
    torch.set_num_threads(1)
    np.random.seed((seed + os.getpid()) % (2**32 - 1))
    model, data = build_loco_world()
    input_dim   = len(get_robot_state(data))
    network     = Network(input_size=input_dim, output_size=model.nu)
    _loco_ctx   = {"model": model, "data": data, "network": network}


def _worker_loco(weights_list: list[float]) -> float:
    assert _loco_ctx is not None
    return run_loco_episode(_loco_ctx["model"], _loco_ctx["data"], _loco_ctx["network"],
                             np.array(weights_list, dtype=np.float32))


def _worker_init_left(seed: int) -> None:
    global _left_ctx
    torch.set_num_threads(1)
    np.random.seed((seed + os.getpid()) % (2**32 - 1))
    model, data = build_loco_world()
    input_dim   = len(get_robot_state(data))
    network     = Network(input_size=input_dim, output_size=model.nu)
    _left_ctx   = {"model": model, "data": data, "network": network}


def _worker_left(weights_list: list[float]) -> float:
    assert _left_ctx is not None
    return run_turn_episode(_left_ctx["model"], _left_ctx["data"], _left_ctx["network"],
                             np.array(weights_list, dtype=np.float32), direction=+1)


def _worker_init_right(seed: int) -> None:
    global _right_ctx
    torch.set_num_threads(1)
    np.random.seed((seed + os.getpid()) % (2**32 - 1))
    model, data = build_loco_world()
    input_dim   = len(get_robot_state(data))
    network     = Network(input_size=input_dim, output_size=model.nu)
    _right_ctx  = {"model": model, "data": data, "network": network}


def _worker_right(weights_list: list[float]) -> float:
    assert _right_ctx is not None
    return run_turn_episode(_right_ctx["model"], _right_ctx["data"], _right_ctx["network"],
                             np.array(weights_list, dtype=np.float32), direction=-1)


def train_skill(name: str, budget: int, num_params: int, worker_fn, worker_init, seed_offset: int) -> np.ndarray:
    console.rule(f"[bold cyan]Skill — {name}[/bold cyan]")
    rng = np.random.default_rng(SEED + seed_offset)
    x0  = rng.uniform(-INIT_SCALE, INIT_SCALE, size=num_params)

    es = cma.CMAEvolutionStrategy(
        x0.tolist(), SIGMA,
        {"maxiter": budget, "popsize": WORKERS, "seed": SEED + seed_offset, "verbose": -9},
    )

    best_fit = float("inf")
    best_w   = x0.copy()
    t0       = time.perf_counter()

    with ProcessPoolExecutor(max_workers=WORKERS, initializer=worker_init,
                              initargs=(SEED + seed_offset,)) as pool:
        gen = 0
        while not es.stop() and gen < budget:
            solutions = es.ask()
            fitnesses = list(pool.map(worker_fn, [s.tolist() for s in solutions]))
            es.tell(solutions, fitnesses)

            gen_best = min(fitnesses)
            if gen_best < best_fit:
                best_fit = gen_best
                best_w   = np.array(solutions[fitnesses.index(gen_best)])

            if gen % 10 == 0 or gen == 0:
                console.log(f"  {name} gen {gen:3d}/{budget}  best={best_fit:.4f}  "
                            f"elapsed={time.perf_counter() - t0:.1f}s")
            gen += 1

    console.log(f"  {name} done: best_fitness={best_fit:.4f}  time={time.perf_counter() - t0:.1f}s")
    out_path = DATA / f"{name}_weights.npy"
    np.save(out_path, best_w)
    console.log(f"  Saved -> {out_path}")
    return best_w


# ── SkillController (matches gecko_food_skills.py) ────────────────────────────


class SkillController:
    def __init__(self, rng: np.random.Generator, commit_steps: int = COMMIT_STEPS,
                 centre_fwd_thresh: float = CENTRE_FWD_THRESH) -> None:
        self.last_turn_dir     = 2 if rng.random() < 0.5 else 3
        self.commit_steps      = commit_steps
        self.centre_fwd_thresh = centre_fwd_thresh
        self._current_skill    = self.last_turn_dir
        self._steps_held       = commit_steps

    def _decide(self, left: float, centre: float, right: float) -> int:
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

    def select(self, left: float, centre: float, right: float) -> int:
        self._steps_held += 1
        if self._steps_held < self.commit_steps:
            return self._current_skill
        candidate = self._decide(left, centre, right)
        if candidate != self._current_skill:
            self._current_skill = candidate
            self._steps_held    = 0
        return self._current_skill


SKILL_LABEL  = {1: "LOCO", 2: "LEFT", 3: "RIGHT"}
SKILL_COLOUR = {1: (100, 220, 100), 2: (220, 160, 80), 3: (80, 160, 220)}
SKILL_SECTION = {1: 1, 2: 0, 3: 2}


def _make_segmented_fpv(fpv_bgr: np.ndarray, vision: list[float], skill: int) -> np.ndarray:
    out = fpv_bgr.copy().astype(np.float32)
    h, w = out.shape[:2]
    third = w // 3
    active_sec = SKILL_SECTION[skill]
    colour = tuple(float(c) / 255.0 for c in SKILL_COLOUR[skill])
    alpha = 0.35
    x0 = active_sec * third
    x1 = (active_sec + 1) * third if active_sec < 2 else w
    for c in range(3):
        out[:, x0:x1, c] = out[:, x0:x1, c] * (1.0 - alpha) + colour[c] * 255.0 * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.line(out, (third, 0), (third, h), (200, 200, 200), 1)
    cv2.line(out, (2 * third, 0), (2 * third, h), (200, 200, 200), 1)
    for i, label in enumerate([f"{v:.2f}" for v in vision]):
        cv2.putText(out, label, (i * third + 3, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, SKILL_LABEL[skill], (active_sec * third + 3, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, SKILL_COLOUR[skill], 1, cv2.LINE_AA)
    return out


def _pip_fpv(base_bgr: np.ndarray, fpv_bgr: np.ndarray, seg_bgr: np.ndarray) -> np.ndarray:
    border, margin, gap = 3, 10, 6
    bh, bw = base_bgr.shape[:2]
    fpv = cv2.resize(fpv_bgr, (FPV_W, FPV_H))
    seg = cv2.resize(seg_bgr, (FPV_W, FPV_H))
    panel_w, panel_h = FPV_W + border * 2, FPV_H + border * 2
    x0, y0 = bw - panel_w - margin, margin
    out = base_bgr.copy()

    def _paste(panel: np.ndarray, y_start: int, label: str) -> None:
        out[y_start:y_start + panel_h, x0:x0 + panel_w] = 255
        out[y_start + border:y_start + border + FPV_H,
            x0 + border:x0 + border + FPV_W] = panel
        cv2.putText(out, label, (x0 + border + 3, y_start + border + FPV_H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    _paste(fpv, y0, "FPV")
    _paste(seg, y0 + panel_h + gap, "SEG")
    return out


def _make_tracking_camera(core_pos: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = core_pos
    cam.distance  = 3.0
    cam.azimuth   = 225.0
    cam.elevation = -35.0
    return cam


def _overlay(frame_bgr: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        cv2.putText(frame_bgr, text, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


# ── Isolated-skill rendering ───────────────────────────────────────────────────


def render_skill_only(skill: str, weights: np.ndarray, out_path: Path) -> None:
    direction = {"loco": 0, "left": +1, "right": -1}[skill]
    duration  = LOCO_DURATION if skill == "loco" else TURN_DURATION

    model, data = build_loco_world()
    input_dim   = len(get_robot_state(data))
    network     = Network(input_size=input_dim, output_size=model.nu)
    fill_parameters(network, weights)

    renderer    = mujoco.Renderer(model, height=args.render_height, width=args.render_width)
    core_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    frame_every = max(1, round(1.0 / (model.opt.timestep * args.render_fps)))

    mujoco.mj_resetData(model, data)
    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    xy0       = np.array([data.qpos[0], data.qpos[1]])
    r_prev    = np.array(data.xmat[core_id]).reshape(3, 3).copy()
    yaw_total = 0.0

    step, current_action = 0, np.zeros(model.nu)
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

        r_curr = np.array(data.xmat[core_id]).reshape(3, 3)
        delta  = signed_vertical_yaw_delta(r_prev, r_curr)
        if direction == +1:
            yaw_total += max(0.0, delta)
        elif direction == -1:
            yaw_total += max(0.0, -delta)
        r_prev = r_curr.copy()

        if step % frame_every == 0:
            cam = _make_tracking_camera(data.xpos[core_id].copy())
            renderer.update_scene(data, camera=cam)
            frame_bgr = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)

            active_time = max(0.0, data.time - SETTLE_DURATION)
            if skill == "loco":
                metric_line = f"Forward-axis displacement: {_forward_displacement(data, xy0):+.3f} m"
            else:
                direction_label = "CCW (left)" if direction == +1 else "CW (right)"
                metric_line = f"Yaw accumulated ({direction_label}): {math.degrees(yaw_total):.1f} deg"

            _overlay(frame_bgr, [f"Skill: {skill}",
                                  f"t = {active_time:.1f} / {duration:.0f} s",
                                  metric_line])
            frames.append(frame_bgr)
        step += 1

    renderer.close()
    if not frames:
        console.log(f"  [red]No frames captured for {skill}[/red]")
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.render_fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()
    console.log(f"  Saved -> {out_path}")


# ── Final food evaluation + render (tracking cam + FPV picture-in-picture) ───


def render_food_episode(
    loco_w: np.ndarray, left_w: np.ndarray, right_w: np.ndarray, out_path: Path, seed: int,
) -> dict:
    model, data, target_mocap_id, cam_name = build_food_world()
    input_dim  = len(get_robot_state(data))
    output_dim = model.nu

    def _load(w: np.ndarray) -> Network:
        net = Network(input_size=input_dim, output_size=output_dim)
        fill_parameters(net, w.astype(np.float32))
        return net

    loco_net, left_net, right_net = _load(loco_w), _load(left_w), _load(right_w)

    vis_renderer = mujoco.Renderer(model, height=96, width=128)
    ov_renderer  = mujoco.Renderer(model, height=args.render_height, width=args.render_width)
    fpv_renderer = mujoco.Renderer(model, height=FPV_H, width=FPV_W)
    core_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    rotor_geom_ids = _rotor_geom_ids(model)
    floor_id     = _floor_id(model)

    rng = np.random.default_rng(seed)

    # First waypoint 1.5 m along the *measured* camera-forward axis, matching
    # gecko_food_skills.py's BodyBrainEvolution.evaluate() convention.
    first_wp = np.array([
        FORWARD_AXIS[0] * 1.5, FORWARD_AXIS[1] * 1.5, GATE_HALF_HEIGHT,
    ])
    waypoints = [first_wp] + sample_waypoints(rng, n=max(args.num_waypoints - 1, 0))
    num_wps = len(waypoints)
    console.log("Waypoints: " + "  ".join(f"({w[0]:.2f},{w[1]:.2f})" for w in waypoints))

    current_wp_idx, waypoints_reached = 0, 0
    current_target = waypoints[0]
    data.mocap_pos[target_mocap_id] = current_target
    min_dist = float("inf")

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)
        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= COLLECT_RADIUS:
            waypoints_reached += 1
            current_wp_idx += 1
            if current_wp_idx < num_wps:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist = float("inf")

    controller = SkillController(rng)
    step, current_action = 0, np.zeros(model.nu)
    frame_every = max(1, round(1.0 / (model.opt.timestep * args.render_fps)))
    skill_log: list[int] = []
    c_hinge = 0
    prev_rotor_contacts: set[int] = set()
    frames: list[np.ndarray] = []

    last_fpv_bgr = np.zeros((FPV_H, FPV_W, 3), dtype=np.uint8)
    last_seg_bgr = np.zeros((FPV_H, FPV_W, 3), dtype=np.uint8)
    last_skill   = controller.last_turn_dir

    episode_end = SETTLE_DURATION + FOOD_EVAL_DURATION
    while data.time < episode_end and current_wp_idx < num_wps:
        if step % CONTROL_STEP_FREQ == 0:
            vis_renderer.update_scene(data, camera=cam_name)
            vision = analyze_sections(isolate_green(vis_renderer.render()))
            left_frac, centre_frac, right_frac = vision
            skill = controller.select(left_frac, centre_frac, right_frac)
            skill_log.append(skill)
            last_skill = skill

            state = get_robot_state(data).astype(np.float32)
            net = {1: loco_net, 2: left_net, 3: right_net}[skill]
            raw_action = net.forward(model, data, state)
            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -math.pi / 2, math.pi / 2,
            )

            fpv_renderer.update_scene(data, camera=cam_name)
            last_fpv_bgr = cv2.cvtColor(fpv_renderer.render().copy(), cv2.COLOR_RGB2BGR)
            last_seg_bgr = _make_segmented_fpv(last_fpv_bgr, vision, skill)

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1

        curr: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor_id and g2 in rotor_geom_ids:
                curr.add(g2)
            elif g2 == floor_id and g1 in rotor_geom_ids:
                curr.add(g1)
        c_hinge += len(curr - prev_rotor_contacts)
        prev_rotor_contacts = curr

        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= COLLECT_RADIUS:
            waypoints_reached += 1
            current_wp_idx += 1
            if current_wp_idx < num_wps:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist = float("inf")

        if step % frame_every == 0:
            cam = _make_tracking_camera(data.xpos[core_id].copy())
            ov_renderer.update_scene(data, camera=cam)
            frame_bgr = cv2.cvtColor(ov_renderer.render().copy(), cv2.COLOR_RGB2BGR)
            _overlay(frame_bgr, [
                f"t={data.time:.1f}s  wp={waypoints_reached}/{num_wps}  "
                f"skill={SKILL_LABEL[last_skill]}",
                f"c_hinge={c_hinge}",
            ])
            frame_bgr = _pip_fpv(frame_bgr, last_fpv_bgr, last_seg_bgr)
            frames.append(frame_bgr)

    vis_renderer.close()
    ov_renderer.close()
    fpv_renderer.close()

    if not frames:
        console.log(f"  [red]No frames captured[/red]")
        return {}

    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.render_fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()

    total = len(skill_log)
    skill_pct = {name: 100.0 * skill_log.count(k) / max(total, 1)
                 for k, name in ((1, "loco"), (2, "left"), (3, "right"))}
    result = {
        "video": str(out_path),
        "waypoints_reached": waypoints_reached,
        "num_waypoints": num_wps,
        "c_hinge": c_hinge,
        "skill_pct": skill_pct,
    }
    console.log(f"  Saved -> {out_path}  wp={waypoints_reached}/{num_wps}  skill_pct={skill_pct}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    torch.set_num_threads(1)
    console.rule("[bold magenta]Gecko Food-In-Front Smoke Test[/bold magenta]")
    console.log(
        f"loco_budget={LOCO_BUDGET}  turn_budget={TURN_BUDGET}  sigma={SIGMA}  "
        f"workers={WORKERS}  seed={SEED}  out_dir={DATA}"
    )

    model, data = build_loco_world()
    input_dim  = len(get_robot_state(data))
    output_dim = model.nu
    network    = Network(input_size=input_dim, output_size=output_dim)
    num_params = sum(p.numel() for p in network.parameters())
    console.log(f"Network: input={input_dim}  output={output_dim}  params={num_params}")

    if args.skip_training:
        console.log("[yellow]--skip-training: loading saved weights[/yellow]")
        loco_w  = np.load(DATA / "loco_weights.npy")
        left_w  = np.load(DATA / "left_weights.npy")
        right_w = np.load(DATA / "right_weights.npy")
    else:
        loco_w  = train_skill("loco",  LOCO_BUDGET, num_params, _worker_loco,  _worker_init_loco,  seed_offset=0)
        left_w  = train_skill("left",  TURN_BUDGET, num_params, _worker_left,  _worker_init_left,  seed_offset=1)
        right_w = train_skill("right", TURN_BUDGET, num_params, _worker_right, _worker_init_right, seed_offset=2)

    console.rule("[bold cyan]Rendering isolated skills[/bold cyan]")
    render_skill_only("loco",  loco_w,  DATA / "loco.mp4")
    render_skill_only("left",  left_w,  DATA / "left.mp4")
    render_skill_only("right", right_w, DATA / "right.mp4")

    console.rule("[bold cyan]Rendering final food evaluation[/bold cyan]")
    render_food_episode(loco_w, left_w, right_w, DATA / "food_episode.mp4", seed=SEED)

    console.rule("[bold green]Done[/bold green]")
    console.log(f"Videos saved under {DATA}")


if __name__ == "__main__":
    main()
