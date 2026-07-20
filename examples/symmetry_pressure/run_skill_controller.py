"""
Deploy three trained gecko skill networks with a hand-coded state-based controller
on the food collection task. Zero additional training.

Controller logic (per control step):
  [left, centre, right] = camera green-pixel fractions
  centre > 0            → Skill 1: forward locomotion
  left   > right        → Skill 2: turn left (CCW)
  right  > left         → Skill 3: turn right (CW)
  else (nothing visible) → Skill 3: turn right (default search)

Usage:
    # Evaluate over N waypoint sets and render one episode:
    python run_skill_controller.py

    # Quick smoke-test (5 evals, shorter episodes):
    python run_skill_controller.py --eval-sets 5 --duration 60

    # Point at custom weights:
    python run_skill_controller.py --weights-dir /path/to/weights

    # Render only (no evaluation):
    python run_skill_controller.py --no-eval
"""

import argparse
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

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
)

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Gecko skill-based controller: eval + render")
parser.add_argument("--weights-dir",    type=Path,
                    default=Path("__data__/gecko_skills"),
                    help="Directory with loco_weights.npy, left_weights.npy, right_weights.npy")
parser.add_argument("--out-dir",        type=Path,
                    default=Path("__data__/gecko_skills/videos"))
parser.add_argument("--eval-sets",      type=int,   default=10,
                    help="Number of random waypoint sets to evaluate over")
parser.add_argument("--num-waypoints",  type=int,   default=10)
parser.add_argument("--collect-radius", type=float, default=0.40)
parser.add_argument("--min-wp-dist",    type=float, default=1.0,
                    help="Minimum distance between any two consecutive waypoints, and from spawn to first waypoint (m)")
parser.add_argument("--duration",       type=float, default=120.0,
                    help="Episode duration in seconds (after settling)")
parser.add_argument("--seed",           type=int,   default=0)
parser.add_argument("--no-eval",        action="store_true",
                    help="Skip evaluation; go straight to rendering")
parser.add_argument("--no-render",      action="store_true")
parser.add_argument("--render-fps",     type=int,   default=30)
parser.add_argument("--render-height",  type=int,   default=480)
parser.add_argument("--render-width",   type=int,   default=640)
parser.add_argument("--render-set",     type=int,   default=0,
                    help="Which waypoint set index to render (0 = first eval set)")
parser.add_argument("--commit-steps",   type=int,   default=50,
                    help="Control steps each skill decision is held for before re-querying vision")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)

# ── Episode constants (must match gecko_skills.py) ────────────────────────────

SETTLE_DURATION   = 3.0
CTRL_ALPHA        = 0.5
CONTROL_STEP_FREQ = 50

REACH_RADIUS   = 0.20   # visual marker cylinder radius
COLLECT_RADIUS = args.collect_radius
MIN_WP_DIST       = args.min_wp_dist
CENTRE_FWD_THRESH = 0.6   # centre fraction must exceed this to trigger forward loco

# ── PIP layout constants (mirrors render_curriculum.py) ───────────────────────

FPV_H      = 180
FPV_W      = 240
PIP_BORDER = 3
PIP_MARGIN = 10
PIP_GAP    = 6   # vertical gap between the two FPV panels

# ── World builder ─────────────────────────────────────────────────────────────


def build_food_world():
    core  = gecko()
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                correct_collision_with_floor=True)

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


# ── Waypoint sampling with minimum spawn distance ─────────────────────────────


def sample_waypoints_min_dist(
    rng: np.random.Generator,
    n: int,
    min_dist: float = MIN_WP_DIST,
    max_attempts: int = 200,
) -> list[np.ndarray]:
    """Like sample_waypoints but enforces min_dist between every consecutive
    pair of waypoints and between spawn and the first waypoint."""
    spawn_xy = np.array([SPAWN_POSITION[0], SPAWN_POSITION[1]])
    for _ in range(max_attempts):
        waypoints = sample_waypoints(rng, n)
        if np.linalg.norm(waypoints[0][:2] - spawn_xy) < min_dist:
            continue
        if all(
            np.linalg.norm(waypoints[i][:2] - waypoints[i - 1][:2]) >= min_dist
            for i in range(1, len(waypoints))
        ):
            return waypoints
    # fallback: keep last set
    return waypoints


# ── Controller logic ──────────────────────────────────────────────────────────


class SkillController:
    """Stateful controller that remembers which side food was last seen on
    and commits to each skill for a minimum number of control steps.

    Decision rules (in priority order):
      1. centre >= CENTRE_FWD_THRESH  OR  centre is the largest segment → LOCO
      2. food visible on left or right → turn toward that side,
         and update last_turn_dir
      3. nothing visible → turn in last_turn_dir (memory-based search)

    Once a skill is chosen it runs for at least commit_steps control steps.
    Vision can only trigger a *different* skill after that minimum has elapsed.
    Staying on the same skill resets the counter (so it keeps running until
    vision says something different for long enough to matter).
    """

    def __init__(self, rng: np.random.Generator, commit_steps: int = 5) -> None:
        self.last_turn_dir:  int = 2 if rng.random() < 0.5 else 3
        self.commit_steps:   int = commit_steps
        self._current_skill: int = self.last_turn_dir
        self._steps_held:    int = commit_steps   # treat start as already committed

    def _decide(self, left: float, centre: float, right: float) -> int:
        """Pure skill selection from vision, with memory update side-effect."""
        any_visible = (left + centre + right) > 0.0
        if any_visible:
            if centre >= CENTRE_FWD_THRESH or (centre >= left and centre >= right):
                return 1
            elif left >= right:
                self.last_turn_dir = 2
                return 2
            else:
                self.last_turn_dir = 3
                return 3
        return self.last_turn_dir

    def select(self, left: float, centre: float, right: float) -> int:
        """Call every control step. The current skill is locked until it has
        run for commit_steps steps; only then can vision switch to a new one."""
        self._steps_held += 1

        if self._steps_held < self.commit_steps:
            # Still within the minimum commitment window — hold.
            return self._current_skill

        # Minimum met — ask vision what it wants.
        candidate = self._decide(left, centre, right)

        if candidate != self._current_skill:
            # Switching: reset the counter so the new skill gets its full window.
            self._current_skill = candidate
            self._steps_held    = 0

        return self._current_skill


SKILL_LABEL  = {1: "LOCO", 2: "LEFT", 3: "RIGHT"}
# BGR colours for overlay text / section highlight
SKILL_COLOUR = {1: (100, 220, 100), 2: (220, 160, 80), 3: (80, 160, 220)}
# Which of the three camera sections each skill "looks at"
SKILL_SECTION = {1: 1, 2: 0, 3: 2}   # 0=left, 1=centre, 2=right


# ── FPV panel builders ────────────────────────────────────────────────────────


def _make_segmented_fpv(
    fpv_bgr: np.ndarray,
    vision: list[float],
    skill: int,
) -> np.ndarray:
    """Return a copy of fpv_bgr with:
      - two vertical dividers splitting it into thirds
      - a semi-transparent colour highlight on the active section
      - a label in the active section corner
    """
    out = fpv_bgr.copy().astype(np.float32)
    h, w = out.shape[:2]
    third = w // 3

    active_sec   = SKILL_SECTION[skill]
    colour_float = tuple(float(c) / 255.0 for c in SKILL_COLOUR[skill])  # BGR 0-1

    # Semi-transparent tint on the active section
    alpha = 0.35
    x0 = active_sec * third
    x1 = (active_sec + 1) * third if active_sec < 2 else w
    for c in range(3):
        out[:, x0:x1, c] = (
            out[:, x0:x1, c] * (1.0 - alpha) + colour_float[c] * 255.0 * alpha
        )

    out = np.clip(out, 0, 255).astype(np.uint8)

    # Divider lines
    cv2.line(out, (third,     0), (third,     h), (200, 200, 200), 1)
    cv2.line(out, (2 * third, 0), (2 * third, h), (200, 200, 200), 1)

    # Section fraction labels along the top
    section_labels = [f"{vision[0]:.2f}", f"{vision[1]:.2f}", f"{vision[2]:.2f}"]
    for i, label in enumerate(section_labels):
        lx = i * third + 3
        cv2.putText(out, label, (lx, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Active skill label bottom-left of active section
    cv2.putText(out, SKILL_LABEL[skill],
                (active_sec * third + 3, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                SKILL_COLOUR[skill], 1, cv2.LINE_AA)

    return out


def _pip_double(
    base_bgr: np.ndarray,
    fpv_bgr: np.ndarray,
    seg_bgr: np.ndarray,
) -> np.ndarray:
    """Composite two FPV panels stacked vertically in the top-right corner.

    Top panel:    raw FPV  (labelled "FPV")
    Bottom panel: segmented FPV (labelled "SEG")
    Both panels are resized to FPV_W × FPV_H if needed.
    """
    bh, bw = base_bgr.shape[:2]
    fpv = cv2.resize(fpv_bgr, (FPV_W, FPV_H))
    seg = cv2.resize(seg_bgr, (FPV_W, FPV_H))

    panel_w = FPV_W + PIP_BORDER * 2
    panel_h = FPV_H + PIP_BORDER * 2

    x0 = bw - panel_w - PIP_MARGIN
    y0 = PIP_MARGIN

    out = base_bgr.copy()

    def _paste(panel: np.ndarray, y_start: int, label: str) -> None:
        # white border
        out[y_start : y_start + panel_h, x0 : x0 + panel_w] = 255
        # image
        out[y_start + PIP_BORDER : y_start + PIP_BORDER + FPV_H,
            x0 + PIP_BORDER       : x0 + PIP_BORDER + FPV_W] = panel
        # small label bottom-left inside the border
        cv2.putText(out, label,
                    (x0 + PIP_BORDER + 3, y_start + PIP_BORDER + FPV_H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    _paste(fpv, y0,                         "FPV")
    _paste(seg, y0 + panel_h + PIP_GAP,     "SEG")

    return out


# ── Episode runner ────────────────────────────────────────────────────────────


def run_controller_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    loco_net: Network,
    left_net: Network,
    right_net: Network,
    vis_renderer: mujoco.Renderer,
    cam_name: str,
    target_mocap_id: int,
    waypoints: list[np.ndarray],
    rng: np.random.Generator,
    record: bool = False,
    ov_renderer: mujoco.Renderer | None = None,
    fpv_renderer: mujoco.Renderer | None = None,
    render_fps: int = 30,
) -> dict:
    mujoco.mj_resetData(model, data)

    num_wps           = len(waypoints)
    current_wp_idx    = 0
    waypoints_reached = 0
    current_target    = waypoints[0]
    data.mocap_pos[target_mocap_id] = current_target
    min_dist = float("inf")

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)
        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= COLLECT_RADIUS:
            waypoints_reached += 1
            current_wp_idx    += 1
            if current_wp_idx < num_wps:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist = float("inf")

    core_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    frame_every = max(1, round(1.0 / (model.opt.timestep * render_fps))) if record else 1
    controller  = SkillController(rng, commit_steps=args.commit_steps)

    step           = 0
    current_action = np.zeros(model.nu)
    frames: list[np.ndarray] = []
    skill_log: list[int] = []

    # Carry last FPV renders into the render branch (updated each control step)
    last_fpv_bgr = np.zeros((FPV_H, FPV_W, 3), dtype=np.uint8)
    last_seg_bgr = np.zeros((FPV_H, FPV_W, 3), dtype=np.uint8)
    last_vision  = [0.0, 0.0, 0.0]
    last_skill   = controller.last_turn_dir

    episode_end = SETTLE_DURATION + args.duration
    while data.time < episode_end and current_wp_idx < num_wps:

        if step % CONTROL_STEP_FREQ == 0:
            # ── vision for controller (small, matches training) ────────────────
            vis_renderer.update_scene(data, camera=cam_name)
            img    = vis_renderer.render()
            vision = analyze_sections(isolate_green(img))
            left_frac, centre_frac, right_frac = vision
            last_vision = list(vision)

            skill = controller.select(left_frac, centre_frac, right_frac)
            skill_log.append(skill)
            last_skill = skill

            state = get_robot_state(data).astype(np.float32)
            if skill == 1:
                raw_action = loco_net.forward(model, data, state)
            elif skill == 2:
                raw_action = left_net.forward(model, data, state)
            else:
                raw_action = right_net.forward(model, data, state)

            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -math.pi / 2, math.pi / 2,
            )

            # ── FPV frames for PIP (only built when we're recording) ───────────
            if record and fpv_renderer is not None:
                fpv_renderer.update_scene(data, camera=cam_name)
                fpv_rgb      = fpv_renderer.render().copy()
                last_fpv_bgr = cv2.cvtColor(fpv_rgb, cv2.COLOR_RGB2BGR)
                last_seg_bgr = _make_segmented_fpv(last_fpv_bgr, last_vision, skill)

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= COLLECT_RADIUS:
            waypoints_reached += 1
            current_wp_idx    += 1
            if current_wp_idx < num_wps:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist = float("inf")

        if record and ov_renderer is not None and step % frame_every == 0:
            core_pos = data.xpos[core_id].copy()
            cam = mujoco.MjvCamera()
            cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = core_pos
            cam.distance  = 2.5
            cam.azimuth   = 225.0
            cam.elevation = -35.0
            ov_renderer.update_scene(data, camera=cam)
            frame     = ov_renderer.render().copy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            active_t = max(0.0, data.time - SETTLE_DURATION)
            colour   = SKILL_COLOUR[last_skill]
            lines    = [
                f"Skill: {SKILL_LABEL[last_skill]}   t={active_t:.1f}/{args.duration:.0f}s",
                f"Waypoints: {waypoints_reached}/{num_wps}",
            ]
            for i, text in enumerate(lines):
                cv2.putText(frame_bgr, text, (10, 28 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 1, cv2.LINE_AA)

            frame_bgr = _pip_double(frame_bgr, last_fpv_bgr, last_seg_bgr)
            frames.append(frame_bgr)

        step += 1

    final_dist = 0.0 if waypoints_reached >= num_wps else min_dist
    d_norm = float(np.clip((RING_R_MAX - final_dist) / RING_R_MAX, 0.0, 1.0))
    score  = waypoints_reached + d_norm

    total_ctrl_steps = len(skill_log)
    skill_pct = {
        s: 100.0 * skill_log.count(s) / max(total_ctrl_steps, 1)
        for s in (1, 2, 3)
    }

    return {
        "waypoints_reached": waypoints_reached,
        "score":             score,
        "skill_pct":         skill_pct,
        "frames":            frames,
    }


# ── Load weights ──────────────────────────────────────────────────────────────


def load_networks(input_dim: int, output_dim: int):
    nets = {}
    for name in ("loco", "left", "right"):
        path = args.weights_dir / f"{name}_weights.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Weights not found: {path}\n"
                f"Run gecko_skills.py first, or pass --weights-dir."
            )
        weights = np.load(path).astype(np.float32)
        net = Network(input_size=input_dim, output_size=output_dim)
        fill_parameters(net, weights)
        nets[name] = net
        console.log(f"  Loaded {name} weights from {path}")
    return nets["loco"], nets["left"], nets["right"]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    torch.set_num_threads(1)
    console.rule("[bold magenta]Gecko Skill Controller[/bold magenta]")
    console.log(
        f"eval_sets={args.eval_sets}  waypoints={args.num_waypoints}  "
        f"collect_radius={args.collect_radius}  min_wp_dist={MIN_WP_DIST}m  "
        f"centre_fwd_thresh={CENTRE_FWD_THRESH}  commit_steps={args.commit_steps}  "
        f"duration={args.duration}s  seed={args.seed}"
    )

    model, data, target_mocap_id, cam_name = build_food_world()
    input_dim  = len(get_robot_state(data))
    output_dim = model.nu
    console.log(f"Network: input={input_dim}  output={output_dim}")

    if not cam_name:
        console.log("[red]Warning: no body-mounted camera found — vision will be empty[/red]")

    loco_net, left_net, right_net = load_networks(input_dim, output_dim)

    vis_renderer = mujoco.Renderer(model, height=96, width=128)

    # ── Evaluation ────────────────────────────────────────────────────────────
    scores: list[float] = []
    waypoints_per_set: list[int] = []

    if not args.no_eval:
        console.rule("[bold cyan]Evaluation[/bold cyan]")

        for i in range(args.eval_sets):
            rng       = np.random.default_rng(args.seed + i)
            waypoints = sample_waypoints_min_dist(rng, args.num_waypoints)

            result = run_controller_episode(
                model, data, loco_net, left_net, right_net,
                vis_renderer, cam_name, target_mocap_id, waypoints,
                rng=np.random.default_rng(args.seed + i + 1000),
            )
            scores.append(result["score"])
            waypoints_per_set.append(result["waypoints_reached"])
            sp = result["skill_pct"]
            console.log(
                f"  Set {i:2d}: {result['waypoints_reached']:2d}/{args.num_waypoints} wps  "
                f"score={result['score']:.3f}  "
                f"LOCO={sp[1]:.0f}%  LEFT={sp[2]:.0f}%  RIGHT={sp[3]:.0f}%"
            )

        table = Table(title="Evaluation Summary")
        table.add_column("Metric",  style="cyan")
        table.add_column("Value",   style="green")
        table.add_row("Mean score",       f"{np.mean(scores):.3f}")
        table.add_row("Std score",        f"{np.std(scores):.3f}")
        table.add_row("Mean wps reached", f"{np.mean(waypoints_per_set):.2f} / {args.num_waypoints}")
        table.add_row("Best wps reached", str(max(waypoints_per_set)))
        console.print(table)

    # ── Render one episode ────────────────────────────────────────────────────
    if not args.no_render:
        console.rule("[bold cyan]Rendering[/bold cyan]")

        rng           = np.random.default_rng(args.seed + args.render_set)
        render_wps    = sample_waypoints_min_dist(rng, args.num_waypoints)
        ov_renderer   = mujoco.Renderer(model, height=args.render_height, width=args.render_width)
        fpv_renderer  = mujoco.Renderer(model, height=FPV_H, width=FPV_W)

        result = run_controller_episode(
            model, data, loco_net, left_net, right_net,
            vis_renderer, cam_name, target_mocap_id, render_wps,
            rng=np.random.default_rng(args.seed + args.render_set + 1000),
            record=True,
            ov_renderer=ov_renderer,
            fpv_renderer=fpv_renderer,
            render_fps=args.render_fps,
        )
        ov_renderer.close()
        fpv_renderer.close()

        console.log(
            f"  Rendered episode: {result['waypoints_reached']}/{args.num_waypoints} wps  "
            f"score={result['score']:.3f}"
        )

        frames = result["frames"]
        if frames:
            out_path = args.out_dir / "skill_controller.mp4"
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                args.render_fps, (w, h),
            )
            for fr in frames:
                writer.write(fr)
            writer.release()
            console.log(f"  Saved → {out_path}")
        else:
            console.log("  [red]No frames captured[/red]")

    vis_renderer.close()
    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
