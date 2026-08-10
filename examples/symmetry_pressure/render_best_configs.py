"""
Re-run CMA-ES with the best HPO configs and render a video of each.

Reads results.jsonl files from one or more tuning data directories, picks the
top-N configs per task by mean fitness, re-runs CMA-ES with those exact
hyperparameters (recovering the best weight vector), then renders an MP4.

Usage (smoke test, 2 configs per task, fast):
    python render_best_configs.py \\
        --data-dirs ../../data/tuning_data/tune_cma_ann_loco_20260717_152354 \\
                    ../../data/tuning_data/tune_cma_ann_food_20260718_102521 \\
        --top-n 1 --budget-override 50 --dur 10

Usage (full, from SLURM):
    python render_best_configs.py \\
        --data-dirs ../../data/tuning_data/tune_cma_ann_loco_20260717_152354 \\
                    ../../data/tuning_data/tune_cma_ann_loco_20260718_102515 \\
                    ../../data/tuning_data/tune_cma_ann_food_20260718_102521 \\
                    ../../data/tuning_data/tune_cma_ann_food_20260717_152400 \\
                    ../../data/tuning_data/tune_cma_ann_food_20260717_152418 \\
        --top-n 3 --out-dir __data__/best_config_videos --workers 64
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cma
import cv2
import mujoco
import numpy as np
import torch
from torch import nn

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

import sys
sys.path.insert(0, str(Path(__file__).parent))
from shared import (
    SPAWN_POSITION,
    analyze_sections,
    fill_parameters,
    isolate_green,
    sample_waypoints,
)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Re-run best HPO configs and render videos")
parser.add_argument("--data-dirs",      nargs="+", type=Path, required=True,
                    help="One or more tuning output directories containing results.jsonl")
parser.add_argument("--top-n",          type=int, default=3,
                    help="Number of top configs to render per task")
parser.add_argument("--out-dir",        type=Path, default=Path("__data__/best_config_videos"))
parser.add_argument("--workers",        type=int, default=64)
parser.add_argument("--reruns",         type=int, default=3,
                    help="Re-run CMA-ES this many times per config; keep the best")
parser.add_argument("--budget-override",type=int, default=None,
                    help="Override the HPO budget (useful for smoke tests)")
parser.add_argument("--dur",            type=float, default=None,
                    help="Override episode duration in seconds")
# food task defaults (must match tuning run)
parser.add_argument("--num-waypoints",  type=int,   default=10)
parser.add_argument("--reach-radius",   type=float, default=0.20)
parser.add_argument("--arena-radius",   type=float, default=2.0)
# video
parser.add_argument("--render-fps",    type=int, default=30)
parser.add_argument("--render-height", type=int, default=480)
parser.add_argument("--render-width",  type=int, default=640)
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)

# ── Episode constants (must match tune_cma_ann.py) ────────────────────────────

SETTLE_DURATION          = 3.0
CTRL_ALPHA               = 0.5
CONTROL_STEP_FREQ        = 100
HINGE_CONTACT_LIMIT      = 200
HINGE_CONTACT_PENALTY    = 0.005
HINGE_GLITCH_FITNESS     = 1.0
HEIGHT_PENALTY_THRESHOLD = 0.21
RING_R_MAX               = 1.0
GATE_HALF_HEIGHT         = 0.15

FOOD_DUR_DEFAULT = 60.0
LOCO_DUR_DEFAULT = 60.0


# ── FlexNetwork (must match tune_cma_ann.py) ──────────────────────────────────


class FlexNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, state: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(state, dtype=torch.float32)
        return (torch.tanh(self.net(x)) * (torch.pi / 2)).numpy()


# ── World builders ────────────────────────────────────────────────────────────


def _build_gecko_food(reach_radius: float, arena_radius: float):
    core  = gecko()
    world = SimpleFlatWorld()
    try:
        world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                    correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                    correct_collision_with_floor=False)

    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=[0.0, 0.0, GATE_HALF_HEIGHT]
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[reach_radius, GATE_HALF_HEIGHT],
        rgba=[0, 1, 0, 0.7],
        contype=0, conaffinity=0,
    )
    world.spec.worldbody.add_camera(
        name="overview_cam",
        pos=[0.0, 0.0, arena_radius * 3.5],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    target_mocap_id = model.body("green_target").mocapid[0]
    cam_name: str | None = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "overview" not in name:
            cam_name = name
            break
    return model, data, target_mocap_id, cam_name


def _build_gecko_loco():
    core  = gecko()
    world = SimpleFlatWorld()
    try:
        world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                    correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                    correct_collision_with_floor=False)
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    return model, data


def _rotor_geom_ids(model: mujoco.MjModel) -> set[int]:
    return {i for i in range(model.ngeom) if model.geom(i).name.endswith("-rotor")}


def _floor_geom_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


# ── Candidate evaluator (runs in subprocess) ──────────────────────────────────


def _evaluate_candidate(
    weights: np.ndarray,
    task: str,
    hidden_sizes: list[int],
    num_waypoints: int,
    reach_radius: float,
    arena_radius: float,
    dur: float,
    seed: int,
) -> float:
    torch.set_num_threads(1)
    np.random.seed(seed)

    if task == "food":
        model, data, target_mocap_id, cam_name = _build_gecko_food(reach_radius, arena_radius)
        renderer  = mujoco.Renderer(model, height=96, width=128)
        rng       = np.random.default_rng(seed)
        waypoints = sample_waypoints(rng, num_waypoints)
        input_dim = len(get_robot_state(data)) + 3 + 1
    else:
        model, data = _build_gecko_loco()
        renderer, target_mocap_id, cam_name, waypoints = None, -1, None, []
        input_dim = len(get_robot_state(data))

    network        = FlexNetwork(input_size=input_dim, output_size=model.nu, hidden_sizes=hidden_sizes)
    rotor_ids      = _rotor_geom_ids(model)
    floor_id       = _floor_geom_id(model)
    fill_parameters(network, weights)

    if task == "food":
        assert renderer is not None
        fit = _run_food_episode(model, data, network, renderer, cam_name,
                                target_mocap_id, waypoints, reach_radius, dur,
                                rotor_ids, floor_id)
        renderer.close()
    else:
        fit = _run_loco_episode(model, data, network, dur, rotor_ids, floor_id)
    return fit


def _run_food_episode(
    model, data, network, renderer, cam_name, target_mocap_id,
    waypoints, reach_radius, dur, rotor_ids, floor_id,
) -> float:
    mujoco.mj_resetData(model, data)
    num_wps        = len(waypoints)
    current_wp_idx = 0
    waypoints_reached = 0
    current_target = waypoints[0]
    data.mocap_pos[target_mocap_id] = current_target
    min_dist       = float("inf")

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    core_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    initial_height = float(data.xpos[core_id, 2])
    step           = 0
    current_action = np.zeros(model.nu)
    c_hinge        = 0
    prev_rotors: set[int] = set()

    while data.time < SETTLE_DURATION + dur and current_wp_idx < num_wps:
        if step % CONTROL_STEP_FREQ == 0:
            renderer.update_scene(data, camera=cam_name)
            img    = renderer.render()
            vision = analyze_sections(isolate_green(img))
            state  = np.concatenate([
                get_robot_state(data),
                vision,
                [current_wp_idx / max(num_wps - 1, 1)],
            ]).astype(np.float32)
            raw    = network.forward(state)
            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw * CTRL_ALPHA,
                -np.pi / 2, np.pi / 2,
            )
        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1

        curr_rotors: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor_id and g2 in rotor_ids:
                curr_rotors.add(g2)
            elif g2 == floor_id and g1 in rotor_ids:
                curr_rotors.add(g1)
        c_hinge += len(curr_rotors - prev_rotors)
        prev_rotors = curr_rotors

        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= reach_radius:
            waypoints_reached += 1
            current_wp_idx    += 1
            if current_wp_idx < num_wps:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist = float("inf")

    if c_hinge > HINGE_CONTACT_LIMIT:
        return HINGE_GLITCH_FITNESS
    final_dist    = 0.0 if waypoints_reached >= num_wps else min_dist
    d_norm        = float(np.clip((RING_R_MAX - final_dist) / RING_R_MAX, 0.0, 1.0))
    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    return -(waypoints_reached + d_norm - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)


def _run_loco_episode(model, data, network, dur, rotor_ids, floor_id) -> float:
    mujoco.mj_resetData(model, data)
    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    core_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    x0             = float(data.qpos[0])
    initial_height = float(data.xpos[core_id, 2])
    step           = 0
    current_action = np.zeros(model.nu)
    c_hinge        = 0
    prev_rotors: set[int] = set()

    while data.time < SETTLE_DURATION + dur:
        if step % CONTROL_STEP_FREQ == 0:
            state  = get_robot_state(data).astype(np.float32)
            raw    = network.forward(state)
            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw * CTRL_ALPHA,
                -np.pi / 2, np.pi / 2,
            )
        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1

        curr_rotors: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor_id and g2 in rotor_ids:
                curr_rotors.add(g2)
            elif g2 == floor_id and g1 in rotor_ids:
                curr_rotors.add(g1)
        c_hinge += len(curr_rotors - prev_rotors)
        prev_rotors = curr_rotors

    if c_hinge > HINGE_CONTACT_LIMIT:
        return HINGE_GLITCH_FITNESS
    x_displacement = float(data.qpos[0]) - x0
    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    return -(x_displacement - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)


# ── CMA-ES re-run ─────────────────────────────────────────────────────────────


def rerun_cma(
    cfg: dict[str, Any],
    run_seed: int,
    workers: int,
    budget_override: int | None,
    dur_override: float | None,
) -> tuple[float, np.ndarray]:
    """Re-run CMA-ES for one config; return (best_fitness, best_weights)."""
    torch.set_num_threads(1)

    task         = cfg["task"]
    hidden_sizes = cfg["hidden_sizes"]
    sigma        = cfg["sigma"]
    pop          = cfg["pop"]
    budget       = budget_override if budget_override is not None else cfg["budget"]
    init_scale   = cfg["init_scale"]
    num_wps      = cfg.get("num_waypoints", args.num_waypoints)
    reach_r      = cfg.get("reach_radius",  args.reach_radius)
    arena_r      = cfg.get("arena_radius",  args.arena_radius)
    dur          = dur_override if dur_override is not None else (
        FOOD_DUR_DEFAULT if task == "food" else LOCO_DUR_DEFAULT
    )

    if task == "food":
        model, data, _, _ = _build_gecko_food(reach_r, arena_r)
        input_dim = len(get_robot_state(data)) + 3 + 1
    else:
        model, data = _build_gecko_loco()
        input_dim = len(get_robot_state(data))

    network    = FlexNetwork(input_size=input_dim, output_size=model.nu, hidden_sizes=hidden_sizes)
    num_params = sum(p.numel() for p in network.parameters())

    rng           = np.random.default_rng(run_seed)
    initial_guess = rng.uniform(-init_scale, init_scale, size=num_params).tolist()
    es = cma.CMAEvolutionStrategy(
        initial_guess,
        sigma,
        {"popsize": pop, "seed": run_seed, "verbose": -9, "maxiter": 10**9},
    )

    best_fit    = float("inf")
    best_weights = np.zeros(num_params)
    evals       = 0

    with ProcessPoolExecutor(max_workers=min(pop, workers)) as pool:
        while evals < budget and not es.stop():
            candidates = es.ask()
            futures = [
                pool.submit(
                    _evaluate_candidate,
                    np.array(w, dtype=np.float32),
                    task, hidden_sizes, num_wps, reach_r, arena_r, dur,
                    run_seed + evals + i,
                )
                for i, w in enumerate(candidates)
            ]
            fits = [f.result() for f in futures]
            es.tell(candidates, fits)
            for i, fit in enumerate(fits):
                evals += 1
                if fit < best_fit:
                    best_fit     = fit
                    best_weights = np.array(candidates[i], dtype=np.float32)

    return best_fit, best_weights


# ── Video renderer ────────────────────────────────────────────────────────────


def render_video(
    cfg: dict[str, Any],
    weights: np.ndarray,
    fitness: float,
    out_path: Path,
    dur_override: float | None,
) -> None:
    task         = cfg["task"]
    hidden_sizes = cfg["hidden_sizes"]
    num_wps      = cfg.get("num_waypoints", args.num_waypoints)
    reach_r      = cfg.get("reach_radius",  args.reach_radius)
    arena_r      = cfg.get("arena_radius",  args.arena_radius)
    dur          = dur_override if dur_override is not None else (
        FOOD_DUR_DEFAULT if task == "food" else LOCO_DUR_DEFAULT
    )

    if task == "food":
        model, data, target_mocap_id, cam_name = _build_gecko_food(reach_r, arena_r)
        vis_renderer = mujoco.Renderer(model, height=96, width=128)
        rng          = np.random.default_rng(0)
        waypoints    = sample_waypoints(rng, num_wps)
        input_dim    = len(get_robot_state(data)) + 3 + 1
    else:
        model, data  = _build_gecko_loco()
        vis_renderer, target_mocap_id, cam_name, waypoints = None, -1, None, []
        input_dim    = len(get_robot_state(data))

    network = FlexNetwork(input_size=input_dim, output_size=model.nu, hidden_sizes=hidden_sizes)
    fill_parameters(network, weights)

    ov_renderer = mujoco.Renderer(model, height=args.render_height, width=args.render_width)
    core_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    CAM_OFFSET  = np.array([-1.5, -1.5, 1.5])

    mujoco.mj_resetData(model, data)
    if task == "food" and waypoints and target_mocap_id >= 0:
        data.mocap_pos[target_mocap_id] = waypoints[0]

    frame_every    = max(1, round(1.0 / (model.opt.timestep * args.render_fps)))
    current_wp_idx = 0
    current_action = np.zeros(model.nu)
    frames: list[np.ndarray] = []
    step   = 0

    episode_end = SETTLE_DURATION + dur

    while data.time < episode_end:
        if task == "food" and waypoints and current_wp_idx >= len(waypoints):
            break

        if step % CONTROL_STEP_FREQ == 0:
            robot_state = get_robot_state(data).astype(np.float32)
            if task == "food" and vis_renderer and cam_name:
                vis_renderer.update_scene(data, camera=cam_name)
                img    = vis_renderer.render()
                vision = np.array(analyze_sections(isolate_green(img)), dtype=np.float32)
                prog   = np.array([current_wp_idx / max((len(waypoints) or 1) - 1, 1)],
                                  dtype=np.float32)
                state  = np.concatenate([robot_state, vision, prog])
            else:
                state = robot_state
            raw = network.forward(state)
            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw * CTRL_ALPHA,
                -np.pi / 2, np.pi / 2,
            )

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

        if task == "food" and waypoints and target_mocap_id >= 0:
            dist = float(np.linalg.norm(np.array(data.qpos[:2]) - waypoints[current_wp_idx][:2]))
            if dist <= reach_r:
                current_wp_idx += 1
                if current_wp_idx < len(waypoints):
                    data.mocap_pos[target_mocap_id] = waypoints[current_wp_idx]

        if step % frame_every == 0:
            core_pos = data.xpos[core_id].copy()
            cam = mujoco.MjvCamera()
            cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = core_pos
            cam.distance  = float(np.linalg.norm(CAM_OFFSET))
            cam.azimuth   = 225.0
            cam.elevation = -35.0
            ov_renderer.update_scene(data, camera=cam)
            frame     = ov_renderer.render().copy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            label = (f"task={task}  layers={cfg['n_layers']}  "
                     f"hidden={cfg['hidden_sizes']}  "
                     f"σ={cfg['sigma']:.3f}  pop={cfg['pop']}  budget={cfg['budget']}")
            cv2.putText(frame_bgr, label,
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame_bgr, f"fit={fitness:.3f}  t={data.time:.1f}s",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            if task == "food" and waypoints:
                cv2.putText(frame_bgr,
                            f"waypoints: {current_wp_idx}/{len(waypoints)}",
                            (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
            frames.append(frame_bgr)

        step += 1

    ov_renderer.close()
    if vis_renderer:
        vis_renderer.close()

    if not frames:
        print(f"  No frames captured for {out_path.name}")
        return

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.render_fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()
    print(f"  Saved → {out_path}")


# ── Load and select configs ───────────────────────────────────────────────────


def load_configs(data_dirs: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Load all results.jsonl files; return dict[task -> sorted list of records]."""
    all_records: list[dict[str, Any]] = []
    for d in data_dirs:
        jsonl = d / "results.jsonl"
        cfg_f = d / "run_config.json"
        if not jsonl.exists():
            print(f"  Skipping {d} — no results.jsonl")
            continue
        run_cfg = json.loads(cfg_f.read_text()) if cfg_f.exists() else {}
        with jsonl.open() as f:
            for line in f:
                rec = json.loads(line)
                # Inject food task geometry from run_config if not already present
                for key in ("num_waypoints", "reach_radius", "arena_radius"):
                    if key not in rec and key in run_cfg:
                        rec[key] = run_cfg[key]
                all_records.append(rec)

    by_task: dict[str, list[dict]] = {}
    for rec in all_records:
        by_task.setdefault(rec["task"], []).append(rec)

    # Sort each task's records by mean_fitness ascending (most negative = best performance)
    for task in by_task:
        by_task[task].sort(key=lambda r: r["mean_fitness"])

    return by_task


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    torch.set_num_threads(1)

    by_task = load_configs(args.data_dirs)
    if not by_task:
        print("No results found — check --data-dirs")
        return

    manifest: list[dict] = []

    for task, records in by_task.items():
        top = records[: args.top_n]
        print(f"\n=== {task.upper()} — top {len(top)} configs ===")
        for rank, cfg in enumerate(top, 1):
            print(f"  #{rank}  mean_fitness={cfg['mean_fitness']:.4f}  "
                  f"layers={cfg['n_layers']}  hidden={cfg['hidden_sizes']}  "
                  f"σ={cfg['sigma']:.3f}  pop={cfg['pop']}  budget={cfg['budget']}  "
                  f"init_scale={cfg['init_scale']:.3f}  mean_time={cfg['mean_time_s']:.0f}s")

        for rank, cfg in enumerate(top, 1):
            print(f"\n  Re-running #{rank} ({task}, {args.reruns} seeds)…")
            best_fit    = float("inf")
            best_weights = None

            for rerun_idx in range(args.reruns):
                seed = 1000 + rank * 100 + rerun_idx
                t0   = time.perf_counter()
                fit, w = rerun_cma(
                    cfg, seed, args.workers,
                    args.budget_override, args.dur,
                )
                elapsed = time.perf_counter() - t0
                print(f"    seed={seed}  fit={fit:.4f}  ({elapsed:.0f}s)")
                if fit < best_fit:
                    best_fit     = fit
                    best_weights = w

            assert best_weights is not None
            label    = (f"{task}_rank{rank:02d}"
                        f"_fit{abs(best_fit):.3f}"
                        f"_L{cfg['n_layers']}"
                        f"_h{'x'.join(str(h) for h in cfg['hidden_sizes'])}"
                        f"_s{cfg['sigma']:.2f}"
                        f"_p{cfg['pop']}"
                        f"_b{cfg['budget']}")
            vid_path = args.out_dir / f"{label}.mp4"

            print(f"  Rendering video → {vid_path.name}")
            render_video(cfg, best_weights, best_fit, vid_path, args.dur)

            manifest.append({
                "task":         task,
                "rank":         rank,
                "hpo_fitness":  cfg["mean_fitness"],
                "rerun_fitness": float(best_fit),
                "config":        cfg,
                "video":         str(vid_path),
            })

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
