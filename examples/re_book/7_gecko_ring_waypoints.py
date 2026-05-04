"""
Gecko brain evolution on the new ring-waypoints task — single fixed body.

Same shape as 5_randomized_waypoints.py (CMA-ES on a fixed gecko body), but
running against the *new* environment defined in 6_body_brain_randomized_waypoints.py:
  • Sequential ring sampler (each waypoint 0.5–1.0 m from the previous).
  • Transparent green cylinder gate, sized exactly to --reach-radius.
  • Pass-through (contype=0/conaffinity=0): the robot walks through the gate.
  • Fitness = -waypoints_reached + (d_min/RING_R_MAX - 1), no clamp.

After training, records two synchronised videos for the best-seen individual:
  • overview MP4 — top-down camera over the arena.
  • POV MP4 (stacked) — top half is the raw 96×128 RGB camera feed (red section
    dividers); bottom half is the post-isolate_green binary mask (magenta lines).

Usage:
    uv run python examples/re_book/7_gecko_ring_waypoints.py \\
        --budget 80 --population 40 --workers 8 --dur 45 --seed 42
"""
# Standard library
import argparse
import multiprocessing as mp
import os
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional, cast

# Third-party
import cv2
import mujoco
import nevergrad as ng
import numpy as np
import torch
from rich.console import Console
from rich.traceback import install

# ARIEL
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.video_recorder import VideoRecorder

# Reuse the env + fitness defined in script 6 so visual / sampler / fitness
# stay in lockstep with the body+brain co-evolution experiment.
SCRIPT_DIR = Path(__file__).resolve().parent

install()
console = Console()

warnings.filterwarnings(
    "ignore",
    message="TPA: apparent inconsistency",
    category=UserWarning,
    module="cma",
)


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Gecko brain evolution on ring waypoints")
parser.add_argument("--budget",        type=int,   default=80,   help="CMA generations")
parser.add_argument("--population",    type=int,   default=40,   help="Requested CMA population (≥ min-lambda, even)")
parser.add_argument("--dur",           type=float, default=45.0, help="Max episode duration (s)")
parser.add_argument("--reach-radius",  type=float, default=0.20, help="Planar reach radius (m); also the gate cylinder radius")
parser.add_argument("--num-waypoints", type=int,   default=10,   help="Waypoints per episode")
parser.add_argument("--workers",       type=int,   default=max(1, os.cpu_count() or 1))
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--no-video",      action="store_true", help="Skip post-evolution video recording")
parser.add_argument("--fps",           type=int,   default=30,   help="Video frame rate")
args = parser.parse_args()

BUDGET        = args.budget
POP_SIZE      = args.population
DURATION      = args.dur
REACH_RADIUS  = max(0.05, args.reach_radius)
NUM_WAYPOINTS = args.num_waypoints
NUM_WORKERS   = max(1, args.workers)
BASE_SEED     = args.seed
FPS           = args.fps

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

# Load helpers from script 6 with the desired REACH_RADIUS so the cylinder
# gate matches the trigger radius. Forwarding --no-video keeps that script's
# main() from running on import.
def _import_env(reach_radius: float):
    import sys
    _orig = sys.argv
    sys.argv = [str(SCRIPT_DIR / "6_body_brain_randomized_waypoints.py"),
                "--no-video", "--reach-radius", str(reach_radius),
                "--num-waypoints", str(NUM_WAYPOINTS),
                "--dur", str(DURATION)]
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_bb_env", SCRIPT_DIR / "6_body_brain_randomized_waypoints.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = _orig

_env = _import_env(REACH_RADIUS)
sample_waypoints  = _env.sample_waypoints
Network           = _env.Network
fill_parameters   = _env.fill_parameters
isolate_green     = _env.isolate_green
analyze_sections  = _env.analyze_sections
run_episode       = _env.run_episode
compute_fitness   = _env.compute_fitness
get_robot_state   = _env.get_robot_state
RING_R_MIN        = _env.RING_R_MIN
RING_R_MAX        = _env.RING_R_MAX
GATE_HALF_HEIGHT  = _env.GATE_HALF_HEIGHT


# ── World construction ───────────────────────────────────────────────────────
#
# Body is a fixed gecko (no morphology evolution). Gate matches script 6
# exactly: vertical cylinder of radius=REACH_RADIUS, half-height=GATE_HALF_HEIGHT,
# pass-through (contype=0/conaffinity=0), alpha 0.7.

_RENDER_INIT_LOCK = mp.Lock() if False else None  # not needed for serial worker calls
_process_local_ctx: Optional[dict[str, Any]] = None


def _build_context() -> dict[str, Any]:
    world = SimpleFlatWorld()
    body  = gecko()
    world.spawn(body.spec, position=[0.0, 0.0, 0.1])

    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=[0.0, 0.0, GATE_HALF_HEIGHT]
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[REACH_RADIUS, GATE_HALF_HEIGHT],
        rgba=[0, 1, 0, 0.7],
        contype=0,
        conaffinity=0,
    )

    arena_extent = max(2.0, RING_R_MAX * NUM_WAYPOINTS * 1.5)
    world.spec.worldbody.add_camera(
        name="overview_cam",
        pos=[0.0, 0.0, arena_extent],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )

    model = world.spec.compile()
    data  = mujoco.MjData(model)

    cam_name: Optional[str] = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "overview" not in name:
            cam_name = name
            break

    target_mocap_id = model.body("green_target").mocapid[0]
    input_dim = len(get_robot_state(data)) + 3 + 2 + 1  # robot + vision + phase + progress

    return {
        "world":           world,
        "model":           model,
        "data":            data,
        "cam_name":        cam_name,
        "target_mocap_id": target_mocap_id,
        "input_dim":       input_dim,
    }


# ── CMA-ES worker pool ───────────────────────────────────────────────────────

def _init_worker(base_seed: int) -> None:
    torch.set_num_threads(1)
    seed = (base_seed + os.getpid()) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_ctx() -> dict[str, Any]:
    global _process_local_ctx
    if _process_local_ctx is None:
        ctx = _build_context()
        ctx["network"]  = Network(input_size=ctx["input_dim"], output_size=ctx["model"].nu)
        ctx["renderer"] = mujoco.Renderer(ctx["model"], height=48 * 2, width=64 * 2)
        _process_local_ctx = ctx
    return cast(dict[str, Any], _process_local_ctx)


def _evaluate_candidate(task: tuple[np.ndarray, list[np.ndarray]]) -> float:
    weights, waypoints = task
    ctx = _get_ctx()
    fill_parameters(ctx["network"], weights)
    mujoco.mj_resetData(ctx["model"], ctx["data"])
    result = run_episode(
        model=ctx["model"], data=ctx["data"], network=ctx["network"],
        waypoints=waypoints, duration=DURATION, reach_radius=REACH_RADIUS,
        target_mocap_id=ctx["target_mocap_id"], renderer=ctx["renderer"],
        cam_name=ctx["cam_name"],
    )
    return compute_fitness(
        waypoints_reached=result["waypoints_reached"],
        min_dist_to_current=result["min_dist_to_current"],
        num_waypoints=len(waypoints),
        completion_time=result["completion_time"],
        duration=DURATION,
    )


# ── Evolution ────────────────────────────────────────────────────────────────

def evolve() -> tuple[np.ndarray, list[np.ndarray], int]:
    ctx        = _build_context()
    input_dim  = ctx["input_dim"]
    model      = ctx["model"]

    dummy_net  = Network(input_size=input_dim, output_size=model.nu)
    num_params = sum(p.numel() for p in dummy_net.parameters())

    min_lambda = 4 + int(3 * np.log(max(num_params, 2)))
    pop_size   = max(POP_SIZE, min_lambda)
    if pop_size % 2 != 0:
        pop_size += 1

    initial_guess = np.random.uniform(-0.5, 0.5, size=num_params)
    param = ng.p.Array(init=initial_guess).set_mutation(sigma=0.3)
    cma_config = ng.optimizers.ParametrizedCMA(popsize=pop_size)
    optimizer  = cma_config(parametrization=param, budget=BUDGET * pop_size, num_workers=pop_size)

    console.rule("[bold magenta]Gecko Ring-Waypoints Brain Evolution[/bold magenta]")
    console.log(
        f"params={num_params}  pop_size={pop_size} (requested {POP_SIZE})  "
        f"budget={BUDGET} gens  workers={NUM_WORKERS}"
    )
    console.log(
        f"waypoints/ep={NUM_WAYPOINTS}  ring=[{RING_R_MIN}, {RING_R_MAX}] m  "
        f"reach={REACH_RADIUS} m  duration={DURATION} s"
    )

    best_seen_fitness:  float                    = float("inf")
    best_seen_weights:  Optional[np.ndarray]     = None
    best_seen_waypoints: Optional[list[np.ndarray]] = None

    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(BASE_SEED,),
    ) as executor:
        for gen in range(BUDGET):
            gen_rng       = np.random.default_rng(BASE_SEED + gen)
            gen_waypoints = sample_waypoints(gen_rng, n=NUM_WAYPOINTS)

            candidates = [optimizer.ask() for _ in range(pop_size)]
            tasks      = [(c.value, gen_waypoints) for c in candidates]
            fitnesses  = list(executor.map(_evaluate_candidate, tasks))

            for cand, fit in zip(candidates, fitnesses):
                optimizer.tell(cand, fit)

            best_idx = int(np.argmin(fitnesses))
            best     = float(fitnesses[best_idx])
            if best < best_seen_fitness:
                best_seen_fitness   = best
                best_seen_weights   = candidates[best_idx].value.copy()
                best_seen_waypoints = gen_waypoints

            best_wps = max(0, int(np.floor(-best))) if best < 0 else 0
            console.log(
                f"gen {gen+1:3d}/{BUDGET}  best={best:.3f}  ({best_wps}/{NUM_WAYPOINTS} reached)  "
                f"best_seen={best_seen_fitness:.3f}"
            )

    elapsed = time.time() - t0
    console.log(f"[green]Evolution done — {elapsed/60:.1f} min[/green]")

    assert best_seen_weights is not None and best_seen_waypoints is not None
    return best_seen_weights, best_seen_waypoints, input_dim


# ── Best-of-run video (overview + stacked POV) ───────────────────────────────

def record_videos(weights: np.ndarray, waypoints: list[np.ndarray], input_dim: int) -> None:
    ctx = _build_context()
    model           = ctx["model"]
    data            = ctx["data"]
    target_mocap_id = ctx["target_mocap_id"]
    cam_name        = ctx["cam_name"]

    net = Network(input_size=input_dim, output_size=model.nu)
    fill_parameters(net, weights)

    mujoco.mj_resetData(model, data)
    num_wps = len(waypoints)
    wp_idx  = 0
    current_target = waypoints[0]
    data.mocap_pos[target_mocap_id] = current_target

    DATA.mkdir(parents=True, exist_ok=True)
    overview = VideoRecorder(file_name="best_overview", output_folder=str(DATA), fps=FPS)

    POV_H, POV_W = 48 * 2, 64 * 2
    pov = VideoRecorder(
        file_name="best_pov",
        output_folder=str(DATA),
        width=POV_W,
        height=POV_H * 2,
        fps=FPS,
    )

    dt = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (FPS * dt))))
    control_step_freq = 50
    current_ctrl = np.zeros(model.nu)
    render_step = 0

    control_renderer = mujoco.Renderer(model, height=POV_H, width=POV_W)
    latest = {"raw":  np.zeros((POV_H, POV_W, 3), dtype=np.uint8),
              "mask": np.zeros((POV_H, POV_W),    dtype=np.uint8)}

    def get_ctrl(d: mujoco.MjData) -> np.ndarray:
        control_renderer.update_scene(d, camera=cam_name)
        img    = control_renderer.render()
        mask   = isolate_green(img)
        latest["raw"]  = img
        latest["mask"] = mask
        vision = analyze_sections(mask)
        rs     = get_robot_state(d)
        phase  = [2.0 * np.sin(d.time * 2.0 * np.pi),
                  2.0 * np.cos(d.time * 2.0 * np.pi)]
        prog   = [wp_idx / max(num_wps - 1, 1)]
        state  = np.concatenate([rs, vision, phase, prog]).astype(np.float32)
        return net.forward(model, d, state)

    def write_pov_frame() -> None:
        x1, x2 = POV_W // 3, (2 * POV_W) // 3
        raw  = latest["raw"].copy()
        mask = cv2.cvtColor(latest["mask"], cv2.COLOR_GRAY2RGB)
        for x in (x1, x2):
            raw[:, x, :]  = (255, 0, 0)
            mask[:, x, :] = (255, 0, 255)
        stacked = np.vstack([raw, mask])
        pov.write(stacked)

    overview_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview_cam")

    with mujoco.Renderer(model, height=480, width=640) as renderer:
        while data.time < DURATION and wp_idx < num_wps:
            for _ in range(steps_per_frame):
                if render_step % control_step_freq == 0:
                    current_ctrl = get_ctrl(data)
                np.copyto(data.ctrl, current_ctrl)
                mujoco.mj_step(model, data)
                render_step += 1

                if wp_idx < num_wps:
                    dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
                    if dist <= REACH_RADIUS:
                        wp_idx += 1
                        if wp_idx < num_wps:
                            current_target = waypoints[wp_idx]
                            data.mocap_pos[target_mocap_id] = current_target

            renderer.update_scene(data, camera=overview_id)
            overview.write(renderer.render())
            write_pov_frame()

    overview.release()
    pov.release()
    control_renderer.close()
    console.log(f"[green]Videos → {DATA}/best_overview*.mp4 and {DATA}/best_pov*.mp4[/green]")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    weights, waypoints, input_dim = evolve()

    np.save(DATA / "best_seen_weights.npy",   weights)
    np.save(DATA / "best_seen_waypoints.npy", np.array(waypoints))
    console.log(f"[green]Saved best weights + waypoints → {DATA}[/green]")

    if not args.no_video:
        record_videos(weights, waypoints, input_dim)


if __name__ == "__main__":
    main()
