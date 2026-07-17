"""
Multi-objective hyperparameter tuning for CMA-ES + ANN on gecko tasks.

Tunes CMA-ES parameters (sigma, population size, budget, weight init scale) and
ANN architecture (number of hidden layers 1-3, size of each layer 1-64) using
Optuna's multi-objective NSGA-II sampler.

Two objectives (both minimised):
  - mean_fitness:     mean best-fitness over N runs
  - mean_wall_time_s: mean wall-clock seconds per run

Tasks:
  food  — collect waypoints; input = [robot_state, vision (3 bins), progress]
  loco  — forward locomotion; input = [robot_state]

Sinusoidal phase inputs are excluded from both tasks.

Usage (smoke test):
  python tune_cma_ann.py --task food --n-trials 4 --runs-per-trial 2 --workers 4 --dur 10
  python tune_cma_ann.py --task loco --n-trials 4 --runs-per-trial 2 --workers 4 --dur 10

Usage (full run, from SLURM script):
  python tune_cma_ann.py --task food --n-trials 200 --runs-per-trial 5 --workers 64
  python tune_cma_ann.py --task loco --n-trials 200 --runs-per-trial 5 --workers 64
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mujoco
import nevergrad as ng
import numpy as np
import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from rich.console import Console
from rich.table import Table
from rich.traceback import install
from torch import nn

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

sys.path.insert(0, str(Path(__file__).parent))
from shared import (
    SPAWN_POSITION,
    action_control_cost,
    analyze_sections,
    fill_parameters,
    isolate_green,
    sample_waypoints,
)

install()
warnings.filterwarnings("ignore", message="TPA: apparent inconsistency",
                        category=UserWarning, module="cma")
optuna.logging.set_verbosity(optuna.logging.WARNING)

console = Console()

# ── Episode constants ─────────────────────────────────────────────────────────

SETTLE_DURATION          = 3.0
HINGE_CONTACT_LIMIT      = 200
HINGE_CONTACT_PENALTY    = 0.005
HINGE_GLITCH_FITNESS     = 1.0
CTRL_ALPHA               = 0.5
HEIGHT_PENALTY_THRESHOLD = 0.21
CONTROL_STEP_FREQ        = 100
RING_R_MAX               = 1.0

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Multi-objective HPO: CMA-ES + ANN on gecko tasks")
parser.add_argument("--task",             choices=["food", "loco"], default="food")
parser.add_argument("--n-trials",         type=int,   default=200)
parser.add_argument("--runs-per-trial",   type=int,   default=5)
parser.add_argument("--seed",             type=int,   default=None,
                    help="Random seed (default: random)")
parser.add_argument("--workers",          type=int,   default=64,
                    help="CPUs used to parallelise each CMA-ES generation (one per candidate)")
parser.add_argument("--trial-time-limit", type=float, default=300.0,
                    help="Wall-clock seconds allowed per individual CMA-ES run (default 300 = 5 min)")
# food-task only
parser.add_argument("--num-waypoints",    type=int,   default=10)
parser.add_argument("--reach-radius",     type=float, default=0.20)
parser.add_argument("--arena-radius",     type=float, default=2.0)
# shared
parser.add_argument("--dur",              type=float, default=60.0)
parser.add_argument("--budget-max",       type=int,   default=None,
                    help="Cap the budget search range (e.g. 50 for smoke tests)")
parser.add_argument("--out-dir",          type=Path,  default=None,
                    help="Output directory (defaults to __data__/tune_cma_ann_{task})")
args = parser.parse_args()
if args.seed is None:
    args.seed = random.randint(0, 2**31 - 1)

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR: Path = args.out_dir if args.out_dir is not None else Path("__data__") / f"tune_cma_ann_{args.task}_{_timestamp}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_LOG  = OUT_DIR / "tune_study.log"
JSONL_PATH = OUT_DIR / "results.jsonl"

with (OUT_DIR / "run_config.json").open("w") as _fh:
    json.dump(vars(args), _fh, indent=2, default=str)

# ── Flexible ANN ──────────────────────────────────────────────────────────────


class FlexNetwork(nn.Module):
    """ANN with a variable number of hidden layers, each of configurable size."""

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


# ── Gecko model builders ──────────────────────────────────────────────────────


def _build_gecko_food(reach_radius: float, arena_radius: float):
    """Build gecko with green mocap target and onboard camera for food task."""
    core  = gecko()
    world = SimpleFlatWorld()
    try:
        world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                    correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                    correct_collision_with_floor=False)

    gate_half_h = 0.15
    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=[0.0, 0.0, gate_half_h]
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[reach_radius, gate_half_h],
        rgba=[0, 1, 0, 0.7],
        contype=0,
        conaffinity=0,
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
    """Build plain gecko for locomotion task (no mocap target, no extra camera)."""
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


def _build_rotor_geom_ids(model: mujoco.MjModel) -> set[int]:
    return {i for i in range(model.ngeom) if model.geom(i).name.endswith("-rotor")}


def _floor_geom_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


# ── Episode runners ───────────────────────────────────────────────────────────


def _run_episode_food(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: FlexNetwork,
    renderer: mujoco.Renderer,
    cam_name: str | None,
    target_mocap_id: int,
    waypoints: list[np.ndarray],
    weights: np.ndarray,
    reach_radius: float,
    dur: float,
    rotor_geom_ids: set[int],
    floor_id: int,
) -> float:
    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    num_wps        = len(waypoints)
    current_wp_idx = 0
    waypoints_reached = 0
    current_target = waypoints[0]
    data.mocap_pos[target_mocap_id] = current_target
    min_dist_to_current = float("inf")

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)
        if current_wp_idx < num_wps:
            dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
            min_dist_to_current = min(min_dist_to_current, dist)
            if dist <= reach_radius:
                waypoints_reached += 1
                current_wp_idx    += 1
                if current_wp_idx < num_wps:
                    current_target = waypoints[current_wp_idx]
                    data.mocap_pos[target_mocap_id] = current_target
                    min_dist_to_current = float("inf")

    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    initial_height = float(data.xpos[core_id, 2])

    step = 0
    current_action = np.zeros(model.nu)
    c_hinge = 0
    prev_rotor_contacts: set[int] = set()

    episode_end = SETTLE_DURATION + dur
    while data.time < episode_end and current_wp_idx < num_wps:
        if step % CONTROL_STEP_FREQ == 0:
            renderer.update_scene(data, camera=cam_name)
            img    = renderer.render()
            vision = analyze_sections(isolate_green(img))

            robot_state = get_robot_state(data)
            progress    = [current_wp_idx / max(num_wps - 1, 1)]
            state = np.concatenate([robot_state, vision, progress]).astype(np.float32)
            raw_action = network.forward(state)
            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -np.pi / 2, np.pi / 2,
            )

        data.ctrl[:] = current_action
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

        if current_wp_idx < num_wps:
            dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
            min_dist_to_current = min(min_dist_to_current, dist)
            if dist <= reach_radius:
                waypoints_reached += 1
                current_wp_idx    += 1
                if current_wp_idx < num_wps:
                    current_target = waypoints[current_wp_idx]
                    data.mocap_pos[target_mocap_id] = current_target
                    min_dist_to_current = float("inf")

    final_dist = 0.0 if waypoints_reached >= num_wps else min_dist_to_current
    d_norm = float(np.clip((RING_R_MAX - final_dist) / RING_R_MAX, 0.0, 1.0))

    if c_hinge > HINGE_CONTACT_LIMIT:
        return HINGE_GLITCH_FITNESS

    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    return -(waypoints_reached + d_norm - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)


def _run_episode_loco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: FlexNetwork,
    weights: np.ndarray,
    dur: float,
    rotor_geom_ids: set[int],
    floor_id: int,
) -> float:
    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    x0             = float(data.qpos[0])
    initial_height = float(data.xpos[core_id, 2])

    step = 0
    current_action = np.zeros(model.nu)
    c_hinge = 0
    prev_rotor_contacts: set[int] = set()

    episode_end = SETTLE_DURATION + dur
    while data.time < episode_end:
        if step % CONTROL_STEP_FREQ == 0:
            robot_state = get_robot_state(data).astype(np.float32)
            raw_action  = network.forward(robot_state)
            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -np.pi / 2, np.pi / 2,
            )

        data.ctrl[:] = current_action
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

    x_displacement = float(data.qpos[0]) - x0
    if c_hinge > HINGE_CONTACT_LIMIT:
        return HINGE_GLITCH_FITNESS

    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    return -(x_displacement - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)


# ── Single CMA-ES run ─────────────────────────────────────────────────────────


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
    """Evaluate one CMA-ES candidate in a subprocess."""
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if task == "food":
        model, data, target_mocap_id, cam_name = _build_gecko_food(reach_radius, arena_radius)
        renderer  = mujoco.Renderer(model, height=96, width=128)
        waypoints = sample_waypoints(rng, num_waypoints)
        base_dim  = len(get_robot_state(data))
        input_dim = base_dim + 3 + 1
    else:
        model, data = _build_gecko_loco()
        renderer, target_mocap_id, cam_name, waypoints = None, -1, None, []
        base_dim  = len(get_robot_state(data))
        input_dim = base_dim

    network        = FlexNetwork(input_size=input_dim, output_size=model.nu, hidden_sizes=hidden_sizes)
    rotor_geom_ids = _build_rotor_geom_ids(model)
    floor_id       = _floor_geom_id(model)

    if task == "food":
        assert renderer is not None
        fit = _run_episode_food(
            model, data, network, renderer, cam_name, target_mocap_id,
            waypoints, weights, reach_radius, dur, rotor_geom_ids, floor_id,
        )
        renderer.close()
    else:
        fit = _run_episode_loco(model, data, network, weights, dur, rotor_geom_ids, floor_id)

    return fit


def run_single_trial(
    task: str,
    hidden_sizes: list[int],
    sigma: float,
    pop: int,
    budget: int,
    init_scale: float,
    run_seed: int,
    num_waypoints: int,
    reach_radius: float,
    arena_radius: float,
    dur: float,
    workers: int = 1,
    time_limit_s: float = 300.0,
) -> dict[str, Any]:
    """Run one CMA-ES optimisation with parallel generation evaluation.

    Each generation's candidates are evaluated concurrently (up to `workers`
    processes). The run stops early if `time_limit_s` wall-clock seconds elapse;
    the number of completed evaluations is recorded alongside the result.
    """
    torch.set_num_threads(1)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    t_start = time.perf_counter()
    rng = np.random.default_rng(run_seed)

    # Build one model just to count params and input dim.
    if task == "food":
        model, data, _, _ = _build_gecko_food(reach_radius, arena_radius)
        base_dim  = len(get_robot_state(data))
        input_dim = base_dim + 3 + 1
    else:
        model, data = _build_gecko_loco()
        input_dim = len(get_robot_state(data))

    network    = FlexNetwork(input_size=input_dim, output_size=model.nu, hidden_sizes=hidden_sizes)
    num_params = sum(p.numel() for p in network.parameters())

    initial_guess = rng.uniform(-init_scale, init_scale, size=num_params)
    param     = ng.p.Array(init=initial_guess).set_mutation(sigma=sigma)
    optimizer = ng.optimizers.CMA(
        parametrization=param,
        budget=budget,
        num_workers=pop,
    )

    best_fit: float = float("inf")
    learning_curve: list[float] = []
    evals = 0
    timed_out = False

    with ProcessPoolExecutor(max_workers=min(pop, workers)) as pool:
        while evals < budget:
            if time.perf_counter() - t_start >= time_limit_s:
                timed_out = True
                break

            # Always ask a full generation of `pop` candidates — never a partial
            # batch. CMA's TPA step-size adaptation calls es.ask() on the first
            # optimizer.ask() of each generation and expects es.tell() to be called
            # with all `pop` results before the next es.ask(). A partial batch
            # leaves _to_be_told below popsize so es.tell() never fires, breaking
            # TPA on the following generation.
            candidates = [optimizer.ask() for _ in range(pop)]

            futures = [
                pool.submit(
                    _evaluate_candidate,
                    np.array(c.value, dtype=np.float32),
                    task, hidden_sizes, num_waypoints,
                    reach_radius, arena_radius, dur,
                    run_seed + evals + i,
                )
                for i, c in enumerate(candidates)
            ]

            fits = [f.result() for f in futures]
            for candidate, fit in zip(candidates, fits):
                optimizer.tell(candidate, fit)
                evals += 1
                if fit < best_fit:
                    best_fit = fit
                    learning_curve.append(best_fit)

            if time.perf_counter() - t_start >= time_limit_s:
                timed_out = True
                break

    return {
        "fitness":        best_fit,
        "wall_time_s":    time.perf_counter() - t_start,
        "learning_curve": learning_curve,
        "num_params":     num_params,
        "evals_completed": evals,
        "timed_out":      timed_out,
    }


# ── Optuna objective ──────────────────────────────────────────────────────────


def make_objective(
    task: str,
    runs_per_trial: int,
    base_seed: int,
    num_waypoints: int,
    reach_radius: float,
    arena_radius: float,
    dur: float,
    max_budget: int = 1000,
    workers: int = 1,
    time_limit_s: float = 300.0,
):
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        n_layers   = trial.suggest_int("n_layers", 1, 3)
        hidden_1   = trial.suggest_int("hidden_1", 1, 64)
        hidden_2   = trial.suggest_int("hidden_2", 1, 64)
        hidden_3   = trial.suggest_int("hidden_3", 1, 64)
        sigma      = trial.suggest_float("sigma", 0.05, 1.0, log=True)
        pop        = trial.suggest_int("pop", 5, 64)
        budget     = trial.suggest_int("budget", min(200, max_budget), max_budget, step=100)
        init_scale = trial.suggest_float("init_scale", 0.1, 2.0, log=True)

        if n_layers == 1:
            hidden_sizes = [hidden_1]
        elif n_layers == 2:
            hidden_sizes = [hidden_1, hidden_2]
        else:
            hidden_sizes = [hidden_1, hidden_2, hidden_3]

        fitnesses:    list[float] = []
        wall_times:   list[float] = []
        evals_done:   list[int]   = []
        timed_out_flags: list[bool] = []

        for run_idx in range(runs_per_trial):
            run_seed = base_seed + trial.number * 1000 + run_idx
            result = run_single_trial(
                task=task,
                hidden_sizes=hidden_sizes,
                sigma=sigma,
                pop=pop,
                budget=budget,
                init_scale=init_scale,
                run_seed=run_seed,
                num_waypoints=num_waypoints,
                reach_radius=reach_radius,
                arena_radius=arena_radius,
                dur=dur,
                workers=workers,
                time_limit_s=time_limit_s,
            )
            fitnesses.append(result["fitness"])
            wall_times.append(result["wall_time_s"])
            evals_done.append(result["evals_completed"])
            timed_out_flags.append(result["timed_out"])

        mean_fitness = float(np.mean(fitnesses))
        mean_time    = float(np.mean(wall_times))

        record = {
            "task":              task,
            "trial":             trial.number,
            "n_layers":          n_layers,
            "hidden_sizes":      hidden_sizes,
            "sigma":             sigma,
            "pop":               pop,
            "budget":            budget,
            "init_scale":        init_scale,
            "mean_fitness":      mean_fitness,
            "std_fitness":       float(np.std(fitnesses)),
            "min_fitness":       float(np.min(fitnesses)),
            "max_fitness":       float(np.max(fitnesses)),
            "mean_time_s":       mean_time,
            "per_run_fitness":   fitnesses,
            "per_run_time_s":    wall_times,
            "per_run_evals":     evals_done,
            "per_run_timed_out": timed_out_flags,
        }
        _append_jsonl(JSONL_PATH, record)

        return mean_fitness, mean_time

    return objective


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ── Pareto summary ────────────────────────────────────────────────────────────


def print_pareto_summary(study: optuna.Study, task: str) -> None:
    pareto = study.best_trials
    if not pareto:
        console.log("[yellow]No Pareto-optimal trials yet.[/yellow]")
        return

    table = Table(title=f"Pareto Front — {task} ({len(pareto)} configs)")
    table.add_column("Trial",         justify="right")
    table.add_column("Layers",        justify="right")
    table.add_column("Hidden sizes")
    table.add_column("sigma",         justify="right")
    table.add_column("pop",           justify="right")
    table.add_column("budget",        justify="right")
    table.add_column("init_scale",    justify="right")
    table.add_column("Mean fitness",  justify="right")
    table.add_column("Mean time (s)", justify="right")

    for t in sorted(pareto, key=lambda t: t.values[0]):
        p = t.params
        n = p["n_layers"]
        hs = [p["hidden_1"]] + ([p["hidden_2"]] if n >= 2 else []) + ([p["hidden_3"]] if n == 3 else [])
        table.add_row(
            str(t.number),
            str(n),
            str(hs),
            f"{p['sigma']:.3f}",
            str(p["pop"]),
            str(p["budget"]),
            f"{p['init_scale']:.3f}",
            f"{t.values[0]:.4f}",
            f"{t.values[1]:.1f}",
        )

    console.print(table)
    console.log(f"Full results → {JSONL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    torch.set_num_threads(1)

    console.rule(f"[bold magenta]CMA-ES + ANN HPO — Gecko {args.task}[/bold magenta]")
    console.log(
        f"task={args.task}  n_trials={args.n_trials}  runs_per_trial={args.runs_per_trial}  "
        f"workers={args.workers}  dur={args.dur}s  trial_time_limit={args.trial_time_limit}s"
    )
    if args.task == "food":
        console.log(f"arena_radius={args.arena_radius}m  num_waypoints={args.num_waypoints}")
    console.log(f"Results → {OUT_DIR}")

    storage = JournalStorage(JournalFileBackend(str(STUDY_LOG)))
    study = optuna.create_study(
        study_name=f"tune_cma_ann_{args.task}",
        storage=storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
    )

    max_budget = args.budget_max if args.budget_max is not None else 1000
    objective = make_objective(
        task=args.task,
        runs_per_trial=args.runs_per_trial,
        base_seed=args.seed,
        num_waypoints=args.num_waypoints,
        reach_radius=args.reach_radius,
        arena_radius=args.arena_radius,
        dur=args.dur,
        max_budget=max_budget,
        workers=args.workers,
        time_limit_s=args.trial_time_limit,
    )

    # Trials run sequentially; parallelism is inside each trial across CMA-ES candidates.
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    print_pareto_summary(study, args.task)


if __name__ == "__main__":
    main()
