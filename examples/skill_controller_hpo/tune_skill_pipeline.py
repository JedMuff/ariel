"""
Multi-objective HPO for the gecko skill-controller pipeline.

Jointly tunes:
  - Network architecture (hidden size / depth, shared across loco + turn skills)
  - CMA-ES training budget for loco and turn phases separately
  - CMA-ES population size and sigma
  - SkillController commit-steps

Pipeline per Optuna trial:
  1. Train loco skill via CMA-ES  (budget up to loco_budget generations, or trial_time_limit wall-clock)
  2. Train left  skill via CMA-ES  (same time budget, turn_budget generations)
  3. Train right skill via CMA-ES  (same)
  4. Evaluate the SkillController (state-based) on the food task for n_eval_runs × 120 s episodes

Objectives (NSGA-II, both minimised internally):
  - neg_mean_score:   negative mean food-task score (minimise → maximise score)
  - mean_wall_time_s: total wall-clock seconds for training + evaluation (minimise)

Results are saved after every trial:
  {OUT_DIR}/results.jsonl                         — one JSON record per trial
  {OUT_DIR}/trial_{N:04d}/loco_weights.npy
  {OUT_DIR}/trial_{N:04d}/left_weights.npy
  {OUT_DIR}/trial_{N:04d}/right_weights.npy

Usage (smoke test, ~2 min):
  python tune_skill_pipeline.py \\
      --n-trials 2 --workers 4 \\
      --loco-budget-max 10 --turn-budget-max 10 \\
      --n-eval-runs 1 --eval-duration 10 --trial-time-limit 120

Usage (full SLURM run):
  python tune_skill_pipeline.py \\
      --n-trials 100 --workers 64 \\
      --n-eval-runs 5 --eval-duration 120 --trial-time-limit 600
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cma
import mujoco
import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from rich.console import Console
from rich.table import Table
from rich.traceback import install

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

# Pull shared helpers from the symmetry_pressure sibling folder.
_SHARED_DIR = Path(__file__).parent.parent / "symmetry_pressure"
sys.path.insert(0, str(_SHARED_DIR))
from shared import (  # noqa: E402
    GATE_HALF_HEIGHT,
    RING_R_MAX,
    SPAWN_POSITION,
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

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Multi-objective HPO for the gecko skill-controller pipeline"
)
parser.add_argument("--n-trials",          type=int,   default=100)
parser.add_argument("--n-eval-runs",       type=int,   default=5,
                    help="Food-task evaluation episodes per trial")
parser.add_argument("--workers",           type=int,   default=max(1, os.cpu_count() or 1),
                    help="Max parallel processes for CMA-ES population evaluation")
parser.add_argument("--trial-time-limit",  type=float, default=600.0,
                    help="Wall-clock seconds budget for ONE full trial (training + eval). "
                         "CMA-ES phases stop early when this is about to be exceeded and "
                         "the best weights found so far are used. Default: 600 (10 min).")
parser.add_argument("--eval-duration",     type=float, default=120.0,
                    help="Episode duration in seconds for the food-task evaluation")
parser.add_argument("--num-waypoints",     type=int,   default=10)
parser.add_argument("--reach-radius",      type=float, default=0.20)
parser.add_argument("--collect-radius",    type=float, default=0.40)
parser.add_argument("--arena-radius",      type=float, default=2.0)
# Budget caps (useful for smoke tests)
parser.add_argument("--loco-budget-max",   type=int,   default=500,
                    help="Maximum CMA-ES generations for loco training")
parser.add_argument("--turn-budget-max",   type=int,   default=300,
                    help="Maximum CMA-ES generations for turn training")
parser.add_argument("--seed",              type=int,   default=None)
parser.add_argument("--out-dir",           type=Path,  default=None)
args = parser.parse_args()

if args.seed is None:
    args.seed = random.randint(0, 2**31 - 1)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR: Path = (
    args.out_dir if args.out_dir is not None
    else Path(__file__).parent.parent.parent / "__data__" / f"tune_skill_pipeline_{_ts}"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_LOG  = OUT_DIR / "tune_study.log"
JSONL_PATH = OUT_DIR / "results.jsonl"

with (OUT_DIR / "run_config.json").open("w") as _fh:
    json.dump(vars(args), _fh, indent=2, default=str)

# ── Episode constants (must match gecko_skills.py / run_skill_controller.py) ──

SETTLE_DURATION          = 3.0
CTRL_ALPHA               = 0.5
CONTROL_STEP_FREQ        = 50
HINGE_CONTACT_LIMIT      = 200
HINGE_CONTACT_PENALTY    = 0.005
HINGE_GLITCH_FITNESS     = 1.0
HEIGHT_PENALTY_THRESHOLD = 0.21
LOCO_DURATION            = 30.0
TURN_DURATION            = 15.0
CENTRE_FWD_THRESH        = 0.6

# ── Flexible ANN ──────────────────────────────────────────────────────────────


class FlexNetwork(nn.Module):
    """ANN with variable hidden layers, each of configurable size.

    Copied from tune_cma_ann.py so this script has no module-level side-effect
    imports from that file.
    """

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


# ── SkillController (copied from run_skill_controller.py) ─────────────────────


class SkillController:
    """Stateful controller: commits to each skill for a minimum number of steps."""

    def __init__(
        self,
        rng: np.random.Generator,
        commit_steps: int = 50,
        centre_fwd_thresh: float = CENTRE_FWD_THRESH,
    ) -> None:
        self.last_turn_dir:       int   = 2 if rng.random() < 0.5 else 3
        self.commit_steps:        int   = commit_steps
        self.centre_fwd_thresh:   float = centre_fwd_thresh
        self._current_skill:      int   = self.last_turn_dir
        self._steps_held:         int   = commit_steps

    def _decide(self, left: float, centre: float, right: float) -> int:
        any_visible = (left + centre + right) > 0.0
        if any_visible:
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


# ── World builders ────────────────────────────────────────────────────────────


def _build_loco_world():
    core  = gecko()
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                correct_collision_with_floor=True)
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    return model, data


def _build_food_world(reach_radius: float):
    core  = gecko()
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                correct_collision_with_floor=True)
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
    data  = mujoco.MjData(model)
    target_mocap_id = model.body("green_target").mocapid[0]
    cam_name = ""
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "overview" not in name:
            cam_name = name
            break
    return model, data, target_mocap_id, cam_name


def _rotor_geom_ids(model: mujoco.MjModel) -> set[int]:
    return {i for i in range(model.ngeom) if model.geom(i).name.endswith("-rotor")}


def _floor_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


# ── Subprocess worker state (one per process per skill) ───────────────────────

_worker_ctx: Optional[dict[str, Any]] = None


def _worker_init(seed: int, skill: str, hidden_sizes: list[int]) -> None:
    global _worker_ctx
    torch.set_num_threads(1)
    np.random.seed((seed + os.getpid()) % (2**32 - 1))
    model, data = _build_loco_world()
    input_dim   = len(get_robot_state(data))
    output_dim  = model.nu
    network     = FlexNetwork(input_size=input_dim, output_size=output_dim,
                              hidden_sizes=hidden_sizes)
    rotor_ids   = _rotor_geom_ids(model)
    floor       = _floor_id(model)
    _worker_ctx = {
        "model": model, "data": data, "network": network,
        "rotor_ids": rotor_ids, "floor": floor, "skill": skill,
    }


def _worker_eval(weights_list: list[float]) -> float:
    assert _worker_ctx is not None
    model     = _worker_ctx["model"]
    data      = _worker_ctx["data"]
    network   = _worker_ctx["network"]
    rotor_ids = _worker_ctx["rotor_ids"]
    floor     = _worker_ctx["floor"]
    skill     = _worker_ctx["skill"]
    weights   = np.array(weights_list, dtype=np.float32)
    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    core_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    initial_height = float(data.xpos[core_id, 2])
    step           = 0
    current_action = np.zeros(model.nu)
    c_hinge        = 0
    prev_contacts: set[int] = set()

    if skill == "loco":
        x0       = float(data.qpos[0])
        duration = LOCO_DURATION
    else:
        xmat     = data.xmat[core_id]
        yaw_prev = math.atan2(float(xmat[3]), float(xmat[0]))
        accumulated = 0.0
        direction   = +1 if skill == "left" else -1
        duration    = TURN_DURATION

    episode_end = SETTLE_DURATION + duration
    while data.time < episode_end:
        if step % CONTROL_STEP_FREQ == 0:
            state      = get_robot_state(data).astype(np.float32)
            raw_action = network.forward(state)
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
            if g1 == floor and g2 in rotor_ids:
                curr.add(g2)
            elif g2 == floor and g1 in rotor_ids:
                curr.add(g1)
        c_hinge += len(curr - prev_contacts)
        prev_contacts = curr

        if skill != "loco":
            xmat     = data.xmat[core_id]
            yaw_curr = math.atan2(float(xmat[3]), float(xmat[0]))
            delta    = (yaw_curr - yaw_prev + math.pi) % (2 * math.pi) - math.pi
            accumulated += max(0.0, delta * direction)
            yaw_prev = yaw_curr

        step += 1

    if c_hinge > HINGE_CONTACT_LIMIT:
        return HINGE_GLITCH_FITNESS

    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    if skill == "loco":
        x_disp = float(data.qpos[0]) - x0
        return -(x_disp - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)
    else:
        return -(accumulated - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)


# ── CMA-ES skill trainer ──────────────────────────────────────────────────────


def _train_skill(
    skill: str,
    hidden_sizes: list[int],
    sigma: float,
    init_scale: float,
    pop: int,
    budget: int,
    workers: int,
    seed: int,
    deadline: float,  # time.perf_counter() deadline — stop when exceeded
) -> tuple[np.ndarray, bool]:
    """Train one skill via CMA-ES; returns (best_weights, timed_out).

    Stops early (saving best-so-far weights) if deadline is reached.
    """
    model, data = _build_loco_world()
    input_dim   = len(get_robot_state(data))
    output_dim  = model.nu
    network     = FlexNetwork(input_size=input_dim, output_size=output_dim,
                              hidden_sizes=hidden_sizes)
    num_params  = sum(p.numel() for p in network.parameters())

    rng = np.random.default_rng(seed)
    x0  = rng.uniform(-init_scale, init_scale, size=num_params).tolist()

    es = cma.CMAEvolutionStrategy(
        x0, sigma,
        {"popsize": pop, "seed": seed, "verbose": -9, "maxiter": 10**9},
    )

    best_fit = float("inf")
    best_w   = np.array(x0, dtype=np.float32)
    timed_out = False

    effective_workers = min(pop, workers)
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=_worker_init,
        initargs=(seed, skill, hidden_sizes),
    ) as pool:
        gen = 0
        while gen < budget:
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            if es.stop():
                break

            solutions = es.ask()
            fits      = list(pool.map(_worker_eval, [s.tolist() for s in solutions]))
            es.tell(solutions, fits)

            gen_best = min(fits)
            if gen_best < best_fit:
                best_fit = gen_best
                best_w   = np.array(solutions[fits.index(gen_best)], dtype=np.float32)

            gen += 1

    label = f"{skill}{'  [timed out]' if timed_out else ''}"
    console.log(f"    {label}: best_fitness={best_fit:.4f}  gens={gen}/{budget}")
    return best_w, timed_out


# ── Food evaluation worker ────────────────────────────────────────────────────

_food_worker_ctx: Optional[dict[str, Any]] = None


def _food_worker_init(
    hidden_sizes: list[int],
    loco_weights: list[float],
    left_weights: list[float],
    right_weights: list[float],
    reach_radius: float,
    collect_radius: float,
    eval_duration: float,
    num_waypoints: int,
    centre_fwd_thresh: float,
) -> None:
    global _food_worker_ctx
    torch.set_num_threads(1)
    model, data, target_mocap_id, cam_name = _build_food_world(reach_radius)
    input_dim  = len(get_robot_state(data))
    output_dim = model.nu

    def _load(w: list[float]) -> FlexNetwork:
        net = FlexNetwork(input_size=input_dim, output_size=output_dim,
                          hidden_sizes=hidden_sizes)
        fill_parameters(net, np.array(w, dtype=np.float32))
        return net

    _food_worker_ctx = {
        "model":           model,
        "data":            data,
        "target_mocap_id": target_mocap_id,
        "cam_name":        cam_name,
        "loco_net":        _load(loco_weights),
        "left_net":        _load(left_weights),
        "right_net":       _load(right_weights),
        "renderer":        mujoco.Renderer(model, height=96, width=128),
        "collect_radius":      collect_radius,
        "eval_duration":       eval_duration,
        "num_waypoints":       num_waypoints,
        "centre_fwd_thresh":   centre_fwd_thresh,
    }


def _food_worker_eval(job: tuple[int, float, int]) -> float:
    """Evaluate one food episode. job = (commit_steps, centre_fwd_thresh, seed)."""
    assert _food_worker_ctx is not None
    ctx            = _food_worker_ctx
    commit_steps, centre_fwd_thresh, seed = job
    model          = ctx["model"]
    data           = ctx["data"]
    target_mocap_id = ctx["target_mocap_id"]
    cam_name       = ctx["cam_name"]
    loco_net       = ctx["loco_net"]
    left_net       = ctx["left_net"]
    right_net      = ctx["right_net"]
    renderer       = ctx["renderer"]
    collect_radius = ctx["collect_radius"]
    eval_duration  = ctx["eval_duration"]
    num_waypoints  = ctx["num_waypoints"]
    centre_fwd_thresh = ctx["centre_fwd_thresh"]

    rng       = np.random.default_rng(seed)
    waypoints = sample_waypoints(rng, num_waypoints)

    mujoco.mj_resetData(model, data)
    current_wp_idx    = 0
    waypoints_reached = 0
    current_target    = waypoints[0]
    data.mocap_pos[target_mocap_id] = current_target
    min_dist = float("inf")

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)
        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= collect_radius:
            waypoints_reached += 1
            current_wp_idx    += 1
            if current_wp_idx < num_waypoints:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist = float("inf")

    controller     = SkillController(rng, commit_steps=commit_steps,
                                     centre_fwd_thresh=centre_fwd_thresh)
    step           = 0
    current_action = np.zeros(model.nu)

    episode_end = SETTLE_DURATION + eval_duration
    while data.time < episode_end and current_wp_idx < num_waypoints:
        if step % CONTROL_STEP_FREQ == 0:
            renderer.update_scene(data, camera=cam_name)
            img    = renderer.render()
            vision = analyze_sections(isolate_green(img))
            left_frac, centre_frac, right_frac = vision
            skill  = controller.select(left_frac, centre_frac, right_frac)

            state = get_robot_state(data).astype(np.float32)
            if skill == 1:
                raw_action = loco_net.forward(state)
            elif skill == 2:
                raw_action = left_net.forward(state)
            else:
                raw_action = right_net.forward(state)

            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -math.pi / 2, math.pi / 2,
            )

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= collect_radius:
            waypoints_reached += 1
            current_wp_idx    += 1
            if current_wp_idx < num_waypoints:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist = float("inf")
        step += 1

    final_dist = 0.0 if waypoints_reached >= num_waypoints else min_dist
    d_norm = float(np.clip((RING_R_MAX - final_dist) / RING_R_MAX, 0.0, 1.0))
    return waypoints_reached + d_norm


# ── JSONL helper ──────────────────────────────────────────────────────────────


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ── Optuna objective ──────────────────────────────────────────────────────────


def make_objective(
    base_seed: int,
    workers: int,
    trial_time_limit: float,
    n_eval_runs: int,
    eval_duration: float,
    num_waypoints: int,
    reach_radius: float,
    collect_radius: float,
    loco_budget_max: int,
    turn_budget_max: int,
):
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        t_trial_start = time.perf_counter()
        # deadline shared across all three CMA-ES phases; eval runs whatever is left
        deadline = t_trial_start + trial_time_limit

        # ── Sample hyperparameters ─────────────────────────────────────────────
        n_layers   = trial.suggest_int("n_layers",    1,    2)
        hidden_1   = trial.suggest_int("hidden_1",    8,   64)
        hidden_2   = trial.suggest_int("hidden_2",    8,   64)
        sigma      = trial.suggest_float("sigma",    0.05, 1.0, log=True)
        init_scale = trial.suggest_float("init_scale", 0.5, 2.0, log=True)
        pop        = trial.suggest_int("pop",         5,   64)
        loco_budget = trial.suggest_int("loco_budget", min(50, loco_budget_max), loco_budget_max, step=50)
        turn_budget = trial.suggest_int("turn_budget", min(50, turn_budget_max), turn_budget_max, step=50)
        commit_steps = trial.suggest_int("commit_steps", 5, 100)
        centre_fwd_thresh = trial.suggest_float("centre_fwd_thresh", 0.1, 0.9)

        hidden_sizes = [hidden_1] if n_layers == 1 else [hidden_1, hidden_2]

        console.rule(
            f"[bold cyan]Trial {trial.number}[/bold cyan]  "
            f"layers={n_layers} hidden={hidden_sizes}  "
            f"sigma={sigma:.3f} init={init_scale:.3f}  "
            f"pop={pop}  loco_budget={loco_budget}  turn_budget={turn_budget}  "
            f"commit={commit_steps}  centre_thresh={centre_fwd_thresh:.2f}"
        )

        trial_seed = base_seed + trial.number * 10_000

        # ── Phase 1: train loco ────────────────────────────────────────────────
        loco_w, loco_to = _train_skill(
            "loco", hidden_sizes, sigma, init_scale, pop,
            loco_budget, workers, trial_seed, deadline,
        )

        # ── Phase 2: train left ────────────────────────────────────────────────
        left_w, left_to = _train_skill(
            "left", hidden_sizes, sigma, init_scale, pop,
            turn_budget, workers, trial_seed + 1, deadline,
        )

        # ── Phase 3: train right ───────────────────────────────────────────────
        right_w, right_to = _train_skill(
            "right", hidden_sizes, sigma, init_scale, pop,
            turn_budget, workers, trial_seed + 2, deadline,
        )

        timed_out_any = loco_to or left_to or right_to

        # ── Phase 4: evaluate SkillController on food task ────────────────────
        eval_seeds = [trial_seed + 3 + i for i in range(n_eval_runs)]
        scores: list[float] = []

        with ProcessPoolExecutor(
            max_workers=min(n_eval_runs, workers),
            initializer=_food_worker_init,
            initargs=(
                hidden_sizes,
                loco_w.tolist(), left_w.tolist(), right_w.tolist(),
                reach_radius, collect_radius, eval_duration, num_waypoints,
                centre_fwd_thresh,
            ),
        ) as pool:
            jobs = [(commit_steps, centre_fwd_thresh, s) for s in eval_seeds]
            scores = list(pool.map(_food_worker_eval, jobs))

        mean_score = float(np.mean(scores))
        std_score  = float(np.std(scores))
        elapsed    = time.perf_counter() - t_trial_start

        console.log(
            f"  Trial {trial.number}: mean_score={mean_score:.3f}±{std_score:.3f}  "
            f"scores={[f'{s:.2f}' for s in scores]}  wall={elapsed:.1f}s"
            + ("  [timed_out]" if timed_out_any else "")
        )

        # ── Save weights ───────────────────────────────────────────────────────
        trial_dir = OUT_DIR / f"trial_{trial.number:04d}"
        trial_dir.mkdir(exist_ok=True)
        np.save(trial_dir / "loco_weights.npy",  loco_w)
        np.save(trial_dir / "left_weights.npy",  left_w)
        np.save(trial_dir / "right_weights.npy", right_w)

        # ── Append JSONL record ────────────────────────────────────────────────
        record: dict[str, Any] = {
            "trial":          trial.number,
            "n_layers":       n_layers,
            "hidden_sizes":   hidden_sizes,
            "sigma":          sigma,
            "init_scale":     init_scale,
            "pop":            pop,
            "loco_budget":    loco_budget,
            "turn_budget":    turn_budget,
            "commit_steps":      commit_steps,
            "centre_fwd_thresh": centre_fwd_thresh,
            "scores":            scores,
            "mean_score":     mean_score,
            "std_score":      std_score,
            "min_score":      float(np.min(scores)),
            "max_score":      float(np.max(scores)),
            "wall_time_s":    elapsed,
            "timed_out":      timed_out_any,
            "weights_dir":    str(trial_dir),
        }
        _append_jsonl(JSONL_PATH, record)

        # Optuna: minimise neg_mean_score and wall_time_s
        return -mean_score, elapsed

    return objective


# ── Pareto summary ────────────────────────────────────────────────────────────


def _print_pareto_summary(study: optuna.Study) -> None:
    pareto = study.best_trials
    if not pareto:
        console.log("[yellow]No Pareto-optimal trials yet.[/yellow]")
        return

    table = Table(title=f"Pareto Front ({len(pareto)} configs)")
    for col in ("Trial", "Layers", "Hidden", "sigma", "pop",
                "loco_bgt", "turn_bgt", "commit", "ctr_thresh", "Mean score", "Wall time (s)"):
        table.add_column(col, justify="right")

    for t in sorted(pareto, key=lambda t: t.values[0]):
        p = t.params
        hs = [p["hidden_1"]] + ([p["hidden_2"]] if p["n_layers"] == 2 else [])
        table.add_row(
            str(t.number), str(p["n_layers"]), str(hs),
            f"{p['sigma']:.3f}", str(p["pop"]),
            str(p["loco_budget"]), str(p["turn_budget"]), str(p["commit_steps"]),
            f"{p['centre_fwd_thresh']:.2f}",
            f"{-t.values[0]:.3f}", f"{t.values[1]:.1f}",
        )

    console.print(table)
    console.log(f"Full results → {JSONL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    torch.set_num_threads(1)

    console.rule("[bold magenta]Gecko Skill-Controller Pipeline HPO[/bold magenta]")
    console.log(
        f"n_trials={args.n_trials}  n_eval_runs={args.n_eval_runs}  "
        f"workers={args.workers}  trial_time_limit={args.trial_time_limit}s  "
        f"eval_duration={args.eval_duration}s  seed={args.seed}"
    )
    console.log(f"Results → {OUT_DIR}")

    storage = JournalStorage(JournalFileBackend(str(STUDY_LOG)))
    study = optuna.create_study(
        study_name="tune_skill_pipeline",
        storage=storage,
        directions=["minimize", "minimize"],  # neg_score, wall_time
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
    )

    objective = make_objective(
        base_seed=args.seed,
        workers=args.workers,
        trial_time_limit=args.trial_time_limit,
        n_eval_runs=args.n_eval_runs,
        eval_duration=args.eval_duration,
        num_waypoints=args.num_waypoints,
        reach_radius=args.reach_radius,
        collect_radius=args.collect_radius,
        loco_budget_max=args.loco_budget_max,
        turn_budget_max=args.turn_budget_max,
    )

    study.optimize(objective, n_trials=args.n_trials, n_jobs=1, show_progress_bar=True)
    _print_pareto_summary(study)


if __name__ == "__main__":
    main()
