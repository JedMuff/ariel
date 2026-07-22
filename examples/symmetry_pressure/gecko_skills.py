"""
Train three motor primitive skills for the fixed gecko morphology using CMA-ES.

  Skill 1 — loco:  maximise forward X displacement (30 s)
  Skill 2 — left:  maximise CCW yaw accumulation   (15 s)
  Skill 3 — right: maximise CW  yaw accumulation   (15 s)

All three use Network(input=11, output=8) trained from scratch.
Weights saved to __data__/gecko_skills/{loco,left,right}_weights.npy.

Usage:
    python gecko_skills.py
    python gecko_skills.py --loco-budget 200 --turn-budget 200 --workers 8
"""

import argparse
import gc
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cma
import mujoco
import numpy as np
import torch
from rich.console import Console
from rich.traceback import install

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

from shared import (
    SPAWN_POSITION,
    Network,
    fill_parameters,
    signed_vertical_yaw_delta,
)

install()
warnings.filterwarnings("ignore", message="TPA: apparent inconsistency",
                        category=UserWarning, module="cma")

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train gecko motor primitive skills")
parser.add_argument("--loco-budget",  type=int,   default=200)
parser.add_argument("--turn-budget",  type=int,   default=200)
parser.add_argument("--sigma",        type=float, default=0.35)
parser.add_argument("--init-scale",   type=float, default=1.3)
parser.add_argument("--workers",      type=int,   default=max(1, os.cpu_count() or 1))
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--skip-loco",    action="store_true", help="Skip loco training (use saved weights)")
parser.add_argument("--skip-left",    action="store_true")
parser.add_argument("--skip-right",   action="store_true")
args = parser.parse_args()

LOCO_BUDGET = args.loco_budget
TURN_BUDGET = args.turn_budget
SIGMA       = args.sigma
INIT_SCALE  = args.init_scale
WORKERS     = args.workers
SEED        = args.seed

DATA = Path.cwd() / "__data__" / "gecko_skills"
DATA.mkdir(exist_ok=True, parents=True)

# ── Episode constants ─────────────────────────────────────────────────────────

SETTLE_DURATION          = 3.0
CTRL_ALPHA               = 0.5
CONTROL_STEP_FREQ        = 50
HINGE_CONTACT_LIMIT      = 200
HINGE_CONTACT_PENALTY    = 0.005
HINGE_GLITCH_FITNESS     = 1.0
HEIGHT_PENALTY_THRESHOLD = 0.21

LOCO_DURATION = 30.0
TURN_DURATION = 15.0

# ── World builder ─────────────────────────────────────────────────────────────


def build_loco_world() -> tuple[mujoco.MjModel, mujoco.MjData]:
    core  = gecko()
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                correct_collision_with_floor=True)
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    return model, data


def _build_rotor_geom_ids(model: mujoco.MjModel) -> set[int]:
    return {i for i in range(model.ngeom) if model.geom(i).name.endswith("-rotor")}


def _floor_geom_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


# ── Episode runners ───────────────────────────────────────────────────────────


def run_loco_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: Network,
    weights: np.ndarray,
) -> float:
    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    rotor_geom_ids = _build_rotor_geom_ids(model)
    floor_id       = _floor_geom_id(model)

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    core_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    x0             = float(data.qpos[0])
    initial_height = float(data.xpos[core_id, 2])

    step = 0
    current_action = np.zeros(model.nu)
    c_hinge = 0
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
        step += 1

    if c_hinge > HINGE_CONTACT_LIMIT:
        return HINGE_GLITCH_FITNESS

    x_displacement = float(data.qpos[0]) - x0
    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    return -(x_displacement - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)


def run_turn_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: Network,
    weights: np.ndarray,
    direction: int,  # +1 = left (CCW), -1 = right (CW)
) -> float:
    """Reward accumulated yaw in the given direction."""
    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    rotor_geom_ids = _build_rotor_geom_ids(model)
    floor_id       = _floor_geom_id(model)

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    core_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    initial_height = float(data.xpos[core_id, 2])

    # Rotation matrix after settling, for tracking incremental turning
    r_prev = np.array(data.xmat[core_id]).reshape(3, 3).copy()

    accumulated    = 0.0
    step           = 0
    current_action = np.zeros(model.nu)
    c_hinge        = 0
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
        step += 1

    if c_hinge > HINGE_CONTACT_LIMIT:
        return HINGE_GLITCH_FITNESS

    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    return -(accumulated - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)


# ── Subprocess worker state ───────────────────────────────────────────────────

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
    return run_loco_episode(
        _loco_ctx["model"], _loco_ctx["data"], _loco_ctx["network"],
        np.array(weights_list, dtype=np.float32),
    )


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
    return run_turn_episode(
        _left_ctx["model"], _left_ctx["data"], _left_ctx["network"],
        np.array(weights_list, dtype=np.float32),
        direction=+1,
    )


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
    return run_turn_episode(
        _right_ctx["model"], _right_ctx["data"], _right_ctx["network"],
        np.array(weights_list, dtype=np.float32),
        direction=-1,
    )


# ── Generic CMA-ES trainer ────────────────────────────────────────────────────


def train_skill(
    name: str,
    budget: int,
    num_params: int,
    worker_fn: Any,
    worker_init: Any,
    seed_offset: int,
) -> np.ndarray:
    console.rule(f"[bold cyan]Skill — {name}[/bold cyan]")
    rng = np.random.default_rng(SEED + seed_offset)
    x0  = rng.uniform(-INIT_SCALE, INIT_SCALE, size=num_params)

    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        SIGMA,
        {"maxiter": budget, "popsize": WORKERS,
         "seed": SEED + seed_offset, "verbose": -9},
    )

    best_fit = float("inf")
    best_w   = x0.copy()
    t0       = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=worker_init,
        initargs=(SEED + seed_offset,),
    ) as pool:
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
                elapsed = time.perf_counter() - t0
                console.log(
                    f"  {name} gen {gen:3d}/{budget}  "
                    f"best={best_fit:.4f}  elapsed={elapsed:.1f}s"
                )
            gen += 1

    elapsed = time.perf_counter() - t0
    console.log(f"  {name} done: best_fitness={best_fit:.4f}  time={elapsed:.1f}s")

    out_path = DATA / f"{name}_weights.npy"
    np.save(out_path, best_w)
    console.log(f"  Saved → {out_path}")
    return best_w


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    torch.set_num_threads(1)

    console.rule("[bold magenta]Gecko Skill Training[/bold magenta]")
    console.log(
        f"loco_budget={LOCO_BUDGET}  turn_budget={TURN_BUDGET}  "
        f"sigma={SIGMA}  init_scale={INIT_SCALE}  workers={WORKERS}  seed={SEED}"
    )

    # Derive network shape from a throw-away world
    model, data = build_loco_world()
    input_dim   = len(get_robot_state(data))
    output_dim  = model.nu
    network     = Network(input_size=input_dim, output_size=output_dim)
    num_params  = sum(p.numel() for p in network.parameters())
    console.log(f"Network: input={input_dim}  output={output_dim}  params={num_params}")

    if not args.skip_loco:
        train_skill("loco",  LOCO_BUDGET, num_params, _worker_loco,  _worker_init_loco,  seed_offset=0)
    else:
        console.log("  [yellow]Skipping loco (--skip-loco)[/yellow]")

    if not args.skip_left:
        train_skill("left",  TURN_BUDGET, num_params, _worker_left,  _worker_init_left,  seed_offset=1)
    else:
        console.log("  [yellow]Skipping left (--skip-left)[/yellow]")

    if not args.skip_right:
        train_skill("right", TURN_BUDGET, num_params, _worker_right, _worker_init_right, seed_offset=2)
    else:
        console.log("  [yellow]Skipping right (--skip-right)[/yellow]")

    console.rule("[bold green]Done[/bold green]")
    console.log(f"Weights saved to {DATA}")


if __name__ == "__main__":
    main()
    gc.disable()
