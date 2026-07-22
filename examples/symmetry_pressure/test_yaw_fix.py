"""
Test a robust replacement for the turn-skill yaw metric in gecko_food_skills.py.

Background: the current metric measures yaw as atan2(xmat[3], xmat[0]) — i.e.
treats R00/R10 of the core's rotation matrix as if they were a pure
rotation-about-Z. For a body that stays upright this is fine, but for a
tumbling/jittering body it is not a heading at all, and the per-step
accumulation (`max(0.0, delta * direction)`, never subtracting the reverse
direction) turns any chaotic wobble into a monotonically growing "fitness"
completely disconnected from real turning. Several evolved bodies (dense,
tangled ~25-module blobs) exploited exactly this: near-zero net displacement,
near-zero camera motion, yet accumulated "yaw" in the tens of thousands of
degrees over a 15s episode.

Fix under test: measure the actual geodesic rotation between consecutive
frames (R_rel = R_curr @ R_prev.T, angle = arccos((trace(R_rel)-1)/2)), then
take only its component about the *world* vertical axis (via the rotation
axis extracted from R_rel's antisymmetric part, dotted with world Z). This
is convention-free -- it doesn't assume which local body axis is "up" or
"forward" the way the original atan2(xmat[3], xmat[0]) did, so it can't be
fooled by a body whose core happens to be oriented sideways relative to the
"standard" gecko spawn. Rotation about a horizontal axis (tumbling) or
translation contributes ~0; only genuine turning about vertical accumulates.

Two checks:
  1. Replay: the 3 known exploit checkpoints (run 31983) should now register
     near-zero accumulated turn fitness (old metric gave -24000..-28000 deg).
  2. Training: run a short CMA-ES turn-training loop on the plain gecko body
     with the fixed metric and confirm it can still learn a real turn (i.e.
     the metric isn't so conservative it can no longer be optimized).

Usage:
    python test_yaw_fix.py replay
    python test_yaw_fix.py train --budget 40 --popsize 16
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cma
import mujoco
import numpy as np

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

from shared import Network, fill_parameters, genome_to_spec

# ── Episode constants (match gecko_food_skills.py) ───────────────────────────

HIDDEN_SIZES      = [32]
CONTROL_STEP_FREQ = 50
CTRL_ALPHA        = 0.5
SETTLE_DURATION   = 3.0
TURN_DURATION     = 15.0
CMA_SIGMA         = 0.7
CMA_POPSIZE       = 16
CMA_INIT_SCALE    = 1.3

SPAWN_POSITION = (0.0, 0.0, 0.1)


def old_yaw(xmat: np.ndarray) -> float:
    """Current (buggy) metric: treats R00/R10 as if rotation were pure yaw."""
    return math.atan2(float(xmat[3]), float(xmat[0]))


def signed_vertical_yaw_delta(R_prev: np.ndarray, R_curr: np.ndarray) -> float:
    """Signed rotation-about-world-Z component of the incremental rotation
    R_prev -> R_curr. Convention-free: does not assume which local body axis
    is "up" or "forward", so it stays correct even if the core's frame is
    rotated relative to the "standard" gecko spawn orientation. Tumbling /
    rotation about a horizontal axis contributes ~0.
    """
    R_rel = R_curr @ R_prev.T
    cos_theta = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos_theta)
    if angle < 1e-9:
        return 0.0
    # Rotation axis from the antisymmetric part: (R - R^T) = 2 sin(theta) [axis]_x
    ax = np.array([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1],
    ])
    sin_theta = math.sin(angle)
    if abs(sin_theta) < 1e-9:
        return 0.0
    axis_z = ax[2] / (2.0 * sin_theta)
    return angle * axis_z


def build_world(genome_dict: dict | None):
    if genome_dict is None:
        core = gecko()
        world = SimpleFlatWorld()
        world.spawn(core.spec, position=SPAWN_POSITION, rotation=(0, 0, 90),
                    correct_collision_with_floor=True)
    else:
        spec = genome_to_spec(genome_dict)
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
    return model, data


def run_turn_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: Network,
    direction: int,
    duration: float = TURN_DURATION,
) -> dict:
    """Replay one turn episode, tracking both old and new yaw metrics
    side by side so they can be compared directly."""
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    mujoco.mj_resetData(model, data)

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    xmat = data.xmat[core_id]
    old_prev = old_yaw(xmat)
    R_prev = np.array(xmat).reshape(3, 3).copy()

    old_accum = 0.0
    new_accum = 0.0
    total_unsigned_angle = 0.0

    step = 0
    action = np.zeros(model.nu)
    episode_end = SETTLE_DURATION + duration

    while data.time < episode_end:
        if step % CONTROL_STEP_FREQ == 0:
            state = get_robot_state(data).astype(np.float32)
            raw_action = network.forward(model, data, state)
            action = np.clip(action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                              -math.pi / 2, math.pi / 2)
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        step += 1

        xmat = data.xmat[core_id]

        old_curr = old_yaw(xmat)
        old_delta = (old_curr - old_prev + math.pi) % (2 * math.pi) - math.pi
        old_accum += max(0.0, old_delta * direction)
        old_prev = old_curr

        R_curr = np.array(xmat).reshape(3, 3)
        new_delta = signed_vertical_yaw_delta(R_prev, R_curr)
        new_accum += max(0.0, new_delta * direction)
        cos_theta = float(np.clip((np.trace(R_curr @ R_prev.T) - 1.0) / 2.0, -1.0, 1.0))
        total_unsigned_angle += math.acos(cos_theta)
        R_prev = R_curr.copy()

    return {
        "old_accum_deg": math.degrees(old_accum),
        "new_accum_deg": math.degrees(new_accum),
        "total_unsigned_deg": math.degrees(total_unsigned_angle),
        "final_xy": [float(data.qpos[0]), float(data.qpos[1])],
    }


# ── Mode 1: replay known exploit checkpoints ─────────────────────────────────

EXPLOIT_CHECKPOINTS = [
    ("data/food_skils/food_skills_20260721_142257_31983/__data__/gecko_food_skills/checkpoints/gen001_body13", "left"),
    ("data/food_skils/food_skills_20260721_142257_31983/__data__/gecko_food_skills/checkpoints/gen001_body09", "left"),
    ("data/food_skils/food_skills_20260721_142257_31983/__data__/gecko_food_skills/checkpoints/gen000_body17", "left"),
]


def cmd_replay() -> None:
    print(f"{'checkpoint':16s} {'skill':6s} {'old_deg':>12s} {'new_deg':>10s} {'total_unsigned_deg':>19s}")
    for ckpt, skill in EXPLOIT_CHECKPOINTS:
        ckpt_dir = Path(ckpt)
        genome = json.loads((ckpt_dir / "best_genome.json").read_text())
        weights = np.load(ckpt_dir / f"{skill}_weights.npy")

        model, data = build_world(genome)
        input_dim = len(get_robot_state(data))
        output_dim = model.nu
        net = Network(input_size=input_dim, output_size=output_dim, hidden_size=HIDDEN_SIZES[0])
        fill_parameters(net, weights.astype(np.float32))

        direction = +1 if skill == "left" else -1
        res = run_turn_episode(model, data, net, direction)
        print(f"{ckpt_dir.name:16s} {skill:6s} {res['old_accum_deg']:12.1f} "
              f"{res['new_accum_deg']:10.2f} {res['total_unsigned_deg']:19.2f}")


# ── Mode 2: train the plain gecko body with the fixed metric ────────────────


def _new_metric_fitness(model, data, network, direction, height_penalty_thresh=0.21):
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    mujoco.mj_resetData(model, data)
    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    initial_height = float(data.xpos[core_id, 2])
    R_prev = np.array(data.xmat[core_id]).reshape(3, 3).copy()
    accumulated = 0.0

    step = 0
    action = np.zeros(model.nu)
    episode_end = SETTLE_DURATION + TURN_DURATION
    while data.time < episode_end:
        if step % CONTROL_STEP_FREQ == 0:
            state = get_robot_state(data).astype(np.float32)
            raw_action = network.forward(model, data, state)
            action = np.clip(action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                              -math.pi / 2, math.pi / 2)
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        step += 1

        R_curr = np.array(data.xmat[core_id]).reshape(3, 3)
        delta = signed_vertical_yaw_delta(R_prev, R_curr)
        accumulated += max(0.0, delta * direction)
        R_prev = R_curr.copy()

    height_penalty = initial_height if initial_height > height_penalty_thresh else 0.0
    return -(accumulated - height_penalty)


def cmd_train(budget: int, popsize: int, skill: str, save_weights: Path | None = None) -> None:
    direction = +1 if skill == "left" else -1
    model, data = build_world(None)
    input_dim = len(get_robot_state(data))
    output_dim = model.nu
    network = Network(input_size=input_dim, output_size=output_dim, hidden_size=HIDDEN_SIZES[0])
    num_params = sum(p.numel() for p in network.parameters())

    rng = np.random.default_rng(42)
    x0 = rng.uniform(-CMA_INIT_SCALE, CMA_INIT_SCALE, size=num_params).tolist()
    es = cma.CMAEvolutionStrategy(
        x0, CMA_SIGMA, {"popsize": popsize, "seed": 42, "verbose": -9, "maxiter": 10**9},
    )

    best_fit = float("inf")
    best_w = np.array(x0, dtype=np.float32)

    gen = 0
    while gen < budget and not es.stop():
        solutions = es.ask()
        fits = []
        for sol in solutions:
            fill_parameters(network, np.array(sol, dtype=np.float32))
            fits.append(_new_metric_fitness(model, data, network, direction))
        es.tell(solutions, fits)
        gen_best = min(fits)
        if gen_best < best_fit:
            best_fit = gen_best
            best_w = np.array(solutions[fits.index(gen_best)], dtype=np.float32)
        print(f"  gen {gen+1:3d}/{budget}  gen_best={gen_best:9.4f}  best_so_far={best_fit:9.4f}")
        gen += 1

    print(f"\nBest fitness (new metric, negated accum rad): {best_fit:.4f}")
    fill_parameters(network, best_w)
    res = run_turn_episode(model, data, network, direction)
    print(f"Re-check with instrumented replay: old_deg={res['old_accum_deg']:.1f}  "
          f"new_deg={res['new_accum_deg']:.1f}  total_unsigned_deg={res['total_unsigned_deg']:.1f}  "
          f"final_xy={res['final_xy']}")

    if save_weights is not None:
        save_weights.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_weights, best_w)
        print(f"Saved best weights -> {save_weights}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("replay")
    p_train = sub.add_parser("train")
    p_train.add_argument("--budget", type=int, default=40)
    p_train.add_argument("--popsize", type=int, default=16)
    p_train.add_argument("--skill", choices=["left", "right"], default="left")
    p_train.add_argument("--save-weights", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "replay":
        cmd_replay()
    else:
        cmd_train(args.budget, args.popsize, args.skill, args.save_weights)


if __name__ == "__main__":
    main()
