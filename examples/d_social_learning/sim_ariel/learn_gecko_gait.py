"""Learn a gait for the fixed gecko morphology under the exact same
evaluator settings as the current social-learning experiments (see
ariel/evaluator.py), and report the resulting mean_jerk.

Unlike ariel_gecko_learn.py (older, CTRL_EVERY=100/no jerk penalty/no
settle phase) and gecko_evaluator.py (own stale CTRL_EVERY=100 and
JERK_PENALTY_WEIGHT=0.01 constants, shared with the parallelization
benchmark and NSGA-II tuning script — not touched here), this mirrors
evaluator.py::run_episode's settle phase, control blending, and fitness
formula verbatim (CTRL_EVERY=9, JERK_PENALTY_WEIGHT=3.0), just built on the
prebuilt gecko topology (gecko_graph()) instead of a TreeGenome morphology.

Usage:
    MUJOCO_GL=egl uv run examples/d_social_learning/ariel/learn_gecko_gait.py \
        [--inner-gens 25] [--inner-pop 20] [--sigma 0.45] [--hidden 16] \
        [--out-dir __data__/social/ariel/gecko]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# Append (not insert at position 0) so the installed ariel package always
# resolves before d_social_learning/ariel/, which would otherwise shadow it
# — matches gecko_evaluator.py's sys.path setup.
_SOCIAL_DIR = Path(__file__).parent.parent  # d_social_learning/
if str(_SOCIAL_DIR) not in sys.path:
    sys.path.append(str(_SOCIAL_DIR))

from rich.console import Console

from gecko_evaluator import build_gecko_world

console = Console()

# Must match ariel/evaluator.py exactly.
DURATION = 30.0
SETTLE_TIME = 3.0
CTRL_EVERY = 9  # 500Hz physics / 9 ~= 55.6Hz control
CTRL_ALPHA = 0.5
HEIGHT_PENALTY_THRESHOLD = 0.5
JERK_PENALTY_WEIGHT = 3.0  # evaluator.py's default; overridable via --jerk-weight
N_NEIGHBORS = 6


def _scale_actions(raw: np.ndarray) -> np.ndarray:
    return np.asarray(raw) * (math.pi / 2)


def make_run_episode(
    model, data, adapter, brain,
    jerk_penalty_weight: float = JERK_PENALTY_WEIGHT,
    jerk_threshold: float | None = None,
):
    """evaluator.py::run_episode, verbatim, closed over a gecko model/data."""
    import mujoco

    def run_episode(theta: np.ndarray) -> dict:
        brain.set_theta(theta)
        mujoco.mj_resetData(model, data)

        core_height = float(data.qpos[2])

        sim_step = 0
        while data.time < SETTLE_TIME:
            mujoco.mj_step(model, data)
            sim_step += 1

        ctrl_step = 0
        prev_ctrl = np.zeros(model.nu, dtype=np.float32)
        jerk_sum = 0.0
        rollout_end = SETTLE_TIME + DURATION
        while data.time < rollout_end:
            if sim_step % CTRL_EVERY == 0:
                node_inputs, t = adapter.get_node_inputs(model, data, ctrl_step)
                raw = brain.forward_all(node_inputs, t)
                target_ctrl = _scale_actions(raw)
                new_ctrl = np.clip(
                    prev_ctrl * (1.0 - CTRL_ALPHA) + target_ctrl * CTRL_ALPHA,
                    -np.pi / 2, np.pi / 2,
                ).astype(np.float32)
                if ctrl_step > 0 and model.nu > 0:
                    jerk_sum += float(np.mean(np.abs(new_ctrl - prev_ctrl)))
                prev_ctrl = new_ctrl.copy()
                data.ctrl[:] = new_ctrl
                ctrl_step += 1

            mujoco.mj_step(model, data)
            sim_step += 1

        mean_jerk = jerk_sum / max(ctrl_step - 1, 1)
        d = float(data.qpos[0])
        height_penalty = core_height if core_height > HEIGHT_PENALTY_THRESHOLD else 0.0
        # Hurdle penalty: below jerk_threshold, jerk is free; at/above it, the
        # full (not excess-over-threshold) jerk term applies -- an
        # intentional step discontinuity, not a smoothed ramp.
        if jerk_threshold is not None and mean_jerk < jerk_threshold:
            jerk_penalty = 0.0
        else:
            jerk_penalty = jerk_penalty_weight * mean_jerk
        fitness = d - height_penalty - jerk_penalty

        return {"fitness": fitness, "distance": d, "mean_jerk": mean_jerk}

    return run_episode


def render_best(model, data, adapter, brain, theta, out_path: Path, fps: int = 50) -> None:
    import imageio
    import mujoco

    brain.set_theta(theta)
    mujoco.mj_resetData(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    dt = model.opt.timestep
    render_every = max(1, int(round(1.0 / (fps * dt))))

    frames = []
    sim_step = 0
    while data.time < SETTLE_TIME:
        mujoco.mj_step(model, data)
        if sim_step % render_every == 0:
            renderer.update_scene(data, camera="pretty-cam")
            frames.append(renderer.render())
        sim_step += 1

    ctrl_step = 0
    prev_ctrl = np.zeros(model.nu, dtype=np.float32)
    rollout_end = SETTLE_TIME + DURATION
    while data.time < rollout_end:
        if sim_step % CTRL_EVERY == 0:
            node_inputs, t = adapter.get_node_inputs(model, data, ctrl_step)
            raw = brain.forward_all(node_inputs, t)
            target_ctrl = _scale_actions(raw)
            new_ctrl = np.clip(
                prev_ctrl * (1.0 - CTRL_ALPHA) + target_ctrl * CTRL_ALPHA,
                -np.pi / 2, np.pi / 2,
            ).astype(np.float32)
            prev_ctrl = new_ctrl.copy()
            data.ctrl[:] = new_ctrl
            ctrl_step += 1
        mujoco.mj_step(model, data)
        if sim_step % render_every == 0:
            renderer.update_scene(data, camera="pretty-cam")
            frames.append(renderer.render())
        sim_step += 1

    renderer.close()
    imageio.mimsave(str(out_path), frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-gens", type=int, default=25)
    parser.add_argument("--inner-pop", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.45)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument(
        "--jerk-weight", type=float, default=JERK_PENALTY_WEIGHT,
        help="Jerk penalty weight (evaluator.py default 3.0). Pass 0 to disable it.",
    )
    parser.add_argument(
        "--jerk-threshold", type=float, default=None,
        help="Hurdle penalty: below this mean_jerk, no penalty at all; at/above "
             "it, the full jerk_weight * mean_jerk term applies (a step, not a "
             "smoothed ramp). Omit for a plain linear penalty everywhere.",
    )
    parser.add_argument("--out-dir", default="__data__/social/ariel/gecko")
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Gecko gait learning (current evaluator.py settings)[/bold cyan]")
    console.log(
        f"inner_gens={args.inner_gens} inner_pop={args.inner_pop} "
        f"sigma={args.sigma} hidden={args.hidden} "
        f"CTRL_EVERY={CTRL_EVERY} JERK_PENALTY_WEIGHT={args.jerk_weight} "
        f"JERK_THRESHOLD={args.jerk_threshold}"
    )

    model, data, adapter, brain, _hinge_geom_ids, _floor_geom_id = build_gecko_world(args.hidden)
    run_episode = make_run_episode(
        model, data, adapter, brain,
        jerk_penalty_weight=args.jerk_weight, jerk_threshold=args.jerk_threshold,
    )

    from ariel.simulation.controllers.cmaes_learner import CMAESLearner

    learner = CMAESLearner(
        n_params=brain.n_params,
        init_mean=None,  # cold start (zeros) -- gecko has no prior individual to inherit from
        sigma=args.sigma,
        pop_size=args.inner_pop,
    )

    t0 = time.perf_counter()
    for gen in range(args.inner_gens):
        candidates = learner.ask()
        eps = [run_episode(theta) for theta in candidates]
        fitnesses = [ep["fitness"] for ep in eps]
        learner.tell(candidates, fitnesses)
        gen_best = max(fitnesses)
        console.log(
            f"gen {gen + 1:3d}/{args.inner_gens}  "
            f"running_best={learner.best_fitness:.4f}  gen_best={gen_best:.4f}  "
            f"gen_mean={np.mean(fitnesses):.4f}"
        )
    wall_time_s = time.perf_counter() - t0

    best_ep = run_episode(np.asarray(learner.best_theta, dtype=np.float64))

    console.rule("[bold green]Result[/bold green]")
    console.log(f"best fitness   = {best_ep['fitness']:.4f}")
    console.log(f"distance       = {best_ep['distance']:.4f}")
    console.log(f"mean_jerk      = {best_ep['mean_jerk']:.4f}")
    console.log(f"wall time      = {wall_time_s:.1f}s")

    theta_path = out_dir / "gecko_best_theta.npy"
    np.save(theta_path, learner.best_theta)
    console.log(f"[green]Saved theta -> {theta_path}[/green]")

    video_path = out_dir / f"gecko_fit{best_ep['fitness']:.3f}_jerk{best_ep['mean_jerk']:.3f}.mp4"
    render_best(model, data, adapter, brain, learner.best_theta, video_path, fps=args.fps)
    console.log(f"[green]Saved video -> {video_path}[/green]")


if __name__ == "__main__":
    main()
