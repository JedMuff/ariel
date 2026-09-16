"""Render videos of gecko controllers from the NSGA-II tuning study.

Loads saved theta from __data__/tuning/weights/trial_<n>.npz and replays it
using gecko_evaluator.py's exact rollout (CTRL_ALPHA smoothing, ctrl_every,
etc.) so the video matches what was actually optimized, then renders frames
with MuJoCo's offscreen renderer.

Usage:
    uv run examples/d_social_learning/analysis/make_tuning_videos.py \
        --trials 30 98 58 42 49 \
        [--weights-dir __data__/tuning/weights] \
        [--out-dir __data__/tuning/videos] \
        [--duration 15.0] [--fps 30]

    # or auto-pick the Pareto front from the study:
    uv run examples/d_social_learning/analysis/make_tuning_videos.py --pareto \
        [--storage sqlite:///__data__/tuning/gecko_nsga2.db] [--study-name gecko_nsga2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent
_ARIEL_DIR = _THIS_DIR.parent / "ariel"
if str(_ARIEL_DIR) not in sys.path:
    sys.path.insert(0, str(_ARIEL_DIR))

import imageio
import mujoco
import numpy as np
from rich.console import Console

import gecko_evaluator as ge

console = Console()

FPS = 30


def render_trial(trial_num: int, weights_dir: Path, out_dir: Path, duration: float, fps: int) -> None:
    npz_path = weights_dir / f"trial_{trial_num}.npz"
    if not npz_path.exists():
        console.log(f"[yellow]Skipping trial {trial_num}: {npz_path} not found[/yellow]")
        return

    d = np.load(npz_path)
    theta = d["theta"]
    fitness = float(d["fitness"])
    time_s = float(d["time_s"])
    sigma = float(d["sigma"])
    pop_size = int(d["pop_size"])
    inner_gens = int(d["inner_gens"])
    hidden = int(d["hidden"])

    console.log(
        f"trial {trial_num}: fitness={fitness:.4f} time={time_s:.1f}s "
        f"(sigma={sigma:.3f} pop_size={pop_size} inner_gens={inner_gens} hidden={hidden})"
    )

    model, data, adapter, brain, hinge_geom_ids, floor_geom_id = ge.build_gecko_world(hidden=hidden)
    brain.set_theta(theta)
    mujoco.mj_resetData(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)

    dt = model.opt.timestep
    render_every = max(1, int(round(1.0 / (fps * dt))))

    sim_step = 0
    ctrl_step = 0
    prev_ctrl = np.zeros(model.nu, dtype=np.float32)
    frames = []

    rollout_end = ge.SETTLE_TIME + duration
    while data.time < rollout_end:
        if sim_step % ge.CTRL_EVERY == 0:
            node_inputs, t = adapter.get_node_inputs(model, data, ctrl_step)
            raw = brain.forward_all(node_inputs, t)
            target_ctrl = ge._scale_actions(raw)
            new_ctrl = np.clip(
                prev_ctrl * (1.0 - ge.CTRL_ALPHA) + target_ctrl * ge.CTRL_ALPHA,
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

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trial_{trial_num}_fit{fitness:.3f}_t{time_s:.0f}s.mp4"
    imageio.mimsave(out_path, frames, fps=fps)
    console.log(f"[green]Saved -> {out_path}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, nargs="*", default=None, help="Trial numbers to render")
    parser.add_argument("--pareto", action="store_true", help="Render every trial on the Pareto front")
    parser.add_argument("--study-name", type=str, default="gecko_nsga2")
    parser.add_argument("--storage", type=str, default="sqlite:///__data__/tuning/gecko_nsga2.db")
    parser.add_argument("--weights-dir", type=Path, default=Path("__data__/tuning/weights"))
    parser.add_argument("--out-dir", type=Path, default=Path("__data__/tuning/videos"))
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    if args.pareto:
        import optuna
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
        trial_nums = sorted(t.number for t in study.best_trials)
        console.log(f"Pareto front: {len(trial_nums)} trials -> {trial_nums}")
    elif args.trials:
        trial_nums = args.trials
    else:
        parser.error("Provide --trials <n...> or --pareto")
        return

    for n in trial_nums:
        render_trial(n, args.weights_dir, args.out_dir, args.duration, args.fps)


if __name__ == "__main__":
    main()
