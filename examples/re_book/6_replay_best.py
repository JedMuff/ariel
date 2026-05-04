"""
Replay the best-of-run individual from a body+brain randomized-waypoints run
and record a video.

A checkpoint dir is written by 6_body_brain_randomized_waypoints.py whenever
a new global-best body/brain is found, and contains:
    best_genome.json   — TreeGenome morphology
    best_weights.npy   — flat NN weight vector
    best_waypoints.npy — (N, 3) waypoint layout for the gen the best was found

Usage
-----
    # Replay best-of-run for every run found under __data__/body_brain_*/
    uv run python examples/re_book/6_replay_best.py

    # Replay one specific checkpoint
    uv run python examples/re_book/6_replay_best.py --checkpoint \\
        __data__/body_brain_14288_0/__data__/6_body_brain_randomized_waypoints/checkpoints/gen011_body14

    # Replay best-of-run for one run dir
    uv run python examples/re_book/6_replay_best.py --run-dir __data__/body_brain_14288_0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from glob import glob
from pathlib import Path

# Headless rendering by default — only set MUJOCO_GL if user hasn't already.
os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")

import cv2
import mujoco
import numpy as np

# Reuse helpers from the experiment script directly so the simulation matches.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def _import_experiment(reach_radius: float):
    """Import the experiment module with a chosen reach_radius.

    The module reads REACH_RADIUS from argparse at import time and uses it
    to size the gate cylinder geom in `_build_world_for_body`. Reload the
    module each time so the visual matches the test threshold.
    """
    _orig_argv = sys.argv
    sys.argv = [str(SCRIPT_DIR / "6_body_brain_randomized_waypoints.py"),
                "--no-video", "--reach-radius", str(reach_radius)]
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "_bb_exp", SCRIPT_DIR / "6_body_brain_randomized_waypoints.py"
        )
        mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
        _spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    finally:
        sys.argv = _orig_argv


# Imported lazily on first call to replay_checkpoint so the user's reach_radius
# choice flows through to the gate geom size as well as the trigger test.
_bb = None

from ariel.utils.video_recorder import VideoRecorder

warnings.filterwarnings("ignore", category=UserWarning, module="cma")


def find_best_checkpoint(run_dir: Path) -> Path | None:
    """The last (highest gen) checkpoint subdir IS the global best of the run,
    because checkpoints are only written when fitness improves."""
    candidates = sorted((run_dir / "__data__").glob("*/checkpoints/gen*_body*"))
    return candidates[-1] if candidates else None


def discover_run_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("body_brain_*") if p.is_dir())


def replay_checkpoint(checkpoint: Path, duration: float, fps: int,
                      output_dir: Path, label: str,
                      reach_radius: float = 0.20) -> Path | None:
    global _bb
    if _bb is None or getattr(_bb, "REACH_RADIUS", None) != reach_radius:
        _bb = _import_experiment(reach_radius)

    genome_path    = checkpoint / "best_genome.json"
    weights_path   = checkpoint / "best_weights.npy"
    waypoints_path = checkpoint / "best_waypoints.npy"
    for p in (genome_path, weights_path, waypoints_path):
        if not p.exists():
            print(f"[skip] missing {p}")
            return None

    genome    = json.loads(genome_path.read_text())
    weights   = np.load(weights_path)
    waypoints = [np.asarray(w) for w in np.load(waypoints_path)]

    model, data, target_mocap_id, cam_name = _bb._build_world_for_body(genome)
    if model.nu == 0:
        print(f"[skip] {label}: body has no actuators")
        return None

    input_dim = _bb._genome_input_dim(model, data)
    net = _bb.Network(input_size=input_dim, output_size=model.nu)
    _bb.fill_parameters(net, weights)

    num_wps = len(waypoints)
    wp_idx = 0
    current_target = waypoints[0]
    mujoco.mj_resetData(model, data)
    data.mocap_pos[target_mocap_id] = current_target

    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = VideoRecorder(file_name=f"replay_{label}", output_folder=str(output_dir))

    # Match the brain's vision render: 96×128 RGB, exactly what the network sees.
    POV_H, POV_W = 48 * 2, 64 * 2
    pov_recorder = VideoRecorder(
        file_name=f"replay_{label}_pov",
        output_folder=str(output_dir),
        width=POV_W,
        height=POV_H * 2,  # raw on top, mask on bottom
    )

    dt = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
    control_step_freq = 50
    current_ctrl = np.zeros(model.nu)
    render_step = 0

    control_renderer = mujoco.Renderer(model, height=POV_H, width=POV_W)

    # Closure cells holding the most recent raw frame + mask, refreshed each
    # control step. The POV video re-uses these between control steps so each
    # frame the brain "saw" is held for control_step_freq physics steps.
    latest = {"raw": np.zeros((POV_H, POV_W, 3), dtype=np.uint8),
              "mask": np.zeros((POV_H, POV_W),    dtype=np.uint8)}

    def get_ctrl(d: mujoco.MjData) -> np.ndarray:
        control_renderer.update_scene(d, camera=cam_name)
        img    = control_renderer.render()
        mask   = _bb.isolate_green(img)
        latest["raw"]  = img
        latest["mask"] = mask
        vision = _bb.analyze_sections(mask)
        rs     = _bb.get_robot_state(d)
        phase  = [2.0 * np.sin(d.time * 2.0 * np.pi),
                  2.0 * np.cos(d.time * 2.0 * np.pi)]
        prog   = [wp_idx / max(num_wps - 1, 1)]
        state  = np.concatenate([rs, vision, phase, prog]).astype(np.float32)
        return net.forward(model, d, state)

    def write_pov_frame() -> None:
        # analyze_sections splits along axis=1 into 3 equal columns. With
        # width=POV_W the boundaries are at floor(POV_W/3) and floor(2*POV_W/3).
        x1, x2 = POV_W // 3, (2 * POV_W) // 3
        raw  = latest["raw"].copy()                       # (H, W, 3) RGB
        mask = cv2.cvtColor(latest["mask"], cv2.COLOR_GRAY2RGB)  # (H, W, 3)
        # Section dividers: red on raw, magenta on mask (visible against B&W).
        for x in (x1, x2):
            raw[:,  x, :]  = (255, 0, 0)
            mask[:, x, :]  = (255, 0, 255)
        stacked = np.vstack([raw, mask])                  # (2H, W, 3) RGB
        pov_recorder.write(stacked)

    overview_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview_cam")

    min_dist = float("inf")
    completion_time: float | None = None

    with mujoco.Renderer(model, height=480, width=640) as renderer:
        while data.time < duration and wp_idx < num_wps:
            for _ in range(steps_per_frame):
                if render_step % control_step_freq == 0:
                    current_ctrl = get_ctrl(data)
                np.copyto(data.ctrl, current_ctrl)
                mujoco.mj_step(model, data)
                render_step += 1

                if wp_idx < num_wps:
                    dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
                    min_dist = min(min_dist, dist)
                    if dist <= reach_radius:
                        wp_idx += 1
                        if wp_idx < num_wps:
                            current_target = waypoints[wp_idx]
                            data.mocap_pos[target_mocap_id] = current_target
                            min_dist = float("inf")
                        else:
                            completion_time = float(data.time)

            renderer.update_scene(data, camera=overview_id)
            recorder.write(renderer.render())
            write_pov_frame()

    recorder.release()
    pov_recorder.release()
    control_renderer.close()

    final_dist = 0.0 if wp_idx >= num_wps else min_dist
    fitness = _bb.compute_fitness(
        waypoints_reached=wp_idx,
        min_dist_to_current=final_dist,
        num_waypoints=num_wps,
        completion_time=completion_time,
        duration=duration,
    )
    out_file = output_dir / f"replay_{label}.mp4"
    print(f"  {label}: reached {wp_idx}/{num_wps}  fitness={fitness:.3f}  → {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", help="Single checkpoint dir to replay.")
    parser.add_argument("--run-dir", action="append", default=[],
                        help="Run dir (e.g. __data__/body_brain_14288_0). Repeatable.")
    parser.add_argument("--data-root", default="__data__",
                        help="Auto-discover all body_brain_* runs under this dir.")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--reach-radius", type=float, default=0.20,
                        help="Trigger radius (m). Use 0.35 to replay against pre-2026-05 brains.")
    parser.add_argument("--output", default="__data__/6_replays",
                        help="Where to drop the .mp4 files.")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        label = ckpt.parent.parent.parent.parent.name + "_" + ckpt.name  # run_<gen_body>
        replay_checkpoint(ckpt, args.duration, args.fps, output_dir, label,
                          reach_radius=args.reach_radius)
        return

    run_dirs = [Path(p) for p in args.run_dir] or discover_run_dirs(Path(args.data_root))
    if not run_dirs:
        raise SystemExit(f"No run dirs found. Pass --run-dir or check {args.data_root}/")

    print(f"Replaying best-of-run for {len(run_dirs)} run(s):")
    for run in run_dirs:
        ckpt = find_best_checkpoint(run)
        if ckpt is None:
            print(f"  {run.name}: no checkpoints found")
            continue
        label = f"{run.name}_{ckpt.name}"
        replay_checkpoint(ckpt, args.duration, args.fps, output_dir, label,
                          reach_radius=args.reach_radius)


if __name__ == "__main__":
    main()
