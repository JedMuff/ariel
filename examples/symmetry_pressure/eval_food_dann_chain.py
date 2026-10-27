"""
Evaluate a gecko_food_dann.py controller on the original food task.

The original task is the food condition of slurm/run_symmetry_pressure_sweep.sh
(gecko_food_skills._run_food_episode): one 120 s episode with a chain of 10
foods, shown one at a time. The first food is 1.5 m straight ahead; the rest
come from shared.sample_waypoints (each 0.5-1 m from the previous waypoint,
any angle, chained from the spawn point), and each appears as soon as the
previous one is collected. The original's ahead is world -Y (the zoo camera's
side view); this world's camera looks along +X, so the waypoints are rotated
by +90 deg to keep the first food straight ahead.

Metrics per episode match the original: food collected, d_norm
(1 - closest approach to the uncollected food / 1 m, clipped to [0, 1]) and
fitness = food + d_norm. --blind also runs every episode with the vision input
zeroed, to show how much the controller relies on it.

Usage:
  python eval_food_dann_chain.py <run>/__data__/gecko_food_dann [--episodes 50] [--blind] [--video 3 --out dir]
"""

import argparse
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco
import numpy as np

import gecko_food_dann as g
from shared import GATE_HALF_HEIGHT, RING_R_MAX, get_robot_state, sample_waypoints

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=Path, help="A run's __data__/gecko_food_dann directory")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--num-waypoints", type=int, default=10)
parser.add_argument("--duration", type=float, default=120.0, help="Episode length after the settle (s)")
parser.add_argument("--seed", type=int, default=1000, help="Episode k uses waypoint seed seed + k")
parser.add_argument("--blind", action="store_true", help="Also run every episode with vision zeroed")
parser.add_argument("--workers", type=int, default=os.cpu_count())
parser.add_argument("--video", type=int, default=0, help="Render the first N episodes")
parser.add_argument("--out", type=Path, default=None, help="Defaults to <data_dir>/chain_eval")
args = parser.parse_args()

config = json.loads((args.data_dir / "run_config.json").read_text())
weights = np.load(args.data_dir / "best_weights.npy")
out_dir = args.out or args.data_dir / "chain_eval"


def waypoints(seed: int) -> list[np.ndarray]:
    """The original gecko_food_skills chain, rotated so 'ahead' is +X."""
    wps = [np.array([0.0, -1.5, GATE_HALF_HEIGHT])]
    wps += sample_waypoints(np.random.default_rng(seed), n=args.num_waypoints - 1)
    return [np.array([-w[1], w[0], w[2]]) for w in wps]


def run_chain(ctx: dict[str, Any], wps: list[np.ndarray], blind: bool = False,
              recorder: Any = None) -> dict[str, Any]:
    model, data, brain, vision_fn = ctx["model"], ctx["data"], ctx["brain"], ctx["vision"]
    brain.reset()
    mujoco.mj_resetData(model, data)
    idx, min_dist, collect_times = 0, float("inf"), []

    def check() -> None:
        nonlocal idx, min_dist
        dist = math.hypot(data.qpos[0] - wps[idx][0], data.qpos[1] - wps[idx][1])
        min_dist = min(min_dist, dist)
        if dist <= ctx["reach_radius"]:
            collect_times.append(float(data.time) - g.SETTLE_DURATION)
            idx += 1
            min_dist = float("inf")
            if idx < len(wps):
                data.mocap_pos[ctx["target_mocap_id"]] = wps[idx]

    data.mocap_pos[ctx["target_mocap_id"]] = wps[0]
    while data.time < g.SETTLE_DURATION and idx < len(wps):  # as in the original
        mujoco.mj_step(model, data)
        check()

    step, action = 0, np.zeros(model.nu)
    trajectory = []
    vision = np.zeros(g.N_VISION)
    while data.time < g.SETTLE_DURATION + args.duration and idx < len(wps):
        if step % g.PROFILE.control_step_freq == 0:
            vision = vision_fn(data)
            seen = np.zeros(g.N_VISION) if blind else vision
            raw = brain.act(data, get_robot_state(data).astype(np.float32), seen)
            action = np.clip(action * (1.0 - g.PROFILE.ctrl_alpha) + raw * g.PROFILE.ctrl_alpha,
                             -math.pi / 2, math.pi / 2)
            trajectory.append([float(data.qpos[0]), float(data.qpos[1])])
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        step += 1
        check()
        if recorder is not None:
            recorder(data, vision, idx, collect_times)

    food = len(collect_times)
    final_dist = 0.0 if food >= len(wps) else min_dist
    d_norm = float(np.clip((RING_R_MAX - final_dist) / RING_R_MAX, 0.0, 1.0))
    return {"food": food, "d_norm": d_norm, "fitness": food + d_norm, "collect_times": collect_times,
            "fell": bool(data.qpos[2] < 0.03), "trajectory": trajectory,
            "waypoints": [w.tolist() for w in wps]}


_ctx: dict[str, Any] | None = None


def _init() -> None:
    global _ctx  # noqa: PLW0603
    g.torch.set_num_threads(1)
    _ctx = g.make_ctx(config["reach_radius"], config["subtask_time"], config.get("vision_norm", "raw"),
                      config.get("hidden", 16))
    _ctx["brain"].set_params(weights)


def _eval(job: tuple[int, bool]) -> dict[str, Any]:
    assert _ctx is not None
    seed, blind = job
    return run_chain(_ctx, waypoints(seed), blind) | {"seed": seed, "blind": blind}


def render(seed: int, path: Path) -> dict[str, Any]:
    import cv2  # noqa: PLC0415

    ctx = g.make_ctx(config["reach_radius"], config["subtask_time"], config.get("vision_norm", "raw"),
                     config.get("hidden", 16))
    ctx["brain"].set_params(weights)
    model = ctx["model"]
    wps = waypoints(seed)
    core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    h, w, fps = 480, 640, 30
    scene, fpv = mujoco.Renderer(model, h, w), mujoco.Renderer(model, h // 4, w // 4)
    every = max(1, round(1.0 / (model.opt.timestep * fps)))
    frames: list[np.ndarray] = []
    step = [0]

    def text(img: np.ndarray, s: str, y: int) -> None:
        for color, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick, cv2.LINE_AA)

    def recorder(d: mujoco.MjData, vision: np.ndarray, idx: int, times: list[float]) -> None:
        step[0] += 1
        if step[0] % every:
            return
        target = wps[min(idx, len(wps) - 1)]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (d.xpos[core] + target) / 2
        cam.distance = 2.4 + 0.5 * math.hypot(*(d.xpos[core][:2] - target[:2]))
        cam.azimuth, cam.elevation = 90.0, -50.0
        scene.update_scene(d, camera=cam)
        img = cv2.cvtColor(scene.render(), cv2.COLOR_RGB2BGR)
        fpv.update_scene(d, camera=ctx["cam_name"])
        inset = cv2.cvtColor(fpv.render(), cv2.COLOR_RGB2BGR)
        cv2.rectangle(inset, (0, 0), (w // 4 - 1, h // 4 - 1), (255, 255, 255), 2)
        img[-h // 4 - 10:-10, -w // 4 - 10:-10] = inset
        text(img, f"chain seed {seed}: food {idx}/{len(wps)} collected", 25)
        text(img, f"t = {d.time - g.SETTLE_DURATION:5.1f} / {args.duration:.0f} s", 50)
        text(img, f"vision L/C/R  {vision[0]:.2f} {vision[1]:.2f} {vision[2]:.2f}", 75)
        frames.append(img)

    result = run_chain(ctx, wps, recorder=recorder)
    raw = path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames + [frames[-1]] * fps:
        writer.write(f)
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(path)], check=True)
    raw.unlink()
    return result


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    food = np.array([r["food"] for r in rows])
    fit = np.array([r["fitness"] for r in rows])
    gaps = [b - a for r in rows for a, b in zip([0.0, *r["collect_times"]], r["collect_times"], strict=False)]
    return {"episodes": len(rows), "food_mean": float(food.mean()), "food_std": float(food.std()),
            "food_median": float(np.median(food)), "food_min": int(food.min()), "food_max": int(food.max()),
            "all_collected": int((food >= args.num_waypoints).sum()), "none_collected": int((food == 0).sum()),
            "fitness_mean": float(fit.mean()), "fitness_std": float(fit.std()),
            "fell": int(sum(r["fell"] for r in rows)),
            "first_food_collected": int(sum(r["food"] >= 1 for r in rows)),
            "mean_time_per_food_s": float(np.mean(gaps)) if gaps else None}


def main() -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.seed + k, False) for k in range(args.episodes)]
    if args.blind:
        jobs += [(args.seed + k, True) for k in range(args.episodes)]
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        rows = list(pool.map(_eval, jobs))
    result = {"config": vars(args) | {"data_dir": str(args.data_dir), "out": str(out_dir)},
              "vision": summary([r for r in rows if not r["blind"]]), "episodes": rows}
    if args.blind:
        result["blind"] = summary([r for r in rows if r["blind"]])
    (out_dir / "chain_eval.json").write_text(json.dumps(result))
    for key in ("vision", "blind"):
        if key in result:
            print(key, json.dumps(result[key]))
    for k in range(args.video):
        r = render(args.seed + k, out_dir / f"chain_seed{args.seed + k}.mp4")
        print(f"video seed {args.seed + k}: {r['food']} food at {[round(t, 1) for t in r['collect_times']]}")


if __name__ == "__main__":
    main()
