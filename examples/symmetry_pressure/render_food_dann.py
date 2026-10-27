"""
Videos of a gecko_food_dann.py controller doing its food subtasks.

Replays best_weights.npy through gecko_food_dann.run_subtask (the training
episode code, so the trajectories are the scored ones) and renders each
post-settle step at --fps: a tracking view of the gecko and the food, with the
robot's own camera view inset and the subtask, time, vision fractions and
score overlaid. Writes one H.264 mp4 per subtask plus all of them in sequence
(all_subtasks.mp4).

Usage:
  python render_food_dann.py <run_dir>/__data__/gecko_food_dann [--heldout] [--out videos/]
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

import gecko_food_dann as g

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=Path, help="A run's __data__/gecko_food_dann directory")
parser.add_argument("--weights", type=Path, default=None, help="Defaults to <data_dir>/best_weights.npy")
parser.add_argument("--heldout", action="store_true", help="Render the held-out angles instead of the training ones")
parser.add_argument("--out", type=Path, default=None, help="Defaults to <data_dir>/videos")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=640)
args = parser.parse_args()

config = json.loads((args.data_dir / "run_config.json").read_text())
foods = config["heldout_foods" if args.heldout else "train_foods"]
weights = np.load(args.weights or args.data_dir / "best_weights.npy")
out_dir = args.out or args.data_dir / "videos"
out_dir.mkdir(parents=True, exist_ok=True)
tag = "heldout" if args.heldout else "train"

ctx = g.make_ctx(config["reach_radius"], config["subtask_time"], config.get("vision_norm", "raw"),
                 config.get("hidden", 16))
model, data = ctx["model"], ctx["data"]
core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
scene = mujoco.Renderer(model, height=args.height, width=args.width)
inset_h, inset_w = args.height // 4, args.width // 4
fpv = mujoco.Renderer(model, height=inset_h, width=inset_w)
frame_every = max(1, round(1.0 / (model.opt.timestep * args.fps)))


def text(img: np.ndarray, s: str, y: int, scale: float = 0.55) -> None:
    for color, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_subtask(k: int, food: list[float]) -> tuple[list[np.ndarray], dict]:
    frames: list[np.ndarray] = []
    step = [0]
    angle = math.degrees(math.atan2(food[1], food[0]))

    def recorder(d: mujoco.MjData, vision: np.ndarray, progress: float) -> None:
        step[0] += 1
        if step[0] % frame_every:
            return
        # Tracking camera looking at the midpoint of the gecko and the food, so both stay in frame.
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (d.xpos[core] + np.array(food)) / 2
        cam.distance = 2.4 + 0.5 * math.hypot(*(d.xpos[core][:2] - np.array(food[:2])))
        cam.azimuth, cam.elevation = 90.0, -50.0
        scene.update_scene(d, camera=cam)
        img = cv2.cvtColor(scene.render(), cv2.COLOR_RGB2BGR)
        fpv.update_scene(d, camera=ctx["cam_name"])
        inset = cv2.cvtColor(fpv.render(), cv2.COLOR_RGB2BGR)
        cv2.rectangle(inset, (0, 0), (inset_w - 1, inset_h - 1), (255, 255, 255), 2)
        img[-inset_h - 10:-10, -inset_w - 10:-10] = inset
        t = d.time - g.SETTLE_DURATION
        text(img, f"{tag} subtask {k + 1}/{len(foods)}: food at {angle:+.0f} deg", 25)
        text(img, f"t = {t:4.1f} / {config['subtask_time']:.0f} s   progress {progress:.2f}", 50)
        text(img, f"vision L/C/R  {vision[0]:.2f} {vision[1]:.2f} {vision[2]:.2f}   food in image {ctx['vision'].coverage:.3f}", 75)
        cv2.putText(img, "robot camera", (args.width - inset_w - 10, args.height - inset_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(img)

    ctx["brain"].set_params(weights)
    result = g.run_subtask(ctx, np.array(food), recorder=recorder)
    # Hold the last frame for a second with the outcome.
    last = frames[-1].copy()
    outcome = "COLLECTED" if result["collected"] else "missed"
    text(last, f"{outcome}  score {result['score']:.2f} / 2, coverage {result['coverage']:.2f}  ({result['sim_time']:.1f} s)",
         args.height // 2, 0.8)
    frames += [last] * args.fps
    return frames, result


def write(path: Path, frames: list[np.ndarray]) -> None:
    """mp4v via OpenCV, then re-encoded to H.264 so browsers and players open it."""
    raw = path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    for f in frames:
        writer.write(f)
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(path)], check=True)
    raw.unlink()


all_frames: list[np.ndarray] = []
episodes = []
for k, food in enumerate(foods):
    frames, result = render_subtask(k, food)
    episodes.append(result)
    write(out_dir / f"{tag}_subtask{k + 1}.mp4", frames)
    all_frames += frames
    print(f"subtask {k + 1}: food at {food[0]:+.2f},{food[1]:+.2f}  score {result['score']:.2f}  "
          f"{'collected' if result['collected'] else 'missed'} in {result['sim_time']:.1f} s", flush=True)
write(out_dir / f"{tag}_all_subtasks.mp4", all_frames)
coverage_weight, speed_weight = config.get("coverage_weight", 0.0), config.get("speed_weight", 1.0)
total = g.total_score(episodes, coverage_weight, speed_weight, config.get("progress_weight", 1.0),
                      config.get("progress_mode", "first"))
print(f"total {total:.2f} / {(1 + speed_weight + coverage_weight) * len(foods):g} -> {out_dir}")
sys.exit(0)
