"""Record a video of a zoo-benchmark champion.

Replays champion.npy from a run dir with exactly the episode loop used for
training (zoo_benchmark.run_episode semantics) and writes champion.mp4 next to
it, with a camera tracking the core.

Usage:
  MUJOCO_GL=egl python render_zoo_run.py runs/gecko_matsuoka [runs/gecko_ann ...]
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from zoo_benchmark import CORE_BODY, build_world, get_state_from_data, make_brain_for


def _tracking_camera(core_pos: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = core_pos
    cam.distance = 1.5
    cam.azimuth = 135.0
    cam.elevation = -30.0
    return cam


def render_run(run_dir: Path, fps: int, width: int, height: int) -> Path:
    cfg = json.loads((run_dir / "config.json").read_text())
    params = np.load(run_dir / "champion.npy")

    model, data = build_world(cfg["body"])
    brain = make_brain_for(cfg["brain"], model, data)
    brain.set_params(params)

    control_freq, duration = cfg["control_freq"], cfg["duration"]
    steps_per_ctrl = round(1.0 / (control_freq * model.opt.timestep))
    steps_per_frame = round(1.0 / (fps * model.opt.timestep))
    n_steps = round(duration * control_freq) * steps_per_ctrl
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CORE_BODY)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    brain.reset()
    x0 = float(data.xpos[core_id, 0])

    out_path = run_dir / "champion.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for step in range(n_steps):
            if step % steps_per_ctrl == 0:
                data.ctrl[:] = brain.act(float(data.time), get_state_from_data(data))
            if step % steps_per_frame == 0:
                renderer.update_scene(data, camera=_tracking_camera(data.xpos[core_id]))
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                dx = float(data.xpos[core_id, 0]) - x0
                for i, text in enumerate([
                    f"{cfg['body']} / {cfg['brain']}  (seed {cfg['seed']})",
                    f"t = {data.time:5.2f} / {duration:.0f} s",
                    f"x displacement {dx:+.3f} m",
                ]):
                    cv2.putText(frame, text, (10, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame)
            mujoco.mj_step(model, data)
    writer.release()

    speed = (float(data.xpos[core_id, 0]) - x0) / float(data.time)
    print(f"{run_dir}: x-speed {speed:+.4f} m/s -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for run_dir in args.run_dirs:
        render_run(run_dir, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
