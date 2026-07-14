"""
Test whether changing control_step_freq from 50 → 100 reduces the glitch fitness
for saved checkpoints, and render videos for visual confirmation.

Usage (from repo root):
    python examples/symmetry_pressure/test_control_freq.py \
        --checkpoints <cp1> [<cp2> ...] \
        [--dur 30] [--freqs 50 100] [--out-dir /tmp/freq_test]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from shared import Network, build_world_for_body, fill_parameters
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
parser.add_argument("--dur",  type=float, default=30.0)
parser.add_argument("--freqs", type=int, nargs="+", default=[50, 100])
parser.add_argument("--out-dir", type=Path,
                    default=Path("data/symmetry_investigation/analysis/freq_test"))
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)

SETTLE_DURATION     = 3.0
HINGE_CONTACT_LIMIT = 200
HINGE_CONTACT_PENALTY = 0.005
HINGE_GLITCH_FITNESS  = 1.0
RENDER_H, RENDER_W    = 480, 640
FPS                   = 30


def _hinge_geom_ids(model):
    return {i for i in range(model.ngeom)
            if model.geom(i).name.endswith(("-stator", "-rotor"))}

def _floor_geom_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


def run_and_render(genome_dict, weights, freq, dur, label, out_path):
    model, data, _, _ = build_world_for_body(genome_dict, reach_radius=0.2, arena_radius=3.0)
    network = Network(input_size=len(get_robot_state(data)), output_size=model.nu)
    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    hinge_ids = _hinge_geom_ids(model)
    floor_id  = _floor_geom_id(model)
    core_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    renderer  = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    # Settling
    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    x0             = float(data.qpos[0])
    initial_height = float(data.xpos[core_id, 2])

    step           = 0
    current_action = np.zeros(model.nu)
    c_hinge        = 0
    z_history: list[float] = []
    frames: list[np.ndarray] = []
    sim_step_per_render = max(1, int(1.0 / (FPS * model.opt.timestep)))

    episode_end = SETTLE_DURATION + dur
    while data.time < episode_end:
        if step % freq == 0:
            state = get_robot_state(data).astype(np.float32)
            current_action = network.forward(model, data, state)
            z_history.append(float(data.xpos[core_id, 2]))

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == floor_id and g2 in hinge_ids) or \
               (g2 == floor_id and g1 in hinge_ids):
                c_hinge += 1

        if step % sim_step_per_render == 0:
            renderer.update_scene(data, camera="overview_cam")
            frame = renderer.render().copy()
            # Overlay
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            x_now = float(data.qpos[0]) - x0
            cv2.putText(bgr, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(bgr, f"t={data.time - SETTLE_DURATION:.1f}s  x={x_now:.2f}m",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
            frames.append(bgr)

        step += 1

    renderer.close()

    x_disp = float(data.qpos[0]) - x0
    if c_hinge >= HINGE_CONTACT_LIMIT:
        fitness = HINGE_GLITCH_FITNESS
    else:
        fitness = -(x_disp - initial_height - HINGE_CONTACT_PENALTY * c_hinge)

    dz      = np.abs(np.diff(z_history)) if len(z_history) >= 2 else np.array([0.0])
    mean_dz = float(np.mean(dz))
    max_dz  = float(np.max(dz))

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (RENDER_W, RENDER_H))
    for f in frames:
        writer.write(f)
    writer.release()

    return {
        "fitness": fitness, "x_displacement": x_disp,
        "initial_height": initial_height, "c_hinge": c_hinge,
        "mean_dz": mean_dz, "max_dz": max_dz,
    }


print(f"\n{'checkpoint':<30}  {'freq':>5}  {'fitness':>10}  {'x_disp':>7}  {'mean_dz':>9}  {'max_dz':>8}  video")
print("-" * 100)

for cp in args.checkpoints:
    cp = cp.resolve()
    with (cp / "best_genome.json").open() as fh:
        genome_dict = json.load(fh)
    weights = np.load(cp / "best_weights.npy")

    meta_path = cp / "meta.json"
    stored_fit = ""
    if meta_path.exists():
        with meta_path.open() as fh:
            m = json.load(fh)
        stored_fit = f"(stored {m.get('fitness', '?'):.3f})"

    for freq in args.freqs:
        label = f"{cp.name}  freq={freq}"
        out_path = args.out_dir / f"{cp.name}_freq{freq}.mp4"
        r = run_and_render(genome_dict, weights, freq, args.dur, label, out_path)
        print(
            f"{cp.name:<30}  {freq:>5}  {r['fitness']:>10.4f}  {r['x_displacement']:>7.3f}  "
            f"{r['mean_dz']:>9.5f}  {r['max_dz']:>8.5f}  {out_path.name}"
        )

print(f"\nVideos → {args.out_dir}")
