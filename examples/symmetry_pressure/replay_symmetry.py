"""
Replay the top-3 individuals from a symmetry-pressure run and save videos.

Usage:
    python examples/symmetry_pressure/replay_symmetry.py --run-dir __data__/gecko_locomotion/
    python examples/symmetry_pressure/replay_symmetry.py --run-dir __data__/gecko_food/ --task food
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np

# Allow importing shared from the same directory
sys.path.insert(0, str(Path(__file__).parent))

from shared import (
    Network,
    RING_R_MAX,
    action_control_cost,
    analyze_sections,
    bilateral_symmetry_score,
    build_world_for_body,
    fill_parameters,
    genome_hash,
    isolate_green,
    sample_waypoints,
)
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Replay top-3 individuals as video")
parser.add_argument("--run-dir", required=True, type=Path)
parser.add_argument("--task",    choices=["locomotion", "food"], default="locomotion")
parser.add_argument("--top-n",  type=int, default=3)
args = parser.parse_args()

RUN_DIR     = args.run_dir.resolve()
JSONL       = RUN_DIR / "run_data.jsonl"
CHECKPOINTS = RUN_DIR / "checkpoints"
OUT_DIR     = RUN_DIR / "analysis"
OUT_DIR.mkdir(exist_ok=True, parents=True)
TASK        = args.task
TOP_N       = args.top_n

# ── Load config ───────────────────────────────────────────────────────────────

config: dict[str, Any] = {}
config_path = RUN_DIR / "run_config.json"
if config_path.exists():
    with config_path.open() as fh:
        config = json.load(fh)

DURATION     = float(config.get("dur", 15.0 if TASK == "locomotion" else 60.0))
REACH_RADIUS = float(config.get("reach_radius", 0.20))
ARENA_RADIUS = float(config.get("arena_radius", 3.0))
NUM_WAYPOINTS = int(config.get("num_waypoints", 10))
BASE_SEED    = int(config.get("seed", 42))

# ── Load JSONL ────────────────────────────────────────────────────────────────

records: list[dict[str, Any]] = []
with JSONL.open() as fh:
    for line in fh:
        line = line.strip()
        if line:
            records.append(json.loads(line))

finite_records = [
    r for r in records
    if r.get("fitness") is not None and np.isfinite(r["fitness"])
]
finite_records.sort(key=lambda r: r["fitness"])

# ── Find checkpoints for top individuals ──────────────────────────────────────

def find_checkpoint(genome_hash_val: Optional[str], gen: int, ind_id: Any) -> Optional[Path]:
    """Return a checkpoint directory that matches this individual."""
    if not CHECKPOINTS.exists():
        return None

    # Try meta.json match on gen + fitness
    for cp in sorted(CHECKPOINTS.iterdir()):
        if not cp.is_dir():
            continue
        meta_path = cp / "meta.json"
        genome_path = cp / "best_genome.json"
        if not genome_path.exists():
            continue
        # Match by genome hash if available
        if genome_hash_val is not None:
            with genome_path.open() as fh:
                stored = json.load(fh)
            if genome_hash(stored) == genome_hash_val:
                return cp
    return None


selected: list[tuple[dict, Path]] = []
seen_hashes: set = set()

for rec in finite_records:
    if len(selected) >= TOP_N:
        break
    gh = rec.get("genome_hash")
    if gh in seen_hashes:
        continue
    cp = find_checkpoint(gh, rec["gen"], rec["ind_id"])
    if cp is None:
        continue
    seen_hashes.add(gh)
    selected.append((rec, cp))

if not selected:
    print(f"No matching checkpoints found under {CHECKPOINTS}. "
          "Run with full budget first so checkpoints are saved.")
    sys.exit(0)

print(f"Replaying {len(selected)} individuals")

# ── Replay one individual ─────────────────────────────────────────────────────

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def render_video(rec: dict, cp: Path) -> None:
    gen      = rec["gen"]
    fitness  = rec["fitness"]
    symmetry = rec.get("yz_symmetry", 0.0) or 0.0

    with (cp / "best_genome.json").open() as fh:
        genome_dict = json.load(fh)
    weights = np.load(cp / "best_weights.npy")

    use_vision = TASK == "food"
    waypoints_path = cp / "best_waypoints.npy"
    waypoints: Optional[list[np.ndarray]] = None
    if use_vision and waypoints_path.exists():
        wp_arr = np.load(str(waypoints_path))
        waypoints = [wp_arr[i] for i in range(len(wp_arr))]
    elif use_vision:
        rng = np.random.default_rng(BASE_SEED + gen)
        waypoints = sample_waypoints(rng, n=NUM_WAYPOINTS)

    model, data, target_mocap_id, cam_name = build_world_for_body(
        genome_dict, REACH_RADIUS, ARENA_RADIUS, add_food_target=use_vision
    )

    input_dim = len(get_robot_state(data))
    if use_vision:
        input_dim += 3 + 2 + 1
    network = Network(input_size=input_dim, output_size=model.nu)
    fill_parameters(network, weights)

    render_h, render_w = 480, 640
    overview_renderer = mujoco.Renderer(model, height=render_h, width=render_w)
    vision_renderer: Optional[mujoco.Renderer] = None
    if use_vision:
        vision_renderer = mujoco.Renderer(model, height=48 * 2, width=64 * 2)

    frames: list[np.ndarray] = []
    mujoco.mj_resetData(model, data)

    control_step_freq = 50
    step = 0
    current_action = np.zeros(model.nu)

    current_wp_idx = 0
    if use_vision and waypoints is not None and target_mocap_id is not None:
        data.mocap_pos[target_mocap_id] = waypoints[0]

    while data.time < DURATION:
        if use_vision and waypoints is not None:
            num_wps = len(waypoints)
            if current_wp_idx >= num_wps:
                break

        if step % control_step_freq == 0:
            if use_vision and vision_renderer is not None and cam_name is not None:
                vision_renderer.update_scene(data, camera=cam_name)
                img    = vision_renderer.render()
                vision = analyze_sections(isolate_green(img))
                robot_state = get_robot_state(data)
                num_wps_l = NUM_WAYPOINTS
                phase   = [2.0 * np.sin(data.time * 2.0 * np.pi),
                           2.0 * np.cos(data.time * 2.0 * np.pi)]
                progress = [current_wp_idx / max(num_wps_l - 1, 1)]
                state = np.concatenate([robot_state, vision, phase, progress]).astype(np.float32)
            else:
                state = get_robot_state(data).astype(np.float32)
            current_action = network.forward(model, data, state)

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

        if use_vision and waypoints is not None and target_mocap_id is not None:
            dist = float(np.linalg.norm(np.array(data.qpos[:2]) - waypoints[current_wp_idx][:2]))
            if dist <= REACH_RADIUS:
                current_wp_idx += 1
                if current_wp_idx < len(waypoints):
                    data.mocap_pos[target_mocap_id] = waypoints[current_wp_idx]

        # Render overview at ~30 fps
        if step % 17 == 0:
            overview_renderer.update_scene(data, camera="overview_cam")
            frame = overview_renderer.render().copy()

            # Overlay text
            if HAS_CV2:
                import cv2 as _cv2
                frame_bgr = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
                _cv2.putText(frame_bgr, f"Gen {gen}  Fit {fitness:.3f}",
                             (10, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                _cv2.putText(frame_bgr, f"Symmetry {symmetry:.3f}",
                             (render_w - 220, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                frame = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)

            frames.append(frame)

        step += 1

    overview_renderer.close()
    if vision_renderer is not None:
        vision_renderer.close()

    if not frames:
        print(f"  No frames rendered for gen={gen} fit={fitness:.3f}")
        return

    # Write video
    fname = f"video_gen{gen:03d}_body{rec.get('ind_id', 0):02d}_fit{fitness:.3f}.mp4"
    out_path = OUT_DIR / fname

    if HAS_CV2:
        import cv2 as _cv2
        fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
        writer = _cv2.VideoWriter(str(out_path), fourcc, 30, (render_w, render_h))
        for f in frames:
            writer.write(_cv2.cvtColor(f, _cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  {out_path}")
    else:
        # Fallback: save as numpy
        np_path = out_path.with_suffix(".npy")
        np.save(str(np_path), np.stack(frames))
        print(f"  (cv2 not available) frames saved to {np_path}")


for rec, cp in selected:
    print(f"  gen={rec['gen']}  fitness={rec['fitness']:.3f}  "
          f"symmetry={rec.get('yz_symmetry', 0.0):.3f}  checkpoint={cp.name}")
    render_video(rec, cp)

print(f"\nVideos written to {OUT_DIR}")
