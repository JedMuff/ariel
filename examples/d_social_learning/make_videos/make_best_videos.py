"""Render a video of the best individual from each scheme in the ARIEL domain.

Candidates are re-scored under the current evaluator (settle phase + hinge-contact
penalty) to pick the best theta per scheme, regardless of stored fitness values.

Usage:
    uv run examples/d_social_learning/make_best_videos.py \
        [--data-dir __data__/social/ariel] [--out-dir __data__/social/videos] \
        [--duration SECONDS] [--settle 3.0] [--fps 50] [--sim-preset legacy|zoo]

The simulation preset (experiment.py --sim-preset) is read from the runs'
run_config.json files unless --sim-preset is given; --duration defaults to
the preset's episode length.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import json
import sys
from pathlib import Path

# Remove d_social_learning/ from sys.path — it shadows the installed ariel package
# because it contains an ariel/ subdirectory. Python inserts the script's directory
# automatically; we drop it and re-add only the safe subdirs.
_SOCIAL_DIR = Path(__file__).parent
for _p in [str(_SOCIAL_DIR), ""]:
    if _p in sys.path:
        sys.path.remove(_p)

# Appended, not prepended: the installed ariel package still wins over the
# ariel/ directory here.
sys.path.append(str(_SOCIAL_DIR.parent))  # for sim_ariel

import imageio
import mujoco
import numpy as np
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.simulation.controllers import DistributedMLP
from ariel.simulation.controllers.morphology_adapter import MorphologyAdapter
from ariel.simulation.environments import SimpleFlatWorld
from sim_ariel import evaluator as sim_evaluator
from sim_ariel.zoo_modules import apply_zoo_modules

console = Console()

SPAWN_POS = (-0.8, 0.0, 0.1)
CTRL_EVERY = 9  # 500Hz physics / 9 ≈ 55.6Hz control (closest integer divisor to 60Hz)
CTRL_ALPHA = 0.5
N_NEIGHBORS = 6
FEATURES_PER_NODE = 8
HEIGHT_PENALTY_THRESHOLD = 0.5
JERK_PENALTY_WEIGHT = 3.0
JERK_THRESHOLD = 0.15
ZOO_MODULES = False
DISPLACEMENT_FROM_START = False

SCHEMES = [
    "darwinian", "lamarckian", "random", "random_many",
    "best", "best_many", "similar_DESCR", "similar_many_DESCR",
    "similar_STRUCT", "similar_many_STRUCT",
    "similar_TOPDOWN", "similar_many_TOPDOWN",
]


def _infer_hidden(theta_len: int) -> int:
    """Solve n_params = hidden * (input_size + 2) + 1 for hidden."""
    input_size = (1 + N_NEIGHBORS) * FEATURES_PER_NODE + 1
    hidden, remainder = divmod(theta_len - 1, input_size + 2)
    if remainder != 0:
        raise ValueError(f"theta length {theta_len} is not a valid DistributedMLP size")
    return hidden


def _scale_actions(raw: np.ndarray) -> np.ndarray:
    return np.asarray(raw) * (math.pi / 2)


def load_all_candidates(db_path: Path) -> list[tuple[dict, np.ndarray]]:
    """Return (morph_dict, theta) for every evaluated individual in the DB."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "SELECT genotype_, tags_ FROM individual WHERE fitness_ IS NOT NULL"
    )
    rows = cur.fetchall()
    con.close()
    candidates = []
    for geno_json, tags_json in rows:
        geno = json.loads(geno_json) if geno_json else {}
        tags = json.loads(tags_json) if tags_json else {}
        theta = tags.get("theta")
        morph = geno.get("morph")
        if theta and morph:
            candidates.append((morph, np.array(theta, dtype=np.float64)))
    return candidates


def _apply_preset(name: str) -> None:
    """Use the evaluator's constants for preset *name* (see experiment.py
    --sim-preset), including the zoo module geometry."""
    sim_evaluator.apply_preset(name)
    globals().update(sim_evaluator.PRESETS[name])


def _runs_preset(data_dir: Path, schemes: list[str]) -> str:
    """The --sim-preset the runs under *data_dir* were made with."""
    presets = set()
    for scheme in schemes:
        for config in data_dir.glob(f"{scheme}/**/run_config.json"):
            presets.add((json.loads(config.read_text()).get("args") or {}).get("sim_preset", "legacy"))
    if len(presets) > 1:
        raise SystemExit(f"runs under {data_dir} mix sim presets {sorted(presets)}; pass --sim-preset")
    return presets.pop() if presets else "legacy"


def _build_model(morph_dict: dict):
    genome = TreeGenome.from_dict(morph_dict)
    graph = genome.to_networkx()
    core = construct_mjspec_from_graph(graph)
    if ZOO_MODULES:
        apply_zoo_modules(core.spec)
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
    model = world.spec.compile()
    adapter = MorphologyAdapter.from_graph(graph)
    return model, adapter


def evaluate(morph_dict: dict, theta: np.ndarray, duration: float, settle: float) -> float:
    """Re-score a theta under the current fitness function (mirrors
    ariel/evaluator.py::run_episode's settle/rollout structure, control
    blending, and fitness formula exactly)."""
    model, adapter = _build_model(morph_dict)
    data = mujoco.MjData(model)

    hidden = _infer_hidden(len(theta))
    brain = DistributedMLP(n_neighbors=N_NEIGHBORS, hidden=hidden)
    brain.set_theta(theta)
    mujoco.mj_resetData(model, data)

    core_height = float(data.qpos[2])

    sim_step = 0
    while data.time < settle:
        mujoco.mj_step(model, data)
        sim_step += 1
    start_x = float(data.qpos[0])

    ctrl_step = 0
    prev_ctrl = np.zeros(model.nu, dtype=np.float32)
    jerk_sum = 0.0
    rollout_end = settle + duration
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
    d = float(data.qpos[0]) - (start_x if DISPLACEMENT_FROM_START else 0.0)
    height_penalty = core_height if core_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    jerk_penalty = JERK_PENALTY_WEIGHT * mean_jerk if mean_jerk >= JERK_THRESHOLD else 0.0
    return d - height_penalty - jerk_penalty


def _tracking_camera(model) -> mujoco.MjvCamera:
    """Camera that follows the robot's core body, so good walkers don't
    leave the frame. Viewed side-on from -y like the world's fixed
    pretty-cam, so forward (+x) motion reads left-to-right."""
    core_id = next(
        i for i in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("_core")
    )
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = core_id
    cam.distance = 2.5
    cam.azimuth = 90.0
    cam.elevation = -30.0
    return cam


def render_individual(
    morph_dict: dict, theta: np.ndarray,
    duration: float, settle: float, fps: int,
) -> list:
    model, adapter = _build_model(morph_dict)
    data = mujoco.MjData(model)
    hidden = _infer_hidden(len(theta))
    brain = DistributedMLP(n_neighbors=N_NEIGHBORS, hidden=hidden)
    brain.set_theta(theta)
    mujoco.mj_resetData(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = _tracking_camera(model)
    dt = model.opt.timestep
    render_every = max(1, int(round(1.0 / (fps * dt))))

    frames = []
    sim_step = 0

    # Settle phase — render so the fall is visible
    while data.time < settle:
        mujoco.mj_step(model, data)
        if sim_step % render_every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
        sim_step += 1

    # Rollout phase
    ctrl_step = 0
    prev_ctrl = np.zeros(model.nu, dtype=np.float32)
    rollout_end = settle + duration
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
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
        sim_step += 1

    renderer.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="__data__/social/ariel")
    parser.add_argument("--out-dir", default="__data__/social/videos")
    parser.add_argument("--duration", type=float, default=None,
                        help="Rollout length (s); default: the sim preset's episode length")
    parser.add_argument("--sim-preset", choices=sorted(sim_evaluator.PRESETS), default=None,
                        help="default: read from the runs' run_config.json")
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--schemes", nargs="*", default=SCHEMES)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preset = args.sim_preset or _runs_preset(data_dir, args.schemes)
    _apply_preset(preset)
    if args.duration is None:
        args.duration = DURATION
    console.log(f"sim preset: {preset}  (duration {args.duration}s, control every {CTRL_EVERY} steps)")

    for scheme in args.schemes:
        all_candidates: list[tuple[dict, np.ndarray, Path]] = []
        for db_path in sorted(data_dir.glob(f"{scheme}/**/database.db")):
            for morph, theta in load_all_candidates(db_path):
                all_candidates.append((morph, theta, db_path))

        if not all_candidates:
            console.log(f"[yellow]No data for scheme={scheme}, skipping[/yellow]")
            continue

        console.log(f"[cyan]{scheme}[/cyan] — re-scoring {len(all_candidates)} candidates...")

        best_morph, best_theta, best_db, best_fitness = None, None, None, -float("inf")
        for morph, theta, db_path in all_candidates:
            fit = evaluate(morph, theta, args.duration, args.settle)
            if fit > best_fitness:
                best_fitness = fit
                best_morph, best_theta, best_db = morph, theta, db_path

        console.log(f"  best fitness={best_fitness:.4f}  db={best_db}")
        frames = render_individual(best_morph, best_theta, args.duration, args.settle, args.fps)

        if not frames:
            console.log(f"[red]No frames for {scheme}[/red]")
            continue

        out_path = out_dir / f"{scheme}_best.mp4"
        imageio.mimsave(str(out_path), frames, fps=args.fps)
        console.log(f"[green]Saved → {out_path}[/green]  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
