"""Render a video of the best individual from each scheme in the ARIEL domain.

Candidates are re-scored under the current evaluator (settle phase + hinge-contact
penalty) to pick the best theta per scheme, regardless of stored fitness values.

Usage:
    uv run examples/d_social_learning/make_best_videos.py \
        [--data-dir __data__/social/ariel] [--out-dir __data__/social/videos] \
        [--duration 30.0] [--settle 3.0] [--fps 50]
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

import imageio
import mujoco
import numpy as np
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.simulation.controllers import DistributedMLP
from ariel.simulation.controllers.morphology_adapter import MorphologyAdapter
from ariel.simulation.environments import SimpleFlatWorld

console = Console()

SPAWN_POS = (-0.8, 0.0, 0.1)
CTRL_EVERY = 50
N_NEIGHBORS = 6
HINGE_CONTACT_LIMIT = 200
HINGE_CONTACT_PENALTY = 0.005

SCHEMES = [
    "darwinian", "lamarckian", "random", "random_many",
    "best", "best_many", "similar", "similar_many",
]


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


def _build_model(morph_dict: dict):
    genome = TreeGenome.from_dict(morph_dict)
    graph = genome.to_networkx()
    core = construct_mjspec_from_graph(graph)
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
    model = world.spec.compile()
    adapter = MorphologyAdapter.from_graph(graph)
    return model, adapter


def _geom_ids(model):
    hinge_geom_ids: set[int] = set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and ("stator" in name or "rotor" in name):
            hinge_geom_ids.add(i)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return hinge_geom_ids, floor_geom_id


def evaluate(morph_dict: dict, theta: np.ndarray, duration: float, settle: float) -> float:
    """Re-score a theta under the current fitness function."""
    model, adapter = _build_model(morph_dict)
    data = mujoco.MjData(model)
    hinge_geom_ids, floor_geom_id = _geom_ids(model)

    brain = DistributedMLP(n_neighbors=N_NEIGHBORS)
    brain.set_theta(theta)
    mujoco.mj_resetData(model, data)

    core_height = float(data.qpos[2])

    sim_step = 0
    while data.time < settle:
        mujoco.mj_step(model, data)
        sim_step += 1

    c_hinge = 0
    ctrl_step = 0
    active_hinge_contacts: set[frozenset[int]] = set()
    rollout_end = settle + duration
    while data.time < rollout_end:
        if sim_step % CTRL_EVERY == 0:
            node_inputs, t = adapter.get_node_inputs(model, data, ctrl_step)
            raw = brain.forward_all(node_inputs, t)
            data.ctrl[:] = _scale_actions(raw)
            ctrl_step += 1
        mujoco.mj_step(model, data)
        current: set[frozenset[int]] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            if ((c.geom1 == floor_geom_id and c.geom2 in hinge_geom_ids) or
                    (c.geom2 == floor_geom_id and c.geom1 in hinge_geom_ids)):
                current.add(frozenset((c.geom1, c.geom2)))
        c_hinge += len(current - active_hinge_contacts)
        active_hinge_contacts = current
        sim_step += 1

    if c_hinge > HINGE_CONTACT_LIMIT:
        return -1.0
    return float(data.qpos[0]) - core_height - HINGE_CONTACT_PENALTY * c_hinge


def render_individual(
    morph_dict: dict, theta: np.ndarray,
    duration: float, settle: float, fps: int,
) -> list:
    model, adapter = _build_model(morph_dict)
    data = mujoco.MjData(model)
    brain = DistributedMLP(n_neighbors=N_NEIGHBORS)
    brain.set_theta(theta)
    mujoco.mj_resetData(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    dt = model.opt.timestep
    render_every = max(1, int(round(1.0 / (fps * dt))))

    frames = []
    sim_step = 0

    # Settle phase — render so the fall is visible
    while data.time < settle:
        mujoco.mj_step(model, data)
        if sim_step % render_every == 0:
            renderer.update_scene(data, camera="pretty-cam")
            frames.append(renderer.render())
        sim_step += 1

    # Rollout phase
    ctrl_step = 0
    rollout_end = settle + duration
    while data.time < rollout_end:
        if sim_step % CTRL_EVERY == 0:
            node_inputs, t = adapter.get_node_inputs(model, data, ctrl_step)
            raw = brain.forward_all(node_inputs, t)
            data.ctrl[:] = _scale_actions(raw)
            ctrl_step += 1
        mujoco.mj_step(model, data)
        if sim_step % render_every == 0:
            renderer.update_scene(data, camera="pretty-cam")
            frames.append(renderer.render())
        sim_step += 1

    renderer.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="__data__/social/ariel")
    parser.add_argument("--out-dir", default="__data__/social/videos")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--schemes", nargs="*", default=SCHEMES)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
