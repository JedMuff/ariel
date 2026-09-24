"""
Render N random initial-population genomes per genome type as 2x2 view grids,
for visually checking the decoders (e.g. the CPPN NONE / face-scan fixes).

Genomes are drawn exactly as generation 0 of gecko_skill_tasks.py draws them:
`genome_adapter.genome_adapter_from_cli(...).create_individual`, after seeding
`random` and `np.random` with --seed. Each genome type is seeded separately
with the same seed, so the images for one type don't depend on which other
types were rendered.

Each body is spawned in the flat world, settled under gravity for
`shared.SETTLE_DURATION` with zero control, then rendered from four cameras:
top-down, side (looking along +y), and two isometric views from opposite
corners.

Output:
    <out>/<genome_type>/<genome_type>_<idx>.png   one 2x2 grid per genome
    <out>/<genome_type>/genomes.json              the genotypes, by idx
    <out>/summary.csv                             idx, modules, hinges, ok

Usage:
    python render_random_genomes.py
    python render_random_genomes.py --seed 7 --n 20 --genome-types cppn
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")

import argparse
import csv
import json
import random
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent))

import genome_adapter
import shared
from descriptor_common import genotype_to_graph
from ariel.body_phenotypes.robogen_lite.config import ModuleType

console = Console()

CORE_BODY_NAME = "robot1_core"

# (title, azimuth, elevation). MuJoCo's azimuth is the direction the camera
# looks along, measured from +x; azimuth=90 looks along +y (a side view of the
# x-z plane).
VIEWS = [
    ("top", 90.0, -90.0),
    ("side (looking +y)", 90.0, -5.0),
    ("iso (from -x,-y)", 45.0, -35.0),
    ("iso (from +x,+y)", 225.0, -35.0),
]

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n", type=int, default=100, help="genomes per genome type")
parser.add_argument("--genome-types", nargs="+", default=["tree", "tree_symmetric", "cppn"],
                    choices=["tree", "tree_symmetric", "cppn"])
parser.add_argument("--max-modules", type=int, default=25)
parser.add_argument("--max-depth", type=int, default=25)
parser.add_argument("--size-px", type=int, default=400, help="pixels per view")
parser.add_argument("--no-settle", action="store_true",
                    help="render the spawn pose instead of settling under gravity")
parser.add_argument("--out", type=Path, default=None,
                    help="default: ../../__data__/random_genomes_seed<seed>")
args = parser.parse_args()


def _render_views(genome_dict: dict, genome_type: str) -> list[np.ndarray]:
    to_spec_fn = (
        partial(genome_adapter.cppn_genome_to_spec, max_modules=args.max_modules)
        if genome_type == "cppn"
        else shared.genome_to_spec
    )
    model, data = shared.build_loco_world_for_body(genome_dict, to_spec_fn)
    if args.no_settle:
        mujoco.mj_forward(model, data)
    else:
        while data.time < shared.SETTLE_DURATION:
            mujoco.mj_step(model, data)

    # Frame on the centre of the robot's bodies (skip the world body and
    # anything at the origin-level floor), sized to their 3D extent.
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CORE_BODY_NAME)
    robot_ids = [i for i in range(1, model.nbody) if model.body_rootid[i] == model.body_rootid[core_id]]
    pts = data.xpos[robot_ids]
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
    extent = float(np.max(np.linalg.norm(pts - centre, axis=1)))
    distance = max(0.6, extent * 3.0 + 0.25)

    frames = []
    renderer = mujoco.Renderer(model, height=args.size_px, width=args.size_px)
    try:
        for _, azimuth, elevation in VIEWS:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = centre
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


def _save_grid(frames: list[np.ndarray], title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.4))
    for ax, frame, (view_title, _, _) in zip(axes.flat, frames, VIEWS):
        ax.imshow(frame)
        ax.set_title(view_title, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    out_root = args.out or (Path(__file__).resolve().parents[2] / "__data__" / f"random_genomes_seed{args.seed}")
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for genome_type in args.genome_types:
        random.seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        adapter = genome_adapter.genome_adapter_from_cli(genome_type, args.max_modules)

        out_dir = out_root / genome_type
        out_dir.mkdir(parents=True, exist_ok=True)
        genomes = {}

        for idx in range(args.n):
            ind = adapter.create_individual(rng, args.max_modules, args.max_depth)
            genome_dict = ind.genotype[adapter.genotype_key]
            genomes[idx] = genome_dict

            graph = genotype_to_graph(genome_dict, genome_type, args.max_modules)
            n_modules = graph.number_of_nodes()
            n_hinges = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == ModuleType.HINGE.name)

            ok = True
            try:
                frames = _render_views(genome_dict, genome_type)
                title = f"{genome_type} #{idx}  (seed {args.seed})  modules={n_modules}  hinges={n_hinges}"
                _save_grid(frames, title, out_dir / f"{genome_type}_{idx:03d}.png")
            except Exception as e:  # noqa: BLE001
                ok = False
                console.log(f"[red]{genome_type} #{idx}: render failed: {e}[/red]")

            rows.append({"genome_type": genome_type, "idx": idx, "modules": n_modules,
                         "hinges": n_hinges, "ok": ok})
            console.log(f"{genome_type} #{idx}: modules={n_modules} hinges={n_hinges}")

        (out_dir / "genomes.json").write_text(json.dumps(genomes))

    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genome_type", "idx", "modules", "hinges", "ok"])
        writer.writeheader()
        writer.writerows(rows)
    console.log(f"[green]Wrote {len(rows)} renders to {out_root}[/green]")


if __name__ == "__main__":
    main()
