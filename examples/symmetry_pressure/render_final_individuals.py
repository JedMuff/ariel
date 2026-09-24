"""
Render a video of the best individual (best fitness across all generations,
skipping checkpoints with missing skill weights) for every run directory
found under __data__/sympress_*. Videos land in a single collated tree at
<sweep_root>/analysis/<task>/<genome>/<run_name>_<checkpoint>[_<skill>].mp4,
alongside plot_run_curves.py's plots for the same run.

Dispatches to render_food_skills_checkpoint.render_checkpoint() for the food
task and render_skill_task_checkpoint.render_checkpoint() for the other
three (forward/multidirection/turn_avg) tasks.

Requires MuJoCo rendering (MUJOCO_GL=egl on a headless/GPU machine, or
MUJOCO_GL=glfw with a display).

Usage:
    python render_final_individuals.py
    python render_final_individuals.py --run-dirs ../../__data__/sympress_food_tree_rep0_37568_3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

import render_skill_task_checkpoint
from render_food_skills_checkpoint import render_checkpoint as render_food_checkpoint
from sweep_common import (
    analysis_dir,
    checkpoints_ranked_by_fitness,
    discover_run_dirs,
    parse_run_tag,
    run_config,
)

console = Console()

parser = argparse.ArgumentParser()
parser.add_argument("--run-dirs", nargs="*", type=Path, default=None)
parser.add_argument("--reach-radius", type=float, default=0.20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--render-fps", type=int, default=30)
parser.add_argument("--render-height", type=int, default=480)
parser.add_argument("--render-width", type=int, default=640)
args = parser.parse_args()


def main() -> None:
    run_dirs = args.run_dirs if args.run_dirs else discover_run_dirs()
    if not run_dirs:
        console.log("No run dirs found.")
        return

    manifest = []
    for d in run_dirs:
        info = parse_run_tag(d.name)
        if info is None:
            console.log(f"  Skipping {d.name} (unrecognized run-dir naming)")
            continue
        task, genome = info["task"], info["genome"]

        candidates = checkpoints_ranked_by_fitness(d, task)
        if not candidates:
            console.log(f"[yellow]{d.name}: no checkpoints found, skipping[/yellow]")
            continue

        console.rule(f"[bold magenta]{d.name}[/bold magenta]")
        out_dir = analysis_dir(d, task, genome)
        out_dir.mkdir(parents=True, exist_ok=True)
        max_modules = run_config(d, task).get("max_modules", 25)

        # Try checkpoints best-fitness-first across the whole run (not just
        # the last generation): a checkpoint can be missing skill-weight
        # files — e.g. gecko_food_skills.py gates left/right-skill training
        # on a loco threshold, so a gated body's checkpoint has no left/right
        # weights — so fall through to the next-best individual rather than
        # losing the whole run's video.
        ckpt = meta = videos = None
        for candidate in candidates:
            meta = json.loads((candidate / "meta.json").read_text())
            gen = meta.get("gen")
            console.log(f"  task={task}  gen={gen}  checkpoint={candidate.name}  fitness={meta.get('fitness')}")
            try:
                if task == "food":
                    out_path = out_dir / f"{d.name}_{candidate.name}.mp4"
                    result = render_food_checkpoint(
                        ckpt_dir=candidate,
                        out_path=out_path,
                        reach_radius=args.reach_radius,
                        seed=args.seed,
                        render_fps=args.render_fps,
                        render_height=args.render_height,
                        render_width=args.render_width,
                        max_modules=max_modules,
                    )
                    videos = [result["video"]] if result.get("video") else []
                else:
                    results = render_skill_task_checkpoint.render_checkpoint(
                        ckpt_dir=candidate,
                        out_dir=out_dir,
                        render_fps=args.render_fps,
                        render_height=args.render_height,
                        render_width=args.render_width,
                        max_modules=max_modules,
                        filename_prefix=f"{d.name}_",
                    )
                    videos = [r["video"] for r in results]
                ckpt = candidate
                break
            except (FileNotFoundError, ValueError) as e:
                console.log(f"  [red]{candidate.name} failed ({e}), trying next-best[/red]")

        if ckpt is None:
            console.log(f"[red]{d.name}: no renderable checkpoint found across any generation, skipping[/red]")
            continue

        manifest.append({
            "run_dir": d.name,
            "task": task,
            "gen": gen,
            "checkpoint": str(ckpt),
            "fitness": meta.get("fitness"),
            "videos": videos,
        })

    console.rule("[bold green]Done[/bold green]")
    for m in manifest:
        console.log(m)


if __name__ == "__main__":
    main()
