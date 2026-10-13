"""
Render videos for every run of the control-stride sweep (and the stride-9
baseline runs from the symmetry-pressure sweep, seeds 42-44): for each of a
few generations, the best, median and worst individual of that generation,
ranked by the fitness in each checkpoint's meta.json. Replays use the run's
own control_step_freq (render_skill_task_checkpoint reads run_config.json).

Videos land in
  <stride-root>/analysis/videos/<genome>/s<stride>/<run>_gen<NNN>_<rank>_<ckpt>_fwd.mp4
with a manifest.json alongside listing which checkpoint each video is and its
fitness. Existing videos are skipped, so the script can be re-run to resume.

Usage:
    MUJOCO_GL=egl python render_stride_sweep_generations.py
    MUJOCO_GL=egl python render_stride_sweep_generations.py --gens 5 15 30 --workers 8 --no-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from rich.console import Console  # noqa: E402

from sweep_common import checkpoint_generations, parse_run_tag, run_config  # noqa: E402

console = Console()

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK = "forward"
BASELINE_STRIDE = 9  # shared.CONTROL_STEP_FREQ, used by the symmetry-pressure sweep
BASELINE_REPS = {0, 1, 2}  # seeds 42-44, matching the stride sweep

_STRIDE_TAG_RE = re.compile(
    r"^ctrlstride_forward_(?P<genome>tree_symmetric|tree|cppn)_s(?P<stride>\d+)_rep(?P<rep>\d+)_\d+_\d+$"
)

parser = argparse.ArgumentParser()
parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
parser.add_argument("--baseline-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_symmetry_pressure_sweep")
parser.add_argument("--no-baseline", action="store_true", help="skip the stride-9 baseline runs")
parser.add_argument("--gens", type=int, nargs="+", default=[5, 15, 30])
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--render-fps", type=int, default=30)
parser.add_argument("--render-height", type=int, default=480)
parser.add_argument("--render-width", type=int, default=640)


def discover_runs(args) -> list[tuple[Path, str, int]]:
    """(run_dir, genome, stride) for every stride-sweep run, plus baseline runs."""
    runs = []
    for d in sorted(args.stride_root.glob("ctrlstride_*")):
        m = _STRIDE_TAG_RE.match(d.name)
        if m and d.is_dir():
            runs.append((d, m["genome"], int(m["stride"])))
    if not args.no_baseline:
        for d in sorted(args.baseline_root.glob("sympress_forward_*")):
            info = parse_run_tag(d.name)
            if info and d.is_dir() and info["rep"] in BASELINE_REPS:
                runs.append((d, info["genome"], BASELINE_STRIDE))
    return runs


def pick_best_median_worst(ckpts: list[Path]) -> dict[str, tuple[Path, float]]:
    """Rank a generation's checkpoints by meta.json fitness (lower is better).
    With an even count, the median is the lower-middle (better-ranked) one."""
    rated = []
    for c in ckpts:
        meta_path = c / "meta.json"
        if meta_path.exists():
            rated.append((json.loads(meta_path.read_text()).get("fitness", float("inf")), c))
    rated.sort(key=lambda x: x[0])
    if not rated:
        return {}
    picks = {"best": rated[0], "median": rated[(len(rated) - 1) // 2], "worst": rated[-1]}
    return {rank: (c, fit) for rank, (fit, c) in picks.items()}


def render_job(ckpt: Path, out_dir: Path, prefix: str, max_modules: int, args_dict: dict) -> list[str]:
    import render_skill_task_checkpoint

    results = render_skill_task_checkpoint.render_checkpoint(
        ckpt_dir=ckpt,
        out_dir=out_dir,
        render_fps=args_dict["render_fps"],
        render_height=args_dict["render_height"],
        render_width=args_dict["render_width"],
        max_modules=max_modules,
        filename_prefix=prefix,
    )
    return [r["video"] for r in results]


def main() -> None:
    args = parser.parse_args()
    runs = discover_runs(args)
    video_root = args.stride_root / "analysis" / "videos"

    jobs = []  # (ckpt, out_dir, prefix, max_modules, manifest_entry)
    for run_dir, genome, stride in runs:
        by_gen = checkpoint_generations(run_dir, TASK)
        max_modules = run_config(run_dir, TASK).get("max_modules", 25)
        out_dir = video_root / genome / f"s{stride}"
        for gen in args.gens:
            picks = pick_best_median_worst(by_gen.get(gen, []))
            if not picks:
                console.log(f"[yellow]{run_dir.name}: no checkpoints for gen {gen}, skipping[/yellow]")
                continue
            for rank, (ckpt, fit) in picks.items():
                prefix = f"{run_dir.name}_gen{gen:03d}_{rank}_"
                entry = {
                    "run_dir": run_dir.name, "genome": genome, "stride": stride, "gen": gen,
                    "rank": rank, "n_in_gen": len(by_gen[gen]), "checkpoint": str(ckpt),
                    "fitness": fit, "video": str(out_dir / f"{prefix}{ckpt.name}_fwd.mp4"),
                }
                jobs.append((ckpt, out_dir, prefix, max_modules, entry))

    todo = [j for j in jobs if not Path(j[4]["video"]).exists()]
    console.log(f"{len(runs)} runs, {len(jobs)} videos ({len(jobs) - len(todo)} already exist), "
                f"rendering {len(todo)} with {args.workers} workers")

    args_dict = vars(args)
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(render_job, ckpt, out_dir, prefix, mm, args_dict): entry
            for ckpt, out_dir, prefix, mm, entry in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            entry = futures[fut]
            try:
                fut.result()
                console.log(f"[{i}/{len(todo)}] {Path(entry['video']).name}")
            except Exception as e:  # noqa: BLE001 -- keep going, report at the end
                failed.append((entry, repr(e)))
                console.log(f"[red][{i}/{len(todo)}] FAILED {entry['run_dir']} gen {entry['gen']} "
                            f"{entry['rank']}: {e!r}[/red]")

    manifest_path = video_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps([j[4] for j in jobs], indent=2))
    console.log(f"Manifest -> {manifest_path}")
    if failed:
        console.log(f"[red]{len(failed)} renders failed[/red]")
        for entry, err in failed:
            console.log(f"  {entry['run_dir']} gen {entry['gen']} {entry['rank']}: {err}")


if __name__ == "__main__":
    main()
