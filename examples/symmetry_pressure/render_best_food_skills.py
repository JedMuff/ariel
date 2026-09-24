"""
Render videos of the 3 best individuals (one per run, ranked by food-task
fitness) for each condition: symmetry-enforced (y_zero) and no symmetry (none).

Reuses render_checkpoint() from render_food_skills_checkpoint.py.

Requires MuJoCo with EGL rendering — run on a SLURM GPU node or a machine
with a GPU (MUJOCO_GL=egl) or a local display (MUJOCO_GL=glfw).

Output layout (--out-dir):
    sym_rank1_<run>_<ckpt>.mp4
    sym_rank2_<run>_<ckpt>.mp4
    sym_rank3_<run>_<ckpt>.mp4
    nosym_rank1_<run>_<ckpt>.mp4
    nosym_rank2_<run>_<ckpt>.mp4
    nosym_rank3_<run>_<ckpt>.mp4
    manifest.json

Usage:
    python render_best_food_skills.py \\
        --sym-dirs   ../../__data__/food_skills/food_skills_..._35117 ... \\
        --nosym-dirs ../../__data__/food_skills/food_skills_..._35301 ... \\
        --out-dir    ../../__data__/food_skills_videos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from render_food_skills_checkpoint import render_checkpoint

GLITCH_FIT = 1.0

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--sym-dirs",    nargs="+", type=Path, required=True)
parser.add_argument("--nosym-dirs",  nargs="+", type=Path, required=True)
parser.add_argument("--out-dir",     type=Path, default=Path("__data__/food_skills_videos"))
parser.add_argument("--top-n",       type=int, default=3,
                    help="Number of best individuals to render per group (one per run)")
parser.add_argument("--reach-radius",type=float, default=0.20)
parser.add_argument("--seed",        type=int, default=None,
                    help="SkillController RNG seed. Default: reconstruct each "
                         "checkpoint's exact training seed (see "
                         "render_food_skills_checkpoint._reconstruct_food_seed).")
parser.add_argument("--render-fps",  type=int, default=30)
parser.add_argument("--render-height",type=int, default=480)
parser.add_argument("--render-width", type=int, default=640)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_record(run_dir: Path) -> dict | None:
    """Return the single best (lowest fitness, non-glitch) record in the run."""
    data_path = run_dir / "__data__" / "gecko_food_skills" / "run_data.jsonl"
    if not data_path.exists():
        return None
    records = [json.loads(l) for l in data_path.open()]
    valid = [
        r for r in records
        if r.get("fitness") is not None
        and r["fitness"] != GLITCH_FIT
        and __import__("math").isfinite(r["fitness"])
    ]
    if not valid:
        return None
    best = min(valid, key=lambda r: r["fitness"])
    best["_run_dir"] = str(run_dir)
    return best


def _find_checkpoint(run_dir: Path, target_fitness: float) -> Path | None:
    """Return the checkpoint directory whose meta.json fitness is closest to target."""
    ckpt_base = run_dir / "__data__" / "gecko_food_skills" / "checkpoints"
    best_ckpt, best_diff = None, float("inf")
    for ckpt in sorted(ckpt_base.iterdir()):
        meta_path = ckpt / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        diff = abs(meta.get("fitness", float("nan")) - target_fitness)
        if diff < best_diff:
            best_diff = diff
            best_ckpt = ckpt
    return best_ckpt


def top_individuals(run_dirs: list[Path], top_n: int) -> list[dict]:
    """
    For each run pick the best individual, then return the globally top-n
    (sorted by fitness ascending, i.e. most negative first).
    One individual per run — different runs only.
    """
    per_run = []
    for d in run_dirs:
        rec = _best_record(d)
        if rec is None:
            continue
        ckpt = _find_checkpoint(d, rec["fitness"])
        if ckpt is None:
            continue
        per_run.append({
            "run_name":  d.name,
            "fitness":   rec["fitness"],
            "gen":       rec["gen"],
            "ckpt_dir":  ckpt,
        })
    per_run.sort(key=lambda x: x["fitness"])
    return per_run[:top_n]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    manifest = []

    for group_label, run_dirs in [
        ("sym",   args.sym_dirs),
        ("nosym", args.nosym_dirs),
    ]:
        print(f"\n{'='*60}")
        print(f"Group: {group_label}")
        individuals = top_individuals(list(run_dirs), args.top_n)

        for rank, ind in enumerate(individuals, 1):
            run_short = ind["run_name"].split("_")[-1]
            vid_name = (
                f"{group_label}_rank{rank:02d}"
                f"_run{run_short}"
                f"_{ind['ckpt_dir'].name}"
                f"_fit{abs(ind['fitness']):.3f}"
                ".mp4"
            )
            out_path = args.out_dir / vid_name

            print(
                f"\n  Rank {rank}: run={run_short}  gen={ind['gen']}"
                f"  fitness={ind['fitness']:.4f}"
                f"\n  Checkpoint: {ind['ckpt_dir']}"
                f"\n  -> {out_path.name}"
            )

            result = render_checkpoint(
                ckpt_dir=ind["ckpt_dir"],
                out_path=out_path,
                reach_radius=args.reach_radius,
                seed=args.seed,
                render_fps=args.render_fps,
                render_height=args.render_height,
                render_width=args.render_width,
            )
            manifest.append({
                "group":    group_label,
                "rank":     rank,
                "run_name": ind["run_name"],
                "gen":      ind["gen"],
                "fitness":  ind["fitness"],
                "ckpt_dir": str(ind["ckpt_dir"]),
                "video":    str(out_path),
                **result,
            })

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
