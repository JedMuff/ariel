"""Render a video of the top-N individuals from a single social-learning run.

Unlike make_best_videos.py (which re-scores every candidate because it pools
runs across scheme/x subdirectories whose fitness_ may predate the current
evaluator), this operates on one rep directory and trusts the stored
fitness_ values — they were produced by the current evaluator.py (verified:
re-scoring a candidate under evaluate() reproduces its stored fitness_
exactly), so re-simulating hundreds of individuals just to rank them would be
wasted work.

Stitches database.db with any database_part*.db continuation files (see
analysis/curve_utils.py's rep_db_parts/load_run) so individuals from resumed
runs aren't missed or double-counted.

Usage:
    uv run examples/d_social_learning/make_run_top_videos.py \
        --run-dir __data__/social/ariel/lamarckian/x10/rep_0 \
        [--out-dir __data__/social/ariel/lamarckian/x10/rep_0/videos] \
        [--top-n 3] [--duration 30.0] [--settle 3.0] [--fps 50]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

# Never put d_social_learning/ itself on sys.path — it contains an ariel/
# subdirectory that would shadow the installed ariel package for every
# subsequent `import ariel...`. Python auto-inserts the script's own
# directory, so drop that; add only the ariel-free analysis/ subdir.
_SOCIAL_DIR = Path(__file__).parent
for _p in [str(_SOCIAL_DIR), ""]:
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_SOCIAL_DIR / "analysis"))

import imageio
import numpy as np
from rich.console import Console

from curve_utils import rep_db_parts

# Load make_best_videos.py by file path (not `import make_best_videos`) so we
# reuse its render_individual() without adding _SOCIAL_DIR to sys.path.
_spec = importlib.util.spec_from_file_location("make_best_videos", _SOCIAL_DIR / "make_best_videos.py")
_mbv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mbv)
render_individual = _mbv.render_individual

console = Console()


def load_ranked_candidates(rep_dir: Path) -> list[tuple[float, int, dict, np.ndarray]]:
    """Return (fitness, birth_gen, morph, theta) for every individual in a
    rep, stitched across db parts (see curve_utils.load_run), sorted best
    first.
    """
    candidates = []
    covered_max = -1
    for db_path in rep_db_parts(rep_dir):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(
            "SELECT time_of_birth, fitness_, genotype_, tags_ FROM individual "
            "WHERE fitness_ IS NOT NULL"
        )
        rows = cur.fetchall()
        con.close()

        part_max = covered_max
        for birth, fit, geno_json, tags_json in rows:
            if birth <= covered_max:
                continue
            geno = json.loads(geno_json) if geno_json else {}
            tags = json.loads(tags_json) if tags_json else {}
            theta = tags.get("theta")
            morph = geno.get("morph")
            if theta and morph and np.isfinite(fit):
                candidates.append((float(fit), birth, morph, np.array(theta, dtype=np.float64)))
            part_max = max(part_max, birth)
        covered_max = part_max

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Rep directory, e.g. .../lamarckian/x10/rep_0")
    parser.add_argument("--out-dir", default=None, help="Defaults to <run-dir>/videos")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_ranked_candidates(run_dir)
    if not candidates:
        console.log(f"[red]No candidates found in {run_dir}[/red]")
        return
    console.log(f"Loaded {len(candidates)} candidates from {run_dir}")

    for rank, (fit, birth, morph, theta) in enumerate(candidates[: args.top_n], start=1):
        console.log(f"[cyan]#{rank}[/cyan] fitness={fit:.4f} (born gen {birth}) — rendering...")
        frames = render_individual(morph, theta, args.duration, args.settle, args.fps)
        if not frames:
            console.log(f"[red]No frames for rank {rank}[/red]")
            continue
        out_path = out_dir / f"top{rank}_fit{fit:.3f}_gen{birth}.mp4"
        imageio.mimsave(str(out_path), frames, fps=args.fps)
        console.log(f"[green]Saved -> {out_path}[/green]  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
