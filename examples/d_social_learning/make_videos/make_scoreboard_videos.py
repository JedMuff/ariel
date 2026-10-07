"""Render one video per (scheme, x): the highest-fitness individual in the
final generation, picked across all reps of that (scheme, x).

This is the same individual the final-generation scoreboard reports on.
Stored fitness_ is trusted (see make_run_top_videos.py), so no re-scoring.

Usage:
    uv run examples/d_social_learning/make_videos/make_scoreboard_videos.py \
        --data-dir __data__/snellius_data/ariel \
        [--out-dir __data__/snellius_data/videos] \
        [--schemes darwinian best ...] [--x-values 0 0.5 1] \
        [--duration 30.0] [--settle 3.0] [--fps 50]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

# Never put d_social_learning/ itself on sys.path — see make_run_top_videos.py.
_VIDEOS_DIR = Path(__file__).parent
for _p in [str(_VIDEOS_DIR), ""]:
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_VIDEOS_DIR.parent / "analysis"))

import imageio
import numpy as np
from rich.console import Console

from curve_utils import SCHEMES, X_VALUES, discover_reps, pure_displacement, rep_db_parts, x_str

_spec = importlib.util.spec_from_file_location("make_best_videos", _VIDEOS_DIR / "make_best_videos.py")
_mbv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mbv)
render_individual = _mbv.render_individual

console = Console()


def final_gen_best(rep_dir: Path) -> tuple[float, int, dict, dict] | None:
    """Return (fitness, generation, morph, tags) of the highest-fitness
    individual born in the rep's last generation, stitched across db parts
    (see curve_utils.load_run)."""
    rows = []
    covered_max = -1
    for db_path in rep_db_parts(rep_dir):
        con = sqlite3.connect(db_path)
        part = con.execute(
            "SELECT time_of_birth, fitness_, genotype_, tags_ FROM individual "
            "WHERE fitness_ IS NOT NULL"
        ).fetchall()
        con.close()
        new = [r for r in part if r[0] > covered_max and np.isfinite(r[1])]
        rows.extend(new)
        if new:
            covered_max = max(r[0] for r in new)
    if not rows:
        return None

    last_gen = max(r[0] for r in rows)
    birth, fit, geno_json, tags_json = max(
        (r for r in rows if r[0] == last_gen), key=lambda r: r[1],
    )
    return float(fit), birth, json.loads(geno_json)["morph"], json.loads(tags_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Experiment dir, e.g. __data__/snellius_data/ariel")
    parser.add_argument("--out-dir", default=None, help="Defaults to <data-dir>/../videos")
    parser.add_argument("--schemes", nargs="*", default=SCHEMES)
    parser.add_argument("--x-values", nargs="*", type=float, default=X_VALUES)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir.parent / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for xv in args.x_values:
        for scheme in args.schemes:
            picks = [
                (best, rep_dir) for rep_dir in discover_reps(data_dir, scheme, xv)
                if (best := final_gen_best(rep_dir)) is not None
            ]
            if not picks:
                console.log(f"[yellow]No data for {scheme} x={xv}[/yellow]")
                continue
            (fit, gen, morph, tags), rep_dir = max(picks, key=lambda p: p[0][0])
            disp = pure_displacement(tags)
            console.log(
                f"[cyan]{scheme} x={xv}[/cyan] {rep_dir.name} gen {gen}: "
                f"fitness={fit:.3f} displacement={disp:.3f} — rendering...",
            )
            frames = render_individual(
                morph, np.array(tags["theta"], dtype=np.float64),
                args.duration, args.settle, args.fps,
            )
            out_path = out_dir / (
                f"x{x_str(xv)}_{scheme}_{rep_dir.name}_gen{gen}"
                f"_fit{fit:.2f}_disp{disp:.2f}.mp4"
            )
            imageio.mimsave(str(out_path), frames, fps=args.fps)
            console.log(f"[green]Saved -> {out_path}[/green]")


if __name__ == "__main__":
    main()
