"""Render one video per (scheme, x): the robot with the highest learned
episode fitness (tags['distance']) across all reps of that (scheme, x),
either among those born in the final generation (--pick final, the
scoreboard's robot) or among every robot ever evaluated (--pick best).

The sim preset (experiment.py --sim-preset: physics, episode length, control
rate) is read from the runs' run_config.json unless --sim-preset is given, so
zoo runs are replayed with zoo physics. Stored fitness is trusted for
picking; --verify re-simulates each pick and prints the difference.

Usage:
    uv run examples/d_social_learning/make_videos/make_scoreboard_videos.py \
        --data-dir __data__/snellius_data/ariel \
        [--out-dir __data__/snellius_data/videos] \
        [--schemes darwinian best ...] [--x-values 0 0.5 1] \
        [--pick final|best] [--verify] [--sim-preset legacy|zoo] \
        [--duration SECONDS] [--settle 3.0] [--fps 50]
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

from curve_utils import SCHEMES, X_VALUES, discover_reps, rep_db_parts, x_str

_spec = importlib.util.spec_from_file_location("make_best_videos", _VIDEOS_DIR / "make_best_videos.py")
_mbv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mbv)
render_individual = _mbv.render_individual

console = Console()


def load_rows(rep_dir: Path) -> list[tuple]:
    """(birth, genotype json, tags json) of every evaluated individual,
    stitched across db parts (see curve_utils.load_run)."""
    rows = []
    covered_max = -1
    for db_path in rep_db_parts(rep_dir):
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        part = con.execute(
            "SELECT time_of_birth, genotype_, tags_ FROM individual WHERE fitness_ IS NOT NULL"
        ).fetchall()
        con.close()
        new = [r for r in part if r[0] > covered_max]
        rows.extend(new)
        if new:
            covered_max = max(r[0] for r in new)
    return rows


def pick(rep_dir: Path, mode: str) -> tuple[float, int, dict, dict] | None:
    """(episode fitness, generation, morph, tags) of the rep's robot with the
    highest learned episode fitness (tags['distance']): among those born in
    the last generation (mode 'final') or among all ('best'). fitness_ is not
    used: it is the pool-normalised selection blend, which at x < 1 also
    rewards novelty."""
    rows = []
    for birth, geno, tags_json in load_rows(rep_dir):
        tags = json.loads(tags_json)
        d = tags.get("distance")
        if d is not None and np.isfinite(d):
            rows.append((float(d), birth, geno, tags))
    if not rows:
        return None
    if mode == "final":
        last = max(r[1] for r in rows)
        rows = [r for r in rows if r[1] == last]
    fit, birth, geno, tags = max(rows, key=lambda r: r[0])
    return fit, birth, json.loads(geno)["morph"], tags


def displacement(tags: dict) -> float:
    """Episode fitness with the jerk penalty added back, under the active
    preset's jerk constants."""
    jerk = tags.get("mean_jerk", 0.0)
    penalty = _mbv.JERK_PENALTY_WEIGHT * jerk if jerk >= _mbv.JERK_THRESHOLD else 0.0
    return tags["distance"] + penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Experiment dir, e.g. __data__/snellius_data/ariel")
    parser.add_argument("--out-dir", default=None, help="Defaults to <data-dir>/../videos")
    parser.add_argument("--schemes", nargs="*", default=SCHEMES)
    parser.add_argument("--x-values", nargs="*", type=float, default=X_VALUES)
    parser.add_argument("--pick", choices=["final", "best"], default="final",
                        help="final: best robot born in the last generation; best: best robot ever found")
    parser.add_argument("--sim-preset", choices=sorted(_mbv.sim_evaluator.PRESETS), default=None,
                        help="default: read from the runs' run_config.json")
    parser.add_argument("--duration", type=float, default=None,
                        help="Rollout length (s); default: the sim preset's episode length")
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--verify", action="store_true",
                        help="Re-simulate each picked robot and compare with its stored fitness")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir.parent / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    preset = args.sim_preset or _mbv._runs_preset(data_dir, args.schemes)
    _mbv._apply_preset(preset)
    duration = args.duration if args.duration is not None else _mbv.DURATION
    console.log(f"sim preset: {preset}  (duration {duration}s, control every {_mbv.CTRL_EVERY} steps)")

    for xv in args.x_values:
        for scheme in args.schemes:
            picks = [
                (best, rep_dir) for rep_dir in discover_reps(data_dir, scheme, xv)
                if (best := pick(rep_dir, args.pick)) is not None
            ]
            if not picks:
                console.log(f"[yellow]No data for {scheme} x={xv}[/yellow]")
                continue
            (fit, gen, morph, tags), rep_dir = max(picks, key=lambda p: p[0][0])
            disp = displacement(tags)
            theta = np.array(tags["theta"], dtype=np.float64)
            msg = (f"[cyan]{scheme} x={xv}[/cyan] {rep_dir.name} gen {gen}: "
                   f"fitness={fit:.3f} displacement={disp:.3f} jerk={tags.get('mean_jerk', float('nan')):.3f}")
            if args.verify:
                rescored = _mbv.evaluate(morph, theta, duration, args.settle)
                msg += f" re-simulated={rescored:.3f} (diff {rescored - fit:+.2e})"
            console.log(msg + " — rendering...")
            frames = render_individual(morph, theta, duration, args.settle, args.fps)
            out_path = out_dir / (
                f"x{x_str(xv)}_{scheme}_{rep_dir.name}_gen{gen}"
                f"_fit{fit:.2f}_disp{disp:.2f}.mp4"
            )
            imageio.mimsave(str(out_path), frames, fps=args.fps)
            console.log(f"[green]Saved -> {out_path}[/green]")


if __name__ == "__main__":
    main()
