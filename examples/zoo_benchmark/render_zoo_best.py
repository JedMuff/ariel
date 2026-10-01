"""Render the best champion of every body x brain cell of the zoo benchmark.

Picks, per (body, brain), the rep with the highest best_xspeed from
<root>/summary.csv (written by aggregate_zoo_benchmark.py) and renders it with
render_zoo_run.render_run into <out-dir>/<body>_<brain>.mp4.

Usage:
  MUJOCO_GL=egl python render_zoo_best.py __data__/ariel_zoo_benchmark [-o OUTDIR] [-j 8]
  MUJOCO_GL=egl python render_zoo_best.py ROOT --bodies gecko ant --brains ann
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

from render_zoo_run import render_run


def _render(run_dir: Path, out_path: Path, fps: int, width: int, height: int) -> Path:
    torch.set_num_threads(1)
    return render_run(run_dir, fps, width, height, out_path=out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="default: <root>/best_videos")
    parser.add_argument("-j", "--jobs", type=int, default=8)
    parser.add_argument("--bodies", nargs="+", default=None)
    parser.add_argument("--brains", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    df = pd.read_csv(args.root / "summary.csv")
    if args.bodies:
        df = df[df["body"].isin(args.bodies)]
    if args.brains:
        df = df[df["brain"].isin(args.brains)]
    best = df.loc[df.groupby(["body", "brain"])["best_xspeed"].idxmax()].sort_values(["body", "brain"])

    out_dir = args.out_dir or args.root / "best_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    best[["body", "brain", "seed", "best_xspeed", "run_dir"]].to_csv(out_dir / "best_runs.csv", index=False)

    tasks = []
    for row in best.itertuples():
        out_path = out_dir / f"{row.body}_{row.brain}.mp4"
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append((args.root / row.run_dir, out_path))
    print(f"{len(best)} cells, {len(tasks)} to render -> {out_dir}")

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_render, run_dir, out_path, args.fps, args.width, args.height)
                   for run_dir, out_path in tasks]
        for fut in as_completed(futures):
            fut.result()


if __name__ == "__main__":
    main()
