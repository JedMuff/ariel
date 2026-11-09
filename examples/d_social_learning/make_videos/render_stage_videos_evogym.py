"""Render the EvoGym rows of __data__/videos/manifest.csv (select_stage_videos.py).

Each video replays the individual's learned controller (tags theta) for one
499-step Walker episode with EvoGym's own renderer, with an info panel and a
live displacement bar on every frame (overlay.py). EvoGym is deterministic,
so the replayed displacement should equal the stored distance; both go to
__data__/videos/render_evogym.csv. Existing videos are skipped.

Needs the Python 3.10 EvoGym venv and a display; from the repo root:
    PYTHONPATH=src xvfb-run -a -s "-screen 0 1280x1024x24" \\
        evogym-venv/bin/python examples/d_social_learning/make_videos/render_stage_videos_evogym.py --workers 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOCIAL = HERE.parent
sys.path.insert(0, str(HERE))
from overlay import InfoPanel, fmt  # noqa: E402

OUT = Path("__data__/videos")
_opts: dict = {}


def panel_lines(r) -> list[str]:
    when = f"at its first selection, generation {r.selection_gen}" if r.generation == 0 \
        else f"at generation {r.selection_gen}'s selection"
    return [
        f"EvoGym  ·  {r.scheme}  ·  x = {r.x:g}  ·  rep {r.rep}",
        f"Generation {r.generation}  ·  {r.rank} of {r.pop_size} survivors (rank {r.pop_rank}/{r.pop_size})",
        f"Displacement {fmt(r.displacement)} voxels  (best learning episode)",
        f"Novelty {fmt(r.novelty)}  ·  selection score {fmt(r.selection_score)}  ({when})",
        f"Head start {fmt(r.init_fitness)} voxels  (inherited controller, before learning)",
    ]


def _setup() -> None:
    # Only the evogym simulator dir and d_social_learning's ariel-free modules:
    # d_social_learning/ itself would shadow the ariel package (see evaluator.py).
    sys.path.insert(0, str(SOCIAL / "sim_evogym"))
    import importlib.util

    for name in ("simulator_dependent_functions",):
        spec = importlib.util.spec_from_file_location(name, SOCIAL / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    import simulator_dependent_functions as sdf

    sdf.simulator = sdf.SIMULATOR_EVOGYM


def render(r) -> dict:
    import imageio.v2 as imageio
    import evaluator  # sim_evogym/evaluator.py
    import simulator_dependent_functions as sdf

    from ariel.simulation.controllers.distributed_mlp import DistributedMLP

    con = sqlite3.connect(f"file:{r.source}?mode=ro", uri=True)
    theta_json, morph_json = con.execute(
        "SELECT json_extract(tags_, '$.theta'), json_extract(genotype_, '$.morph') FROM individual WHERE id = ?",
        (int(r.individual_id),)).fetchone()
    con.close()
    theta = np.asarray(json.loads(theta_json), dtype=np.float64)
    brain = DistributedMLP(n_neighbors=sdf.n_neighbours(), hidden=16)

    spec = evaluator.initialize_world(json.loads(morph_json), render_mode="rgb_array")
    viewer = spec["env"].unwrapped.default_viewer
    viewer.set_resolution((_opts["width"], _opts["height"]))
    viewer.set_view_size((_opts["view"], _opts["view"] * _opts["height"] / _opts["width"]))
    viewer.track_objects("robot")
    sim = spec["sim"]
    x0 = float(sim.object_pos_at_time(sim.get_time(), "robot")[0].mean())
    panel = InfoPanel(panel_lines(r), _opts["width"], _opts["height"])

    path = Path(r.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.mp4")
    writer = imageio.get_writer(tmp, fps=_opts["fps"], codec="libx264", format="FFMPEG", macro_block_size=16,
                                ffmpeg_log_level="error",
                                ffmpeg_params=["-crf", str(_opts["crf"]), "-pix_fmt", "yuv420p",
                                               "-movflags", "+faststart"])
    n = 0

    def recorder(step, sim, env):
        nonlocal n
        frame = env.render()
        x = float(sim.object_pos_at_time(sim.get_time(), "robot")[0].mean())
        writer.append_data(panel.draw(np.ascontiguousarray(frame[..., :3]),
                                      live=f"step {step + 1:3d}/{evaluator.N_STEPS}"
                                           f"      displacement so far {x - x0:+.2f} voxels"))
        n += 1

    try:
        result = evaluator.run_episode(theta, brain, spec, recorder=recorder)
    finally:
        writer.close()
        evaluator.kill_world(spec)
    tmp.replace(path)
    return dict(path=r.path, replayed_displacement=float(result["fitness"]),
                stored_displacement=float(r.displacement), frames=n)


def _init(opts: dict) -> None:
    _opts.update(opts)
    _setup()


def _work(job: dict) -> tuple[dict | None, str | None]:
    r = SimpleNamespace(**job)
    try:
        return render(r), None
    except Exception:  # noqa: BLE001 - report and carry on
        return None, f"{r.path}\n{traceback.format_exc()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=OUT / "manifest.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="only the first N missing videos (testing)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--view", type=float, default=24.0, help="voxels visible across the width")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--crf", type=int, default=23)
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    todo = m[(m.domain == "evogym") & ~m.path.map(lambda p: Path(p).exists())]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"EvoGym: {len(todo)} of {(m.domain == 'evogym').sum()} videos to render", flush=True)
    opts = dict(width=args.width, height=args.height, view=args.view, fps=args.fps, crf=args.crf)

    log = OUT / "render_evogym.csv"
    done = [pd.read_csv(log)] if log.exists() else []
    rows, failed, t0 = [], 0, time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(opts,)) as pool:
        for i, (row, err) in enumerate(pool.imap(_work, todo.to_dict("records"), chunksize=1), 1):
            if err:
                failed += 1
                print(f"FAILED {err}", flush=True)
                continue
            rows.append(row)
            diff = abs(row["replayed_displacement"] - row["stored_displacement"])
            flag = "  <-- replay differs" if diff > 1e-3 else ""
            print(f"[{i}/{len(todo)}] {row['path']}  replay {row['replayed_displacement']:.3f} "
                  f"stored {row['stored_displacement']:.3f}{flag}  ({time.time() - t0:.0f}s)", flush=True)
            pd.concat(done + [pd.DataFrame(rows)]).drop_duplicates("path", keep="last").to_csv(log, index=False)
    print(f"Done: {len(rows)} rendered, {failed} failed. Log: {log}")


if __name__ == "__main__":
    main()
