"""Render the ARIEL rows of __data__/videos/manifest.csv (select_stage_videos.py).

Each individual has 10 stored final evaluations (jittered starts drawn from
its eval_seed). Replays on this machine don't reproduce them bit for bit: the
runs were simulated on a cluster, and contact-rich gaits amplify floating-point
differences, so robots that move diverge a little. To film a representative
episode, all 10 are replayed here (physics only) and the one whose
displacement is closest to the stored mean is recorded, through
evosight_adapter.SocialLearningAdapter, with an info panel and a live
displacement bar on every frame (overlay.py). The chosen episode and its
replayed displacement go to __data__/videos/render_ariel.csv.
Existing videos are skipped, so a rerun only fills in what is missing.

Usage (from the repo root, GPU rendering):
    MUJOCO_GL=egl uv run examples/d_social_learning/make_videos/render_stage_videos_ariel.py --workers 16
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from overlay import InfoPanel, fmt  # noqa: E402

ADAPTER = str(HERE.parent / "evosight_adapter.py") + ":SocialLearningAdapter"
OUT = Path("__data__/videos")
_cache: dict[str, tuple] = {}
_opts: dict = {}


def panel_lines(r) -> list[str]:
    when = f"at its first selection, generation {r.selection_gen}" if r.generation == 0 \
        else f"at generation {r.selection_gen}'s selection"
    return [
        f"ARIEL  ·  {r.scheme}  ·  x = {r.x:g}  ·  rep {r.rep}",
        f"Generation {r.generation}  ·  {r.rank} of {r.pop_size} survivors (rank {r.pop_rank}/{r.pop_size})",
        f"Displacement {fmt(r.displacement)} m  (mean of 10 evaluations)",
        f"Novelty {fmt(r.novelty)}  ·  selection score {fmt(r.selection_score)}"
        f"  ({when})",
        f"Head start {fmt(r.init_fitness)} m  (inherited controller, before learning)",
    ]


def _adapter(run_dir: str):
    if run_dir not in _cache:
        from ariel.visualisation.evosight.adapter import load_adapter

        adapter = load_adapter(ADAPTER)
        adapter.prepare(Path(run_dir))
        adapter.setup_worker()
        _cache.clear()  # one run at a time per worker
        _cache[run_dir] = (adapter, adapter.load(Path(run_dir)))
    return _cache[run_dir]


def render(r) -> dict:
    from ariel.visualisation.evosight.recorders import VideoRecorder
    from ariel.visualisation.evosight.rollout import play

    adapter, run = _adapter(r.source)
    ind = run.individuals[int(r.individual_id)]
    n_eps = len(ind.tags.get("final_eval_displacement") or [None])
    disps = []
    for ep in range(n_eps):
        adapter.options["episode"] = str(ep)
        disps.append(float(play(adapter.make_rollout(ind), [])["displacement"]))
    episode = min(range(n_eps), key=lambda k: abs(disps[k] - r.displacement))
    adapter.options["episode"] = str(episode)
    rollout = adapter.make_rollout(ind)
    panel = InfoPanel(panel_lines(r), _opts["width"], _opts["height"])
    start: list[float] = []

    def overlay(frame, t, data):
        x = float(data.qpos[0])
        if not start:
            start.append(x)
        return panel.draw(frame, live=f"t = {t:4.1f} s      displacement so far {x - start[0]:+.2f} m")

    path = Path(r.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = VideoRecorder(rollout.model, path, width=_opts["width"], height=_opts["height"], fps=_opts["fps"],
                        crf=_opts["crf"], core_body=rollout.core_body, overlay=overlay)
    try:
        result = play(rollout, [rec])
    except BaseException:
        rec.abort()
        raise
    rec.close()
    return {"path": r.path, "episode": episode, "replayed_displacement": float(result["displacement"]),
                "stored_mean_displacement": float(r.displacement), "replayed_mean_displacement": float(sum(disps) / n_eps),
                "frames": rec.n_frames}


def _init(opts: dict) -> None:
    _opts.update(opts)


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
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=23)
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    todo = m[(m.domain == "ariel") & ~m.path.map(lambda p: Path(p).exists())]
    todo = todo.sort_values(["source", "generation"])  # a worker's consecutive jobs share a run
    if args.limit:
        todo = todo.head(args.limit)
    print(f"ARIEL: {len(todo)} of {(m.domain == 'ariel').sum()} videos to render", flush=True)
    opts = dict(width=args.width, height=args.height, fps=args.fps, crf=args.crf)

    log = OUT / "render_ariel.csv"
    done = [pd.read_csv(log)] if log.exists() else []
    rows, failed, t0 = [], 0, time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(opts,)) as pool:
        for i, (row, err) in enumerate(pool.imap(_work, todo.to_dict("records"), chunksize=1), 1):
            if err:
                failed += 1
                print(f"FAILED {err}", flush=True)
                continue
            rows.append(row)
            print(f"[{i}/{len(todo)}] {row['path']}  episode {row['episode']}  replay {row['replayed_displacement']:.4f} "
                  f"stored mean {row['stored_mean_displacement']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            pd.concat([*done, pd.DataFrame(rows)]).drop_duplicates("path", keep="last").to_csv(log, index=False)
    print(f"Done: {len(rows)} rendered, {failed} failed. Log: {log}")


if __name__ == "__main__":
    main()
