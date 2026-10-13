"""
Render videos across the fitness range of a control-stride sweep: for each
genome x stride, pool every individual of every rep and generation from
rowing/individuals.csv and pick the ones at a few fitness quantiles (0 = best,
1 = worst). Replays use the run's own control_step_freq and brain
(render_skill_task_checkpoint reads run_config.json), so this works for the
ANN and CPG sweeps alike.

Writes <stride-root>/fitness_range/:
  videos/<genome>_s<stride>_fit<fitness>_rep<rep>_<ckpt>_fwd.mp4
                                existing videos are skipped, so re-runs resume
  manifest.json                 which checkpoint each video is, its fitness and gait label
  fitness_range.html            one row per genome x stride, videos ordered best -> worst

Usage:
    MUJOCO_GL=egl python render_fitness_range.py --stride-root ../../__data__/ariel_control_stride_sweep_cpg
    MUJOCO_GL=egl python render_fitness_range.py --quantiles 0 0.5 1 --workers 8
"""

from __future__ import annotations

import argparse
import html
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import pandas as pd  # noqa: E402

from sweep_common import checkpoints_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK = "forward"
GENOMES = ("tree", "tree_symmetric")
GENOME_LABEL = {"tree": "tree", "tree_symmetric": "tree (symmetric)"}


def pick(ind: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """Per genome x stride, the individual nearest each fitness quantile (by rank)."""
    picks = []
    for (genome, stride), df in ind.groupby(["genome", "stride"]):
        df = df.sort_values("stored_fitness").reset_index(drop=True)
        for q in quantiles:
            row = df.iloc[round(q * (len(df) - 1))].copy()
            row["quantile"] = q
            picks.append(row)
    return pd.DataFrame(picks).drop_duplicates(["run", "gen", "body_idx"])


def video_prefix(e: dict) -> str:
    """Filename prefix carrying the fitness; render_checkpoint appends <ckpt>_fwd.mp4."""
    return f"{e['genome']}_s{e['stride']:03d}_fit{e['fitness']:+.3f}_rep{e['rep']}_"


def render_job(ckpt: Path, video_dir: Path, prefix: str) -> str:
    import render_skill_task_checkpoint

    results = render_skill_task_checkpoint.render_checkpoint(ckpt, video_dir, filename_prefix=prefix)
    return results[0]["video"] if results else ""


def write_page(entries: list[dict], out: Path) -> None:
    rows = []
    for (genome, stride), group in pd.DataFrame(entries).groupby(["genome", "stride"], sort=True):
        cells = []
        for e in group.sort_values("fitness").to_dict("records"):
            caption = (f"<b>{e['fitness']:.3f}</b> &middot; q{e['quantile']:.2f}<br>"
                       f"{html.escape(e['rowing_label'])} &middot; rep{e['rep']} gen{e['gen']} body{e['body_idx']}")
            cells.append(f'<figure><video src="videos/{html.escape(Path(e["video"]).name)}" controls loop muted '
                         f'preload="metadata"></video><figcaption>{caption}</figcaption></figure>')
        rows.append(f"<h2>{GENOME_LABEL.get(genome, genome)}, stride {stride} (~{500 / stride:.0f} Hz)</h2>"
                    f'<div class="row">{"".join(cells)}</div>')
    out.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Fitness range videos</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 16px; background: #fafaf8; color: #222; }}
.row {{ display: flex; gap: 10px; overflow-x: auto; }}
figure {{ margin: 0; flex: 0 0 300px; }}
video {{ width: 300px; border-radius: 4px; background: #000; }}
figcaption {{ font-size: 12px; color: #555; }}
h2 {{ font-size: 15px; margin: 22px 0 6px; }}
</style></head><body>
<h1>Gaits across the fitness range</h1>
<p>Per genome x stride, all reps and generations pooled; videos ordered best (most negative fitness) to worst.
Label = rowing-detector gait class. Click a video to play.</p>
{"".join(rows)}
</body></html>
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0, 0.05, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.stride_root / "fitness_range"
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    ind = pd.read_csv(args.stride_root / "rowing" / "individuals.csv")
    ind = ind[ind["genome"].isin(GENOMES)]
    picks = pick(ind, args.quantiles)

    entries, todo = [], []
    for r in picks.to_dict("records"):
        ckpt = checkpoints_dir(args.stride_root / r["run"], TASK) / f"gen{r['gen']:03d}_body{r['body_idx']:02d}"
        e = {"run": r["run"], "genome": r["genome"], "stride": int(r["stride"]), "rep": int(r["rep"]),
             "gen": int(r["gen"]), "body_idx": int(r["body_idx"]), "quantile": r["quantile"],
             "fitness": r["stored_fitness"], "rowing_label": r["rowing_label"], "checkpoint": str(ckpt)}
        e["video"] = str(video_dir / f"{video_prefix(e)}{ckpt.name}_fwd.mp4")
        entries.append(e)
        if not Path(e["video"]).exists():
            todo.append((ckpt, video_prefix(e)))
    print(f"{len(entries)} videos ({len(entries) - len(todo)} already exist), rendering {len(todo)} "
          f"with {args.workers} workers")

    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_job, ckpt, video_dir, prefix): (ckpt, prefix) for ckpt, prefix in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            ckpt, run = futures[fut]
            try:
                fut.result()
                print(f"[{i}/{len(todo)}] {run} {ckpt.name}")
            except Exception as e:  # noqa: BLE001 -- keep going, report at the end
                failed.append(f"{run} {ckpt.name}: {e!r}")
                print(f"[{i}/{len(todo)}] FAILED {run} {ckpt.name}: {e!r}")

    entries = [e for e in entries if Path(e["video"]).exists()]
    (out_dir / "manifest.json").write_text(json.dumps(entries, indent=2))
    write_page(entries, out_dir / "fitness_range.html")
    print(f"-> {out_dir / 'fitness_range.html'} ({len(entries)} videos)")
    for f in failed:
        print(f"  FAILED {f}")


if __name__ == "__main__":
    main()
