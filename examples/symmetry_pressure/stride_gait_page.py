"""
Interactive page for stride_gait_similarity.py output: the most central gait
of each genome x stride experiment and how far the rest of its population is
from it.

For each experiment (final generation, 3 reps pooled) the page shows:
  - an overview table of the medoid's median distance and IQR to the pool,
    tree vs tree_symmetric per stride (click a cell to open that experiment);
  - a 2D MDS map of the pool's gait distances (as zoo_benchmark's
    gait_similarity_page.py); later variants are rotated onto the first;
  - a strip of the medoid's distances to every pooled individual with the IQR
    band;
  - a detail card per individual: video, fitness, gait-cycle stats, behavioural
    descriptors and the pool ranked by distance.
Videos are rendered only for the medoid's own run: the medoid, its --k nearest
and --k farthest individuals and --n-mid in between (by cycle_shape distance),
with render_skill_task_checkpoint.render_checkpoint, then transcoded to H.264
(gait_similarity_page.browser_video) next to the page, which references them by
relative path: keep the page and videos/ together.

Usage:
    MUJOCO_GL=egl python stride_gait_page.py [--k 5] [--n-mid 3] [--workers 16] [--no-videos]
Writes <stride-root>/gait_similarity/page/stride_gait_similarity.html.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "zoo_benchmark"))

from gait_similarity_page import CYCLE_STATS, DESCRIPTOR_GROUPS, browser_video  # noqa: E402
from plot_gait_similarity import mds_embedding  # noqa: E402
from stride_gait_similarity import GENOMES, VARIANTS, medoid_stats  # noqa: E402

PAGE_TEMPLATE = Path(__file__).with_name("stride_gait_page.html")
TASK = "forward"
CYCLE_STATS = {**CYCLE_STATS, "aperiodic": "no period found (whole episode used)"}
VARIANT_INFO = {
    "cycle_shape": {"name": "Gait cycle shape", "unit": None,
                    "note": "Mean gait cycles scaled to the same amplitude: only the shape and timing of the motion "
                            "count. Dimensionless (0 = identical shape, about 1.4 = unrelated)."},
    "cycle": {"name": "Gait cycle (mm)", "unit": "m",
              "note": "Mean gait cycles in metres: amplitude counts too, and it grows with body size."},
}


def _num(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def checkpoint_dir(root: Path, run: str, gen: int, body_idx: int) -> Path:
    return root / run / "__data__" / "gecko_skill_tasks" / TASK / "checkpoints" / f"gen{gen:03d}_body{body_idx:02d}"


def pick_groups(dist: pd.DataFrame, meta: pd.DataFrame, medoid: str, k: int, n_mid: int) -> dict[str, str]:
    """Group of every pooled individual: the medoid, and within its run the k
    nearest / k farthest / n_mid spread in between; the rest of the run is
    'run', other reps 'pool'."""
    groups = {label: "pool" for label in dist.index}
    same_run = meta.index[meta["run"] == meta.loc[medoid, "run"]].drop(medoid)
    ranked = dist.loc[medoid, same_run].sort_values().index.tolist()
    for label in ranked:
        groups[label] = "run"
    near, far = ranked[:k], ranked[max(k, len(ranked) - k):]
    middle = ranked[len(near):len(ranked) - len(far)]
    if middle and n_mid > 0:
        idx = np.unique(np.linspace(0, len(middle) - 1, n_mid + 2)[1:-1].round().astype(int))
        for i in idx:
            groups[middle[i]] = "mid"
    for label in near:
        groups[label] = "near"
    for label in far:
        groups[label] = "far"
    groups[medoid] = "medoid"
    return groups


def load_descriptors(root: Path, subdir: str, run: str) -> pd.DataFrame:
    path = root / subdir / run / "descriptors.csv"
    df = pd.read_csv(path)
    return df[df["skill"] == "fwd"].set_index(["gen", "body_idx"])


def pool_layout(data_dir: Path, key: str, k: int, n_mid: int) -> dict | None:
    """Everything the page and plot_gait_similarity_stride_sweep.py draw for one
    pooled experiment: meta, distance matrices, medoid stats, groups (picked
    by the first variant's medoid) and MDS maps (later variants rotated onto
    the first)."""
    paths = {v: data_dir / f"pool_{key}_{v}.csv" for v in VARIANTS}
    if not all(p.exists() for p in paths.values()):
        return None
    meta = pd.read_csv(data_dir / f"pool_{key}_meta.csv", index_col="label")
    dists = {v: pd.read_csv(p, index_col=0).loc[meta.index, meta.index] for v, p in paths.items()}
    labels = list(meta.index)
    stats = {v: medoid_stats(d.to_numpy()) for v, d in dists.items()}
    medoid = labels[stats[VARIANTS[0]]["medoid"]]
    xy, stress, ref = {}, {}, None
    for v, d in dists.items():
        xy[v], stress[v] = mds_embedding(d.to_numpy())
        xy[v] -= xy[v].mean(axis=0)
        if ref is not None:
            xy[v] = xy[v] @ orthogonal_procrustes(xy[v], ref)[0]
        else:
            ref = xy[v]
    return {"key": key, "labels": labels, "meta": meta, "dists": dists, "stats": stats, "medoid": medoid,
            "groups": pick_groups(dists[VARIANTS[0]], meta, medoid, k, n_mid), "xy": xy, "stress": stress}


def experiment_data(data_dir: Path, root: Path, subdir: str, genome: str, stride: int,
                    k: int, n_mid: int) -> tuple[dict, list[tuple]] | None:
    key = f"{genome}_s{stride}"
    lay = pool_layout(data_dir, key, k, n_mid)
    if lay is None:
        return None
    meta, dists, labels, stats = lay["meta"], lay["dists"], lay["labels"], lay["stats"]
    medoid, groups = lay["medoid"], lay["groups"]
    desc = {run: load_descriptors(root, subdir, run) for run in meta["run"].unique()}

    variants = {}
    for v, d in dists.items():
        xy, stress = lay["xy"][v], lay["stress"][v]
        align = pd.read_csv(data_dir / f"pool_{key}_{v}_alignment.csv")
        pos = {label: i for i, label in enumerate(labels)}
        n = len(labels)
        mirror, yaw, phase = np.zeros((n, n), int), np.zeros((n, n)), np.zeros((n, n))
        for row in align.itertuples(index=False):
            i, j = pos[row.a], pos[row.b]
            mirror[i, j] = mirror[j, i] = int(row.mirrored)
            yaw[i, j] = yaw[j, i] = row.yaw_deg
            phase[i, j], phase[j, i] = row.phase_shift, (1 - row.phase_shift) % 1
        st = stats[v]
        variants[v] = {
            **VARIANT_INFO[v], "stress": stress,
            "xy": [[round(float(x), 5), round(float(y), 5)] for x, y in xy],
            "dist": np.round(d.to_numpy(), 5).tolist(),
            "mirror": mirror.tolist(), "yaw": np.round(yaw).astype(int).tolist(), "phase": np.round(phase, 3).tolist(),
            "medoid_median": st["medoid_median"], "medoid_q25": st["medoid_q25"], "medoid_q75": st["medoid_q75"],
            "medoid_iqr": st["medoid_iqr"], "pairs_median": st["pairs_median"], "pairs_iqr": st["pairs_iqr"],
            "medoid": labels[st["medoid"]],
        }

    individuals, renders = [], []
    for label in labels:
        m = meta.loc[label]
        gen, body = int(m["gen"]), int(m["body_idx"])
        try:
            d = desc[m["run"]].loc[(gen, body)]
            descriptors = {name: _num(d[f"d_{name}"]) for names in DESCRIPTOR_GROUPS.values() for name in names}
        except KeyError:
            descriptors = {}
        individuals.append({
            "label": label, "run": m["run"], "rep": int(m["rep"]), "gen": gen, "body_idx": body,
            "ind_id": int(m["ind_id"]), "fitness": _num(m["stored_fitness"]), "n_hinges": int(m["n_joints"]),
            "group": groups[label], "video": None, "descriptors": descriptors,
            "cycle": {name: _num(m[name]) for name in CYCLE_STATS},
        })
        if groups[label] not in ("run", "pool"):
            renders.append((key, label, checkpoint_dir(root, m["run"], gen, body), m["run"]))
    exp = {"key": key, "genome": genome, "stride": stride, "medoid": medoid,
           "medoid_run": meta.loc[medoid, "run"], "individuals": individuals, "variants": variants}
    return exp, renders


def _pool_init() -> None:
    import torch
    torch.set_num_threads(1)


def render_one(job: tuple) -> tuple[str, str, Path | None, str]:
    key, label, ckpt, run, raw_dir = job
    import render_skill_task_checkpoint

    out = raw_dir / f"{run}_{ckpt.name}_fwd.mp4"
    if out.exists():
        return key, label, out, ""
    try:
        results = render_skill_task_checkpoint.render_checkpoint(ckpt, raw_dir, filename_prefix=f"{run}_")
        return key, label, Path(results[0]["video"]) if results else None, ""
    except Exception as exc:  # noqa: BLE001 -- a failed render must not kill the page
        return key, label, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
    parser.add_argument("--behavior-subdir", default="behavior_50hz")
    parser.add_argument("--data-dir", type=Path, default=None, help="default: <stride-root>/gait_similarity")
    parser.add_argument("--k", type=int, default=5, help="nearest and farthest individuals shown with video")
    parser.add_argument("--n-mid", type=int, default=3, help="individuals in between shown with video")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir or args.stride_root / "gait_similarity"
    page_dir = data_dir / "page"
    page_dir.mkdir(parents=True, exist_ok=True)

    medoids = pd.read_csv(data_dir / "medoid_summary.csv")
    strides = sorted(medoids.loc[medoids["scope"] == "pool", "stride"].unique())
    experiments, renders = [], []
    for stride in strides:
        for genome in GENOMES:
            res = experiment_data(data_dir, args.stride_root, args.behavior_subdir, genome, int(stride),
                                  args.k, args.n_mid)
            if res is None:
                print(f"[warn] no pooled distances for {genome} s{stride}")
                continue
            experiments.append(res[0])
            renders += res[1]
    if not experiments:
        raise SystemExit(f"No pool_*.csv in {data_dir}; run stride_gait_similarity.py")

    if not args.no_videos and renders:
        raw_dir = page_dir / "videos_raw"
        raw_dir.mkdir(exist_ok=True)
        by_key = {e["key"]: {i["label"]: i for i in e["individuals"]} for e in experiments}
        print(f"Rendering {len(renders)} videos ({args.workers} workers) ...")
        # Spawned, single-threaded workers: forking after the MDS above (BLAS/OpenMP
        # threads in this process) left some forked workers spinning forever.
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=_pool_init) as pool:
            futures = [pool.submit(render_one, (*r, raw_dir)) for r in renders]
            for future in as_completed(futures):
                key, label, src, err = future.result()
                if src is None:
                    print(f"[warn] {key} {label}: no video {err}")
                    continue
                video = browser_video(src, page_dir / "videos")
                by_key[key][label]["video"] = video.relative_to(page_dir).as_posix()

    overview = medoids[medoids["scope"] == "pool"][
        ["genome", "stride", "variant", "n", "medoid_label", "medoid_median", "medoid_q25", "medoid_q75",
         "medoid_iqr", "pairs_median", "pairs_iqr"]]
    data = {
        "title": "Gait convergence under symmetry",
        "variants": list(VARIANTS), "experiments": experiments,
        "overview": overview.to_dict(orient="records"), "k": args.k,
        "groups": {**DESCRIPTOR_GROUPS, "Gait cycle": list(CYCLE_STATS)}, "cycle_names": CYCLE_STATS,
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = page_dir / "stride_gait_similarity.html"
    out.write_text(PAGE_TEMPLATE.read_text().replace("/*__DATA__*/null", payload))
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
