"""
Review page for rowing_detect.py: label a sample of individuals by hand to
check the detector.

The sample (--n, default 96) is stratified over genome x stride. Within each
stratum it mixes:
  - one individual per detector label (round robin);
  - borderline cases (rowing_frac in [0.25, 0.75] or rowing_score in [0.4, 0.8]);
  - a spread over rowing_score quantiles, mostly from generation >= --late-gen.
Survivors re-evaluated in later generations are counted once.

Per individual the page shows:
  - the replay video;
  - an animated skeleton of the mean gait cycle in the split-plane frame (top and
    side view): left modules blue, right orange, pairs joined, the heading
    arrow and the split line, so you can check the split and the pairing;
  - left vs mirrored-right traces per pair;
  - the detector's metrics.
In blind mode (default) the verdict, metrics and traces are hidden until you
label the individual.

Labels autosave in the browser (localStorage, shared by every rebuild of the
page) and are exported with "Export labels" as rowing_labels.json; feed that to
rowing_validate.py. --only-disagreements rebuilds the page with only the
individuals your labels disagree with the detector on.

Videos: render_skill_task_checkpoint via stride_gait_page.render_one, H.264
copies next to the page (keep the page and videos/ together). Raw renders
already made for stride_gait_page.py are reused.

Usage:
    MUJOCO_GL=egl python rowing_review_page.py [--n 96] [--seed 0] [--workers 16] [--no-videos]
    python rowing_review_page.py --only-disagreements rowing_labels.json
Writes <stride-root>/rowing/page/rowing_review.html.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "examples" / "zoo_benchmark"))

from gait_similarity_page import browser_video  # noqa: E402
from rowing_detect import LABELS, analyse_episode, review_detail  # noqa: E402
from rowing_validate import disagreements, load_labels  # noqa: E402
from stride_gait_page import _pool_init, checkpoint_dir, render_one  # noqa: E402
from stride_gait_similarity import META_COLS, iter_episode_poses  # noqa: E402

PAGE_TEMPLATE = Path(__file__).with_name("rowing_review_page.html")
IND_COLS = ("rowing_label", "rowing_score", "rowing_frac", "alternating_frac", "aperiodic_frac", "paired_frac",
            "heading_deg", "split_offset_deg", "heading_consistency", "speed", "body_size_m", "n_moving",
            "n_pairs", "n_rowing_pairs", "n_alternating_pairs", "morph_plane_offset_deg", "midline_amp_frac")
PAIR_COLS = ("left", "right", "pair_kind", "mirror_sync", "phase_lag", "freq_l", "freq_r", "periodicity_l",
             "periodicity_r", "amp_l", "amp_r", "amp_ratio", "stroke_fore", "stroke_lat", "stroke_up", "rest_dist")


def key_of(run: str, gen: int, body_idx: int) -> str:
    return f"{run}/{int(gen)}/{int(body_idx)}"


# ── Sample ────────────────────────────────────────────────────────────────────


def stratified_sample(df: pd.DataFrame, n: int, seed: int, late_gen: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["uid"] = np.where(df["ind_id"] >= 0, df["run"] + ":" + df["ind_id"].astype(str), df.index.astype(str))
    # A re-evaluated survivor appears in several gens: keep its latest evaluation.
    df = df.sort_values("gen").drop_duplicates("uid", keep="last")
    strata = sorted(df.groupby(["genome", "stride"]).groups.items())
    per = int(np.ceil(n / len(strata)))
    picked: list[int] = []
    for _, idx in strata:
        s = df.loc[idx]
        mine: list[int] = []

        def take(cands: pd.Index, k: int) -> None:
            cands = [i for i in cands if i not in mine]
            if cands and k > 0:
                w = 1.0 + df.loc[cands, "gen"].to_numpy() / 10.0  # lean towards later generations
                mine.extend(rng.choice(cands, size=min(k, len(cands)), replace=False, p=w / w.sum()).tolist())

        for lab in LABELS:                                            # one per detector label
            take(s.index[s["rowing_label"] == lab], 1)
        border = s.index[s["rowing_frac"].between(0.25, 0.75) | s["rowing_score"].between(0.4, 0.8)]
        take(border, per // 4)
        late = s[s["gen"] >= late_gen].sort_values("rowing_score")    # quantile spread
        rest = [i for i in late.index if i not in mine]
        k = per - len(mine)
        if rest and k > 0:
            mine += [rest[j] for j in np.unique(np.linspace(0, len(rest) - 1, k).round().astype(int))]
        take(s.index, per - len(mine))                                # top up from anywhere
        picked += mine[:per]
    return df.loc[picked].drop(columns="uid")


# ── Details ───────────────────────────────────────────────────────────────────


def _details_job(job: tuple[Path, list[tuple[int, int]], float]) -> dict[tuple[int, int], dict]:
    npz, wanted, snap_deg = job
    meta, episodes = iter_episode_poses(npz, META_COLS + ("body_names_json",))
    lookup = {(int(g), int(b)): e for e, (g, b) in enumerate(zip(meta["gen"], meta["body_idx"]))}
    want_e = {lookup[w]: w for w in wanted if w in lookup}
    out = {}
    for e, P, Q, dt in episodes:
        if e not in want_e:
            continue
        ind, pairs, d = analyse_episode(P, Q, dt, detail=True, snap_deg=snap_deg)
        rd = review_detail(d)
        for t in rd["traces"]:
            t.pop("lat_l"), t.pop("lat_r")
        names = [n.removeprefix("robot1_") for n in json.loads(meta.loc[e, "body_names_json"])]
        out[want_e[e]] = {"detail": rd, "heading_rel_deg": float(np.rad2deg(d["heading_rel"])), "names": names,
                          "pairs": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()
                                     if k in PAIR_COLS} for p in pairs]}
    return out


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return round(x, 4) if np.isfinite(x) else None


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
    parser.add_argument("--behavior-subdir", default="behavior_50hz")
    parser.add_argument("--rowing-dir", type=Path, default=None, help="default: <stride-root>/rowing")
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--late-gen", type=int, default=15)
    parser.add_argument("--only-disagreements", type=Path, default=None, metavar="LABELS_JSON",
                        help="page of only the individuals these labels disagree with the detector on")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--reuse-videos", type=Path, default=None,
                        help="raw renders to reuse (default: <stride-root>/gait_similarity/page/videos_raw)")
    parser.add_argument("--out", type=Path, default=None, help="default: <rowing-dir>/page/rowing_review.html")
    args = parser.parse_args()
    rowing_dir = args.rowing_dir or args.stride_root / "rowing"
    out = args.out or rowing_dir / "page" / "rowing_review.html"
    page_dir = out.parent
    page_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(rowing_dir / "individuals.csv")
    if args.only_disagreements:
        dis = disagreements(load_labels(args.only_disagreements), df)
        sample = df.merge(dis[["run", "gen", "body_idx"]], on=["run", "gen", "body_idx"])
        title = f"Rowing review: {len(sample)} disagreements"
    else:
        sample = stratified_sample(df, args.n, args.seed, args.late_gen)
        title = "Rowing review"
    if sample.empty:
        raise SystemExit("Empty sample")
    sample = sample.sort_values(["genome", "stride", "rep", "gen", "body_idx"]).reset_index(drop=True)
    print(f"{len(sample)} individuals:\n{sample.groupby(['genome', 'stride'])['rowing_label'].value_counts().unstack(fill_value=0)}")

    snap = float(sample["snap_deg"].iloc[0])
    jobs = [(args.stride_root / args.behavior_subdir / run / "behavior.npz",
             list(zip(g["gen"].astype(int), g["body_idx"].astype(int))), snap) for run, g in sample.groupby("run")]
    details: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for (npz, _, _), res in zip(jobs, pool.map(_details_job, jobs)):
            for (gen, body), d in res.items():
                details[key_of(npz.parent.name, gen, body)] = d

    individuals = []
    for r in sample.itertuples(index=False):
        key = key_of(r.run, r.gen, r.body_idx)
        if key not in details:
            print(f"[warn] {key}: episode not found in behavior.npz")
            continue
        individuals.append({
            "key": key, "run": r.run, "genome": r.genome, "stride": int(r.stride), "rep": int(r.rep),
            "gen": int(r.gen), "body_idx": int(r.body_idx), "ind_id": int(r.ind_id),
            "fitness": _num(r.stored_fitness), "n_hinges": int(r.n_joints), "video": None,
            "detector": {c: _num(getattr(r, c)) for c in IND_COLS}, **details[key],
        })

    if not args.no_videos:
        raw_dir = page_dir / "videos_raw"
        raw_dir.mkdir(exist_ok=True)
        reuse = args.reuse_videos or args.stride_root / "gait_similarity" / "page" / "videos_raw"
        by_key = {i["key"]: i for i in individuals}
        todo = []
        for i in individuals:
            ckpt = checkpoint_dir(args.stride_root, i["run"], i["gen"], i["body_idx"])
            name = f"{i['run']}_{ckpt.name}_fwd.mp4"
            if not (raw_dir / name).exists() and (reuse / name).exists():
                shutil.copy2(reuse / name, raw_dir / name)
            todo.append((i["key"], i["key"], ckpt, i["run"]))
        print(f"Rendering up to {len(todo)} videos ({args.workers} workers) ...")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=_pool_init) as pool:
            futures = [pool.submit(render_one, (*t, raw_dir)) for t in todo]
            for future in as_completed(futures):
                key, _, src, err = future.result()
                if src is None:
                    print(f"[warn] {key}: no video {err}")
                    continue
                by_key[key]["video"] = browser_video(src, page_dir / "videos").relative_to(page_dir).as_posix()

    data = {"title": title, "individuals": individuals, "labels": list(LABELS), "snap_deg": snap,
            "seed": args.seed}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(PAGE_TEMPLATE.read_text().replace("/*__DATA__*/null", payload))
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
