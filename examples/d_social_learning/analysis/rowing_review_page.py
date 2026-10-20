"""Review page for the rowing detector on the zoo fe10 replays: label a sample
of robots by hand to check it (port of ariel_symmetry's
examples/symmetry_pressure/rowing_review_page.py).

The sample (--n, default 96) is drawn from behavior_replay.py --select all's
per-run episodes.csv (the cluster replay, learned weights, episode 0),
stratified by the detector's label there: a quota per label (more for
unpaired, the largest class and where asymmetric rowers were missed before),
plus borderline cases (rowing_frac 0.25-0.75 or rowing_score 0.4-0.8). Within
a label the schemes are balanced and later generations are favoured.

Each sampled robot is then replayed here once (learned weights, the
perturbation of stored final evaluation 0) for both the video and the
detector's verdict, so you and the detector judge the same motion: replays on
this machine drift from the cluster's by float noise (about 1 in 10 labels
differ), so the cluster label is shown only as information.

Per robot the page shows:
  - the replay video (side view, tracking the core);
  - an animated skeleton of the mean gait cycle in the split-plane frame (top
    and side view): left modules blue, right orange, pairs joined, the heading
    arrow and the split line, so you can check the split and the pairing;
  - left vs mirrored-right traces per pair;
  - the detector's metrics.
In blind mode (default) the verdict, metrics and traces are hidden until you
label the robot.

Labels autosave in the browser (localStorage, one store per --seed, shared by
every rebuild of that page) and are exported with "Export labels" as
rowing_labels.json; feed that to rowing_validate.py. --only-disagreements
rebuilds the page with only the robots your labels disagree with the detector
on (videos are reused).

Writes <behavior-dir>/rowing/review/seed<N>/:
    rowing_review.html, videos/   keep together: the page links the videos
    sample.csv                    the sample, with this replay's detector
                                  metrics and the cluster's label
    pairs.csv                     per-pair metrics (for rowing_validate.py's
                                  threshold sweep)

Usage:
    MUJOCO_GL=egl uv run examples/d_social_learning/analysis/rowing_review_page.py --n 96 --seed 0
    uv run examples/d_social_learning/analysis/rowing_review_page.py --seed 0 \\
        --only-disagreements ~/Downloads/rowing_labels.json
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

sys.path.append(str(Path(__file__).resolve().parent.parent))

from behavior_replay import RECORD_EVERY, RUN_RE, _layout, _world, select_robots  # noqa: E402
from gait_metrics import TRANSIENT_S, estimate_period, rowing  # noqa: E402
from paper_figures import label as scheme_label  # noqa: E402

PAGE_TEMPLATE = Path(__file__).with_name("rowing_review_page.html")
RUN = ["budget", "scheme", "x", "k", "rep"]
# Share of the sample per detector label (cluster replay); the rest are borderline cases.
QUOTA = {"unpaired": 0.25, "rowing": 0.17, "mixed": 0.17, "alternating": 0.12, "midline": 0.10, "static": 0.06,
         "aperiodic": 0.04}
N_PHASE = 64  # samples per averaged gait cycle
IND_COLS = ("rowing_label", "rowing_score", "rowing_frac", "alternating_frac", "aperiodic_frac", "paired_frac",
            "heading_deg", "split_offset_deg", "heading_consistency", "speed", "body_size_m", "n_moving",
            "n_pairs", "n_rowing_pairs", "n_alternating_pairs", "morph_plane_offset_deg", "midline_amp_frac",
            "stroke_total_m")
PAIR_COLS = ("left", "right", "pair_kind", "mirror_sync", "phase_lag", "freq_l", "freq_r", "periodicity_l",
             "periodicity_r", "amp_l", "amp_r", "amp_ratio", "stroke_fore", "stroke_lat", "stroke_up", "rest_dist",
             "weight")


def key_of(r) -> str:
    return f"{r['budget']}/{r['scheme']}/x{int(round(r['x'] * 10)):02d}/rep{int(r['rep'])}/{int(r['id'])}"


# ── Sample ────────────────────────────────────────────────────────────────────


def load_candidates(behavior_dir: Path) -> pd.DataFrame:
    """One row per robot: the cluster replay's learned-weight episode 0."""
    cols = RUN + ["id", "generation", "weights", "episode", "fitness", "rowing_label", "rowing_frac",
                  "rowing_score"]
    parts = []
    for p in sorted((behavior_dir / "runs").glob("ig*/*/x*/k*/rep_*/episodes.csv")):
        ep = pd.read_csv(p, usecols=cols)
        parts.append(ep[(ep.weights == "learned") & (ep.episode == 0)])
    if not parts:
        raise SystemExit(f"no runs/**/episodes.csv under {behavior_dir}")
    df = pd.concat(parts, ignore_index=True).drop(columns=["weights", "episode"])
    print(f"{len(df)} robots from {len(parts)} runs")
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    picked: set[int] = set()

    def take(cands: pd.Index, k: int) -> None:
        cands = cands[~cands.isin(list(picked))]
        if len(cands) and k > 0:
            s = df.loc[cands]
            # Balance the schemes, lean towards later generations.
            w = (1.0 + s.generation.to_numpy() / 10.0) / s.scheme.map(s.scheme.value_counts()).to_numpy()
            picked.update(rng.choice(cands, size=min(k, len(cands)), replace=False, p=w / w.sum()).tolist())

    for lab, share in QUOTA.items():
        take(df.index[df.rowing_label == lab], round(share * n))
    take(df.index[df.rowing_frac.between(0.25, 0.75) | df.rowing_score.between(0.4, 0.8)], n - len(picked))
    take(df.index, n - len(picked))
    return df.loc[sorted(picked)].rename(columns={"rowing_label": "cluster_label", "fitness": "cluster_fitness"})


# ── Replay ────────────────────────────────────────────────────────────────────


def average_cycle(traj: np.ndarray, dt: float, period: float, transient_s: float = TRANSIENT_S) -> np.ndarray:
    """Mean cycle (M, N_PHASE, 3) over the complete cycles after the transient,
    each sampled at the same phases by linear interpolation; centred on its
    mean per module."""
    t = np.arange(traj.shape[1]) * dt
    n_cycles = int((t[-1] - transient_s) // period)
    if n_cycles < 1:
        raise ValueError(f"No complete cycle of {period:.2f} s after the {transient_s} s transient")
    phase = np.arange(N_PHASE) / N_PHASE
    ts = transient_s + (np.arange(n_cycles)[:, None] + phase[None]) * period
    flat = traj.transpose(0, 2, 1).reshape(-1, traj.shape[1])
    samples = np.stack([np.interp(ts.ravel(), t, row) for row in flat])
    cycle = samples.reshape(traj.shape[0], 3, n_cycles, N_PHASE).mean(axis=2).transpose(0, 2, 1)
    return cycle - cycle.mean(axis=1, keepdims=True)


def review_detail(d: dict, trace_s: float = 6.0) -> dict:
    """Small JSON-able summary of rowing(detail=True)'s detail: the mean gait
    cycle in sagittal coordinates (rest + cycle) and trace_s of each pair's
    left / reflected-right fore and up coordinates."""
    sag, rest, dt = d["sag"], d["rest"], d["dt"]
    traj = (sag - rest).transpose(1, 0, 2)
    period, _ = estimate_period(traj, dt)
    aperiodic = not np.isfinite(period)
    if aperiodic:
        period = (traj.shape[1] - 1) * dt - TRANSIENT_S
    cyc = average_cycle(traj, dt, period) + rest[:, None]
    n = min(len(sag), round(trace_s / dt))
    step = max(1, round(0.04 / dt))  # 25 Hz is plenty to draw
    mirror = np.array([1.0, -1.0, 1.0])
    traces = []
    for left, right in d["pairs"]:
        tl, tr = sag[:n:step, left] - rest[left], (sag[:n:step, right] - rest[right]) * mirror
        traces.append({"fore_l": tl[:, 0], "fore_r": tr[:, 0], "up_l": tl[:, 2], "up_r": tr[:, 2]})
    r4 = lambda a: np.round(np.asarray(a, dtype=float), 4).tolist()  # noqa: E731
    return {
        "cycle": r4(cyc), "rest": r4(rest), "side": d["side"].tolist(), "moving": d["moving"].tolist(),
        "amp": r4(d["amp"]), "pairs": [list(p) for p in d["pairs"]], "period_s": float(period),
        "aperiodic": bool(aperiodic), "trace_dt": dt * step,
        "traces": [{k: r4(v) for k, v in t.items()} for t in traces],
    }


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return round(x, 4) if np.isfinite(x) else None


def replay_one(job: dict, video: Path | None) -> dict:
    """Learned weights, final-evaluation episode 0: detector detail, plus the video if `video`."""
    import mujoco

    import simulator_dependent_functions as sdf
    from ariel.simulation.controllers.distributed_mlp import DistributedMLP
    from ariel.visualisation.evosight.recorders import VideoRecorder
    from sim_ariel import evaluator

    spec = _world(job["morph"], job["sim_preset"])
    model = spec["model"]
    brain = DistributedMLP(n_neighbors=sdf.n_neighbours(), hidden=job["hidden"])
    bodies, hinges, _, _ = _layout(model)
    cam = VideoRecorder(model, video, width=512, height=384, fps=25, crf=26) if video else None
    P, Q = [], []

    def record(sim_step: int, d: mujoco.MjData) -> None:
        if cam is not None:
            cam(sim_step, d)
        if sim_step % RECORD_EVERY == 0:
            P.append(d.xpos[bodies])
            Q.append(d.xquat[bodies])

    rng = np.random.default_rng(np.random.SeedSequence([job["tags"]["eval_seed"], 1]))
    try:
        res = evaluator.run_episode(np.asarray(job["tags"]["theta"], dtype=np.float64), brain, spec,
                                    perturb_rng=rng, recorder=record)
    except BaseException:
        if cam is not None:
            cam.abort()
        raise
    if cam is not None:
        cam.close()
    ind, pairs, d = rowing(np.asarray(P), np.asarray(Q), RECORD_EVERY * model.opt.timestep, detail=True)
    ind["split_offset_deg"] = float(-np.rad2deg(d["heading_rel"]))
    ind["n_rowing_pairs"] = sum(p["pair_kind"] == "rowing" for p in pairs)
    ind["n_alternating_pairs"] = sum(p["pair_kind"] == "alternating" for p in pairs)
    stored = job["tags"].get("final_eval_fitness") or [np.nan]
    return {
        "key": job["key"], "fitness": float(res["fitness"]), "stored_fitness": float(stored[0]),
        "n_hinges": len(hinges), "ind": ind, "pairs": pairs, "detail": review_detail(d),
        "heading_rel_deg": float(np.rad2deg(d["heading_rel"])),
        "names": [model.body(b).name.removeprefix("robot1_") for b in bodies],
    }


def _work(args: tuple) -> dict:
    job, video = args
    try:
        return replay_one(job, video)
    except Exception as exc:  # noqa: BLE001 -- one failed robot must not kill the page
        return {"key": job["key"], "error": f"{type(exc).__name__}: {exc}"}


def jobs_for(sample: pd.DataFrame, data_root: Path) -> list[dict]:
    """The replay jobs (genome, weights, eval_seed) of the sampled robots, read from their run databases."""
    run_dirs = {}
    for db in data_root.glob("ig*/*/x*/k*/rep_*/database.db"):
        m = RUN_RE.search(db.parent.as_posix())
        run_dirs[(m["budget"], m["scheme"], int(m["x"]) / 10, int(m["k"]), int(m["rep"]))] = db.parent
    jobs = []
    for keys, g in sample.groupby(RUN):
        run_jobs, _ = select_robots(run_dirs[(keys[0], keys[1], float(keys[2]), int(keys[3]), int(keys[4]))],
                                    [], 0, "all")
        want = set(g.id.astype(int))
        jobs += [{**j, "key": key_of(j)} for j in run_jobs if j["id"] in want]
    return jobs


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--behavior-dir", type=Path, default=Path("__data__/social/behavior_zoo_all"))
    parser.add_argument("--data-root", type=Path, default=Path("__data__/social/ariel_zoo_fe10"))
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-disagreements", type=Path, default=None, metavar="LABELS_JSON",
                        help="page of only the robots these labels disagree with the detector on")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()
    page_dir = args.behavior_dir / "rowing" / "review" / f"seed{args.seed}"
    page_dir.mkdir(parents=True, exist_ok=True)

    if args.only_disagreements:
        from rowing_validate import disagreements, load_labels

        sample = pd.read_csv(page_dir / "sample.csv")
        dis = disagreements(load_labels(args.only_disagreements), sample)
        sample = sample[sample.key.isin(dis.key)]
        out, title = page_dir / "rowing_review_disagreements.html", f"Rowing review: {len(sample)} disagreements"
    else:
        sample = stratified_sample(load_candidates(args.behavior_dir), args.n, args.seed)
        sample["key"] = [key_of(r) for _, r in sample.iterrows()]
        out, title = page_dir / "rowing_review.html", f"Rowing review (zoo fe10, seed {args.seed})"
    if sample.empty:
        raise SystemExit("Empty sample")
    print(f"{len(sample)} robots; cluster labels: {sample.cluster_label.value_counts().to_dict()}")

    jobs = jobs_for(sample, args.data_root)
    video_dir = page_dir / "videos"
    work = [(j, None if args.no_videos else video_dir / f"{j['key'].replace('/', '_')}.mp4") for j in jobs]
    results: dict[str, dict] = {}
    print(f"Replaying {len(work)} robots ({args.workers} workers) ...")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(_work, w) for w in work]):
            r = future.result()
            if "error" in r:
                print(f"[warn] {r['key']}: {r['error']}")
            else:
                results[r["key"]] = r

    sample = sample[sample.key.isin(results)].sort_values(RUN + ["generation", "id"]).reset_index(drop=True)
    individuals, rows, pair_rows = [], [], []
    for _, s in sample.iterrows():
        r = results[s.key]
        video = video_dir / f"{s.key.replace('/', '_')}.mp4"
        individuals.append({
            "key": s.key, "rep": int(s.rep), "gen": int(s.generation), "id": int(s.id),
            "group": f"{scheme_label(s.scheme)} · {s.budget.removeprefix('ig')} gens, x={s.x:g}",
            "run": f"{s.budget}/{s.scheme}/x{int(round(s.x * 10)):02d}/k{int(s.k)}/rep_{int(s.rep)}",
            "fitness": _num(r["fitness"]), "n_hinges": r["n_hinges"], "cluster_label": s.cluster_label,
            "video": video.relative_to(page_dir).as_posix() if video.exists() else None,
            "detector": {c: _num(r["ind"].get(c)) for c in IND_COLS}, "detail": r["detail"],
            "heading_rel_deg": r["heading_rel_deg"], "names": r["names"],
            "pairs": [{k: _num(v) if k not in ("pair_kind",) else v for k, v in p.items() if k in PAIR_COLS}
                      for p in r["pairs"]],
        })
        rows.append({**s.to_dict(), "fitness": r["fitness"], "stored_fitness": r["stored_fitness"],
                     "n_hinges": r["n_hinges"], **{c: r["ind"].get(c) for c in IND_COLS}})
        pair_rows += [{"key": s.key, **{k: p[k] for k in PAIR_COLS}} for p in r["pairs"]]

    if not args.only_disagreements:
        pd.DataFrame(rows).to_csv(page_dir / "sample.csv", index=False)
        pd.DataFrame(pair_rows, columns=["key", *PAIR_COLS]).to_csv(page_dir / "pairs.csv", index=False)
        same = (sample.cluster_label.values == pd.DataFrame(rows).rowing_label.values).mean()
        print(f"This replay's label = the cluster's for {same:.0%} of the sample")
    data = {"title": title, "individuals": individuals, "seed": args.seed}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(PAGE_TEMPLATE.read_text().replace("/*__DATA__*/null", payload))
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(individuals)} robots)")


if __name__ == "__main__":
    main()
