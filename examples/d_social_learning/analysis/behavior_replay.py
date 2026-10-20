"""Replay stored zoo robots and compute their behaviour (gait) per episode.

For every run under --root, takes either the --top best survivors (by
distance) of each generation in --gens (--select top), or every robot ever
born (--select all), and replays each with its learned theta and its starting
init_theta, --episodes perturbed episodes each, through the evaluator's own
run_episode (sim_ariel/evaluator.py, with the run's sim_preset). The
perturbations come from the individual's eval_seed exactly as evaluate.py's
final evaluations, so learned-theta episode i is stored final evaluation i (it
reproduces to float noise across machines: contact-rich gaits drift, within
their own episode-to-episode spread).

Each episode is recorded at 50 Hz after the settle phase and reduced in the
worker to gait_metrics.behavior_descriptors and gait_metrics.rowing. With
--traces the recorded arrays are kept too (see below).

--select top (default) writes <out-dir>/episodes.csv: run keys, generation,
id, parent_id, birth_gen, weights ("learned"/"init"), episode, replayed fitness
and displacement, the stored final-evaluation fitness of that episode (learned
only), and the metrics.

--select all writes one directory per run, so a sharded cluster job
(--shard I/N, see run_behavior_replay.sh) can be resubmitted and only fills in
the runs that are missing:
    <out-dir>/runs/<budget>/<scheme>/x<x>/k<k>/rep_<n>/
        episodes.csv  as above (generation = birth_gen; 0 = initial population)
        history.csv   generation, id, survivor: every selection pool, plus the
                      initial population as generation 0
        traces.npz    with --traces ep0 (episode 0 of both weight sets) or all:
                      "<id>/<weights>/<ep>/{P,Q,A,C,U}" (body positions and
                      quaternions, core first; hinge angles; floor contact per
                      body; controls), "<id>/anchors" (hinge anchors in the
                      core frame at rest), "<id>/body_names", and "dt".
                      load_trace() reads one episode back.
--merge concatenates the per-run episodes.csv into <out-dir>/episodes.csv.

Usage:
    uv run examples/d_social_learning/analysis/behavior_replay.py \\
        --root __data__/social/ariel_zoo_fe10 --gens 60 --top 5 --episodes 3 \\
        --out-dir __data__/social/behavior_zoo
    uv run examples/d_social_learning/analysis/behavior_replay.py --select all \\
        --traces ep0 --shard 0/64 --out-dir /scratch-shared/$USER/behavior_zoo_all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# Appended, not prepended: d_social_learning/ariel would shadow the ariel package.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from curve_utils import read_individuals  # noqa: E402
from core.novelty_history import read_run_history  # noqa: E402

RUN_RE = re.compile(r"(?P<budget>ig\d+)/(?P<scheme>[^/]+)/x(?P<x>\d+)/k(?P<k>\d+)/rep_(?P<rep>\d+)_")
RECORD_EVERY = 10  # sim steps: 50 Hz at the 500 Hz physics
TRACE_KEYS = ("P", "Q", "A", "C", "U")


def run_keys(rep_dir: Path) -> dict:
    keys = RUN_RE.search(rep_dir.as_posix()).groupdict()
    keys["x"] = int(keys["x"]) / 10
    return keys


def run_out_dir(out_dir: Path, rep_dir: Path) -> Path:
    """<out-dir>/runs/<budget>/<scheme>/x<x>/k<k>/rep_<n>, without the run-dir timestamp."""
    m = RUN_RE.search(rep_dir.as_posix())
    return out_dir / "runs" / m["budget"] / m["scheme"] / f"x{m['x']}" / f"k{m['k']}" / f"rep_{m['rep']}"


def run_history(hist: pd.DataFrame, parents: dict[int, object]) -> pd.DataFrame:
    """generation, id, survivor of every selection pool, plus the initial population as generation 0."""
    h = hist.drop_duplicates(["generation", "individual_id"]).rename(columns={"individual_id": "id"})
    h = h[["generation", "id", "survivor"]].astype(int)
    init = pd.DataFrame({"generation": 0, "id": sorted(i for i, p in parents.items() if p is None), "survivor": 1})
    return pd.concat([init, h], ignore_index=True).sort_values(["generation", "id"]).reset_index(drop=True)


def select_robots(rep_dir: Path, gens: list[int], top: int, select: str = "top") -> tuple[list[dict], pd.DataFrame]:
    """The `top` best survivors by distance of each generation in `gens`, or
    (select="all") every robot ever born, once, at its birth generation.
    Also returns the run's history (see run_history)."""
    keys = run_keys(rep_dir)
    cfg = json.loads((rep_dir / "run_config.json").read_text())["args"]
    hist = read_run_history(rep_dir)
    ind = read_individuals(rep_dir / "database.db", "genotype_, tags_").set_index("id")
    tags = {int(i): json.loads(t) for i, t in ind["tags_"].items()}
    history = run_history(hist, {i: t.get("parent_id") for i, t in tags.items()})
    birth = history.groupby("id").generation.min()
    if select == "all":
        picked = [(int(birth[i]), i) for i in sorted(tags) if i in birth.index]
    else:
        surv = history[history.survivor.astype(bool) & history.id.isin(tags)]
        picked = [(gen, i) for gen in gens
                  for i in sorted(surv[surv.generation == gen].id, key=lambda i: -tags[i]["distance"])[:top]]
    jobs = [{**keys, "generation": gen, "id": int(i), "parent_id": tags[i].get("parent_id"),
             "birth_gen": int(birth.get(i, -1)), "sim_preset": cfg.get("sim_preset", "legacy"),
             "hidden": cfg["hidden"], "morph": json.loads(ind.at[i, "genotype_"])["morph"], "tags": tags[i]}
            for gen, i in picked]
    return jobs, history


def replay_robot(job: dict, n_episodes: int, keep_traces: str = "none") -> tuple[list[dict], dict]:
    """Rows of metrics per episode, and the traces kept (`keep_traces`: none / ep0 / all)."""
    import mujoco

    import simulator_dependent_functions as sdf
    from ariel.simulation.controllers.distributed_mlp import DistributedMLP
    from gait_metrics import behavior_descriptors, rowing
    from sim_ariel import evaluator

    if sdf.sim_preset != job["sim_preset"]:
        sdf.simulator = sdf.SIMULATOR_ARIEL
        sdf.set_sim_preset(job["sim_preset"])
    spec = evaluator.initialize_world(job["morph"])
    model, data = spec["model"], spec["data"]
    brain = DistributedMLP(n_neighbors=sdf.n_neighbours(), hidden=job["hidden"])

    bodies = sorted({int(b) for b in model.geom_bodyid if model.body(int(b)).name.startswith("robot1_")})
    core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    bodies = [core] + [b for b in bodies if b != core]
    body_arr = np.array(bodies)
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr = np.array([model.jnt_qposadr[j] for j in hinges], dtype=int)
    floor = spec["floor_geom_id"]

    mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)
    anchors = (data.xanchor[hinges] - data.xpos[core]) @ data.xmat[core].reshape(3, 3)

    tags, rows = job["tags"], []
    traces = {}
    if keep_traces != "none":
        traces[f"{job['id']}/anchors"] = anchors.astype(np.float32)
        traces[f"{job['id']}/body_names"] = np.array([model.body(b).name for b in bodies])
        traces["dt"] = np.float64(RECORD_EVERY * model.opt.timestep)
    meta = {k: job[k] for k in ("budget", "scheme", "x", "k", "rep", "generation", "id", "parent_id", "birth_gen")}
    for weights, key in (("learned", "theta"), ("init", "init_theta")):
        theta = np.asarray(tags[key], dtype=np.float64)
        rng = np.random.default_rng(np.random.SeedSequence([tags["eval_seed"], 1]))
        for ep in range(n_episodes):
            P, Q, A, C, U = [], [], [], [], []

            def record(sim_step: int, d) -> None:
                if sim_step % RECORD_EVERY:
                    return
                P.append(d.xpos[bodies].copy())
                Q.append(d.xquat[bodies].copy())
                A.append(d.qpos[qadr].copy())
                U.append(d.ctrl.copy())
                g1, g2 = d.contact.geom1[: d.ncon], d.contact.geom2[: d.ncon]
                other = np.concatenate([g2[g1 == floor], g1[g2 == floor]])
                C.append(np.isin(body_arr, model.geom_bodyid[other]))

            res = evaluator.run_episode(theta, brain, spec, perturb_rng=rng, recorder=record)
            P, Q, A, C, U = map(np.asarray, (P, Q, A, C, U))
            dt = RECORD_EVERY * model.opt.timestep
            stored = tags.get("final_eval_fitness", [])
            rows.append({
                **meta, "weights": weights, "episode": ep, "n_bodies": len(bodies), "n_hinges": len(hinges),
                "fitness": res["fitness"], "displacement": res["displacement"],
                "stored_fitness": stored[ep] if weights == "learned" and ep < len(stored) else np.nan,
                **behavior_descriptors(P, Q, A, C, U, anchors, dt),
                **rowing(P, Q, dt),
            })
            if keep_traces == "all" or (keep_traces == "ep0" and ep == 0):
                for name, arr in zip(TRACE_KEYS, (P, Q, A, C, U)):
                    traces[f"{job['id']}/{weights}/{ep}/{name}"] = arr if name == "C" else arr.astype(np.float32)
    return rows, traces


def load_trace(npz_path: Path | str, robot_id: int, weights: str = "learned", episode: int = 0) -> dict:
    """One episode from a run's traces.npz: P, Q, A, C, U, anchors, body_names and dt,
    so gait_metrics.behavior_descriptors(P, Q, A, C, U, anchors, dt) and rowing(P, Q, dt) can be rerun."""
    with np.load(npz_path) as z:
        out = {k: z[f"{robot_id}/{weights}/{episode}/{k}"] for k in TRACE_KEYS}
        out.update(anchors=z[f"{robot_id}/anchors"], body_names=z[f"{robot_id}/body_names"], dt=float(z["dt"]))
    return out


def _work(args: tuple) -> tuple[list[dict], dict]:
    job, n_episodes, keep_traces = args
    try:
        return replay_robot(job, n_episodes, keep_traces)
    except Exception as exc:  # noqa: BLE001
        print(f"[behavior_replay] {job['scheme']} rep {job['rep']} id {job['id']}: {exc}", flush=True)
        return [], {}


def replay_run(pool: Pool, rep_dir: Path, out_dir: Path, args: argparse.Namespace) -> None:
    """--select all: replay every robot of one run into its own directory (skipped if already done)."""
    out = run_out_dir(out_dir, rep_dir)
    if (out / "episodes.csv").exists() and not args.force:
        print(f"skip {out} (done)", flush=True)
        return
    t0 = time.perf_counter()
    jobs, history = select_robots(rep_dir, args.gens, args.top, "all")
    rows, traces = [], {}
    work = [(j, args.episodes, args.traces) for j in jobs]
    for r, tr in pool.imap_unordered(_work, work, chunksize=2):
        rows.extend(r)
        traces.update(tr)
    out.mkdir(parents=True, exist_ok=True)
    # Written to .tmp and renamed, episodes.csv last, so a killed job leaves no half-done run behind.
    history.to_csv(out / "history.csv.tmp", index=False)
    os.replace(out / "history.csv.tmp", out / "history.csv")
    if traces:
        with open(out / "traces.npz.tmp", "wb") as f:
            np.savez_compressed(f, **traces)
        os.replace(out / "traces.npz.tmp", out / "traces.npz")
    pd.DataFrame(rows).to_csv(out / "episodes.csv.tmp", index=False)
    os.replace(out / "episodes.csv.tmp", out / "episodes.csv")
    print(f"{len(jobs)} robots, {len(rows)} episodes in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)


def merge(out_dir: Path) -> None:
    parts = sorted((out_dir / "runs").glob("ig*/*/x*/k*/rep_*/episodes.csv"))
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(out_dir / "episodes.csv", index=False)
    print(f"merged {len(parts)} runs -> {out_dir / 'episodes.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="__data__/social/ariel_zoo_fe10")
    parser.add_argument("--out-dir", default="__data__/social/behavior_zoo")
    parser.add_argument("--select", choices=["top", "all"], default="top",
                        help="top: the --top best survivors of each of --gens; all: every robot ever born, per run")
    parser.add_argument("--gens", type=int, nargs="+", default=[60])
    parser.add_argument("--top", type=int, default=5, help="best survivors per run and generation")
    parser.add_argument("--episodes", type=int, default=3, help="perturbed episodes per weight set")
    parser.add_argument("--traces", choices=["none", "ep0", "all"], default="none",
                        help="--select all: save the 50 Hz traces of episode 0 or of every episode")
    parser.add_argument("--shard", default="0/1", help="I/N: only every N-th run, starting at I")
    parser.add_argument("--force", action="store_true", help="--select all: redo runs already done")
    parser.add_argument("--merge", action="store_true", help="only merge the per-run episodes.csv")
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="only the first N runs (testing)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    if args.merge:
        merge(out)
        return
    shard, n_shards = map(int, args.shard.split("/"))
    runs = sorted(p.parent for p in Path(args.root).glob("ig*/*/x*/k*/rep_*/database.db"))[: args.limit]
    runs = runs[shard::n_shards]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        if args.select == "all":
            print(f"{len(runs)} runs (shard {args.shard})", flush=True)
            for rep_dir in runs:
                replay_run(pool, rep_dir, out, args)
            print(f"done in {time.perf_counter() - t0:.0f}s")
            return
        jobs = [j for js, _ in pool.starmap(select_robots, [(r, args.gens, args.top) for r in runs]) for j in js]
        print(f"{len(jobs)} robots from {len(runs)} runs", flush=True)
        rows = [r for rs, _ in pool.imap_unordered(_work, [(j, args.episodes, "none") for j in jobs], chunksize=2)
                for r in rs]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)
    print(f"{len(rows)} episodes in {time.perf_counter() - t0:.0f}s -> {out / 'episodes.csv'}")


if __name__ == "__main__":
    main()
