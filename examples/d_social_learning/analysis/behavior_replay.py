"""Replay stored zoo robots and compute their behaviour (gait) per episode.

For every run under --root, takes either the --top best survivors (by
distance) of each generation in --gens (--select top), or every robot ever
born (--select all), and replays each with its learned theta (--episodes
perturbed episodes) and its starting init_theta (--init-episodes), through
the evaluator's own run_episode (sim_ariel/evaluator.py, with the run's
sim_preset). The perturbations come from the individual's eval_seed exactly as
evaluate.py's final evaluations, so learned-theta episode i is stored final
evaluation i (it reproduces to float noise across machines: contact-rich gaits
drift, within their own episode-to-episode spread).

Each episode is recorded at 50 Hz after the settle phase and reduced in the
worker to gait_metrics.behavior_descriptors and gait_metrics.rowing, from the
full-precision state. With --traces all the recorded states are kept too (see
below).

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
        traces.zip    with --traces all: one "<id>.npz" per robot holding, per
                      "<weights>/<ep>/", qpos and ctrl as 16-bit fixed point
                      (int16 times qpos_scale / ctrl_scale; ~1e-4 rad steps,
                      ten times finer than float16, which flips 0.2% of rowing
                      labels), root (the free joint's position, float32) and
                      contact (floor contact per body, packbits), plus the robot's morph,
                      sim_preset, body_names, anchors and dt. Body poses are
                      rebuilt from qpos by forward kinematics: load_trace()
                      returns an episode's P, Q, A, C, U as the metrics take
                      them.
--merge concatenates the per-run episodes.csv into <out-dir>/episodes.csv.

Usage:
    uv run examples/d_social_learning/analysis/behavior_replay.py \\
        --root __data__/social/ariel_zoo_fe10 --gens 60 --top 5 --episodes 3 \\
        --out-dir __data__/social/behavior_zoo
    uv run examples/d_social_learning/analysis/behavior_replay.py --select all \\
        --traces all --shard 0/64 --out-dir /scratch-shared/$USER/behavior_zoo_all
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
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
INT16_MAX = 32767


def _to_int16(x: np.ndarray) -> tuple[np.ndarray, float]:
    """16-bit fixed point: (int16 array, scale) with x ~ array * scale / INT16_MAX."""
    scale = float(max(np.abs(x).max(initial=0.0), np.pi))
    return np.round(x * (INT16_MAX / scale)).astype(np.int16), scale


def _from_int16(a: np.ndarray, scale: float) -> np.ndarray:
    return a.astype(np.float64) * (float(scale) / INT16_MAX)


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


def _world(morph, sim_preset: str) -> dict:
    """evaluator.initialize_world(morph) under `sim_preset`."""
    import simulator_dependent_functions as sdf
    from sim_ariel import evaluator

    if sdf.sim_preset != sim_preset:
        sdf.simulator = sdf.SIMULATOR_ARIEL
        sdf.set_sim_preset(sim_preset)
    return evaluator.initialize_world(morph)


def _layout(model) -> tuple[list[int], list[int], np.ndarray, slice]:
    """The robot's bodies (core first), hinge joints, their qpos addresses, and
    the qpos slice of the free joint's position."""
    import mujoco

    bodies = sorted({int(b) for b in model.geom_bodyid if model.body(int(b)).name.startswith("robot1_")})
    core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    bodies = [core] + [b for b in bodies if b != core]
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr = np.array([model.jnt_qposadr[j] for j in hinges], dtype=int)
    free = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    root = slice(int(model.jnt_qposadr[free]), int(model.jnt_qposadr[free]) + 3)
    return bodies, hinges, qadr, root


def replay_robot(job: dict, n_episodes: int, n_init_episodes: int = 1,
                 keep_traces: bool = False) -> tuple[list[dict], bytes | None]:
    """Rows of metrics per episode, and (keep_traces) the robot's traces as npz bytes."""
    import mujoco

    import simulator_dependent_functions as sdf
    from ariel.simulation.controllers.distributed_mlp import DistributedMLP
    from gait_metrics import behavior_descriptors, rowing
    from sim_ariel import evaluator

    spec = _world(job["morph"], job["sim_preset"])
    model, data = spec["model"], spec["data"]
    brain = DistributedMLP(n_neighbors=sdf.n_neighbours(), hidden=job["hidden"])
    bodies, hinges, qadr, root = _layout(model)
    core = bodies[0]
    # geom -> index of its body in `bodies`, or -1 (floor, other robots' geoms)
    body_index = np.full(model.nbody, -1)
    body_index[bodies] = np.arange(len(bodies))
    geom_index = body_index[model.geom_bodyid]
    floor = spec["floor_geom_id"]

    mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)
    anchors = (data.xanchor[hinges] - data.xpos[core]) @ data.xmat[core].reshape(3, 3)
    dt = RECORD_EVERY * model.opt.timestep

    tags, rows = job["tags"], []
    traces = {}
    meta = {k: job[k] for k in ("budget", "scheme", "x", "k", "rep", "generation", "id", "parent_id", "birth_gen")}
    for weights, key, n_eps in (("learned", "theta", n_episodes), ("init", "init_theta", n_init_episodes)):
        theta = np.asarray(tags[key], dtype=np.float64)
        rng = np.random.default_rng(np.random.SeedSequence([tags["eval_seed"], 1]))
        for ep in range(n_eps):
            P, Q, A, X, C, U = [], [], [], [], [], []

            def record(sim_step: int, d) -> None:
                if sim_step % RECORD_EVERY:
                    return
                P.append(d.xpos[bodies])
                Q.append(d.xquat[bodies])
                A.append(d.qpos[qadr])
                # xpos/xquat are the kinematics of the state before this step,
                # qpos is already a step on: step it back so a trace rebuilds P, Q exactly.
                q = d.qpos.copy()
                mujoco.mj_integratePos(model, q, d.qvel, -model.opt.timestep)
                X.append(q)
                U.append(d.ctrl.copy())
                g1, g2 = d.contact.geom1[: d.ncon], d.contact.geom2[: d.ncon]
                touching = geom_index[np.concatenate([g2[g1 == floor], g1[g2 == floor]])]
                c = np.zeros(len(bodies), dtype=bool)
                c[touching[touching >= 0]] = True
                C.append(c)

            res = evaluator.run_episode(theta, brain, spec, perturb_rng=rng, recorder=record)
            P, Q, A, X, C, U = map(np.asarray, (P, Q, A, X, C, U))
            stored = tags.get("final_eval_fitness", [])
            rows.append({
                **meta, "weights": weights, "episode": ep, "n_bodies": len(bodies), "n_hinges": len(hinges),
                "fitness": res["fitness"], "displacement": res["displacement"],
                "stored_fitness": stored[ep] if weights == "learned" and ep < len(stored) else np.nan,
                **behavior_descriptors(P, Q, A, C, U, anchors, dt),
                **rowing(P, Q, dt),
            })
            if keep_traces:
                pre = f"{weights}/{ep}/"
                rest = X.copy()
                rest[:, root] = 0.0  # stored apart: metres would stretch the angles' scale
                traces[pre + "qpos"], traces[pre + "qpos_scale"] = _to_int16(rest)
                traces[pre + "root"] = X[:, root].astype(np.float32)
                traces[pre + "ctrl"], traces[pre + "ctrl_scale"] = _to_int16(U)
                traces[pre + "contact"] = np.packbits(C, axis=1)
    if not keep_traces:
        return rows, None
    buf = io.BytesIO()
    np.savez_compressed(buf, **traces, morph=json.dumps(job["morph"]), sim_preset=job["sim_preset"],
                        body_names=np.array([model.body(b).name for b in bodies]),
                        anchors=anchors.astype(np.float32), dt=dt)
    return rows, buf.getvalue()


def load_trace(zip_path: Path | str, robot_id: int, weights: str = "learned", episode: int = 0) -> dict:
    """One episode from a run's traces.zip as the metrics take it: P, Q (body
    poses, core first, rebuilt from qpos), A (hinge angles), C, U, anchors,
    body_names and dt, so gait_metrics.behavior_descriptors(P, Q, A, C, U,
    anchors, dt) and rowing(P, Q, dt) can be rerun without simulating."""
    import mujoco

    with zipfile.ZipFile(zip_path) as zf, np.load(io.BytesIO(zf.read(f"{robot_id}.npz"))) as z:
        pre = f"{weights}/{episode}/"
        X, root = _from_int16(z[pre + "qpos"], z[pre + "qpos_scale"]), z[pre + "root"]
        U = _from_int16(z[pre + "ctrl"], z[pre + "ctrl_scale"])
        contact = z[pre + "contact"]
        morph, preset = json.loads(str(z["morph"])), str(z["sim_preset"])
        out = {"anchors": z["anchors"].astype(np.float64), "body_names": z["body_names"], "dt": float(z["dt"])}
    model = _world(morph, preset)["model"]
    bodies, _, qadr, root_slice = _layout(model)
    assert [model.body(b).name for b in bodies] == list(out["body_names"])
    X[:, root_slice] = root
    data = mujoco.MjData(model)
    P, Q = np.empty((len(X), len(bodies), 3)), np.empty((len(X), len(bodies), 4))
    for t, q in enumerate(X):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        P[t], Q[t] = data.xpos[bodies], data.xquat[bodies]
    C = np.unpackbits(contact, axis=1, count=len(bodies)).astype(bool)
    return {**out, "P": P, "Q": Q, "A": X[:, qadr], "C": C, "U": U}


def _work(args: tuple) -> tuple[str | None, int, list[dict], bytes | None]:
    job, n_episodes, n_init_episodes, keep_traces, dest = args
    try:
        return dest, job["id"], *replay_robot(job, n_episodes, n_init_episodes, keep_traces)
    except Exception as exc:  # noqa: BLE001
        print(f"[behavior_replay] {job['scheme']} rep {job['rep']} id {job['id']}: {exc}", flush=True)
        return dest, job["id"], [], None


def _write_run(out: Path, history: pd.DataFrame, rows: list[dict], traces: dict[int, bytes]) -> None:
    """A run's files, each written to .tmp and renamed, episodes.csv last, so a
    killed job leaves no half-done run behind."""
    out.mkdir(parents=True, exist_ok=True)
    history.to_csv(out / "history.csv.tmp", index=False)
    os.replace(out / "history.csv.tmp", out / "history.csv")
    if traces:
        # Each member is already a compressed npz: store, don't recompress.
        with zipfile.ZipFile(out / "traces.zip.tmp", "w", zipfile.ZIP_STORED) as zf:
            for i in sorted(traces):
                zf.writestr(f"{i}.npz", traces[i])
        os.replace(out / "traces.zip.tmp", out / "traces.zip")
    rows = sorted(rows, key=lambda r: (r["id"], r["weights"] != "learned", r["episode"]))
    pd.DataFrame(rows).to_csv(out / "episodes.csv.tmp", index=False)
    os.replace(out / "episodes.csv.tmp", out / "episodes.csv")


def replay_all(pool: Pool, runs: list[Path], out_dir: Path, args: argparse.Namespace) -> None:
    """--select all: every robot of every run, in one stream of tasks so no
    worker waits for a run to finish; each run is written as soon as its last
    robot is back (and skipped if it is already done)."""
    pending: dict[str, dict] = {}

    def tasks():
        # Runs are selected lazily by the pool's feeder thread, so a run's
        # bookkeeping exists before any of its robots can come back.
        for rep_dir in runs:
            dest = run_out_dir(out_dir, rep_dir)
            if (dest / "episodes.csv").exists() and not args.force:
                print(f"skip {dest} (done)", flush=True)
                continue
            jobs, history = select_robots(rep_dir, args.gens, args.top, "all")
            if not jobs:
                _write_run(dest, history, [], {})
                continue
            pending[str(dest)] = {"left": len(jobs), "history": history, "rows": [], "traces": {},
                                  "t0": time.perf_counter()}
            for job in jobs:
                yield job, args.episodes, args.init_episodes, args.traces == "all", str(dest)

    for dest, robot_id, rows, blob in pool.imap_unordered(_work, tasks(), chunksize=2):
        run = pending[dest]
        run["rows"].extend(rows)
        if blob is not None:
            run["traces"][robot_id] = blob
        run["left"] -= 1
        if run["left"] == 0:
            _write_run(Path(dest), run["history"], run["rows"], run["traces"])
            del pending[dest]
            print(f"{len(run['history'].id.unique())} robots, {len(run['rows'])} episodes in "
                  f"{time.perf_counter() - run['t0']:.0f}s -> {dest}", flush=True)


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
    parser.add_argument("--episodes", type=int, default=3, help="perturbed episodes with the learned weights")
    parser.add_argument("--init-episodes", type=int, default=1, help="perturbed episodes with the starting weights")
    parser.add_argument("--traces", choices=["none", "all"], default="none",
                        help="--select all: save every episode's 50 Hz state (traces.zip)")
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
            replay_all(pool, runs, out, args)
            print(f"done in {time.perf_counter() - t0:.0f}s")
            return
        jobs = [j for js, _ in pool.starmap(select_robots, [(r, args.gens, args.top) for r in runs]) for j in js]
        print(f"{len(jobs)} robots from {len(runs)} runs", flush=True)
        work = [(j, args.episodes, args.init_episodes, False, None) for j in jobs]
        rows = [r for _, _, rs, _ in pool.imap_unordered(_work, work, chunksize=2) for r in rs]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)
    print(f"{len(rows)} episodes in {time.perf_counter() - t0:.0f}s -> {out / 'episodes.csv'}")


if __name__ == "__main__":
    main()
