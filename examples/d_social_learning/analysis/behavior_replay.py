"""Replay stored zoo robots and compute their behaviour (gait) per episode.

For every run under --root, takes the --top best survivors (by distance) of
each generation in --gens and replays each with its learned theta and its
starting init_theta, --episodes perturbed episodes each, through the
evaluator's own run_episode (sim_ariel/evaluator.py, with the run's
sim_preset). The perturbations come from the individual's eval_seed exactly
as evaluate.py's final evaluations, so learned-theta episode i is stored
final evaluation i (it reproduces to float noise across machines: contact-rich
gaits drift, within their own episode-to-episode spread).

Each episode is recorded at 50 Hz after the settle phase and reduced in the
worker to gait_metrics.behavior_descriptors and gait_metrics.rowing, so
nothing large is written; local replays are deterministic, so any episode can
be replayed again for its traces.

Writes <out-dir>/episodes.csv: run keys, generation, id, parent_id, weights
("learned"/"init"), episode, replayed fitness and displacement, the stored
final-evaluation fitness of that episode (learned only), and the metrics.

Usage:
    uv run examples/d_social_learning/analysis/behavior_replay.py \\
        --root __data__/social/ariel_zoo_fe10 --gens 60 --top 5 --episodes 3 \\
        --out-dir __data__/social/behavior_zoo
"""

from __future__ import annotations

import argparse
import json
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


def select_robots(rep_dir: Path, gens: list[int], top: int) -> list[dict]:
    """The `top` best survivors by distance of each generation in `gens`."""
    keys = RUN_RE.search(rep_dir.as_posix()).groupdict()
    keys["x"] = int(keys["x"]) / 10
    cfg = json.loads((rep_dir / "run_config.json").read_text())["args"]
    hist = read_run_history(rep_dir)
    hist = hist[hist.survivor.astype(bool)].drop_duplicates(["generation", "individual_id"])
    ind = read_individuals(rep_dir / "database.db", "genotype_, tags_").set_index("id")
    jobs = []
    for gen in gens:
        ids = [i for i in hist[hist.generation == gen].individual_id if i in ind.index]
        tags = {i: json.loads(ind.at[i, "tags_"]) for i in ids}
        for i in sorted(ids, key=lambda i: -tags[i]["distance"])[:top]:
            jobs.append({**keys, "generation": gen, "id": int(i), "parent_id": tags[i].get("parent_id"),
                         "sim_preset": cfg.get("sim_preset", "legacy"), "hidden": cfg["hidden"],
                         "morph": json.loads(ind.at[i, "genotype_"])["morph"], "tags": tags[i]})
    return jobs


def replay_robot(job: dict, n_episodes: int) -> list[dict]:
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
    meta = {k: job[k] for k in ("budget", "scheme", "x", "k", "rep", "generation", "id", "parent_id")}
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
    return rows


def _work(args: tuple) -> list[dict]:
    job, n_episodes = args
    try:
        return replay_robot(job, n_episodes)
    except Exception as exc:  # noqa: BLE001
        print(f"[behavior_replay] {job['scheme']} rep {job['rep']} id {job['id']}: {exc}", flush=True)
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="__data__/social/ariel_zoo_fe10")
    parser.add_argument("--out-dir", default="__data__/social/behavior_zoo")
    parser.add_argument("--gens", type=int, nargs="+", default=[60])
    parser.add_argument("--top", type=int, default=5, help="best survivors per run and generation")
    parser.add_argument("--episodes", type=int, default=3, help="perturbed episodes per weight set")
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="only the first N runs (testing)")
    args = parser.parse_args()

    runs = sorted(p.parent for p in Path(args.root).glob("ig*/*/x*/k*/rep_*/database.db"))[: args.limit]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        jobs = [j for js in pool.starmap(select_robots, [(r, args.gens, args.top) for r in runs]) for j in js]
        print(f"{len(jobs)} robots from {len(runs)} runs", flush=True)
        rows = [r for rs in pool.imap_unordered(_work, [(j, args.episodes) for j in jobs], chunksize=2) for r in rs]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)
    print(f"{len(rows)} episodes in {time.perf_counter() - t0:.0f}s -> {out / 'episodes.csv'}")


if __name__ == "__main__":
    main()
