"""
Replay checkpoints and measure their gaits (E3).

Each checkpoint (genGGG_bodyII of an evolution run, or a train_fixed_body.py
checkpoint) is replayed with the training episode itself
(sim.run_skill_episode, with the run's brain and hinge actuator), so the
replayed fitness reproduces the stored one. Body poses are recorded every
RECORD_EVERY_N physics steps (50 Hz) and stored as float16 / float32 time,
as the investigation's behavior_extract.py did for the traces the gait
detector was validated on; gait.py then gives the gait class, phase relations,
phase quality and mean gait cycle of each episode.

DANN runs trained before the neighbour ties were made deterministic (the
investigation's DANN sweep, the 200-evaluation point of the budget series)
resolved them by their job's hash seed: if the default does not reproduce the
stored fitness, the other resolutions are tried (up to MAX_TIE_CHOICES).

Writes to <out-dir>/<run name>/:
  episodes.csv   one row per checkpoint: run config, gen / body_idx or evals,
                 stored and replayed fitness, body size, gait metrics
  pairs.csv      one row per left/right pair (gait.analyse_episode)
  cycles.npz     mean gait cycle per episode, for gait_species.py

Usage:
    python replay.py --run-dir RUN [--gens last|all|0,10,29] [--workers 20] -o OUT
    python replay.py --list-runs ROOT [ROOT ...]     # run dirs, sorted (for Slurm arrays)
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
import json
import math
import re
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import pandas as pd

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.utils.morphological_descriptor import MorphologicalMeasures

import gait
import runs
from brains import SETTLE_DURATION, make_brain_for_model
from sim import PROFILE, build_world, joint_count, robot_bodies, run_skill_episode, zoo_spec_fn

RECORD_EVERY_N = 10     # physics steps between recorded poses: 50 Hz
FIT_TOL = 1e-6          # replayed vs stored fitness, for the DANN tie search
MAX_TIE_CHOICES = 64
_GEN_CKPT_RE = re.compile(r"^gen(\d+)_body(\d+)$")
_EVALS_CKPT_RE = re.compile(r"^evals(\d+)$")


# ── Checkpoints ───────────────────────────────────────────────────────────────


def ckpt_config(ckpt: Path) -> dict:
    """run_config.json of the run a checkpoint belongs to (<run>/checkpoints/<ckpt>)."""
    p = ckpt.parent.parent / "run_config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def is_fixed_body_run(run_dir: Path) -> bool:
    return (run_dir / "run_config.json").exists() and (run_dir / "curve.jsonl").exists()


def select_checkpoints(run_dir: Path, gens: str) -> list[Path]:
    """Evolution run: the checkpoints of the chosen generations (last, all or
    a comma list). Fixed-body run (train_fixed_body.py): all its checkpoints."""
    if is_fixed_body_run(run_dir):
        return sorted(c for c in (run_dir / "checkpoints").iterdir() if _EVALS_CKPT_RE.match(c.name))
    by_gen = runs.checkpoints_by_gen(run_dir)
    if not by_gen:
        return []
    if gens == "last":
        chosen = [max(by_gen)]
    elif gens == "all":
        chosen = sorted(by_gen)
    else:
        chosen = [int(g) for g in gens.split(",") if int(g) in by_gen]
    return [c for g in chosen for c in by_gen[g]]


def list_runs(roots: list[Path]) -> list[Path]:
    out = []
    for root in roots:
        out += [d for d in root.iterdir() if d.is_dir() and (runs.data_dir(d) is not None or is_fixed_body_run(d))]
    return sorted(out, key=lambda p: (str(p.parent), p.name))


# ── Recording replay ──────────────────────────────────────────────────────────


def replay_episode(genome: dict, weights: np.ndarray, brain_kind: str, cfg: dict,
                   tie_choice: Optional[tuple[int, ...]] = None) -> tuple[float, dict, np.ndarray, np.ndarray, float]:
    """(fitness, info, P (T, B, 3), Q (T, B, 4), dt) of one replayed episode:
    post-settle poses of the bodies with geoms, core first."""
    to_spec = zoo_spec_fn(runs.hinge_actuator(cfg))
    model, data = build_world(genome, to_spec)
    brain = make_brain_for_model(brain_kind, model, data, np.asarray(weights, dtype=np.float32), tie_choice,
                                 duration=PROFILE.loco_duration)
    bodies = robot_bodies(model)
    t, pos, quat = [], [], []
    step = itertools.count()

    def record(d: mujoco.MjData) -> None:
        if next(step) % RECORD_EVERY_N == 0:
            t.append(d.time)
            pos.append(d.xpos[bodies].copy())
            quat.append(d.xquat[bodies].copy())

    fitness, info = run_skill_episode(model, data, brain, PROFILE, recorder=record, on_settle_step=record)
    t32 = np.asarray(t, dtype=np.float32)
    keep = t32 >= SETTLE_DURATION
    P = np.asarray(pos, dtype=np.float16)[keep].astype(np.float64)
    Q = np.asarray(quat, dtype=np.float16)[keep].astype(np.float64)
    dt = float(np.median(np.diff(t32[keep])))
    return fitness, info, P, Q, dt


def dann_tie_choices(genome: dict, cfg: dict) -> list[tuple[int, ...]]:
    """Every resolution of the DANN's exact neighbour ties for this body, default first."""
    model, data = build_world(genome, zoo_spec_fn(runs.hinge_actuator(cfg)))
    ties = make_brain_for_model("dann", model, data).neighbour_ties
    return list(itertools.islice(itertools.product(*(range(len(c)) for _, _, c in ties)), MAX_TIE_CHOICES))


def body_descriptors(genome: dict) -> dict:
    """Module, hinge and limb (leaf module) counts of a tree genome."""
    tree = TreeGenome.from_dict(genome)
    m = MorphologicalMeasures(tree.to_networkx())
    return {"n_modules": len(tree.nodes), "n_hinges": joint_count(tree), "n_limbs": int(m.num_single_neighbour_modules)}


def _replay_job(ckpt_s: str) -> tuple[dict, list[dict], Optional[np.ndarray], str]:
    ckpt = Path(ckpt_s)
    try:
        cfg = ckpt_config(ckpt)
        meta = json.loads((ckpt / "meta.json").read_text())
        genome = json.loads((ckpt / "best_genome.json").read_text())
        weights = np.load(ckpt / "fwd_weights.npy")
        brain = cfg.get("brain", "ann")
        stored = float(meta.get("fitness", math.nan))

        fitness, info, P, Q, dt = replay_episode(genome, weights, brain, cfg)
        tie = None
        if brain == "dann" and math.isfinite(stored) and abs(fitness - stored) > FIT_TOL:
            for choice in dann_tie_choices(genome, cfg)[1:]:
                alt = replay_episode(genome, weights, brain, cfg, choice)
                if abs(alt[0] - stored) <= FIT_TOL:
                    (fitness, info, P, Q, dt), tie = alt, choice
                    break

        ind, pairs = gait.analyse_episode(P, Q, dt)
        cycle, cyc_stats = gait.episode_cycle(P, Q, dt)
        m_gen, m_evals = _GEN_CKPT_RE.match(ckpt.name), _EVALS_CKPT_RE.match(ckpt.name)
        row = {
            "ckpt": ckpt.name,
            "gen": int(m_gen.group(1)) if m_gen else -1, "body_idx": int(m_gen.group(2)) if m_gen else -1,
            "evals": int(m_evals.group(1)) if m_evals else -1,
            "stored_fitness": stored, "replayed_fitness": fitness, "score": info["score"],
            "raw_score": info["raw_score"], "tie_choice": json.dumps(tie),
            **body_descriptors(genome), **ind, **cyc_stats,
        }
        return row, pairs, cycle.astype(np.float32), ""
    except Exception as exc:  # noqa: BLE001 -- one bad body must not stop the run
        return {"ckpt": ckpt.name}, [], None, f"{type(exc).__name__}: {exc}"


# ── Main ──────────────────────────────────────────────────────────────────────


def run_identity(run_dir: Path) -> dict:
    """Run-level columns from the run's config."""
    cfg = json.loads((run_dir / "run_config.json").read_text()) if is_fixed_body_run(run_dir) else runs.run_config(run_dir)
    act = runs.hinge_actuator(cfg)
    return {"run": run_dir.name, "brain": cfg.get("brain", "ann"), "genome": cfg["genome_type"],
            "seed": int(cfg["seed"]), "skill_budget": int(cfg.get("skill_budget", -1)), "hinge": act.label,
            "body_id": cfg.get("body_id", ""), "body_source": cfg.get("body_source", "evolved")}


def replay_run(run_dir: Path, out: Path, gens: str, workers: int) -> str:
    ckpts = select_checkpoints(run_dir, gens)
    if not ckpts:
        return f"{run_dir.name}: no checkpoints"
    ident = run_identity(run_dir)
    jobs = [str(c) for c in ckpts]
    if workers > 1 and len(jobs) > 1:
        with Pool(processes=workers) as pool:
            results = pool.map(_replay_job, jobs, chunksize=1)
    else:
        results = [_replay_job(j) for j in jobs]

    rows, pair_rows, cycles, keys, errors = [], [], [], [], []
    for row, pairs, cycle, err in results:
        if err:
            errors.append(f"{row['ckpt']}: {err}")
            continue
        rows.append({**ident, **row})
        pair_rows += [{"ckpt": row["ckpt"], **p} for p in pairs]
        cycles.append(cycle)
        keys.append(row["ckpt"])

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "pairs.csv", index=False)
    if cycles:
        np.savez_compressed(out / "cycles.npz", ckpt=np.array(keys), cycles=np.concatenate(cycles),
                            n_modules=np.array([len(c) for c in cycles]))
    for e in errors:
        print(f"  ERROR {e}", flush=True)
    df = pd.DataFrame(rows)
    mism = int((np.abs(df["replayed_fitness"] - df["stored_fitness"]) > FIT_TOL).sum()) if len(df) else 0
    labels = ", ".join(f"{k}={v}" for k, v in df["rowing_label"].value_counts().items()) if len(df) else ""
    return f"{run_dir.name}: {len(df)} episodes ({mism} not reproduced, {len(errors)} errors) {labels}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", help="run directories to replay")
    parser.add_argument("--list-runs", type=Path, nargs="+", help="print the run dirs under these roots and exit")
    parser.add_argument("--gens", default="last", help="evolution runs: last, all or a comma list")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("-o", "--out-dir", type=Path, required=False, help="results go to <out-dir>/<run name>/")
    args = parser.parse_args()

    if args.list_runs:
        for d in list_runs(args.list_runs):
            print(d)
        return
    if not args.run_dir or not args.out_dir:
        parser.error("--run-dir and --out-dir are required (or --list-runs)")
    for run_dir in args.run_dir:
        print(replay_run(run_dir, args.out_dir / run_dir.name, args.gens, args.workers), flush=True)


if __name__ == "__main__":
    main()
