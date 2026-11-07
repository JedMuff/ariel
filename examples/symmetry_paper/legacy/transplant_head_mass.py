"""
Head-mass experiment, step 2: move trained rowing controllers to other physics.

Replays old-simulator symmetric champions labelled rowing (ANN and bang-bang,
stride 20, last generations) under every head mass x hinge x module set,
without retraining, and relabels the gait. This shows how fragile a learned
gait is to the physics. It does not show whether learning would find rowing
under that physics.

The cell with the old physics is the check: every controller must reproduce
its stored fitness and label there.

Usage:
    python transplant_head_mass.py --workers 32 --per-brain 60 -o __data__/ariel_head_mass/transplant
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
import json
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import gait
import replay
from sim import LOCAL_HINGE, ZOO_HINGE

REPO = Path(__file__).resolve().parents[2]
SOURCES = {
    "ann": (REPO / "__data__/ariel_control_stride_sweep/rowing/individuals.csv",
            REPO / "__data__/ariel_control_stride_sweep", lambda run: run),
    "bang_bang": (REPO / "__data__/ariel_bangbang_oldsim/rowing/individuals.csv",
                  REPO / "__data__/ariel_bangbang_oldsim", lambda run: run.replace("_0_0", "_local")),
}
MASSES = (0.25, 0.5, 1.0, 1.363, 2.0)
HINGES = {"local": LOCAL_HINGE, "zoo": ZOO_HINGE}
MODULE_SETS = ("local", "zoo")


def rowing_champions(brain: str, n: int, seed: int = 42) -> list[dict]:
    """Up to n rowing symmetric individuals of this brain's last 5 generations."""
    csv, root, run_dir = SOURCES[brain]
    df = pd.read_csv(csv)
    df = df[(df.genome == "tree_symmetric") & (df.rowing_label == "rowing")]
    if "stride" in df:
        df = df[df.stride == 20]
    df = df[df.gen >= df.groupby("run").gen.transform("max") - 4]
    df = df.sample(min(n, len(df)), random_state=seed)
    out = []
    for r in df.itertuples():
        ckpt = root / run_dir(r.run) / f"__data__/gecko_skill_tasks/forward/checkpoints/gen{r.gen:03d}_body{r.body_idx:02d}"
        out.append({"brain": brain, "run": r.run, "gen": int(r.gen), "body_idx": int(r.body_idx), "ckpt": str(ckpt)})
    return out


def _job(job: dict) -> dict:
    ckpt = Path(job["ckpt"])
    genome = json.loads((ckpt / "best_genome.json").read_text())
    stored = float(json.loads((ckpt / "meta.json").read_text())["fitness"])
    cfg = {"eval_profile": "legacy", "hinge_actuator": asdict(HINGES[job["hinge"]]),
           "body_physics": {"module_set": job["module_set"], "core_mass": job["core_mass"]}}
    row = {k: v for k, v in job.items() if k != "ckpt"} | {"stored_fitness": stored}
    try:
        fitness, info, P, Q, dt = replay.replay_episode(genome, np.load(ckpt / "fwd_weights.npy"), job["brain"], cfg)
        ind, _ = gait.analyse_episode(P, Q, dt)
    except Exception as exc:  # noqa: BLE001  a transplant that fails is recorded, not fatal
        return row | {"error": repr(exc)}
    return row | {"fitness": fitness, "displacement": info["raw_score"], "rowing_label": ind["rowing_label"],
                  "rowing_frac": ind["rowing_frac"], "alternating_frac": ind["alternating_frac"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--per-brain", type=int, default=60)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    champs = [c for b in SOURCES for c in rowing_champions(b, args.per_brain)]
    jobs = [c | {"core_mass": m, "hinge": h, "module_set": s}
            for c, m, h, s in itertools.product(champs, MASSES, HINGES, MODULE_SETS)]
    print(f"{len(champs)} controllers x {len(MASSES) * len(HINGES) * len(MODULE_SETS)} cells = {len(jobs)} replays", flush=True)
    with Pool(args.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=4)
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "transplant.csv", index=False)

    ok = df[df.fitness.notna()] if "fitness" in df else df
    df["rows"] = df.rowing_label.eq("rowing")
    home = ok[(ok.core_mass == 1.0) & (ok.hinge == "local") & (ok.module_set == "local")]
    print(f"errors: {len(df) - len(ok)}; old-physics cell: {int((np.abs(home.fitness - home.stored_fitness) > 1e-9).sum())}/{len(home)} "
          f"fitness mismatches, {home.rowing_label.eq('rowing').mean():.0%} still labelled rowing")
    print(df.groupby(["brain", "module_set", "hinge", "core_mass"]).agg(
        rowing=("rows", "mean"), displacement=("displacement", "median")).round(2).unstack("core_mass").to_string())


if __name__ == "__main__":
    main()
