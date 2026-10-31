"""
Phase-shift replay: does left-right phase locking cause the symmetric advantage?

E3 found that symmetric bodies whose sides are phase-locked carry the gap, and
unlocked symmetric bodies score like tree bodies. That is a correlation; this
replays trained controllers with the left side's timing changed and nothing
else, and measures the score and gait again.

The body's sides come from its rest pose at spawn: the vertical plane through
the core that best mirrors the modules onto each other (gait.morph_plane), and
each hinge's anchor is left, right or midline (|lat| < gait.MIDLINE_TOL). Tree
bodies are split the same way, by their best mirror plane.

Conditions (one episode each, from the trained checkpoint), graded because a
tuned open-loop gait breaks under any large timing change:
  baseline          unchanged (reproduces the stored fitness)
  side_shift_S      left hinges S cycles later, S in SHIFTS (0.5: anti-phase
                    <-> in-phase)
  side_detune_D     left hinges' frequency x (1 + D), D in DETUNES: the
                    left-right lag drifts, by D x 15 s x f cycles over the episode
  rand_shift_S      control: the same number of lateral hinges, drawn at random
  rand_detune_D     but never one whole side (N_DRAWS draws each)
Midline hinges are never perturbed. If the lock causes the gap, side
perturbations should cost locked symmetric bodies more than the random ones,
and cost tree bodies no more than the random ones.

bang_bang, bang_bang_drift, bang_bang_free: the hinge's phase offset (cycles)
and frequency. dann: the sine clock input of the left hinges is shifted by the
same fraction of its period, or runs (1 + D) faster; the rest of the MLP's
inputs are unchanged, so a closed-loop controller can pull the sides back.

Checkpoints: a fixed-body training's last checkpoint (--ckpts all: every one),
an evolution run's best last-generation checkpoint.

Writes <out-dir>/<run name>/phase_shift.csv, one row per checkpoint and
condition: condition, draw, n_perturbed, hinges_left / _right / _midline,
stored and replayed fitness, score, and the gait metrics of gait.analyse_episode.

Usage:
    python phase_shift.py --run-dir RUN [RUN ...] [--ckpts last|all] [--workers 20] -o OUT
    python replay.py --list-runs ROOT ...     # run dirs, for Slurm arrays
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
import json
import math
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np
import pandas as pd

import gait
import runs
from brains import DANN_FEATURES, DANN_TIME_PERIOD, HALF_PI, N_SLOTS, DannBrain, ModulatedValue
from replay import _EVALS_CKPT_RE, body_descriptors, ckpt_config, is_fixed_body_run, replay_episode, run_identity
from sim import robot_bodies

SHIFTS = (0.05, 0.1, 0.25, 0.5)    # cycles
DETUNES = (0.02, 0.15)              # frequency factor - 1
N_DRAWS = 3             # random-subset draws per random condition
CLOCK_COL = (1 + N_SLOTS) * DANN_FEATURES  # the DANN's sine clock input

CONDITIONS: list[dict] = [{"condition": "baseline", "subset": None, "shift": 0.0, "rate": 1.0, "draw": 0}]
for _subset, _draws in (("side", 1), ("rand", N_DRAWS)):
    CONDITIONS += [{"condition": f"{_subset}_shift_{s}", "subset": _subset, "shift": s, "rate": 1.0, "draw": d}
                   for s in SHIFTS for d in range(_draws)]
    CONDITIONS += [{"condition": f"{_subset}_detune_{r}", "subset": _subset, "shift": 0.0, "rate": 1.0 + r, "draw": d}
                   for r in DETUNES for d in range(_draws)]


# ── Body sides ────────────────────────────────────────────────────────────────


def hinge_sides(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Per actuator: 1 (left), -1 (right) or 0 (midline) of its hinge anchor,
    relative to the plane that best mirrors the body's rest pose at spawn."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    rot = data.xmat[core].reshape(3, 3)
    xy = np.array([(rot.T @ (data.xpos[b] - data.xpos[core]))[:2] for b in robot_bodies(model) if b != core])
    angle, _ = gait.morph_plane(xy)
    normal = np.array([-math.sin(angle), math.cos(angle)])
    joints = model.actuator_trnid[:, 0]
    lat = np.array([(rot.T @ (data.xanchor[j] - data.xpos[core]))[:2] @ normal for j in joints])
    return np.where(lat > gait.MIDLINE_TOL, 1, np.where(lat < -gait.MIDLINE_TOL, -1, 0))


def random_subset(sides: np.ndarray, rng: np.random.Generator) -> Optional[np.ndarray]:
    """A random set of lateral hinges as large as the left side that is
    neither whole side (shifting the right side by half a cycle is the same
    as shifting the left), or None if there is none."""
    pool = np.flatnonzero(sides != 0)
    left, right = set(np.flatnonzero(sides == 1)), set(np.flatnonzero(sides == -1))
    k = len(left)
    if k == 0 or k >= len(pool):
        return None
    if math.comb(len(pool), k) <= 1000:
        mixed = [c for c in itertools.combinations(pool, k) if set(c) not in (left, right)]
        return np.array(mixed[rng.integers(len(mixed))]) if mixed else None
    while True:
        c = rng.choice(pool, size=k, replace=False)
        if set(c) not in (left, right):
            return c


# ── Perturbed controllers ─────────────────────────────────────────────────────


class ClockShiftedDann:
    """A DannBrain whose sine clock input runs, per hinge, offset cycles later
    and rate times faster; act() is otherwise DannBrain.act."""

    def __init__(self, brain: DannBrain, offset: np.ndarray, rate: np.ndarray) -> None:
        self.brain, self.offset, self.rate = brain, offset, rate

    def reset(self) -> None:
        self.brain.reset()

    def act(self, data: Any, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        b = self.brain
        x = b.node_inputs(state)
        x[:, CLOCK_COL] = np.sin(2 * math.pi * (b._step * self.rate / DANN_TIME_PERIOD + self.offset))
        h = np.maximum(0.0, x @ b._w1.T + b._b1)
        b._step += 1
        return np.tanh(h @ b._w2.T + b._b2)[:, 0] * HALF_PI


def perturb_brain(brain: Any, mask: np.ndarray, shift: float, rate: float) -> Any:
    """The brain with the masked hinges shift cycles later and rate times faster."""
    offset, factor = shift * mask, np.where(mask, rate, 1.0)
    if isinstance(brain, DannBrain):
        return ClockShiftedDann(brain, offset, factor)
    f = brain.freq
    brain.freq = ModulatedValue(f.start * factor, np.asarray(f.end) * factor, np.asarray(f.wobble_amp) * factor, f.wobble_rate)
    brain.phase_offset = np.mod(np.asarray(brain.phase_offset) + offset, 1.0)
    brain.reset()
    return brain


class Perturbation:
    """replay_episode's perturb hook for one condition; records the hinge split."""

    def __init__(self, cond: dict, seed_key: str) -> None:
        self.cond, self.seed_key = cond, seed_key
        self.info: dict = {}

    def __call__(self, brain: Any, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        sides = hinge_sides(model, data)
        self.info = {"hinges_left": int((sides == 1).sum()), "hinges_right": int((sides == -1).sum()),
                     "hinges_midline": int((sides == 0).sum()), "n_perturbed": 0}
        c = self.cond
        if c["subset"] is None:
            return brain
        if c["subset"] == "side":
            idx = np.flatnonzero(sides == 1)
        else:
            rng = np.random.default_rng([zlib.crc32(self.seed_key.encode()), c["draw"]])
            idx = random_subset(sides, rng)
            if idx is None:
                raise _NoSubset
        mask = np.zeros(model.nu, dtype=bool)
        mask[idx] = True
        self.info["n_perturbed"] = int(mask.sum())
        if not mask.any():
            raise _NoSubset
        return perturb_brain(brain, mask, c["shift"], c["rate"])


class _NoSubset(Exception):  # noqa: N818
    """The body has no hinges to perturb under this condition."""


# ── Replay ────────────────────────────────────────────────────────────────────


def select_checkpoints(run_dir: Path, which: str) -> list[Path]:
    """Fixed-body training: the last checkpoint (or all); evolution run: the
    best last-generation checkpoint."""
    if is_fixed_body_run(run_dir):
        ckpts = sorted(c for c in (run_dir / "checkpoints").iterdir() if _EVALS_CKPT_RE.match(c.name))
        return ckpts if which == "all" else ckpts[-1:]
    best = runs.best_checkpoint(run_dir)
    return [best] if best is not None else []


def _job(args: tuple[str, dict]) -> tuple[dict, str]:
    ckpt_s, cond = args
    ckpt = Path(ckpt_s)
    base = {"ckpt": ckpt.name, "condition": cond["condition"], "draw": cond["draw"]}
    try:
        cfg = ckpt_config(ckpt)
        meta = json.loads((ckpt / "meta.json").read_text())
        genome = json.loads((ckpt / "best_genome.json").read_text())
        weights = np.load(ckpt / "fwd_weights.npy")
        brain = cfg.get("brain", "ann")
        pert = Perturbation(cond, f"{ckpt.parent.parent.name}/{ckpt.name}")
        try:
            fitness, info, P, Q, dt = replay_episode(genome, weights, brain, cfg, perturb=pert)
        except _NoSubset:
            return {**base, **pert.info, "skipped": True}, ""
        ind, _ = gait.analyse_episode(P, Q, dt)
        m = _EVALS_CKPT_RE.match(ckpt.name)
        return {**base, **pert.info, "skipped": False, "evals": int(m.group(1)) if m else -1,
                "stored_fitness": float(meta.get("fitness", math.nan)), "replayed_fitness": fitness,
                "score": info["score"], "raw_score": info["raw_score"], **body_descriptors(genome), **ind}, ""
    except Exception as exc:  # noqa: BLE001 -- one bad episode must not stop the run
        return base, f"{type(exc).__name__}: {exc}"


def run_phase_shift(run_dir: Path, out: Path, which: str, workers: int) -> str:
    ckpts = select_checkpoints(run_dir, which)
    if not ckpts:
        return f"{run_dir.name}: no checkpoints"
    jobs = [(str(c), cond) for c in ckpts for cond in CONDITIONS]
    if workers > 1:
        with Pool(processes=min(workers, len(jobs))) as pool:
            results = pool.map(_job, jobs, chunksize=1)
    else:
        results = [_job(j) for j in jobs]

    ident = run_identity(run_dir)
    rows = [{**ident, **row} for row, err in results if not err]
    for row, err in results:
        if err:
            print(f"  ERROR {row['ckpt']} {row['condition']}: {err}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "phase_shift.csv", index=False)
    base = df[(df["condition"] == "baseline") & ~df["skipped"].astype(bool)]
    mism = int((np.abs(base["replayed_fitness"] - base["stored_fitness"]) > 1e-6).sum()) if len(base) else 0
    return f"{run_dir.name}: {len(df)} episodes, {int(df.get('skipped', pd.Series(dtype=bool)).sum())} skipped, {mism} baselines not reproduced"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--ckpts", choices=("last", "all"), default="last", help="fixed-body trainings: which checkpoints")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("-o", "--out-dir", type=Path, required=True, help="results go to <out-dir>/<run name>/")
    args = parser.parse_args()
    for run_dir in args.run_dir:
        print(run_phase_shift(run_dir, args.out_dir / run_dir.name, args.ckpts, args.workers), flush=True)


if __name__ == "__main__":
    main()
