"""
What makes the symmetric bodies that phase-lock different? (R2)

The phase-shift replay showed that left-right timing is no more critical to a
trained gait than any other timing, and that locking is mostly a property of
the body: under bang_bang, most symmetric E2 bodies lock in every training
seed or in none, and the always-locked ones score highest. This measures each
E2 body at rest and relates the measures to how it trains.

Body measures (after the 3 s settle with zero control, world frame: forward
is -y, lateral x, up z; bodies spawn unrotated):
  n_modules, n_hinges, n_limbs       replay.body_descriptors
  hinges_lateral, hinges_midline     phase_shift.hinge_sides
  length, width, height, aspect      extent of the module bodies; aspect = length / (length + width)
  core_height, mass
  n_grounded, contact_len/_wid       modules touching the floor; spread (std) of
                                     the floor contacts along forward / lateral
  per hinge, by its axis's dominant world direction:
    pitch (axis lateral: swings fore-aft), roll (axis forward: lifts sideways),
    yaw (axis vertical: sweeps horizontally)
  leg hinges                         hinges whose distal part touches the floor;
                                     n_leg, n_leg_pitch / _roll / _yaw, leg_reach
                                     (mean distance from a leg hinge to its distal
                                     floor contacts)
  distal_mass                        mean share of the body's mass a hinge moves
  n_diag, n_leg_diag                 hinges (leg hinges) whose axis is diagonal, off
                                     every world axis: the 45 deg rotations, which the
                                     symmetric genome rarely makes (midline hinges
                                     are forced to DEG_0)

Training outcomes per body (E3 replays of the E2 trainings' final checkpoints,
mean over seeds): score and locked share (seeds with phase_locked_frac >= 0.5).

Writes to --out:
  body_features.csv     one row per body: measures and outcomes per brain
  bodies_locked.csv     symmetric bodies locked in every seed vs none (under
                        --lock-brain): mean of each measure, Mann-Whitney p
  bodies_score_corr.csv Spearman of each measure with the body's score, per
                        genome and brain
Exploratory: ~25 measures on 40 bodies per genome; read p-values accordingly.

Usage:
    python analyse_bodies.py --bodies $FINAL_ROOT/e2/bodies_random.jsonl \\
        --replays $FINAL_ROOT/e3/e2_random --out $FINAL_ROOT/e3/bodies
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from brains import SETTLE_DURATION
from phase_shift import hinge_sides
from replay import body_descriptors
from sim import build_world, robot_bodies, zoo_spec

LOCKED_MIN = 0.5
DIAG_MAX = 0.9  # a hinge axis is diagonal if no world component reaches this (45 deg rotations)
AXIS_CLASS = ("pitch", "roll", "yaw")  # dominant world axis of the hinge axis: x, y, z


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    out = {root}
    for b in range(root + 1, model.nbody):
        if model.body_parentid[b] in out:
            out.add(b)
    return out


def body_measures(genome: dict) -> dict:
    """Rest-pose measures of one body (see the module docstring)."""
    model, data = build_world(genome, zoo_spec)
    mujoco.mj_resetData(model, data)
    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    bodies = robot_bodies(model)
    core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    pos = data.xpos[bodies]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    contacts: dict[int, list[np.ndarray]] = {}
    for c in data.contact[: data.ncon]:
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            contacts.setdefault(int(model.geom_bodyid[other]), []).append(c.pos.copy())
    cpos = np.array([p for ps in contacts.values() for p in ps]) if contacts else np.zeros((0, 3))
    mass = float(sum(model.body_mass[b] for b in bodies))

    joints = model.actuator_trnid[:, 0]
    hinge_rows = []
    for j in joints:
        sub = _subtree(model, int(model.jnt_bodyid[j]))
        axis = AXIS_CLASS[int(np.argmax(np.abs(data.xaxis[j])))]
        diag = float(np.abs(data.xaxis[j]).max()) < DIAG_MAX
        feet = [p for b in sub if b in contacts for p in contacts[b]]
        reach = float(np.mean([np.linalg.norm(p - data.xanchor[j]) for p in feet])) if feet else np.nan
        hinge_rows.append({"axis": axis, "diag": diag, "leg": bool(feet), "reach": reach,
                           "distal_mass": sum(model.body_mass[b] for b in sub) / mass})
    h = pd.DataFrame(hinge_rows)
    legs = h[h["leg"]]
    sides = hinge_sides(model, data)

    out = {
        **body_descriptors(genome),
        "hinges_lateral": int((sides != 0).sum()), "hinges_midline": int((sides == 0).sum()),
        "length": float(np.ptp(pos[:, 1])), "width": float(np.ptp(pos[:, 0])), "height": float(np.ptp(pos[:, 2])),
        "core_height": float(data.xpos[core][2]), "mass": mass,
        "n_grounded": len(contacts),
        "contact_len": float(cpos[:, 1].std()) if len(cpos) else 0.0,
        "contact_wid": float(cpos[:, 0].std()) if len(cpos) else 0.0,
        "n_leg": int(len(legs)), "leg_reach": float(legs["reach"].mean()) if len(legs) else 0.0,
        "distal_mass": float(h["distal_mass"].mean()) if len(h) else 0.0,
        "n_diag": int(h["diag"].sum()) if len(h) else 0, "n_leg_diag": int(legs["diag"].sum()) if len(legs) else 0,
    }
    out["aspect"] = out["length"] / max(out["length"] + out["width"], 1e-6)
    for a in AXIS_CLASS:
        out[f"n_{a}"] = int((h["axis"] == a).sum()) if len(h) else 0
        out[f"n_leg_{a}"] = int((legs["axis"] == a).sum()) if len(legs) else 0
    return out


def outcomes(replays: Path) -> pd.DataFrame:
    """Per body and brain: mean final score and share of seeds phase-locked."""
    eps = pd.concat([pd.read_csv(p) for p in sorted(replays.glob("*/episodes.csv"))], ignore_index=True)
    final = eps[eps["evals"] == eps.groupby("run")["evals"].transform("max")].copy()
    final["locked"] = final["phase_locked_frac"] >= LOCKED_MIN
    agg = final.groupby(["body_id", "brain"]).agg(score=("score", "mean"), locked=("locked", "mean")).unstack()
    agg.columns = [f"{v}_{b}" for v, b in agg.columns]
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bodies", type=Path, required=True, help="sample_bodies.py output (jsonl)")
    parser.add_argument("--replays", type=Path, required=True, help="replay.py output root of the E2 trainings")
    parser.add_argument("--lock-brain", default="bang_bang")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in args.bodies.read_text().splitlines():
        b = json.loads(line)
        rows.append({"body_id": b["body_id"], "genome": b["genome_type"], **body_measures(b["genome"])})
    df = pd.DataFrame(rows).set_index("body_id").join(outcomes(args.replays)).reset_index()
    df.to_csv(args.out / "body_features.csv", index=False)
    measures = [c for c in rows[0] if c not in ("body_id", "genome")]
    brains = sorted(c.removeprefix("score_") for c in df if c.startswith("score_"))

    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print("measures by genome (mean):")
        print(df.groupby("genome")[measures].mean().T.round(2).to_string())

        sym = df[df["genome"] == "tree_symmetric"]
        lk = sym[f"locked_{args.lock_brain}"]
        always, never = sym[lk == 1.0], sym[lk == 0.0]
        tab = pd.DataFrame([{"measure": m, "always_locked": always[m].mean(), "never_locked": never[m].mean(),
                             "tree": df.loc[df["genome"] == "tree", m].mean(),
                             "p": mannwhitneyu(always[m], never[m]).pvalue if always[m].nunique() + never[m].nunique() > 2 else np.nan}
                            for m in measures]).sort_values("p")
        tab.to_csv(args.out / "bodies_locked.csv", index=False)
        print(f"\nsymmetric bodies locked in every seed (n={len(always)}) vs none (n={len(never)}) under {args.lock_brain}:")
        print(tab.round(3).to_string(index=False))

        corr = []
        for genome, g in df.groupby("genome"):
            for brain in brains:
                for m in measures:
                    r = spearmanr(g[m], g[f"score_{brain}"]) if g[m].nunique() > 1 else None
                    corr.append({"genome": genome, "brain": brain, "measure": m,
                                 "rho": r.statistic if r else np.nan, "p": r.pvalue if r else np.nan})
        corr = pd.DataFrame(corr)
        corr.to_csv(args.out / "bodies_score_corr.csv", index=False)
        wide = corr[corr["brain"].isin(["bang_bang", "dann"])].pivot_table(index="measure", columns=["genome", "brain"], values="rho")
        print("\nSpearman rho of each measure with the body's score:")
        print(wide.round(2).sort_values(("tree_symmetric", "bang_bang")).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
