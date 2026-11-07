"""
Body measures, for the morphological space (summarise.py) and the gait
analyses (mechanics.py).

descriptors(genome)      from the genome alone: module, hinge and limb counts,
                         and the Miras et al. (2018) descriptors of
                         ariel.utils.morphological_descriptor (size, joints,
                         limbs, branching, length of limbs, coverage, symmetry),
                         plus this study's mirror-symmetry share (genomes.py)
body_measures(genome)    from the body at rest after the 3 s settle (world
                         frame: forward -y, lateral x, up z): extent, height,
                         contacts, hinge axes and leg hinges (hinges whose
                         distal part touches the floor)

Moved from legacy/analyse_bodies.py and legacy/phase_shift.py.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pandas as pd

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.utils.morphological_descriptor import MorphologicalMeasures

import gait
import genomes
from config import PROTOCOL
from sim import build_world, robot_bodies

DIAG_MAX = 0.9  # a hinge axis is diagonal if no world component reaches this (45 deg rotations)
AXIS_CLASS = ("pitch", "roll", "yaw")  # dominant world axis of the hinge axis: x, y, z


def descriptors(genome: dict) -> dict:
    m = MorphologicalMeasures(TreeGenome.from_dict(genome).to_networkx())

    def safe(f):
        try:
            return float(f())
        except Exception:  # noqa: BLE001 -- some measures are undefined on tiny bodies
            return math.nan

    return {
        **genomes.describe(genome),
        "n_limbs": int(m.num_single_neighbour_modules),
        **{k: safe(lambda k=k: getattr(m, k)) for k in
           ("size", "joints", "limbs", "branching", "length_of_limbs", "coverage", "symmetry")},
    }


def subtree(model: mujoco.MjModel, root: int) -> set[int]:
    """Bodies of the kinematic subtree from root (MuJoCo orders parents first)."""
    out = {root}
    for b in range(root + 1, model.nbody):
        if model.body_parentid[b] in out:
            out.add(b)
    return out


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


def body_measures(genome: dict) -> dict:
    """Rest-pose measures of one body (see the module docstring)."""
    model, data = build_world(genome)
    mujoco.mj_resetData(model, data)
    while data.time < PROTOCOL.settle_s:
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

    hinge_rows = []
    for j in model.actuator_trnid[:, 0]:
        sub = subtree(model, int(model.jnt_bodyid[j]))
        feet = [p for b in sub if b in contacts for p in contacts[b]]
        hinge_rows.append({
            "axis": AXIS_CLASS[int(np.argmax(np.abs(data.xaxis[j])))],
            "diag": float(np.abs(data.xaxis[j]).max()) < DIAG_MAX,
            "leg": bool(feet),
            "reach": float(np.mean([np.linalg.norm(p - data.xanchor[j]) for p in feet])) if feet else np.nan,
            "distal_mass": sum(model.body_mass[b] for b in sub) / mass,
        })
    h = pd.DataFrame(hinge_rows, columns=["axis", "diag", "leg", "reach", "distal_mass"])
    legs = h[h["leg"].astype(bool)]
    sides = hinge_sides(model, data)

    out = {
        "hinges_lateral": int((sides != 0).sum()), "hinges_midline": int((sides == 0).sum()),
        "length": float(np.ptp(pos[:, 1])), "width": float(np.ptp(pos[:, 0])), "height": float(np.ptp(pos[:, 2])),
        "core_height": float(data.xpos[core][2]), "mass": mass,
        "n_grounded": len(contacts),
        "contact_len": float(cpos[:, 1].std()) if len(cpos) else 0.0,
        "contact_wid": float(cpos[:, 0].std()) if len(cpos) else 0.0,
        "n_leg": len(legs), "leg_reach": float(legs["reach"].mean()) if len(legs) else 0.0,
        "distal_mass": float(h["distal_mass"].mean()) if len(h) else 0.0,
        "n_diag": int(h["diag"].sum()), "n_leg_diag": int(legs["diag"].sum()),
    }
    out["aspect"] = out["length"] / max(out["length"] + out["width"], 1e-6)
    for a in AXIS_CLASS:
        out[f"n_{a}"] = int((h["axis"] == a).sum())
        out[f"n_leg_{a}"] = int((legs["axis"] == a).sum())
    return out
