"""
Recording replay + behavioural descriptors for gecko_skill_tasks.py
checkpoints (forward / multidirection / turn_avg locomotion runs).

``replay_skill_episode`` mirrors ``shared._skill_worker_eval`` step for step
(settle, stride-gated control with CTRL_ALPHA smoothing, rotor-floor contact
counting, yaw accumulation, jerk / height penalties) so its recomputed fitness
reproduces the stored per-skill fitness, and additionally records:

  - saved traces, subsampled every ``record_every_n`` sim steps: hinge
    angles/velocities, world pose + floor contact of every robot body, and
    controller state / raw / applied actions (every
    ``max(1, record_every_n // control_step_freq)``-th control update);
  - full-rate rollout arrays (core pose, hinge angles, body contacts, every
    control update), used only by ``compute_behavior_descriptors`` and then
    dropped, so subsampling never biases the descriptors.

Also home to the task -> skill table and checkpoint replay config shared with
render_skill_task_checkpoint.py.
"""

from __future__ import annotations

import functools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

import genome_adapter
from shared import (
    CONTROL_STEP_FREQ,
    CTRL_ALPHA,
    FORWARD_AXIS,
    HEIGHT_PENALTY_THRESHOLD,
    JERK_PENALTY_WEIGHT,
    JERK_THRESHOLD,
    LOCO_DURATION,
    SETTLE_DURATION,
    TURN_DURATION,
    _rotate_2d,
    build_loco_world_for_body,
    floor_id,
    genome_to_spec,
    make_brain_for_model,
    rotor_geom_ids,
    signed_vertical_yaw_delta,
)

N_DIRECTIONS = 5  # must match gecko_skill_tasks.py


def skills_for_task(task: str) -> list[tuple[str, dict]]:
    """Duplicated from gecko_skill_tasks.py:_skills_for_task (importing that
    module would trigger its own module-level argparse.parse_args() and
    fail). Each entry is (skill_name, reward_spec) where reward_spec is a
    dict describing the SkillReward used at training time.
    """
    if task == "forward":
        return [("fwd", {"kind": "translate", "angle_deg": 0.0, "duration": LOCO_DURATION})]
    if task == "multidirection":
        step = 360.0 / N_DIRECTIONS
        return [
            (f"dir{i}", {"kind": "translate", "angle_deg": i * step, "duration": LOCO_DURATION})
            for i in range(N_DIRECTIONS)
        ]
    if task == "turn_avg":
        return [
            ("fwd", {"kind": "translate", "angle_deg": 0.0, "duration": LOCO_DURATION}),
            ("left", {"kind": "rotate", "turn_sign": +1, "duration": TURN_DURATION}),
            ("right", {"kind": "rotate", "turn_sign": -1, "duration": TURN_DURATION}),
        ]
    raise ValueError(f"Unknown task: {task!r}")


def resolve_replay_config(ckpt_dir: Path, max_modules: Optional[int] = None) -> tuple:
    """(to_spec_fn, control_step_freq, brain_kind) for replaying a checkpoint
    exactly as it was trained. Checkpoints live at <run>/checkpoints/<ckpt>;
    run_config.json sits in <run>. Runs predating --control-step-freq / --brain
    have no such keys and used the shared default / the ANN brain. ``max_modules`` overrides run_config's value
    (itself defaulting to 25) for the cppn decoder.
    """
    meta_path = ckpt_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    run_config_path = ckpt_dir.parent.parent / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}

    if max_modules is None:
        max_modules = run_config.get("max_modules", 25)
    to_spec_fn = (
        functools.partial(genome_adapter.cppn_genome_to_spec, max_modules=max_modules)
        if meta.get("genome_type") == "cppn"
        else genome_to_spec
    )
    control_step_freq = run_config.get("control_step_freq", CONTROL_STEP_FREQ)
    brain_kind = run_config.get("brain", "ann")
    return to_spec_fn, control_step_freq, brain_kind


# ── Recording replay ──────────────────────────────────────────────────────────


@dataclass
class EpisodeTrace:
    skill: str
    reward_spec: dict
    control_step_freq: int
    dt: float
    replayed_fitness: float = float("nan")
    raw_score: float = float("nan")  # displacement along skill axis, or accumulated yaw
    mean_jerk: float = 0.0
    c_hinge: int = 0
    initial_height: float = 0.0
    wall_time_s: float = 0.0

    joint_names: list[str] = field(default_factory=list)    # hinge order used below
    body_names: list[str] = field(default_factory=list)     # robot bodies with geoms
    actuator_joint: list[int] = field(default_factory=list)  # actuator -> index into joint_names

    # Saved (subsampled) rows.
    joint_rows: list[tuple] = field(default_factory=list)  # (sim_step, t, angles[J], vels[J])
    pose_rows: list[tuple] = field(default_factory=list)   # (sim_step, t, pos[B,3], quat[B,4], contact[B])
    ctrl_rows: list[tuple] = field(default_factory=list)   # (ctrl_step, sim_step, t, state, raw, applied)

    # Full-rate rollout arrays (post-settle only), for descriptors.
    core_xy: Optional[np.ndarray] = None       # (T, 2)
    core_z: Optional[np.ndarray] = None        # (T,)
    core_tilt: Optional[np.ndarray] = None     # (T,) angle between core local z and world z
    yaw_delta: Optional[np.ndarray] = None     # (T,) signed vertical yaw per step
    joint_angles: Optional[np.ndarray] = None  # (T, J)
    body_contact: Optional[np.ndarray] = None  # (T, B) bool
    ctrl_applied: Optional[np.ndarray] = None  # (C, nu)
    joint_anchors_core: Optional[np.ndarray] = None  # (J, 3) rest pose, in core frame


def _robot_bodies(model: mujoco.MjModel) -> list[int]:
    """Robot bodies that own at least one geom (core, stators, rotors, bricks)."""
    has_geom = set(int(b) for b in model.geom_bodyid)
    return [
        b for b in range(model.nbody)
        if model.body(b).name.startswith("robot1_") and b in has_geom
    ]


def replay_skill_episode(
    genome: dict,
    weights: np.ndarray,
    skill_name: str,
    reward_spec: dict,
    to_spec_fn=genome_to_spec,
    control_step_freq: int = CONTROL_STEP_FREQ,
    record_every_n: int = 100,
    brain_kind: str = "ann",
) -> EpisodeTrace:
    """Replay one trained skill; see module docstring. The control/fitness
    logic must stay in lockstep with shared._skill_worker_eval.
    """
    from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state

    t0 = time.perf_counter()
    model, data = build_loco_world_for_body(genome, to_spec_fn=to_spec_fn)
    brain = make_brain_for_model(brain_kind, model, data, np.asarray(weights, dtype=np.float32))

    rotor_ids = rotor_geom_ids(model)
    floor = floor_id(model)
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")

    hinge_joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qpos_adr = np.array([model.jnt_qposadr[j] for j in hinge_joints], dtype=int)
    dof_adr = np.array([model.jnt_dofadr[j] for j in hinge_joints], dtype=int)
    bodies = _robot_bodies(model)
    body_col = {b: i for i, b in enumerate(bodies)}
    jnt_col = {j: i for i, j in enumerate(hinge_joints)}

    trace = EpisodeTrace(
        skill=skill_name, reward_spec=reward_spec,
        control_step_freq=control_step_freq, dt=float(model.opt.timestep),
        joint_names=[model.joint(j).name for j in hinge_joints],
        body_names=[model.body(b).name for b in bodies],
        actuator_joint=[jnt_col.get(int(model.actuator_trnid[a, 0]), -1) for a in range(model.nu)],
    )
    ctrl_record_every = max(1, record_every_n // control_step_freq)

    def body_contacts() -> np.ndarray:
        mask = np.zeros(len(bodies), dtype=bool)
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if g1 == floor else g1 if g2 == floor else -1
            if other >= 0:
                col = body_col.get(int(model.geom_bodyid[other]))
                if col is not None:
                    mask[col] = True
        return mask

    def record_step(sim_step: int, contacts: Optional[np.ndarray] = None) -> None:
        if sim_step % record_every_n != 0:
            return
        if contacts is None:
            contacts = body_contacts()
        trace.joint_rows.append((
            sim_step, data.time, data.qpos[qpos_adr].copy(), data.qvel[dof_adr].copy(),
        ))
        trace.pose_rows.append((
            sim_step, data.time, data.xpos[bodies].copy(), data.xquat[bodies].copy(), contacts,
        ))

    mujoco.mj_resetData(model, data)
    # Hinge anchors in the core frame at the genome's rest pose (before settle
    # deformation), for pairing left/right mirror joints.
    mujoco.mj_kinematics(model, data)
    core_R = data.xmat[core_id].reshape(3, 3)
    trace.joint_anchors_core = (data.xanchor[hinge_joints] - data.xpos[core_id]) @ core_R

    sim_step = 0
    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)
        record_step(sim_step)
        sim_step += 1

    initial_height = float(data.xpos[core_id, 2])

    step = 0
    current_action = np.zeros(model.nu)
    c_hinge = 0
    prev_contacts: set[int] = set()
    prev_ctrl: Optional[np.ndarray] = None
    jerk_sum = 0.0
    ctrl_step = 0

    kind = reward_spec["kind"]
    if kind == "translate":
        xy0 = np.array([data.qpos[0], data.qpos[1]])
        reward_axis = _rotate_2d(FORWARD_AXIS, reward_spec["angle_deg"])
    else:
        accumulated = 0.0
    r_prev = np.array(data.xmat[core_id]).reshape(3, 3).copy()

    core_xy, core_z, core_tilt, yaw_delta = [], [], [], []
    joint_angles, body_contact, ctrl_applied = [], [], []

    episode_end = SETTLE_DURATION + reward_spec["duration"]
    while data.time < episode_end:
        if step % control_step_freq == 0:
            state = get_robot_state(data).astype(np.float32)
            raw_action = brain.act(data, state)
            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -math.pi / 2, math.pi / 2,
            )
            if prev_ctrl is not None:
                jerk_sum += float(np.mean(np.abs(current_action - prev_ctrl)))
            prev_ctrl = current_action.copy()
            if ctrl_step % ctrl_record_every == 0:
                trace.ctrl_rows.append((
                    ctrl_step, sim_step, data.time, state.copy(),
                    np.asarray(raw_action).copy(), current_action.copy(),
                ))
            ctrl_applied.append(current_action.copy())
            ctrl_step += 1
        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

        curr: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor and g2 in rotor_ids:
                curr.add(g2)
            elif g2 == floor and g1 in rotor_ids:
                curr.add(g1)
        c_hinge += len(curr - prev_contacts)
        prev_contacts = curr

        r_curr = np.array(data.xmat[core_id]).reshape(3, 3)
        delta = signed_vertical_yaw_delta(r_prev, r_curr)
        if kind == "rotate":
            accumulated += max(0.0, delta * reward_spec["turn_sign"])
        r_prev = r_curr.copy()

        contacts = body_contacts()
        core_xy.append((data.qpos[0], data.qpos[1]))
        core_z.append(data.xpos[core_id, 2])
        core_tilt.append(math.acos(max(-1.0, min(1.0, float(r_curr[2, 2])))))
        yaw_delta.append(delta)
        joint_angles.append(data.qpos[qpos_adr].copy())
        body_contact.append(contacts)
        record_step(sim_step, contacts)

        step += 1
        sim_step += 1

    mean_jerk = jerk_sum / max(ctrl_step - 1, 1)
    jerk_penalty = JERK_PENALTY_WEIGHT * mean_jerk if mean_jerk >= JERK_THRESHOLD else 0.0
    height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    if kind == "translate":
        xy_now = np.array([data.qpos[0], data.qpos[1]])
        raw_score = float(np.dot(xy_now - xy0, reward_axis))
    else:
        raw_score = accumulated
    # c_hinge is a behavioural descriptor only; training no longer scores it.
    fitness = -(raw_score - height_penalty - jerk_penalty)

    trace.replayed_fitness = fitness
    trace.raw_score = raw_score
    trace.mean_jerk = mean_jerk
    trace.c_hinge = c_hinge
    trace.initial_height = initial_height
    trace.core_xy = np.asarray(core_xy, dtype=np.float64).reshape(-1, 2)
    trace.core_z = np.asarray(core_z, dtype=np.float64)
    trace.core_tilt = np.asarray(core_tilt, dtype=np.float64)
    trace.yaw_delta = np.asarray(yaw_delta, dtype=np.float64)
    trace.joint_angles = np.asarray(joint_angles, dtype=np.float64).reshape(-1, len(hinge_joints))
    trace.body_contact = np.asarray(body_contact, dtype=bool).reshape(-1, len(bodies))
    trace.ctrl_applied = np.asarray(ctrl_applied, dtype=np.float64).reshape(-1, model.nu)
    trace.wall_time_s = time.perf_counter() - t0
    return trace


def drop_full_rate(trace: EpisodeTrace) -> None:
    """Free the full-rate arrays once descriptors are computed (they dominate
    the pickle size when shipping traces back from pool workers)."""
    trace.core_xy = trace.core_z = trace.core_tilt = trace.yaw_delta = None
    trace.joint_angles = trace.body_contact = trace.ctrl_applied = None


# ── Behavioural descriptors ───────────────────────────────────────────────────

MIRROR_TOLERANCE = 0.03  # m, max distance between a hinge and its partner's mirror image
ON_PLANE_TOLERANCE = 0.01  # m, hinges this close to the mirror plane have no partner

DESCRIPTOR_NAMES = [
    # locomotion
    "net_displacement", "path_length", "mean_speed", "straightness",
    "axis_displacement", "lateral_drift", "yaw_total", "mean_abs_yaw_rate",
    # posture
    "core_height_mean", "core_height_std", "core_tilt_mean", "core_tilt_std",
    # contact
    "duty_factor_mean", "n_contact_bodies_mean", "frac_bodies_ever_contact",
    "contact_entropy_bits", "c_hinge",
    # gait
    "gait_frequency_hz", "joint_amplitude_mean", "joint_corr_abs_mean",
    # left/right symmetry
    "mirror_pair_frac", "mirror_pair_corr_mean", "mirror_pair_phase_lag",
    "mirror_pair_amp_asym", "mirror_plane",
    # effort
    "ctrl_abs_mean", "ctrl_jerk_mean",
]


def _mirror_pairs(anchors: np.ndarray, axis: int) -> list[tuple[int, int]]:
    """Mutual-nearest hinge pairs under reflection of core-frame coordinate
    ``axis``. Hinges within ON_PLANE_TOLERANCE of the plane are unpaired."""
    n = len(anchors)
    if n < 2:
        return []
    mirrored = anchors.copy()
    mirrored[:, axis] *= -1
    dist = np.linalg.norm(anchors[:, None, :] - mirrored[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    off_plane = np.abs(anchors[:, axis]) > ON_PLANE_TOLERANCE
    nearest = np.argmin(dist, axis=1)
    pairs = []
    for i in range(n):
        j = int(nearest[i])
        if (i < j and nearest[j] == i and dist[i, j] <= MIRROR_TOLERANCE
                and off_plane[i] and off_plane[j]):
            pairs.append((i, j))
    return pairs


def _dominant_frequency(x: np.ndarray, dt: float) -> tuple[float, np.ndarray]:
    """(dominant non-DC frequency in Hz, rfft) of a mean-removed signal."""
    spec = np.fft.rfft(x - x.mean())
    freqs = np.fft.rfftfreq(len(x), d=dt)
    if len(spec) < 2:
        return float("nan"), spec
    k = 1 + int(np.argmax(np.abs(spec[1:])))
    return float(freqs[k]), spec


def compute_behavior_descriptors(
    core_xy: np.ndarray,
    core_z: np.ndarray,
    core_tilt: np.ndarray,
    yaw_delta: np.ndarray,
    joint_angles: np.ndarray,
    body_contact: np.ndarray,
    ctrl_applied: np.ndarray,
    joint_anchors_core: np.ndarray,
    dt: float,
    axis_angle_deg: float = 0.0,
    c_hinge: int = 0,
) -> dict[str, float]:
    """Per-episode behavioural descriptors from full-rate post-settle arrays.

    ``axis_angle_deg`` is the translate skill's reward direction (FORWARD_AXIS
    rotated CCW); axis/lateral displacement use it. ``mirror_pair_frac`` is
    the fraction of off-plane hinges that have a mirror partner (1.0 for a
    perfectly bilateral body). ``mirror_plane`` is the
    core-frame axis (0=x, 1=y) whose reflection pairs up the most hinges —
    tree_symmetric genomes mirror across a fixed plane, but cppn / tree
    bodies need not, so both candidates are tried.
    """
    nan = float("nan")
    d: dict[str, float] = dict.fromkeys(DESCRIPTOR_NAMES, nan)
    T = len(core_xy)
    if T < 2:
        return d
    duration = T * dt

    # Locomotion
    disp = core_xy[-1] - core_xy[0]
    steps = np.linalg.norm(np.diff(core_xy, axis=0), axis=1)
    path = float(steps.sum())
    axis = _rotate_2d(FORWARD_AXIS, axis_angle_deg)
    lateral = np.array([-axis[1], axis[0]])
    d["net_displacement"] = float(np.linalg.norm(disp))
    d["path_length"] = path
    d["mean_speed"] = path / duration
    d["straightness"] = d["net_displacement"] / path if path > 1e-9 else nan
    d["axis_displacement"] = float(disp @ axis)
    d["lateral_drift"] = float(abs(disp @ lateral))
    d["yaw_total"] = float(yaw_delta.sum())
    d["mean_abs_yaw_rate"] = float(np.abs(yaw_delta).sum() / duration)

    # Posture
    d["core_height_mean"] = float(core_z.mean())
    d["core_height_std"] = float(core_z.std())
    d["core_tilt_mean"] = float(core_tilt.mean())
    d["core_tilt_std"] = float(core_tilt.std())

    # Contact
    if body_contact.shape[1] > 0:
        d["duty_factor_mean"] = float(body_contact.mean())
        d["n_contact_bodies_mean"] = float(body_contact.sum(axis=1).mean())
        d["frac_bodies_ever_contact"] = float(body_contact.any(axis=0).mean())
        _, counts = np.unique(np.packbits(body_contact, axis=1), axis=0, return_counts=True)
        p = counts / counts.sum()
        d["contact_entropy_bits"] = max(0.0, float(-(p * np.log2(p)).sum()))
    d["c_hinge"] = float(c_hinge)

    # Gait
    J = joint_angles.shape[1]
    if J > 0:
        amps = joint_angles.std(axis=0)
        d["joint_amplitude_mean"] = float(amps.mean())
        freqs_specs = [_dominant_frequency(joint_angles[:, j], dt) for j in range(J)]
        freqs = np.array([f for f, _ in freqs_specs])
        if amps.sum() > 1e-9:
            d["gait_frequency_hz"] = float(np.average(freqs, weights=amps))
        moving = amps > 1e-6
        if moving.sum() >= 2:
            corr = np.corrcoef(joint_angles[:, moving].T)
            iu = np.triu_indices(len(corr), k=1)
            d["joint_corr_abs_mean"] = float(np.nanmean(np.abs(corr[iu])))

        # Left/right symmetry
        best_axis, best_pairs, best_off = None, [], 0
        for ax in (0, 1):
            pairs = _mirror_pairs(joint_anchors_core, ax)
            if len(pairs) > len(best_pairs):
                best_axis, best_pairs = ax, pairs
                best_off = int((np.abs(joint_anchors_core[:, ax]) > ON_PLANE_TOLERANCE).sum())
        if best_pairs:
            d["mirror_pair_frac"] = 2 * len(best_pairs) / best_off
        else:
            d["mirror_pair_frac"] = 0.0
        if best_pairs:
            d["mirror_plane"] = float(best_axis)
            corrs, lags, asyms = [], [], []
            for i, j in best_pairs:
                if amps[i] > 1e-6 and amps[j] > 1e-6:
                    corrs.append(float(np.corrcoef(joint_angles[:, i], joint_angles[:, j])[0, 1]))
                    spec_i, spec_j = freqs_specs[i][1], freqs_specs[j][1]
                    k = 1 + int(np.argmax(np.abs(spec_i[1:]) + np.abs(spec_j[1:])))
                    lags.append(abs(float(np.angle(spec_i[k] * np.conj(spec_j[k])))))
                if amps[i] + amps[j] > 1e-9:
                    asyms.append(float(abs(amps[i] - amps[j]) / (amps[i] + amps[j])))
            if corrs:
                d["mirror_pair_corr_mean"] = float(np.mean(corrs))
                d["mirror_pair_phase_lag"] = float(np.mean(lags))
            if asyms:
                d["mirror_pair_amp_asym"] = float(np.mean(asyms))

    # Effort
    if ctrl_applied.size:
        d["ctrl_abs_mean"] = float(np.abs(ctrl_applied).mean())
        if len(ctrl_applied) > 1:
            d["ctrl_jerk_mean"] = float(np.abs(np.diff(ctrl_applied, axis=0)).mean())
    return d


def trace_descriptors(trace: EpisodeTrace) -> dict[str, float]:
    return compute_behavior_descriptors(
        core_xy=trace.core_xy, core_z=trace.core_z, core_tilt=trace.core_tilt,
        yaw_delta=trace.yaw_delta, joint_angles=trace.joint_angles,
        body_contact=trace.body_contact, ctrl_applied=trace.ctrl_applied,
        joint_anchors_core=trace.joint_anchors_core, dt=trace.dt,
        axis_angle_deg=trace.reward_spec.get("angle_deg", 0.0), c_hinge=trace.c_hinge,
    )
