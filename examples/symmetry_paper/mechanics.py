"""
How do trained gaits move? Measures that need no left-right pairing, so tree,
symmetric and tied bodies are measured alike, and a hinge knockout.

The gait detector (gait.py) pairs modules across the sagittal plane, so it says
little about bodies with few such pairs: the stroke it covers is 0.74 of a
symmetric E1 champion's but 0.30 of a tied one's. This replays the same trained
controllers with the training episode itself (sim.run_episode, so the stored
score is reproduced) and records what happens at the hinges and the floor.

--mode measure, one episode per checkpoint:
  Decomposition   displacement = cycles x path per cycle x straightness, exactly:
                  cycles = gait frequency x episode length; path per cycle = core
                  path sampled once per gait period, per cycle; straightness =
                  displacement / that path (signed).
  Coordination    all hinge pairs, classed by genome relation: twin (a tie of a
                  tied genome), mirror (the node at the mirrored tree position) or
                  other. gait.phase_quality of the joint angles and of the limb:
                  the first principal component of the mass centre of everything
                  distal to the hinge, in the core frame. A shared clock makes
                  joint angles steady nearly by construction, so limb quality is
                  the informative one. Limb lags are only known modulo half a
                  cycle (the component's sign is arbitrary); joint lags are not
                  physically comparable either (hinge axes differ), so lags are
                  reported for joints but mean little across unrelated pairs.
                  Nested pairs (one hinge carries the other) are flagged: the distal
                  limb moves with the proximal hinge, so their limb quality is
                  inflated; limb_locked_unnested leaves them out.
  Dimensions      participation ratio of the active hinges' angle covariance
                  (unit variance per hinge): 1 = all hinges move as one, n_active
                  = independent.
  Floor           bodies in contact, stepping feet (in contact 5-95% of the time)
                  and their duty, always-on contacts, contact changes per second;
                  horizontal speed of bodies while in contact relative to core
                  speed (foot_fwd_frac: their forward speed relative to the core's;
                  0 = planted feet, 1 = feet dragged along).
  Core            tilt mean and RMS, height mean and RMS, net and total yaw.
  Effort          positive mechanical work of the hinge actuators (every physics
                  step), per metre, cost of transport, and how many hinges share
                  the work (inverse participation of per-hinge work).

--mode knockout, per checkpoint: baseline, all hinges frozen, and two episodes per
hinge. The hinge's raw command is held at its mean angle in the baseline (score_mean,
which removes its movement and keeps its pose) or at 0, the rest pose (score, which
also changes the body's shape if the hinge holds a bent pose). The rest of the
controller is unchanged, and a DANN still sees the frozen hinge's true state.

Episodes (runs.episodes, --which): an E1 run's champion (default) or final
population; for an E2 training, its best seed (champion) or every seed.

Writes <out-dir>/<run name>/ (a run with its files present is skipped):
  mechanics.csv, hinges.csv, pairs.csv    --mode measure
  knockout.csv                            --mode knockout

Usage:
    python mechanics.py --mode measure --roots ROOT [ROOT ...] [--match GLOB] [--which champion] [--workers 60] -o OUT
    python mechanics.py --mode knockout --run-dir RUN [RUN ...] -o OUT
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import fnmatch
import math
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np
import pandas as pd

from ariel.ec.genotypes.tree.collision import mirror_node
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

import gait
import runs
from bodies import AXIS_CLASS, subtree
from brains import SETTLE_DURATION, make_brain_for_model
from config import PROTOCOL
from replay import SCORE_TOL
from sim import FORWARD_AXIS, build_world, robot_bodies, run_episode

RECORD_EVERY_N = 10         # physics steps between samples: 50 Hz
ACTIVE_RANGE_DEG = 10.0     # a hinge is active if its angle's 5-95% range reaches this
STEPPING = (0.05, 0.95)     # share of time in contact of a stepping foot
MIN_DISPLACEMENT = 0.05     # m; below this, per-metre measures are undefined


# ── Pure measures ─────────────────────────────────────────────────────────────


def decompose(xy: np.ndarray, displacement: float, freq: float, dt: float, duration: float) -> dict:
    """cycles x path_per_cycle x straightness = displacement (see the module docstring)."""
    nan = {"cycles": math.nan, "path_per_cycle": math.nan, "straightness": math.nan, "wobble": math.nan}
    if not np.isfinite(freq) or freq <= 0:
        return nan
    n = max(round(1.0 / (freq * dt)), 1)
    idx = np.unique(np.r_[np.arange(0, len(xy), n), len(xy) - 1])
    path = float(np.linalg.norm(np.diff(xy[idx], axis=0), axis=1).sum())
    if path <= 0:
        return nan
    full = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    cycles = freq * duration
    return {"cycles": cycles, "path_per_cycle": path / cycles, "straightness": displacement / path, "wobble": full / path}


def participation_ratio(x: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2 of the columns' correlation matrix."""
    if x.shape[1] == 0:
        return math.nan
    if x.shape[1] == 1:
        return 1.0
    z = (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)
    lam = np.linalg.eigvalsh(np.cov(z, rowvar=False))
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def inverse_participation(w: np.ndarray) -> float:
    """1 / sum share^2 of non-negative contributions: how many share them equally."""
    w = np.maximum(np.asarray(w, dtype=np.float64), 0.0)
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.sum() > 0 else math.nan


def first_component(x: np.ndarray) -> np.ndarray:
    """Projection of a (T, 3) trajectory on its first principal component."""
    c = x - x.mean(0)
    return c @ np.linalg.svd(c, full_matrices=False)[2][0]


def contact_stats(contact: np.ndarray, pos: np.ndarray, dt: float) -> dict:
    """Floor measures from contact (T, B, bool) and body positions (T, B, 3)."""
    share = contact.mean(0)
    stepping = (share >= STEPPING[0]) & (share <= STEPPING[1])
    vel = np.gradient(pos[:, :, :2], dt, axis=0)
    core_vel = vel[:, 0]    # the core is first
    speed = float(np.linalg.norm(core_vel, axis=1).mean())
    net = pos[-1, 0, :2] - pos[0, 0, :2]
    direction = net / max(np.linalg.norm(net), 1e-9)
    if contact.any():
        t, b = np.nonzero(contact)
        slip = float(np.linalg.norm(vel[t, b], axis=1).mean())
        along = float((vel[t, b] @ direction).mean())
    else:
        slip, along = math.nan, math.nan
    core_along = float((core_vel @ direction).mean())
    return {
        "bodies_in_contact": float(contact.sum(1).mean()),
        "n_stepping": int(stepping.sum()), "n_always": int((share > STEPPING[1]).sum()),
        "stepping_duty": float(share[stepping].mean()) if stepping.any() else math.nan,
        "contact_changes_per_s": float(np.abs(np.diff(contact.astype(np.int8), axis=0)).sum() / (len(contact) * dt)),
        "foot_speed": slip, "foot_speed_rel": slip / speed if speed > 1e-6 else math.nan,
        "foot_fwd_frac": along / core_along if abs(core_along) > 1e-3 else math.nan,
        "core_speed_path": speed,
    }


def pair_rows(angles: np.ndarray, limbs: np.ndarray, nodes: list[int], twin: dict[int, int], mirror: dict[int, int],
              active: np.ndarray, amp: np.ndarray, freq: float, dt: float,
              nested: frozenset[frozenset[int]] = frozenset()) -> list[dict]:
    """One row per pair of active hinges. nested: pairs where one hinge carries the
    other, so the distal limb moves with the proximal hinge whatever the timing."""
    rows = []
    for i, j in combinations(np.flatnonzero(active), 2):
        a, b = nodes[i], nodes[j]
        kind = "twin" if twin.get(a) == b else "mirror" if mirror.get(a) == b else "other"
        jq = gait.phase_quality(angles[:, [i]], angles[:, [j]], freq, dt)
        lq = gait.phase_quality(limbs[:, [i]], limbs[:, [j]], freq, dt)
        rows.append({"node_a": a, "node_b": b, "kind": kind, "nested": frozenset((a, b)) in nested, "joint_quality": jq["phase_quality"],
                     "joint_lag": jq["mean_lag"], "limb_quality": lq["phase_quality"], "weight": float(amp[i] * amp[j])})
    return rows


# ── Bodies ────────────────────────────────────────────────────────────────────


def actuator_nodes(model: mujoco.MjModel) -> list[int]:
    """The genome node of each actuator's hinge: joint names end "<parent>-<child>-<face>-servo"."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in model.actuator_trnid[:, 0]]
    return [int(n.removesuffix("-servo").split("-")[-2]) for n in names]


def hinge_relations(genome: dict, nodes: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    """(twin, mirror): node -> its partner, for hinges whose partner is also a hinge."""
    present = set(nodes)
    twin: dict[int, int] = {}
    for t in genome.get("ties", []):
        if t["a"] in present and t["b"] in present:
            twin[t["a"]], twin[t["b"]] = t["b"], t["a"]
    tree = TreeGenome.from_dict(genome)
    mirror: dict[int, int] = {}
    for n in nodes:
        m = mirror_node(tree, n, MirrorAxis.Y_ZERO)
        if m is not None and m != n and m in present:
            mirror[n] = m
    return twin, mirror


def nested_pairs(genome: dict, nodes: list[int]) -> frozenset[frozenset[int]]:
    """Hinge pairs in which one hinge is an ancestor of the other in the body tree."""
    parent = {e["child"]: e["parent"] for e in TreeGenome.from_dict(genome).edges}
    present, out = set(nodes), set()
    for n in nodes:
        p = parent.get(n)
        while p is not None:
            if p in present:
                out.add(frozenset((n, p)))
            p = parent.get(p)
    return frozenset(out)


# ── Episodes ──────────────────────────────────────────────────────────────────


def measure_episode(genome: dict, weights: np.ndarray, brain_kind: str) -> tuple[dict, list[dict], list[dict], float]:
    """(episode row, hinge rows, pair rows, score) of one replayed episode."""
    model, data = build_world(genome)
    brain = make_brain_for_model(brain_kind, model, data, np.asarray(weights, dtype=np.float32))
    joints = model.actuator_trnid[:, 0]
    qadr, dadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    h = model.opt.timestep
    core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    geom_bodies = sorted({int(b) for b in model.geom_bodyid if model.body(int(b)).name.startswith("robot1_")})
    ordered = [core, *[b for b in geom_bodies if b != core]]
    column = {b: i for i, b in enumerate(ordered)}

    work = np.zeros(model.nu)
    ts, angles, ipos, xpos, cmat, contacts = [], [], [], [], [], []
    step = [0]

    def record(d: mujoco.MjData) -> None:
        work[:] += np.maximum(d.actuator_force * d.qvel[dadr], 0.0) * h
        if step[0] % RECORD_EVERY_N == 0:
            ts.append(d.time)
            angles.append(d.qpos[qadr].copy())
            ipos.append(d.xipos.copy())
            xpos.append(d.xpos[ordered].copy())
            cmat.append(d.xmat[core].copy())
            row = np.zeros(len(ordered), dtype=bool)
            for c in d.contact[:d.ncon]:
                if floor in (c.geom1, c.geom2):
                    other = int(model.geom_bodyid[c.geom2 if c.geom1 == floor else c.geom1])
                    if other in column:
                        row[column[other]] = True
            contacts.append(row)
        step[0] += 1

    displacement = run_episode(model, data, brain, recorder=record)
    dt = RECORD_EVERY_N * h
    angles_a, ipos_a, xpos_a = np.asarray(angles), np.asarray(ipos), np.asarray(xpos)
    rot = np.asarray(cmat).reshape(-1, 3, 3)
    contact = np.asarray(contacts)
    keep = np.asarray(ts) - SETTLE_DURATION >= gait.TRANSIENT_S

    # Limbs: the mass centre of what each hinge moves, in the core frame.
    mass = model.body_mass
    core_pos = xpos_a[:, 0]
    limbs_3d = np.zeros((len(ts), model.nu, 3))
    for a, j in enumerate(joints):
        sub = sorted(subtree(model, int(model.jnt_bodyid[j])))
        m = mass[sub] / mass[sub].sum()
        com = (ipos_a[:, sub] * m[None, :, None]).sum(1) - core_pos
        limbs_3d[:, a] = np.einsum("tji,tj->ti", rot, com)
    limbs = np.stack([first_component(limbs_3d[keep, a]) for a in range(model.nu)], axis=1) if model.nu else np.zeros((keep.sum(), 0))
    angles_k = angles_a[keep]

    rng_deg = np.degrees(np.percentile(angles_k, 95, axis=0) - np.percentile(angles_k, 5, axis=0))
    active = rng_deg >= ACTIVE_RANGE_DEG
    amp = limbs.std(0)
    period, regularity = (math.nan, 0.0)
    if active.any():
        centred = limbs_3d[:, active] - limbs_3d[keep][:, active].mean(0, keepdims=True)
        period, regularity = gait.estimate_period(np.swapaxes(centred, 0, 1), dt)
    freq = 1.0 / period if np.isfinite(period) and period > 0 else math.nan

    nodes = actuator_nodes(model)
    twin, mirror = hinge_relations(genome, nodes)
    pairs = pair_rows(angles_k, limbs, nodes, twin, mirror, active, amp, freq, dt, nested_pairs(genome, nodes)) if np.isfinite(freq) else []

    up = rot[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(np.abs(up[:, 2]), 0.0, 1.0)))
    yaw = np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))
    total_mass = float(sum(mass[b] for b in robot_bodies(model)))
    total_work = float(work.sum())
    moved = displacement >= MIN_DISPLACEMENT
    weight = np.array([p["weight"] for p in pairs])
    def locked_share(kinds: Optional[set[str]] = None, col: str = "limb_quality", unnested: bool = False) -> float:
        use = [p for p in pairs if (kinds is None or p["kind"] in kinds) and np.isfinite(p[col])
               and not (unnested and p["nested"])]
        den = sum(p["weight"] for p in use)
        return sum(p["weight"] for p in use if p[col] >= gait.LOCK_MIN) / den if den > 0 else math.nan

    row = {
        "score": displacement, "displacement": displacement,
        "n_hinges": int(model.nu), "n_active": int(active.sum()), "n_modules": len(TreeGenome.from_dict(genome).nodes),
        "mass": total_mass, "hinge_range_deg": float(rng_deg.mean()) if len(rng_deg) else math.nan,
        "period_s": period, "regularity": float(regularity),
        **decompose(xpos_a[:, 0, :2], displacement, freq, dt, PROTOCOL.episode_s),
        "lateral_m": float(abs((xpos_a[-1, 0, :2] - xpos_a[0, 0, :2]) @ np.array([-FORWARD_AXIS[1], FORWARD_AXIS[0]]))),
        "participation": participation_ratio(angles_k[:, active]),
        "participation_limb": participation_ratio(limbs[:, active]),
        "limb_locked_all": locked_share(), "limb_locked_unnested": locked_share(unnested=True),
        "n_pairs_unnested": sum(not p["nested"] for p in pairs), "limb_locked_twin": locked_share({"twin"}),
        "limb_locked_mirror": locked_share({"mirror"}), "limb_locked_other": locked_share({"other"}),
        "joint_locked_all": locked_share(col="joint_quality"),
        "limb_quality_mean": float(np.nanmean([p["limb_quality"] for p in pairs])) if pairs else math.nan,
        "limb_quality_weighted": (float(np.nansum(weight * np.array([p["limb_quality"] for p in pairs])) / weight.sum())
                                  if pairs and weight.sum() > 0 else math.nan),
        "n_pairs_twin": sum(p["kind"] == "twin" for p in pairs), "n_pairs_mirror": sum(p["kind"] == "mirror" for p in pairs),
        "n_twin_hinges": len(twin), "n_mirror_hinges": len(mirror),
        **contact_stats(contact[keep], xpos_a[keep], dt),
        "tilt_mean_deg": float(tilt[keep].mean()), "tilt_rms_deg": float(tilt[keep].std()),
        "core_z_mean": float(core_pos[keep, 2].mean()), "core_z_rms": float(core_pos[keep, 2].std()),
        "yaw_net_deg": float(np.degrees(yaw[-1] - yaw[0])), "yaw_total_deg": float(np.degrees(np.abs(np.diff(yaw)).sum())),
        "work_j": total_work, "work_per_m": total_work / displacement if moved else math.nan,
        "cost_of_transport": total_work / (total_mass * 9.81 * displacement) if moved else math.nan,
        "work_hinges": inverse_participation(work),
    }
    mujoco.mj_forward(model, data)
    hinges = [{"hinge": a, "node": nodes[a], "twin": twin.get(nodes[a], -1), "mirror": mirror.get(nodes[a], -1),
               "active": bool(active[a]), "range_deg": float(rng_deg[a]), "limb_amp_m": float(amp[a]), "work_j": float(work[a]),
               "axis": AXIS_CLASS[int(np.argmax(np.abs(data.xaxis[joints[a]])))],
               "distal_mass": float(sum(mass[b] for b in subtree(model, int(model.jnt_bodyid[joints[a]]))) / total_mass)}
              for a in range(model.nu)]
    return row, hinges, pairs, displacement


class FrozenHinges:
    """A brain whose chosen hinges are commanded to fixed targets (default 0, the rest pose)."""

    def __init__(self, brain: Any, frozen: np.ndarray, target: float = 0.0) -> None:
        self.brain, self.frozen, self.target = brain, frozen, target

    def reset(self) -> None:
        self.brain.reset()

    def act(self, data: Any, state: np.ndarray) -> np.ndarray:
        out = np.array(self.brain.act(data, state), dtype=np.float64)
        out[self.frozen] = self.target
        return out


def knockout_episode(genome: dict, weights: np.ndarray, brain_kind: str, frozen: Optional[list[int]],
                     target: float = 0.0) -> float:
    """Displacement of one episode with these hinges held at target (None: unchanged brain)."""
    model, data = build_world(genome)
    brain = make_brain_for_model(brain_kind, model, data, np.asarray(weights, dtype=np.float32))
    if frozen is not None:
        brain = FrozenHinges(brain, np.asarray(frozen, dtype=int), target)
    return run_episode(model, data, brain)


def mean_angles(genome: dict, weights: np.ndarray, brain_kind: str) -> np.ndarray:
    """Each hinge's mean angle over the unperturbed episode, after the transient."""
    model, data = build_world(genome)
    brain = make_brain_for_model(brain_kind, model, data, np.asarray(weights, dtype=np.float32))
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    total, count = np.zeros(model.nu), [0]

    def record(d: mujoco.MjData) -> None:
        if d.time - SETTLE_DURATION >= gait.TRANSIENT_S:
            total[:] += d.qpos[qadr]
            count[0] += 1

    run_episode(model, data, brain, recorder=record)
    return total / max(count[0], 1)


# ── Jobs ──────────────────────────────────────────────────────────────────────


def _measure_job(args: tuple[str, str, str]) -> str:
    run_s, out_s, which = args
    run_dir, out = Path(run_s), Path(out_s)
    if (out / "mechanics.csv").exists():
        return "skip"
    try:
        ident = runs.identity(run_dir)
        rows, hinges, pairs = [], [], []
        for ep in runs.episodes(run_dir, which):
            row, h, p, _ = measure_episode(ep.genome, ep.weights, ep.brain)
            rows.append({**ident, "key": ep.key, "stored_score": ep.score, **row})
            hinges += [{"run": ident["run"], "key": ep.key, **r} for r in h]
            pairs += [{"run": ident["run"], "key": ep.key, **r} for r in p]
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(hinges).to_csv(out / "hinges.csv", index=False)
        pd.DataFrame(pairs).to_csv(out / "pairs.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "mechanics.csv", index=False)   # last: marks the run done
        bad = sum(abs(r["score"] - r["stored_score"]) > SCORE_TOL for r in rows)
        return "NOT REPRODUCED" if bad else "ok"
    except Exception as exc:  # noqa: BLE001 -- one bad body must not stop the sweep
        return f"ERROR {runs.run_name(run_dir)}: {type(exc).__name__}: {exc}"


def _knockout_job(args: tuple[str, str, str]) -> str:
    run_s, out_s, which = args
    run_dir, out = Path(run_s), Path(out_s)
    if (out / "knockout.csv").exists():
        return "skip"
    try:
        ident = runs.identity(run_dir)
        rows = []
        for ep in runs.episodes(run_dir, which):
            g, w, b = ep.genome, ep.weights, ep.brain
            model, _ = build_world(g)
            n, nodes = model.nu, actuator_nodes(model)
            base = knockout_episode(g, w, b, None)
            frozen_all = knockout_episode(g, w, b, list(range(n)))
            held = mean_angles(g, w, b)
            for a in range(n):
                rows.append({**ident, "key": ep.key, "hinge": a, "node": nodes[a], "baseline": base,
                             "stored_score": ep.score, "all_frozen": frozen_all,
                             "score": knockout_episode(g, w, b, [a]),
                             "score_mean": knockout_episode(g, w, b, [a], float(held[a])),
                             "held_angle": float(held[a])})
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "knockout.csv", index=False)
        bad = any(abs(r["baseline"] - r["stored_score"]) > SCORE_TOL for r in rows)
        return "NOT REPRODUCED" if bad else "ok"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {runs.run_name(run_dir)}: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("measure", "knockout"), required=True)
    parser.add_argument("--run-dir", type=Path, nargs="*", default=[])
    parser.add_argument("--roots", type=Path, nargs="*", default=[], help="every E1 / E2 run under these")
    parser.add_argument("--match", nargs="*", default=["*"], help="keep run names matching any of these globs")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--which", default="champion", help="champion, final (E1 population / every E2 seed) or g:<gens>")
    parser.add_argument("--limit", type=int, default=0, help="test: only this many runs")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = list(args.run_dir) + runs.find(args.roots, "e1") + runs.find(args.roots, "e2")
    run_dirs = [r for r in run_dirs if any(fnmatch.fnmatch(runs.run_name(r), m) for m in args.match)]
    if args.limit:
        run_dirs = run_dirs[:args.limit]
    jobs = [(str(r), str(args.out_dir / runs.run_name(r)), args.which) for r in run_dirs]
    print(f"{len(jobs)} runs", flush=True)
    job = _measure_job if args.mode == "measure" else _knockout_job
    counts: dict[str, int] = {}
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, jobs), 1):
            key = res.split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
            if res.startswith(("ERROR", "NOT")):
                print(res, flush=True)
            if i % 200 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {counts}", flush=True)


if __name__ == "__main__":
    main()
