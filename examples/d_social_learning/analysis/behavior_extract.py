"""Replay stored (morphology, theta) pairs and extract per-timestep behaviour.

For individuals in a single ``database.db`` (one experiment: one scheme /
x-value / rep), rebuilds the MuJoCo model, re-runs the exact same rollout
loop as ``ariel/evaluator.py::run_episode``, and records:

  - joint angles/velocities per module, every ``record_every_n``-th sim step
  - module world poses (position + quaternion) and floor-contact state,
    every ``record_every_n``-th sim step
  - ANN inputs/outputs per actuator, every control step

into a single compressed ``behavior.npz`` per experiment (one call to
``numpy.savez_compressed``), aggregating all individuals in that experiment
into flat arrays keyed by ``{table}__{field}`` and disambiguated by an
``individual_id`` column, plus an ``episode_meta`` table (also inside the
same file) recording the recomputed fitness alongside the originally stored
fitness. Trajectory fields (angles, velocities, positions, quaternions, ANN
I/O) are stored as float16 -- all observed value ranges are within roughly
[-3.3, 3.3], well inside float16's range, giving ~3-4 significant decimal
digits of precision (relative eps ~1e-3). Scalar summary fields in
``episode_meta`` (fitness, wall time, etc.) stay float32.

At dt=0.002s, a 33s episode is ~16,500 sim steps. ``--record-every-n``
(default 100, i.e. 5Hz) subsamples the joint_angles/module_poses tables;
ann_steps is unaffected since it's already throttled to control steps
(every 100 sim steps). Measured on real data (620-individual experiment,
5Hz + float16 + npz compression): ~24KB/individual, ~15MB/experiment
(vs. ~8.4MB/individual, ~5.2GB/experiment at the previous 50Hz + float64 +
SQLite settings) -- roughly a 350x reduction.

Scope: single ``--db-path`` (one experiment) per invocation, producing one
``behavior.npz`` in ``--out-dir``. See ``--time-only`` for a timing-only
dry run that writes nothing.

Usage:
    uv run examples/d_social_learning/analysis/behavior_extract.py \\
        --db-path __data__/snellius_test_data/social/ariel/lamarckian/x10/rep_0/database.db \\
        --top-n 3 --workers 4

    # Just measure timing, don't write anything:
    uv run examples/d_social_learning/analysis/behavior_extract.py \\
        --db-path .../database.db --time-only --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Prevent d_social_learning/ from shadowing the installed ariel package
_SOCIAL_DIR = Path(__file__).parent.parent
for _p in [str(_SOCIAL_DIR), ""]:
    if _p in sys.path:
        sys.path.remove(_p)

import numpy as np
from rich.console import Console

console = Console()

DURATION = 30.0
SETTLE_TIME = 3.0
SPAWN_POS = (-0.8, 0.0, 0.1)
CTRL_EVERY = 100
CTRL_ALPHA = 0.5
N_NEIGHBORS = 6
FEATURES_PER_NODE = 8
HINGE_CONTACT_LIMIT = 200
HINGE_CONTACT_PENALTY = 0.005
JERK_PENALTY_WEIGHT = 0.01
HEIGHT_PENALTY_THRESHOLD = 0.21


def _infer_hidden(theta_len: int) -> int:
    """Solve n_params = hidden * (input_size + 2) + 1 for hidden."""
    input_size = (1 + N_NEIGHBORS) * FEATURES_PER_NODE + 1
    hidden, remainder = divmod(theta_len - 1, input_size + 2)
    if remainder != 0:
        raise ValueError(f"theta length {theta_len} is not a valid DistributedMLP size")
    return hidden


def _scale_actions(raw: np.ndarray) -> np.ndarray:
    return np.asarray(raw) * (math.pi / 2)


@dataclass
class ReplayResult:
    individual_id: int
    hidden: int
    theta_len: int
    replayed_fitness: float
    mean_jerk: float
    c_hinge: int
    wall_time_s: float
    joint_rows: list[tuple] = field(default_factory=list)
    pose_rows: list[tuple] = field(default_factory=list)
    ann_rows: list[tuple] = field(default_factory=list)


def _build_model(morph_dict: dict):
    import mujoco
    from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
    from ariel.ec.genotypes.tree.tree_genome import TreeGenome
    from ariel.simulation.controllers.morphology_adapter import MorphologyAdapter
    from ariel.simulation.environments import SimpleFlatWorld

    genome = TreeGenome.from_dict(morph_dict)
    graph = genome.to_networkx()
    core = construct_mjspec_from_graph(graph)
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
    model = world.spec.compile()
    adapter = MorphologyAdapter.from_graph(graph)
    module_to_body_id = adapter.resolve_module_body_ids(model)

    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    hinge_geom_ids: set[int] = set()
    body_geom_ids: dict[int, set[int]] = {}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and name.endswith("-rotor"):
            hinge_geom_ids.add(i)
        gbody = model.geom(i).bodyid[0]
        body_geom_ids.setdefault(gbody, set()).add(i)

    return model, adapter, module_to_body_id, floor_geom_id, hinge_geom_ids, body_geom_ids


def replay_individual(
    individual_id: int,
    morph_dict: dict,
    theta: np.ndarray,
    seed: int | None = None,
    record_every_n: int = 1,
) -> ReplayResult:
    """Re-run one episode, recording behavioural traces at every step.

    Mirrors ``ariel/evaluator.py::run_episode``'s settle/rollout structure
    and fitness formula, but additionally records joint angles, module
    poses/contacts (every ``record_every_n``-th sim step) and ANN
    inputs/outputs (every control step). MuJoCo stepping and
    ``DistributedMLP.forward_all`` are both deterministic given ``theta``,
    so ``seed`` is a no-op safety net (kept for parity with
    ``examples/c_genotypes/simulate_from_db.py``'s replay convention)
    rather than something the current sim path consumes.
    """
    import mujoco
    from ariel.simulation.controllers.distributed_mlp import DistributedMLP

    if seed is not None:
        np.random.seed(seed)

    t0 = time.perf_counter()

    model, adapter, module_to_body_id, floor_geom_id, hinge_geom_ids, body_geom_ids = _build_model(morph_dict)
    data = mujoco.MjData(model)

    hidden = _infer_hidden(len(theta))
    brain = DistributedMLP(n_neighbors=N_NEIGHBORS, hidden=hidden)
    brain.set_theta(np.asarray(theta, dtype=np.float64))
    mujoco.mj_resetData(model, data)

    core_height = float(data.qpos[2])

    result = ReplayResult(
        individual_id=individual_id,
        hidden=hidden,
        theta_len=len(theta),
        replayed_fitness=0.0,
        mean_jerk=0.0,
        c_hinge=0,
        wall_time_s=0.0,
    )

    def record_step(sim_step: int) -> None:
        if sim_step % record_every_n != 0:
            return
        joint_pos, joint_vel = adapter._read_joint_states(model, data)
        t = data.time
        for mod_idx in adapter.module_to_body_name:
            if mod_idx in joint_pos:
                result.joint_rows.append(
                    (individual_id, sim_step, t, mod_idx, joint_pos[mod_idx], joint_vel[mod_idx])
                )

        contacting_bodies: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            if c.geom1 == floor_geom_id:
                contacting_bodies.add(model.geom(c.geom2).bodyid[0])
            elif c.geom2 == floor_geom_id:
                contacting_bodies.add(model.geom(c.geom1).bodyid[0])

        for mod_idx, body_id in module_to_body_id.items():
            pos = data.xpos[body_id]
            quat = data.xquat[body_id]
            in_contact = body_id in contacting_bodies
            result.pose_rows.append((
                individual_id, sim_step, t, mod_idx,
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
                int(in_contact),
            ))

    # Settle phase: let robot fall into place, no control, no counting
    sim_step = 0
    while data.time < SETTLE_TIME:
        mujoco.mj_step(model, data)
        record_step(sim_step)
        sim_step += 1

    # Rollout phase
    c_hinge = 0
    ctrl_step = 0
    active_hinge_contacts: set[frozenset[int]] = set()
    prev_ctrl = np.zeros(model.nu, dtype=np.float32)
    jerk_sum = 0.0
    rollout_end = SETTLE_TIME + DURATION
    while data.time < rollout_end:
        if sim_step % CTRL_EVERY == 0:
            node_inputs, t_signal = adapter.get_node_inputs(model, data, ctrl_step)
            raw = brain.forward_all(node_inputs, t_signal)
            target_ctrl = _scale_actions(raw)
            new_ctrl = np.clip(
                prev_ctrl * (1.0 - CTRL_ALPHA) + target_ctrl * CTRL_ALPHA,
                -np.pi / 2, np.pi / 2,
            ).astype(np.float32)
            if ctrl_step > 0 and model.nu > 0:
                jerk_sum += float(np.mean(np.abs(new_ctrl - prev_ctrl)))

            for actuator_idx, mod_idx in enumerate(adapter.actuator_to_module):
                self_obs, neighbor_obs = node_inputs[actuator_idx]
                input_vec = np.concatenate(
                    [self_obs.to_vector()] + [n.to_vector() for n in neighbor_obs] + [[t_signal]]
                ).tolist()
                result.ann_rows.append((
                    individual_id, ctrl_step, sim_step, data.time, actuator_idx, mod_idx,
                    json.dumps(input_vec),
                    float(raw[actuator_idx]),
                    float(target_ctrl[actuator_idx]),
                    float(new_ctrl[actuator_idx]),
                ))

            prev_ctrl = new_ctrl.copy()
            data.ctrl[:] = new_ctrl
            ctrl_step += 1

        mujoco.mj_step(model, data)
        record_step(sim_step)

        current: set[frozenset[int]] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            if ((c.geom1 == floor_geom_id and c.geom2 in hinge_geom_ids) or
                    (c.geom2 == floor_geom_id and c.geom1 in hinge_geom_ids)):
                current.add(frozenset((c.geom1, c.geom2)))
        c_hinge += len(current - active_hinge_contacts)
        active_hinge_contacts = current
        sim_step += 1

    mean_jerk = jerk_sum / max(ctrl_step - 1, 1)
    d = float(data.qpos[0])
    if c_hinge > HINGE_CONTACT_LIMIT or not np.isfinite(d):
        fitness = -1.0
    else:
        height_penalty = core_height if core_height > HEIGHT_PENALTY_THRESHOLD else 0.0
        fitness = (
            d - height_penalty
            - HINGE_CONTACT_PENALTY * c_hinge
            - JERK_PENALTY_WEIGHT * mean_jerk
        )

    result.replayed_fitness = fitness
    result.mean_jerk = mean_jerk
    result.c_hinge = c_hinge
    result.wall_time_s = time.perf_counter() - t0
    return result


# --------------------------------------------------------------------------
# DB selection / loading
# --------------------------------------------------------------------------

def load_individuals(db_path: Path, individual_ids=None, top_n=None, select_all=False):
    """Return list of (id, morph_dict, theta, stored_fitness) from a database.db."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    if individual_ids:
        placeholders = ",".join("?" * len(individual_ids))
        cur.execute(
            f"SELECT id, genotype_, tags_, fitness_ FROM individual "
            f"WHERE id IN ({placeholders})",
            individual_ids,
        )
    elif select_all:
        cur.execute(
            "SELECT id, genotype_, tags_, fitness_ FROM individual "
            "WHERE fitness_ IS NOT NULL"
        )
    else:
        n = top_n or 3
        cur.execute(
            "SELECT id, genotype_, tags_, fitness_ FROM individual "
            "WHERE fitness_ IS NOT NULL ORDER BY fitness_ DESC LIMIT ?",
            (n,),
        )
    rows = cur.fetchall()
    con.close()

    result = []
    for ind_id, geno_json, tags_json, fitness in rows:
        geno = json.loads(geno_json) if geno_json else {}
        tags = json.loads(tags_json) if tags_json else {}
        morph = geno.get("morph")
        theta = tags.get("theta")
        if morph and theta:
            result.append((ind_id, morph, np.array(theta, dtype=np.float64), fitness))
    return result


# --------------------------------------------------------------------------
# npz output
# --------------------------------------------------------------------------
#
# One compressed archive per experiment (``behavior.npz``), aggregating all
# individuals' rows into flat typed arrays. Keys are named
# ``{table}__{field}`` so a loader can group them back by table; every table
# carries an ``individual_id`` column to disambiguate rows (replacing the
# ``source_db`` string column from the old per-row SQLite schema, since
# there's now exactly one source db per output file).

def write_results(
    out_dir: Path,
    source_db: str,
    results: list[ReplayResult],
    morph_by_id: dict[int, dict],
    stored_fitness_by_id: dict[int, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_rows = [row for r in results for row in r.joint_rows]
    pose_rows = [row for r in results for row in r.pose_rows]
    ann_rows = [row for r in results for row in r.ann_rows]

    arrays: dict[str, np.ndarray] = {}

    if joint_rows:
        # row: (individual_id, sim_step, time, module_idx, angle, velocity)
        j = np.array(joint_rows, dtype=np.float64)
        arrays["joint_angles__individual_id"] = j[:, 0].astype(np.int32)
        arrays["joint_angles__sim_step"] = j[:, 1].astype(np.int32)
        arrays["joint_angles__time"] = j[:, 2].astype(np.float32)
        arrays["joint_angles__module_idx"] = j[:, 3].astype(np.int16)
        arrays["joint_angles__angle"] = j[:, 4].astype(np.float16)
        arrays["joint_angles__velocity"] = j[:, 5].astype(np.float16)

    if pose_rows:
        # row: (individual_id, sim_step, time, module_idx, pos_x, pos_y,
        #       pos_z, quat_w, quat_x, quat_y, quat_z, in_contact)
        p = np.array(pose_rows, dtype=np.float64)
        arrays["module_poses__individual_id"] = p[:, 0].astype(np.int32)
        arrays["module_poses__sim_step"] = p[:, 1].astype(np.int32)
        arrays["module_poses__time"] = p[:, 2].astype(np.float32)
        arrays["module_poses__module_idx"] = p[:, 3].astype(np.int16)
        arrays["module_poses__pos"] = p[:, 4:7].astype(np.float16)
        arrays["module_poses__quat"] = p[:, 7:11].astype(np.float16)
        arrays["module_poses__in_contact"] = p[:, 11].astype(np.int8)

    if ann_rows:
        # row: (individual_id, ctrl_step, sim_step, time, actuator_idx,
        #       module_idx, input_json, output_raw, output_scaled, output_applied)
        arrays["ann_steps__individual_id"] = np.array([row[0] for row in ann_rows], dtype=np.int32)
        arrays["ann_steps__ctrl_step"] = np.array([row[1] for row in ann_rows], dtype=np.int32)
        arrays["ann_steps__sim_step"] = np.array([row[2] for row in ann_rows], dtype=np.int32)
        arrays["ann_steps__time"] = np.array([row[3] for row in ann_rows], dtype=np.float32)
        arrays["ann_steps__actuator_idx"] = np.array([row[4] for row in ann_rows], dtype=np.int16)
        arrays["ann_steps__module_idx"] = np.array([row[5] for row in ann_rows], dtype=np.int16)
        arrays["ann_steps__input_vec"] = np.array([json.loads(row[6]) for row in ann_rows], dtype=np.float16)
        arrays["ann_steps__output_raw"] = np.array([row[7] for row in ann_rows], dtype=np.float16)
        arrays["ann_steps__output_scaled"] = np.array([row[8] for row in ann_rows], dtype=np.float16)
        arrays["ann_steps__output_applied"] = np.array([row[9] for row in ann_rows], dtype=np.float16)

    arrays["episode_meta__individual_id"] = np.array([r.individual_id for r in results], dtype=np.int32)
    arrays["episode_meta__morph_json"] = np.array(
        [json.dumps(morph_by_id[r.individual_id]) for r in results]
    )
    arrays["episode_meta__theta_len"] = np.array([r.theta_len for r in results], dtype=np.int32)
    arrays["episode_meta__hidden"] = np.array([r.hidden for r in results], dtype=np.int16)
    arrays["episode_meta__stored_fitness"] = np.array(
        [stored_fitness_by_id[r.individual_id] for r in results], dtype=np.float32
    )
    arrays["episode_meta__replayed_fitness"] = np.array([r.replayed_fitness for r in results], dtype=np.float32)
    arrays["episode_meta__mean_jerk"] = np.array([r.mean_jerk for r in results], dtype=np.float32)
    arrays["episode_meta__c_hinge"] = np.array([r.c_hinge for r in results], dtype=np.int32)
    arrays["episode_meta__wall_time_s"] = np.array([r.wall_time_s for r in results], dtype=np.float32)
    arrays["episode_meta__source_db"] = np.array([source_db])

    np.savez_compressed(out_dir / "behavior.npz", **arrays)


# --------------------------------------------------------------------------
# Worker (picklable, top-level) for multiprocessing
# --------------------------------------------------------------------------

def _replay_worker(args: tuple) -> ReplayResult:
    individual_id, morph_dict, theta, seed, record_every_n = args
    return replay_individual(individual_id, morph_dict, theta, seed=seed, record_every_n=record_every_n)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def run_timing(individuals, workers: int, n_experiment: int, record_every_n: int = 1) -> None:
    if not individuals:
        console.print("[red]No individuals to time.[/red]")
        return

    ind_id, morph, theta, stored_fitness = individuals[0]
    console.rule("[bold cyan]Single-individual timing")
    r = replay_individual(ind_id, morph, theta, record_every_n=record_every_n)
    console.print(f"individual {ind_id}: wall_time={r.wall_time_s:.2f}s")
    console.print(f"  joint_rows={len(r.joint_rows)} pose_rows={len(r.pose_rows)} ann_rows={len(r.ann_rows)}")
    console.print(f"  stored_fitness={stored_fitness:.6f} replayed_fitness={r.replayed_fitness:.6f} "
                  f"delta={abs(stored_fitness - r.replayed_fitness):.2e}")

    serial_experiment_s = r.wall_time_s * n_experiment
    console.print(f"\nSerial extrapolation for this db (~{n_experiment} individuals): "
                  f"{serial_experiment_s:.0f}s ({serial_experiment_s / 60:.1f} min)")

    n_trial = min(len(individuals), max(workers * 2, 8))
    trial = individuals[:n_trial]
    console.rule(f"[bold cyan]Multiprocessing trial: {len(trial)} individuals, {workers} workers")
    worker_args = [(iid, morph, theta, None, record_every_n) for iid, morph, theta, _ in trial]
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        results = pool.map(_replay_worker, worker_args)
    elapsed = time.perf_counter() - t0
    per_individual = elapsed / len(trial)
    console.print(f"total={elapsed:.2f}s  per_individual_amortized={per_individual:.2f}s  "
                  f"speedup={r.wall_time_s / per_individual:.2f}x")

    parallel_experiment_s = per_individual * n_experiment
    console.print(f"\nExtrapolated for this db (~{n_experiment} individuals) at {workers} workers: "
                  f"{parallel_experiment_s:.0f}s ({parallel_experiment_s / 60:.1f} min)")

    n_all = 80 * n_experiment
    console.print(
        f"\n[dim]Reference only — NOT executed by this script — if scaled to all "
        f"80 experiments (~{n_all} individuals):[/dim]"
    )
    console.print(f"[dim]  serial:   {serial_experiment_s * 80 / 3600:.1f} h[/dim]")
    console.print(f"[dim]  {workers} workers: {parallel_experiment_s * 80 / 3600:.1f} h[/dim]")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", required=True, help="Path to a single database.db")
    parser.add_argument("--individual-id", type=int, action="append", default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-dir", default="__data__/social/behavior")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--time-only", action="store_true")
    parser.add_argument(
        "--record-every-n", type=int, default=100,
        help="Record joint_angles/module_poses rows every Nth sim step "
             "(default 100, i.e. 5Hz at dt=0.002s). Use 1 for full resolution. "
             "ANN I/O is unaffected (already throttled to control steps).",
    )
    args = parser.parse_args()
    if args.record_every_n < 1:
        parser.error("--record-every-n must be >= 1")

    db_path = Path(args.db_path)
    if not db_path.exists():
        console.print(f"[red]No such database: {db_path}[/red]")
        return

    con = sqlite3.connect(db_path)
    n_experiment = con.execute(
        "SELECT COUNT(*) FROM individual WHERE fitness_ IS NOT NULL"
    ).fetchone()[0]
    con.close()

    if args.time_only:
        n_trial = max(args.workers * 2, 8)
        individuals = load_individuals(db_path, top_n=n_trial)
        run_timing(individuals, workers=args.workers, n_experiment=n_experiment, record_every_n=args.record_every_n)
        return

    individuals = load_individuals(
        db_path,
        individual_ids=args.individual_id,
        top_n=args.top_n if not args.all else None,
        select_all=args.all,
    )
    if not individuals:
        console.print(f"[yellow]No matching evaluated individuals in {db_path}[/yellow]")
        return

    console.rule(f"[bold cyan]Extracting {len(individuals)} individuals from {db_path}")
    worker_args = [(iid, morph, theta, args.seed, args.record_every_n) for iid, morph, theta, _ in individuals]

    t0 = time.perf_counter()
    if args.workers > 1 and len(individuals) > 1:
        with Pool(processes=args.workers) as pool:
            results = pool.map(_replay_worker, worker_args)
    else:
        results = [_replay_worker(a) for a in worker_args]
    elapsed = time.perf_counter() - t0

    morph_by_id = {iid: morph for iid, morph, _, _ in individuals}
    stored_fitness_by_id = {iid: fit for iid, _, _, fit in individuals}

    out_dir = Path(args.out_dir)
    write_results(out_dir, str(db_path), results, morph_by_id, stored_fitness_by_id)

    console.print(f"Extracted {len(results)} individuals in {elapsed:.1f}s -> {out_dir}")
    deltas = [abs(stored_fitness_by_id[r.individual_id] - r.replayed_fitness) for r in results]
    console.print(f"Fitness reproduction: mean|delta|={np.mean(deltas):.2e} max|delta|={np.max(deltas):.2e}")


if __name__ == "__main__":
    main()
