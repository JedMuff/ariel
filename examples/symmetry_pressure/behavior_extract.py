"""
Replay gecko_skill_tasks.py checkpoints and extract per-timestep behaviour
plus per-episode behavioural descriptors, for one locomotion sweep run
(sympress_{forward,multidirection,turn_avg}_* or ctrlstride_forward_*).

Ported from the social-learning branch's
examples/d_social_learning/analysis/behavior_extract.py. Controller weights
are not in the run's database here, so episodes come from the per-body
checkpoints (<run>/__data__/gecko_skill_tasks/<task>/checkpoints/genGGG_bodyII),
one episode per (checkpoint, trained skill). See behavior_common.py for what
is recorded and the descriptor definitions.

Writes one compressed ``behavior.npz`` per run into --out-dir, with flat
arrays keyed ``{table}__{field}``; every table carries an ``episode_idx``
column indexing the ``episode_meta`` table:

  joint_states  every --record-every-n sim steps: hinge angle / velocity
  body_poses    every --record-every-n sim steps: world pos / quat / floor contact per robot body
  ctrl_steps    ~every --record-every-n sim steps: controller state (ragged, flattened with
                offsets), raw and applied action per actuator
  episode_meta  gen, body_idx, ind_id, skill, stored vs replayed fitness, name tables, ...
  descriptors   one row per episode (see behavior_common.DESCRIPTOR_NAMES)

Trajectory fields are float16, as in the social branch. descriptors.csv
(episode_meta scalars + descriptors) is written alongside for quick pandas use.
Pass --no-traces to skip the trace tables and keep only meta + descriptors.

Usage:
    uv run examples/symmetry_pressure/behavior_extract.py \\
        --run-dir __data__/ariel_symmetry_pressure_sweep/sympress_turn_avg_tree_symmetric_rep0_37568_6 \\
        --gens last --workers 4

    # Timing estimate only, writes nothing:
    uv run examples/symmetry_pressure/behavior_extract.py --run-dir ... --time-only --workers 16

    # Sorted locomotion run dirs under sweep roots (indexed by the SLURM array):
    uv run examples/symmetry_pressure/behavior_extract.py --list-runs \\
        /scratch/jed/ariel_symmetry_pressure_sweep /scratch/jed/ariel_control_stride_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rich.console import Console

from behavior_common import (
    DESCRIPTOR_NAMES,
    EpisodeTrace,
    drop_full_rate,
    replay_skill_episode,
    resolve_replay_config,
    skills_for_task,
    trace_descriptors,
)
from sweep_common import checkpoint_generations, checkpoints_ranked_by_fitness, run_data_path

console = Console()

LOCOMOTION_TASKS = ("forward", "multidirection", "turn_avg")
_LOCO_RUN_RE = re.compile(r"^(sympress_(forward|multidirection|turn_avg)_|ctrlstride_)")
_CKPT_RE = re.compile(r"^gen(\d+)_body(\d+)$")


# ── Run discovery / episode selection ─────────────────────────────────────────


def list_locomotion_runs(sweep_roots: list[Path]) -> list[Path]:
    runs = []
    for root in sweep_roots:
        runs.extend(p for p in root.iterdir() if p.is_dir() and _LOCO_RUN_RE.match(p.name))
    return sorted(runs, key=lambda p: (str(p.parent), p.name))


def run_task(run_dir: Path) -> str:
    """The run's task, from its single gecko_skill_tasks/<task>/run_config.json."""
    base = run_dir / "__data__" / "gecko_skill_tasks"
    tasks = [p.name for p in base.iterdir() if (p / "run_config.json").exists()] if base.exists() else []
    if len(tasks) != 1 or tasks[0] not in LOCOMOTION_TASKS:
        raise ValueError(f"{run_dir}: expected one locomotion task under {base}, found {tasks}")
    return tasks[0]


def select_checkpoints(run_dir: Path, task: str, gens: str, top_n: int | None) -> list[Path]:
    if top_n is not None:
        return checkpoints_ranked_by_fitness(run_dir, task)[:top_n]
    by_gen = checkpoint_generations(run_dir, task)
    if not by_gen:
        return []
    if gens == "all":
        chosen = sorted(by_gen)
    elif gens == "last":
        chosen = [max(by_gen)]
    elif gens.startswith("every:"):
        n = int(gens.split(":", 1)[1])
        chosen = [g for g in sorted(by_gen) if g % n == 0]
        if max(by_gen) not in chosen:
            chosen.append(max(by_gen))
    else:
        chosen = [int(g) for g in gens.split(",") if int(g) in by_gen]
    return sorted(c for g in chosen for c in by_gen[g])


def load_stored_records(run_dir: Path, task: str) -> dict[tuple[int, int], dict]:
    """(gen, body_idx) -> run_data.jsonl record. gecko_skill_tasks.py writes one
    record per evaluated body, in the same order it assigns genGGG_bodyII
    checkpoint tags, so body_idx is the record's position within its gen."""
    path = run_data_path(run_dir, task)
    records: dict[tuple[int, int], dict] = {}
    if not path.exists():
        return records
    per_gen: dict[int, int] = {}
    for line in path.open():
        rec = json.loads(line)
        idx = per_gen.get(rec["gen"], 0)
        per_gen[rec["gen"]] = idx + 1
        records[(rec["gen"], idx)] = rec
    return records


def build_jobs(ckpts: list[Path], task: str, record_every_n: int, max_modules: int | None) -> list[tuple]:
    jobs = []
    for ckpt in ckpts:
        m = _CKPT_RE.match(ckpt.name)
        if not m:
            continue
        gen, body_idx = int(m.group(1)), int(m.group(2))
        for skill, reward_spec in skills_for_task(task):
            if (ckpt / f"{skill}_weights.npy").exists():
                jobs.append((str(ckpt), gen, body_idx, skill, reward_spec, record_every_n, max_modules))
    return jobs


# ── Worker ────────────────────────────────────────────────────────────────────


def _pool_init() -> None:
    import torch
    torch.set_num_threads(1)


def _replay_worker(job: tuple) -> tuple[int, int, str, EpisodeTrace | None, dict | None, str]:
    ckpt_s, gen, body_idx, skill, reward_spec, record_every_n, max_modules = job
    ckpt = Path(ckpt_s)
    try:
        genome = json.loads((ckpt / "best_genome.json").read_text())
        weights = np.load(ckpt / f"{skill}_weights.npy")
        to_spec_fn, stride = resolve_replay_config(ckpt, max_modules=max_modules)
        trace = replay_skill_episode(
            genome, weights, skill, reward_spec,
            to_spec_fn=to_spec_fn, control_step_freq=stride, record_every_n=record_every_n,
        )
        desc = trace_descriptors(trace)
        drop_full_rate(trace)
        return gen, body_idx, skill, trace, desc, ""
    except Exception as exc:  # noqa: BLE001 -- one bad body must not kill the run
        return gen, body_idx, skill, None, None, f"{type(exc).__name__}: {exc}"


def run_jobs(jobs: list[tuple], workers: int) -> list[tuple]:
    if workers > 1 and len(jobs) > 1:
        with Pool(processes=workers, initializer=_pool_init) as pool:
            return pool.map(_replay_worker, jobs, chunksize=1)
    _pool_init()
    return [_replay_worker(j) for j in jobs]


# ── npz output ────────────────────────────────────────────────────────────────


def write_results(
    out_dir: Path,
    run_dir: Path,
    task: str,
    genome_type: str,
    results: list[tuple],
    stored: dict[tuple[int, int], dict],
    save_traces: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}

    meta_rows = []
    js = {k: [] for k in ("episode_idx", "sim_step", "time", "joint_idx", "angle", "velocity")}
    bp = {k: [] for k in ("episode_idx", "sim_step", "time", "body_idx", "pos", "quat", "in_contact")}
    cs = {k: [] for k in ("episode_idx", "ctrl_step", "sim_step", "time", "actuator_idx",
                          "output_raw", "output_applied")}
    state_flat, state_ep, state_offset, state_len = [], [], [], []
    n_state = 0

    for ep, (gen, body_idx, skill, trace, desc, _) in enumerate(results):
        rec = stored.get((gen, body_idx), {})
        stored_fit = rec.get("per_skill", {}).get(skill, {}).get("fitness", float("nan"))
        meta_rows.append({
            "episode_idx": ep, "gen": gen, "body_idx": body_idx,
            "ind_id": rec.get("ind_id", -1), "skill": skill,
            "task": task, "genome_type": genome_type,
            "control_step_freq": trace.control_step_freq,
            "n_joints": len(trace.joint_names), "n_bodies": len(trace.body_names),
            "stored_fitness": stored_fit if stored_fit is not None else float("nan"),
            "replayed_fitness": trace.replayed_fitness,
            "raw_score": trace.raw_score, "mean_jerk": trace.mean_jerk,
            "initial_height": trace.initial_height,
            "yz_symmetry": rec.get("yz_symmetry", float("nan")),
            "wall_time_s": trace.wall_time_s,
            "joint_names_json": json.dumps(trace.joint_names),
            "body_names_json": json.dumps(trace.body_names),
            "actuator_joint_json": json.dumps(trace.actuator_joint),
            **{f"d_{k}": desc[k] for k in DESCRIPTOR_NAMES},
        })
        if not save_traces:
            continue

        J, B = len(trace.joint_names), len(trace.body_names)
        for sim_step, t, angles, vels in trace.joint_rows:
            js["episode_idx"].append(np.full(J, ep)); js["sim_step"].append(np.full(J, sim_step))
            js["time"].append(np.full(J, t)); js["joint_idx"].append(np.arange(J))
            js["angle"].append(angles); js["velocity"].append(vels)
        for sim_step, t, pos, quat, contact in trace.pose_rows:
            bp["episode_idx"].append(np.full(B, ep)); bp["sim_step"].append(np.full(B, sim_step))
            bp["time"].append(np.full(B, t)); bp["body_idx"].append(np.arange(B))
            bp["pos"].append(pos); bp["quat"].append(quat); bp["in_contact"].append(contact)
        for ctrl_step, sim_step, t, state, raw, applied in trace.ctrl_rows:
            nu = len(applied)
            cs["episode_idx"].append(np.full(nu, ep)); cs["ctrl_step"].append(np.full(nu, ctrl_step))
            cs["sim_step"].append(np.full(nu, sim_step)); cs["time"].append(np.full(nu, t))
            cs["actuator_idx"].append(np.arange(nu))
            cs["output_raw"].append(raw); cs["output_applied"].append(applied)
            state_flat.append(state); state_ep.append(ep)
            state_offset.append(n_state); state_len.append(len(state))
            n_state += len(state)

    dtypes = {
        "episode_idx": np.int32, "sim_step": np.int32, "ctrl_step": np.int32, "time": np.float32,
        "joint_idx": np.int16, "body_idx": np.int16, "actuator_idx": np.int16, "in_contact": np.int8,
    }

    def cat(table: str, cols: dict) -> None:
        if not cols["episode_idx"]:
            return
        for k, parts in cols.items():
            arrays[f"{table}__{k}"] = np.concatenate(parts).astype(dtypes.get(k, np.float16))

    if save_traces:
        cat("joint_states", js)
        cat("body_poses", bp)
        cat("ctrl_steps", cs)
        if state_flat:
            # One state vector per (episode, ctrl_step); its length varies with the
            # body (3 core-orientation dims + one per hinge), hence flat + offsets.
            arrays["ctrl_state__episode_idx"] = np.array(state_ep, dtype=np.int32)
            arrays["ctrl_state__offset"] = np.array(state_offset, dtype=np.int64)
            arrays["ctrl_state__length"] = np.array(state_len, dtype=np.int16)
            arrays["ctrl_state__values"] = np.concatenate(state_flat).astype(np.float16)

    str_cols = {"skill", "task", "genome_type", "joint_names_json", "body_names_json", "actuator_joint_json"}
    int_cols = {"episode_idx", "gen", "body_idx", "ind_id", "control_step_freq", "n_joints", "n_bodies"}
    for k in meta_rows[0] if meta_rows else []:
        vals = [r[k] for r in meta_rows]
        if k.startswith("d_"):
            arrays[f"descriptors__{k[2:]}"] = np.array(vals, dtype=np.float32)
        elif k in str_cols:
            arrays[f"episode_meta__{k}"] = np.array(vals)
        elif k in int_cols:
            arrays[f"episode_meta__{k}"] = np.array(vals, dtype=np.int32)
        else:
            arrays[f"episode_meta__{k}"] = np.array(vals, dtype=np.float32)
    arrays["episode_meta__run_dir"] = np.array([str(run_dir)])

    np.savez_compressed(out_dir / "behavior.npz", **arrays)

    csv_cols = [k for k in (meta_rows[0] if meta_rows else {}) if not k.endswith("_json")]
    with (out_dir / "descriptors.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run"] + csv_cols, extrasaction="ignore")
        w.writeheader()
        for r in meta_rows:
            w.writerow({"run": run_dir.name, **r})


# ── Timing ────────────────────────────────────────────────────────────────────


def run_timing(jobs: list[tuple], n_total: int, workers: int) -> None:
    if not jobs:
        console.print("[red]No episodes to time.[/red]")
        return
    console.rule("[bold cyan]Single-episode timing")
    gen, body_idx, skill, trace, _, err = _replay_worker(jobs[0])
    if trace is None:
        console.print(f"[red]gen{gen} body{body_idx} {skill}: {err}[/red]")
        return
    console.print(f"gen{gen:03d}_body{body_idx:02d} {skill}: wall={trace.wall_time_s:.2f}s "
                  f"replayed_fitness={trace.replayed_fitness:.4f}")

    trial = jobs[:max(workers * 2, 8)]
    console.rule(f"[bold cyan]Pool trial: {len(trial)} episodes, {workers} workers")
    t0 = time.perf_counter()
    run_jobs(trial, workers)
    per_ep = (time.perf_counter() - t0) / len(trial)
    est = per_ep * n_total
    console.print(f"per_episode_amortized={per_ep:.2f}s -> {n_total} episodes ≈ {est:.0f}s ({est / 3600:.2f} h)")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, help="One sweep run dir (sympress_* / ctrlstride_*)")
    parser.add_argument("--list-runs", type=Path, nargs="+", metavar="SWEEP_ROOT",
                        help="Print sorted locomotion run dirs under these sweep roots and exit")
    parser.add_argument("--gens", default="all",
                        help="all | last | every:N | comma-separated gens (default: all)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Instead of --gens: the N best checkpoints across all gens")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <sweep_root>/behavior/<run_name>")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--record-every-n", type=int, default=100,
                        help="Save trace rows every Nth sim step (default 100 = 5Hz at dt=0.002s). "
                             "Descriptors always use every step.")
    parser.add_argument("--max-modules", type=int, default=None,
                        help="cppn decoder cap (default: run_config.json's max_modules)")
    parser.add_argument("--no-traces", action="store_true", help="Only save episode_meta + descriptors")
    parser.add_argument("--time-only", action="store_true")
    args = parser.parse_args()

    if args.list_runs:
        for p in list_locomotion_runs(args.list_runs):
            print(p)
        return
    if args.run_dir is None:
        parser.error("--run-dir is required (unless --list-runs)")
    if args.record_every_n < 1:
        parser.error("--record-every-n must be >= 1")

    run_dir = args.run_dir.resolve()
    task = run_task(run_dir)
    run_cfg = json.loads((run_dir / "__data__" / "gecko_skill_tasks" / task / "run_config.json").read_text())
    genome_type = run_cfg.get("genome_type", "unknown")

    ckpts = select_checkpoints(run_dir, task, args.gens, args.top_n)
    jobs = build_jobs(ckpts, task, args.record_every_n, args.max_modules)
    console.rule(f"[bold cyan]{run_dir.name}: task={task} genome={genome_type} "
                 f"stride={run_cfg.get('control_step_freq', 'default')}  "
                 f"{len(ckpts)} checkpoints -> {len(jobs)} episodes")

    if args.time_only:
        run_timing(jobs, len(jobs), args.workers)
        return
    if not jobs:
        console.print("[yellow]No episodes selected.[/yellow]")
        return

    t0 = time.perf_counter()
    raw = run_jobs(jobs, args.workers)
    elapsed = time.perf_counter() - t0

    results = [r for r in raw if r[3] is not None]
    for gen, body_idx, skill, _, _, err in raw:
        if err:
            console.print(f"[red]gen{gen:03d}_body{body_idx:02d} {skill}: {err}[/red]")

    stored = load_stored_records(run_dir, task)
    out_dir = args.out_dir or run_dir.parent / "behavior" / run_dir.name
    write_results(out_dir, run_dir, task, genome_type, results, stored, save_traces=not args.no_traces)

    console.print(f"Extracted {len(results)}/{len(jobs)} episodes in {elapsed:.1f}s -> {out_dir}")
    deltas = []
    for gen, body_idx, skill, trace, _, _ in results:
        fit = stored.get((gen, body_idx), {}).get("per_skill", {}).get(skill, {}).get("fitness")
        if fit is not None and math.isfinite(fit):
            deltas.append(abs(fit - trace.replayed_fitness))
    if deltas:
        console.print(f"Fitness reproduction over {len(deltas)} episodes: "
                      f"mean|delta|={np.mean(deltas):.2e} max|delta|={np.max(deltas):.2e}")
    else:
        console.print("[yellow]No stored per-skill fitness to compare against.[/yellow]")


if __name__ == "__main__":
    main()
