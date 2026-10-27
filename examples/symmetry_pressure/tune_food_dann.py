"""
Tune gecko_food_dann.py's reward and learning settings for training speed.

Each Optuna trial picks a training setup (fitness weights, number of foods on
the training circle, their distance and the ring's rotation, subtask time,
DANN hidden size, optimiser, population and step size) and trains it from scratch on the fixed
gecko with --seeds-per-trial seeds side by side. A run stops as soon as its best
candidate solves the probe (gecko_food_dann.py's fixed test: 8 foods at 2 m
between the eight compass directions, 30 s each, so "collects every food"
means the same whatever the training setup), when it exceeds the compute
budget, or when it looks hopeless (no training food collected by
--hopeless-gen, or --stall-gens without a new best).

Objective (minimised, mean over the trial's seeds): training compute until the
probe was solved, in simulated seconds (every training episode's settle and
run time; the probe itself is not counted). A run that never solves it scores
budget x (2 - probe_best / 8): always worse than any solved run, and better the
more probe foods it reached. Simulated seconds, not generations, because the
setups differ in candidates and subtasks per generation; wall time is
recorded too (user attribute elapsed_s).

The study lives in a journal file (<study-dir>/study.log), so it can be
resumed and several processes (or Slurm array tasks) can share it. Each run
writes to <study-dir>/trial_XXXX/seed_S/__data__/gecko_food_dann/.
--enqueue-baseline queues the current defaults (the 300-generation setup in
FOOD_DANN.md) as the first trial, as a reference.

ring_offset_frac rotates the training ring: the first food sits
ring_offset_frac x (360 / subtasks) deg from straight ahead (positive = to the
gecko's left), the rest clockwise. 0 puts a food straight ahead; +-0.5 puts
straight ahead halfway between two foods (-0.5: the first food is the one to
the right, 0.5: to the left). The first food matters because progress credit
goes to the first missed food in clockwise order.

--search a,b,... searches only those parameters and fixes the rest at the
current defaults (BASELINE); with --grid it runs every combination of their
GRID values (Optuna GridSampler) instead of TPE, for a clean comparison, e.g.
--search subtasks,ring_offset_frac --grid. --report groups the trials by
the searched parameters (or --by).

Usage:
  python tune_food_dann.py --study-dir tune/ --trials 40 --jobs 2 --seeds-per-trial 2 --workers-per-run 16 --enqueue-baseline
  python tune_food_dann.py --study-dir ring/ --search subtasks,ring_offset_frac --grid --jobs 2 --seeds-per-trial 2
  python tune_food_dann.py --study-dir tune/ --report [--by subtasks,ring_offset_frac]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from rich.console import Console

SCRIPT = Path(__file__).resolve().parent / "gecko_food_dann.py"
console = Console(soft_wrap=True)

parser = argparse.ArgumentParser()
parser.add_argument("--study-dir", type=Path, required=True)
parser.add_argument("--study-name", default="food_dann")
parser.add_argument("--trials", type=int, default=40, help="Trials this process runs")
parser.add_argument("--jobs", type=int, default=1, help="Trials run side by side in this process")
parser.add_argument("--seeds-per-trial", type=int, default=2)
parser.add_argument("--base-seed", type=int, default=42)
parser.add_argument("--workers-per-run", type=int, default=16)
parser.add_argument("--budget-sim-s", type=float, default=8e5,
                    help="Training compute cap per run, simulated seconds (the 300-generation baseline used ~6e5)")
parser.add_argument("--max-gens", type=int, default=2000)
parser.add_argument("--hopeless-gen", type=int, default=60,
                    help="Stop a run if no candidate has collected a training food by this generation")
parser.add_argument("--stall-gens", type=int, default=150, help="Stop a run after this many generations without a new best")
parser.add_argument("--enqueue-baseline", action="store_true", help="Queue the current defaults as a trial")
parser.add_argument("--sampler-seed", type=int, default=None)
parser.add_argument("--search", default=None,
                    help="Comma-separated parameters to search; the rest stay at BASELINE (default: all)")
parser.add_argument("--grid", action="store_true", help="Run every combination of the searched parameters' GRID values")
parser.add_argument("--report", action="store_true", help="Print the trials so far and write trials.csv, then exit")
parser.add_argument("--by", default=None, help="Comma-separated parameters to group the report by (default: --search)")
args = parser.parse_args()

BASELINE = {"subtasks": 5, "ring_offset_frac": 0.0, "food_radius": 2.0, "subtask_time": 20.0, "speed_weight": 1.0,
            "progress_weight": 1.0, "progress_mode": "first", "coverage_weight": 1.0, "hidden": 16, "optimizer": "cma",
            "pop": 20, "sigma": 0.5}

# TPE ranges. The collection bonus is fixed at 1: the optimisers rank
# candidates, so only the other weights' ratios to it matter.
SPACE: dict[str, tuple] = {
    "subtasks": ("int", 2, 8, 1),
    "ring_offset_frac": ("float", -0.5, 0.5, 0.125),
    "food_radius": ("float", 1.0, 2.0, 0.25),
    "subtask_time": ("float", 15.0, 30.0, 5.0),
    "speed_weight": ("float", 0.0, 2.0, 0.25),
    "progress_weight": ("float", 0.25, 2.0, 0.25),
    "progress_mode": ("cat", ["first", "all"]),
    "coverage_weight": ("float", 0.0, 2.0, 0.25),
    "hidden": ("cat", [8, 16, 32]),
    "optimizer": ("cat", ["cma", "sepcma"]),
    "pop": ("int", 10, 40, 2),
    "sigma": ("log", 0.1, 1.0),
}
# --grid values (coarser than SPACE).
GRID: dict[str, list] = {
    "subtasks": [2, 3, 4, 5, 6, 8],
    "ring_offset_frac": [-0.5, -0.25, 0.0, 0.25, 0.5],
    "food_radius": [1.0, 1.5, 2.0],
    "subtask_time": [15.0, 20.0, 30.0],
    "speed_weight": [0.0, 1.0, 2.0],
    "progress_weight": [0.5, 1.0, 2.0],
    "progress_mode": ["first", "all"],
    "coverage_weight": [0.0, 1.0, 2.0],
    "hidden": [8, 16, 32],
    "optimizer": ["cma", "sepcma"],
    "pop": [10, 20, 40],
    "sigma": [0.2, 0.5, 1.0],
}
SEARCH = args.search.split(",") if args.search else list(SPACE)
if unknown := set(SEARCH) - set(SPACE):
    parser.error(f"unknown --search parameters: {sorted(unknown)}")


def sample(trial: optuna.Trial) -> dict[str, Any]:
    params = dict(BASELINE)
    for name in SEARCH:
        kind, *spec = SPACE[name]
        if args.grid:  # GridSampler hands out the values; any distribution over them will do
            params[name] = trial.suggest_categorical(name, GRID[name])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[0], spec[1], step=spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[0], spec[1], step=spec[2])
        elif kind == "log":
            params[name] = trial.suggest_float(name, spec[0], spec[1], log=True)
        else:
            params[name] = trial.suggest_categorical(name, spec[0])
    return params


def cli_flags(params: dict[str, Any]) -> list[str]:
    """gecko_food_dann.py flags; ring_offset_frac becomes --ring-offset in degrees."""
    flags = []
    for key, value in params.items():
        if key == "ring_offset_frac":
            key, value = "ring_offset", value * 360.0 / params["subtasks"]
        flags += [f"--{key.replace('_', '-')}", str(value)]
    return flags


def run_cost(summary: dict[str, Any]) -> float:
    if summary["solved"] is not None:
        return float(summary["solved"]["sim_s"])
    return args.budget_sim_s * (2.0 - summary["probe_best"] / summary["probe_n"])


def objective(trial: optuna.Trial) -> float:
    params = sample(trial)
    trial_dir = args.study_dir / f"trial_{trial.number:04d}"
    procs = []
    for k in range(args.seeds_per_trial):
        seed = args.base_seed + k
        run_dir = trial_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(SCRIPT), "--seed", str(seed), "--workers", str(args.workers_per_run),
               "--gens", str(args.max_gens), "--max-sim-seconds", str(args.budget_sim_s), "--stop-on-solve",
               "--hopeless-gen", str(args.hopeless_gen), "--stall-gens", str(args.stall_gens),
               "--checkpoint-every", "100000"]
        cmd += cli_flags(params)
        log = (run_dir / "train.log").open("w")
        procs.append((run_dir, subprocess.Popen(cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                                                env={**os.environ, "MUJOCO_GL": "egl"})))
    summaries = []
    for run_dir, proc in procs:
        if proc.wait() != 0:
            msg = f"training failed, see {run_dir / 'train.log'}"
            raise RuntimeError(msg)
        summaries.append(json.loads((run_dir / "__data__" / "gecko_food_dann" / "summary.json").read_text()))

    costs = [run_cost(s) for s in summaries]
    trial.set_user_attr("params", params)  # all of them, including the fixed ones
    trial.set_user_attr("costs", costs)
    trial.set_user_attr("solved", sum(s["solved"] is not None for s in summaries))
    trial.set_user_attr("solve_gens", [s["solved"]["gen"] if s["solved"] else None for s in summaries])
    trial.set_user_attr("first_food_gens", [s["first_food_gen"] for s in summaries])
    trial.set_user_attr("probe_best", [s["probe_best"] for s in summaries])
    trial.set_user_attr("stop_reasons", [s["stop_reason"] for s in summaries])
    trial.set_user_attr("elapsed_s", [s["elapsed_s"] for s in summaries])
    trial.set_user_attr("num_params", summaries[0]["num_params"])
    return float(np.mean(costs))


def report(study: optuna.Study) -> None:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    done.sort(key=lambda t: t.value)
    keys = list(BASELINE)
    rows = []
    for t in done:
        a = t.user_attrs
        rows.append({"trial": t.number, "cost_sim_s": round(t.value), "solved": f"{a['solved']}/{len(a['costs'])}",
                     "solve_gens": a["solve_gens"], "first_food_gens": a["first_food_gens"],
                     "probe_best": a["probe_best"], "stop": a["stop_reasons"],
                     "wall_min": [round(e / 60, 1) for e in a["elapsed_s"]], **a.get("params", t.params)})
    if not rows:
        console.print("no completed trials")
        return
    with (args.study_dir / "trials.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"{len(done)} completed trials (best first); full table in {args.study_dir / 'trials.csv'}")
    for r in rows[:15]:
        console.print(f"#{r['trial']:<4d} cost {r['cost_sim_s']:>9,d} sim-s  solved {r['solved']}  "
                      f"gens {r['solve_gens']}  first food {r['first_food_gens']}  stop {r['stop']}  "
                      f"wall {r['wall_min']} min", highlight=False)
        console.print("       " + "  ".join(f"{k}={r[k]}" for k in keys if k in r), highlight=False)
    # When did eventually-solved runs first collect a food? Bounds a safe --hopeless-gen.
    first = [f for t in done for f, g in zip(t.user_attrs["first_food_gens"], t.user_attrs["solve_gens"], strict=True)
             if g is not None and f is not None]
    if first:
        console.print(f"first food generation in solved runs: max {max(first)}, 90th pct {np.percentile(first, 90):.0f}, "
                      f"median {np.median(first):.0f} (hopeless-gen is {args.hopeless_gen})")
    by = args.by.split(",") if args.by else (SEARCH if args.search else [])
    if by:
        groups: dict[tuple, list] = {}
        for r in rows:
            groups.setdefault(tuple(r[k] for k in by), []).append(r)
        console.print(f"\nby {', '.join(by)}: mean cost (sim-s), runs solved, solve generations")
        for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
            g = groups[key]
            solved = [int(r["solved"].split("/")[0]) for r in g]
            runs = [int(r["solved"].split("/")[1]) for r in g]
            gens = [x for r in g for x in r["solve_gens"] if x is not None]
            label = "  ".join(f"{k}={v}" for k, v in zip(by, key, strict=True))
            console.print(f"  {label:40s} {np.mean([r['cost_sim_s'] for r in g]):>11,.0f}  "
                          f"{sum(solved)}/{sum(runs)}  {sorted(gens)}", highlight=False)
    if len(done) >= 10 and len(SEARCH) > 1 and not args.grid:
        imp = optuna.importance.get_param_importances(study)
        console.print("parameter importance: " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))


def main() -> None:
    args.study_dir.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            str(args.study_dir / "study.log"),
            lock_obj=optuna.storages.journal.JournalFileOpenLock(str(args.study_dir / "study.log")),
        ),
    )
    if args.grid:
        sampler = optuna.samplers.GridSampler({k: GRID[k] for k in SEARCH}, seed=args.sampler_seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True, n_startup_trials=10)
    study = optuna.create_study(study_name=args.study_name, storage=storage, direction="minimize",
                                load_if_exists=True, sampler=sampler)
    if args.report:
        report(study)
        return
    if args.enqueue_baseline and not args.grid and not any(t.user_attrs.get("params") == BASELINE for t in study.trials):
        study.enqueue_trial({k: BASELINE[k] for k in SEARCH})
    (args.study_dir / "tune_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs, catch=(RuntimeError,))
    report(study)


if __name__ == "__main__":
    main()
