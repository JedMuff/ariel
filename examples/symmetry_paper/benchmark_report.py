"""
Cluster cost of E1, E2 and E3 from the benchmark (slurm benchmark.sh).

  E1  a full run's wall time = the benchmark's generations 0-3 x the ratio of
      full-run time to generations 0-3 in the hinge grid's bang_bang runs
      (bodies grow, so later generations are slower); mean and the 90th
      percentile ratio. As a check, the bang_bang tree / sym arms repeat the
      grid's zoo-cell seed-42 runs, whose generations 0-3 give the node's
      speed against the nodes the grid ran on (the "node speed" line).
  E2  a task (one body x brain, 3 training seeds in turn) = 3 x the
      benchmark's per-training wall time, per body set and brain.
  E3  per run: replaying an evolution run's last generation, or one
      fixed-body training's checkpoints.
  E0  for reference, the measured wall time of the finished budget series.

Wall time assumes --concurrent jobs of 20 CPUs at once: max(job-hours /
concurrent, longest job). "--time" is a suggested Slurm limit: the longest
estimated job x 1.5, rounded up to the hour.

Usage:
    python benchmark_report.py --bench-root /scratch/$USER/ariel_paper/bench [--concurrent 7]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import runs

BENCH_GENS = 4   # benchmark.sh runs --budget 3: generations 0-3
E1_RUNS_PER_ARM = 10
E2_BODIES = {"random": 80, "evolved": 40}
E2_SEEDS = 3
E3_RUNS = {"E1 champions": 60, "budget series champions": 30}
E3_TRAININGS = {"random": 80 * 3 * E2_SEEDS, "evolved": 40 * 3 * E2_SEEDS}
ZOO_CELL = "kp1.36051_kv0.35555_arm0.0811705_lim1"
CPUS = 20


def gen_times(run_dir: Path) -> list[float]:
    d = runs.data_dir(run_dir)
    if d is None or not (d / "timing.jsonl").exists():
        return []
    return [json.loads(line)["wall_time_s"] for line in (d / "timing.jsonl").read_text().splitlines() if line]


def bench_steps(bench_root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(bench_root.glob("*/bench_times.jsonl")):
        rows += [{"task": f.parent.name, **json.loads(line)} for line in f.read_text().splitlines() if line]
    return pd.DataFrame(rows)


def cost_row(name: str, n_jobs: int, mean_h: float, max_h: float, concurrent: int) -> dict:
    job_h = n_jobs * mean_h
    return {"experiment": name, "jobs": n_jobs, "mean job (h)": mean_h, "longest job (h)": max_h,
            "CPU-h": job_h * CPUS, f"wall @{concurrent} (h)": max(job_h / concurrent, max_h),
            "--time": f"{max(1, math.ceil(1.5 * max_h)):02d}:00:00"}


def e1_rows(bench_root: Path, grid_root: Path, concurrent: int) -> tuple[list[dict], list[str]]:
    ratios, ref03 = [], {}
    for run_dir in sorted(grid_root.glob("actgrid_bang_bang_*")):
        t = gen_times(run_dir)
        if len(t) > BENCH_GENS:
            ratios.append(sum(t) / sum(t[:BENCH_GENS]))
            if ZOO_CELL in run_dir.name and "_s42_" in run_dir.name:
                ref03["tree_symmetric" if "tree_symmetric" in run_dir.name else "tree"] = sum(t[:BENCH_GENS])
    if not ratios:
        raise SystemExit(f"no bang_bang grid runs with timing under {grid_root}")
    r_mean, r_p90 = float(np.mean(ratios)), float(np.percentile(ratios, 90))
    notes = [f"E1 full-run / generations 0-3 ratio over {len(ratios)} grid bang_bang runs: "
             f"mean {r_mean:.1f}, 90th percentile {r_p90:.1f}"]

    rows, arm_h = [], []
    for task_dir in sorted(bench_root.glob("e1_*")):
        t = gen_times(task_dir)
        if len(t) < BENCH_GENS:
            notes.append(f"{task_dir.name}: only {len(t)} of {BENCH_GENS} generations, skipped")
            continue
        s03 = sum(t[:BENCH_GENS])
        cfg = runs.run_config(task_dir)
        genome, brain = cfg["genome_type"], cfg["brain"]
        if brain == "bang_bang" and genome in ref03:
            notes.append(f"node speed, {genome} / bang_bang: generations 0-3 took {s03:.0f} s here vs "
                         f"{ref03[genome]:.0f} s in the grid ({s03 / ref03[genome]:.2f}x)")
        mean_h, max_h = s03 * r_mean / 3600, s03 * r_p90 / 3600
        arm_h.append((mean_h, max_h))
        rows.append(cost_row(f"E1 {brain} / {genome}", E1_RUNS_PER_ARM, mean_h, max_h, concurrent))
    if arm_h:
        mean_h = float(np.mean([m for m, _ in arm_h]))
        rows.append(cost_row("E1 total", E1_RUNS_PER_ARM * len(arm_h), mean_h, max(m for _, m in arm_h), concurrent))
    return rows, notes


def e2_rows(steps: pd.DataFrame, concurrent: int) -> list[dict]:
    train = steps[steps["step"] == "e2_train"]
    rows = []
    for (body_set, brain), g in train.groupby(["body_set", "brain"]):
        n_jobs = E2_BODIES[body_set]
        mean_h, max_h = E2_SEEDS * g["wall_s"].mean() / 3600, E2_SEEDS * g["wall_s"].max() / 3600
        rows.append(cost_row(f"E2 {body_set} / {brain}", n_jobs, mean_h, max_h, concurrent))
    for body_set in E2_BODIES:
        sub = [r for r in rows if r["experiment"].startswith(f"E2 {body_set} ")]
        if sub:
            n = sum(r["jobs"] for r in sub)
            rows.append(cost_row(f"E2 {body_set} total", n, sum(r["jobs"] * r["mean job (h)"] for r in sub) / n,
                                 max(r["longest job (h)"] for r in sub), concurrent))
    return rows


def e3_rows(steps: pd.DataFrame, concurrent: int) -> list[dict]:
    rows = []
    evo = steps[steps["step"] == "e3_replay_evolution"]["wall_s"]
    if len(evo):
        for name, n in E3_RUNS.items():
            rows.append(cost_row(f"E3 {name}", n, evo.mean() / 3600, evo.max() / 3600, concurrent))
    tr = steps[steps["step"] == "e3_replay_training"]
    for body_set, g in tr.groupby("body_set"):
        rows.append(cost_row(f"E3 E2 {body_set} trainings", E3_TRAININGS[body_set], g["wall_s"].mean() / 3600,
                             g["wall_s"].max() / 3600, concurrent))
    return rows


def e0_rows(budget_roots: list[Path], concurrent: int) -> list[dict]:
    rows = []
    for root in budget_roots:
        hours = [sum(t) / 3600 for d in sorted(root.iterdir()) if d.is_dir() and (t := gen_times(d))]
        if hours:
            rows.append(cost_row(f"E0 budget {root.name} (measured)", len(hours), float(np.mean(hours)),
                                 max(hours), concurrent))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--grid-root", type=Path, default=Path("__data__/ariel_actuator_grid"))
    parser.add_argument("--budget-roots", type=Path, nargs="*",
                        default=[Path("__data__/ariel_dann_budget/sb40"), Path("__data__/ariel_dann_budget/sb80")])
    parser.add_argument("--concurrent", type=int, default=7, help="20-CPU jobs running at once")
    parser.add_argument("--out", type=Path, help="CSV (default <bench-root>/benchmark_report.csv)")
    args = parser.parse_args()

    steps = bench_steps(args.bench_root)
    if steps.empty:
        raise SystemExit(f"no bench_times.jsonl under {args.bench_root}")
    print(f"benchmark nodes: {', '.join(sorted(steps['node'].unique()))}")

    rows, notes = e1_rows(args.bench_root, args.grid_root, args.concurrent)
    rows += e2_rows(steps, args.concurrent) + e3_rows(steps, args.concurrent)
    rows += e0_rows([r for r in args.budget_roots if r.exists()], args.concurrent)
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.2f}".format):
        print(df.to_string(index=False))
    for note in notes:
        print(f"  {note}")
    out = args.out or args.bench_root / "benchmark_report.csv"
    df.to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
