"""
Analyse all symmetry-investigation runs and produce combined plots.

Directory naming: sym_<JOBID>_<ARRAY_ID>
Array index encoding (from run_symmetry_investigation.sh):
    TASK_IDX     = array_id % 2              → [gecko_locomotion, gecko_food]
    STRATEGY_IDX = (array_id / 2) % 4        → [plus, comma, nsga2-efficiency, nsga2-symmetry]
    BUDGET_IDX   = (array_id / 8) % 2        → [200, 500]
    REEVAL_IDX   = (array_id / 16) % 2       → [off, on]
    REP          = (array_id / 32) % 5       → 0–4

Usage:
    python examples/symmetry_pressure/analyse_investigation.py \\
        --results-dir data/symmetry_investigation \\
        [--task locomotion|food]  \\
        [--strategy plus|comma|nsga2-efficiency|nsga2-symmetry] \\
        [--budget 200|500|1000] \\
        [--repeat-evals] \\
        [--top-n 3]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=Path, required=True,
                    help="Root dir containing sym_* subdirs")
parser.add_argument("--task", choices=["locomotion", "food"],
                    help="Filter to a specific task")
parser.add_argument("--strategy",
                    choices=["plus", "comma", "nsga2-efficiency", "nsga2-symmetry"],
                    help="Filter to a specific strategy")
parser.add_argument("--budget", type=int, choices=[200, 500],
                    help="Filter to a specific brain budget")
parser.add_argument("--repeat-evals", action="store_true", default=None,
                    help="Filter to runs with repeat-evals=on")
parser.add_argument("--no-repeat-evals", dest="repeat_evals", action="store_false",
                    help="Filter to runs with repeat-evals=off")
parser.add_argument("--top-n", type=int, default=3,
                    help="Top-N individuals to render as video (per condition)")
parser.add_argument("--run-ids", nargs="+", metavar="ID",
                    help="Only include runs whose directory name contains one of these")
parser.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory for plots and videos (default: <results-dir>/analysis)")
args = parser.parse_args()

RESULTS_DIR = args.results_dir.resolve()
OUT_DIR     = args.out_dir.resolve() if args.out_dir else RESULTS_DIR / "analysis"
OUT_DIR.mkdir(exist_ok=True, parents=True)

TASKS      = ["gecko_locomotion", "gecko_food"]
STRATEGIES = ["plus", "comma", "nsga2-efficiency", "nsga2-symmetry"]
BUDGETS    = [200, 500]

STRATEGY_COLORS = {
    "plus":             "steelblue",
    "comma":            "seagreen",
    "nsga2-efficiency": "darkorange",
    "nsga2-symmetry":   "mediumpurple",
}
STRATEGY_LABELS = {
    "plus":             "(μ+λ)-ES",
    "comma":            "(μ,λ)-ES",
    "nsga2-efficiency": "NSGA-II efficiency",
    "nsga2-symmetry":   "NSGA-II symmetry",
}
BUDGET_STYLES = {200: "--", 500: "-"}

# ── Array-index decoder ───────────────────────────────────────────────────────

def decode_array_id(array_id: int) -> dict:
    task_idx     = array_id % 2
    strategy_idx = (array_id // 2) % 4
    budget_idx   = (array_id // 8) % 2
    reeval_idx   = (array_id // 16) % 2
    rep          = (array_id // 32) % 5
    return {
        "task":         TASKS[task_idx],
        "strategy":     STRATEGIES[strategy_idx],
        "brain_budget": BUDGETS[budget_idx],
        "repeat_evals": bool(reeval_idx),
        "rep":          rep,
    }


def parse_run_dir_name(name: str) -> Optional[dict]:
    """Return decoded condition dict if name matches sym_<job>_<array>, else None."""
    parts = name.split("_")
    if len(parts) < 3 or parts[0] != "sym":
        return None
    try:
        array_id = int(parts[2])
        job_id   = int(parts[1])
    except ValueError:
        return None
    info = decode_array_id(array_id)
    info["job_id"]   = job_id
    info["array_id"] = array_id
    return info


# ── Load a single run ─────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> Optional[dict]:
    info = parse_run_dir_name(run_dir.name)
    if info is None:
        return None

    task_name = info["task"].replace("gecko_", "")  # locomotion | food
    data_dir  = run_dir / "__data__" / info["task"]
    jsonl     = data_dir / "run_data.jsonl"
    timing    = data_dir / "timing.jsonl"
    config    = data_dir / "run_config.json"

    if not jsonl.exists():
        return None

    records: list[dict] = []
    with jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    timing_records: list[dict] = []
    if timing.exists():
        with timing.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    timing_records.append(json.loads(line))

    cfg: dict = {}
    if config.exists():
        with config.open() as fh:
            cfg = json.load(fh)

    return {
        "run_dir":  run_dir,
        "data_dir": data_dir,
        "task":     task_name,
        "strategy": info["strategy"],
        "budget":   info["brain_budget"],
        "reeval":   info["repeat_evals"],
        "rep":      info["rep"],
        "job_id":   info["job_id"],
        "array_id": info["array_id"],
        "cfg":      cfg,
        "records":  records,
        "timing":   timing_records,
    }


# ── Discover and filter runs ──────────────────────────────────────────────────

all_runs: list[dict] = []
for candidate in sorted(RESULTS_DIR.iterdir()):
    if not candidate.is_dir() or candidate.name == "analysis":
        continue
    if args.run_ids and not any(rid in candidate.name for rid in args.run_ids):
        continue
    run = load_run(candidate)
    if run is None:
        continue
    if args.task and run["task"] != args.task:
        continue
    if args.strategy and run["strategy"] != args.strategy:
        continue
    if args.budget and run["budget"] != args.budget:
        continue
    if args.repeat_evals is not None and run["reeval"] != args.repeat_evals:
        continue
    all_runs.append(run)
    reeval_tag = "reeval=on" if run["reeval"] else "reeval=off"
    print(f"  {run['strategy']:22s}  task={run['task']:11s}  budget={run['budget']:4d}  "
          f"{reeval_tag}  rep={run['rep']}  "
          f"{len(run['records'])} records  ({candidate.name})")

if not all_runs:
    raise SystemExit(f"No valid runs found under {RESULTS_DIR}")

print(f"\nLoaded {len(all_runs)} run(s)\n")


def by_gen(records: list[dict]) -> dict[int, list[dict]]:
    d: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        d[r["gen"]].append(r)
    return d


def condition_key(run: dict) -> tuple:
    return (run["task"], run["strategy"], run["budget"], run["reeval"])


def condition_label(run: dict) -> str:
    reeval = "reeval" if run["reeval"] else "no-reeval"
    return f"{STRATEGY_LABELS.get(run['strategy'], run['strategy'])}  B={run['budget']}  {reeval}"


# Group runs by condition (task, strategy, budget, reeval)
by_condition: dict[tuple, list[dict]] = defaultdict(list)
for run in all_runs:
    by_condition[condition_key(run)].append(run)

conditions = sorted(by_condition.keys())
n_conditions = len(conditions)

# ── Helpers ───────────────────────────────────────────────────────────────────

def rep_mean_curve(runs_for_cond: list[dict], field: str = "fitness"
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gens, mean_across_reps, std_across_reps) for the mean-per-gen of `field`."""
    all_gen_means: dict[int, list[float]] = defaultdict(list)
    for run in runs_for_cond:
        bg = by_gen(run["records"])
        for g, recs in bg.items():
            vals = [r[field] for r in recs
                    if r.get(field) is not None and np.isfinite(r[field])]
            if vals:
                all_gen_means[g].append(float(np.mean(vals)))
    gens = np.array(sorted(all_gen_means.keys()))
    means = np.array([np.mean(all_gen_means[g]) for g in gens])
    stds  = np.array([np.std(all_gen_means[g])  for g in gens])
    return gens, means, stds


def rep_best_curve(runs_for_cond: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gens, mean_best_so_far, std_best_so_far) across reps."""
    all_bests: dict[int, list[float]] = defaultdict(list)
    for run in runs_for_cond:
        bg = by_gen(run["records"])
        gens_sorted = sorted(bg.keys())
        best_so_far = float("inf")
        for g in gens_sorted:
            fits = [r["fitness"] for r in bg[g]
                    if r.get("fitness") is not None and np.isfinite(r["fitness"])]
            if fits:
                best_so_far = min(best_so_far, min(fits))
            all_bests[g].append(best_so_far)
    gens = np.array(sorted(all_bests.keys()))
    means = np.array([np.mean(all_bests[g]) for g in gens])
    stds  = np.array([np.std(all_bests[g])  for g in gens])
    return gens, means, stds


# ── Build a colour map for conditions ────────────────────────────────────────

# Within each task, group by strategy (colour) + budget (linestyle) + reeval (alpha)
def cond_color(cond: tuple) -> str:
    _, strategy, _, _ = cond
    return STRATEGY_COLORS.get(strategy, "grey")


def cond_linestyle(cond: tuple) -> str:
    _, _, budget, _ = cond
    return BUDGET_STYLES.get(budget, "-")


def cond_alpha(cond: tuple) -> float:
    _, _, _, reeval = cond
    return 0.9 if reeval else 0.5


# ── Plot 1 — Fitness curves, faceted by task ──────────────────────────────────

tasks_present = sorted(set(c[0] for c in conditions))
fig, axes = plt.subplots(len(tasks_present), 2,
                         figsize=(14, 5 * len(tasks_present)), squeeze=False)

for row, task in enumerate(tasks_present):
    ax_mean, ax_best = axes[row]
    task_conds = [c for c in conditions if c[0] == task]
    for cond in task_conds:
        runs_c = by_condition[cond]
        color  = cond_color(cond)
        ls     = cond_linestyle(cond)
        alpha  = cond_alpha(cond)
        _, strategy, budget, reeval = cond
        reeval_tag = "reeval" if reeval else "no-reeval"
        label  = f"{STRATEGY_LABELS.get(strategy, strategy)}  B={budget}  {reeval_tag}"

        gens, means, stds = rep_mean_curve(runs_c)
        if len(gens) == 0:
            continue
        n_reps = len(runs_c)
        lbl = f"{label}  (n={n_reps})"
        ax_mean.plot(gens, means, label=lbl, color=color, linestyle=ls, alpha=alpha)
        ax_mean.fill_between(gens, means - stds, means + stds, alpha=0.10, color=color)

        gens_b, means_b, stds_b = rep_best_curve(runs_c)
        ax_best.plot(gens_b, means_b, label=lbl, color=color, linestyle=ls, alpha=alpha)
        ax_best.fill_between(gens_b, means_b - stds_b, means_b + stds_b, alpha=0.10, color=color)

    for ax, title in zip([ax_mean, ax_best],
                         [f"{task} — mean±std fitness", f"{task} — running-best fitness"]):
        ax.set_xlabel("Outer generation")
        ax.set_ylabel("Fitness (lower is better)")
        ax.set_title(title)
        ax.legend(fontsize=6, ncol=2)
        ax.annotate("lower is better", xy=(0.02, 0.04), xycoords="axes fraction",
                    fontsize=7, color="grey")

fig.suptitle("Symmetry investigation — fitness curves\n"
             "(linestyle=budget: solid=1000, dashed=500, dotted=200; "
             "opacity: high=reeval, low=no-reeval)")
fig.tight_layout()
fig.savefig(OUT_DIR / "fitness_curves.png", dpi=150)
plt.close(fig)
print("  fitness_curves.png")

# ── Plot 2 — Effect of repeat-evals: paired comparison ───────────────────────
# For each (task, strategy, budget) show reeval=off vs reeval=on best curves

paired_conds: dict[tuple, dict] = defaultdict(dict)
for cond in conditions:
    task, strategy, budget, reeval = cond
    key = (task, strategy, budget)
    paired_conds[key][reeval] = cond

has_pairs = any(len(v) == 2 for v in paired_conds.values())
if has_pairs:
    pair_keys = [k for k, v in paired_conds.items() if len(v) == 2]
    n_pairs = len(pair_keys)
    ncols = min(4, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, pkey in zip(axes_flat, pair_keys):
        task, strategy, budget = pkey
        for reeval, cond in sorted(paired_conds[pkey].items()):
            runs_c = by_condition[cond]
            color  = STRATEGY_COLORS.get(strategy, "grey")
            ls     = "-" if reeval else "--"
            alpha  = 0.9 if reeval else 0.6
            label  = ("reeval=on" if reeval else "reeval=off") + f"  n={len(runs_c)}"
            gens_b, means_b, stds_b = rep_best_curve(runs_c)
            if len(gens_b) == 0:
                continue
            ax.plot(gens_b, means_b, label=label, color=color, linestyle=ls, alpha=alpha)
            ax.fill_between(gens_b, means_b - stds_b, means_b + stds_b,
                            alpha=0.12, color=color)
        ax.set_title(f"{STRATEGY_LABELS.get(strategy, strategy)}\n{task}  B={budget}",
                     fontsize=8)
        ax.set_xlabel("Generation", fontsize=7)
        ax.set_ylabel("Running-best fitness", fontsize=7)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(pair_keys):]:
        ax.set_visible(False)

    fig.suptitle("Effect of repeat-evals  (solid=on, dashed=off)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reeval_comparison.png", dpi=150)
    plt.close(fig)
    print("  reeval_comparison.png")

# ── Plot 3 — Effect of brain budget ──────────────────────────────────────────

budget_keys: dict[tuple, dict] = defaultdict(dict)
for cond in conditions:
    task, strategy, budget, reeval = cond
    key = (task, strategy, reeval)
    budget_keys[key][budget] = cond

fig_budget_keys = [k for k, v in budget_keys.items() if len(v) > 1]
if fig_budget_keys:
    n_plots = len(fig_budget_keys)
    ncols = min(4, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, bkey in zip(axes_flat, fig_budget_keys):
        task, strategy, reeval = bkey
        color = STRATEGY_COLORS.get(strategy, "grey")
        cmap  = plt.get_cmap("Blues")
        budget_list = sorted(budget_keys[bkey].keys())
        n_b = len(budget_list)
        for i, budget in enumerate(budget_list):
            cond   = budget_keys[bkey][budget]
            runs_c = by_condition[cond]
            shade  = 0.4 + 0.5 * (i / max(n_b - 1, 1))
            c      = cmap(shade)
            gens_b, means_b, stds_b = rep_best_curve(runs_c)
            if len(gens_b) == 0:
                continue
            ax.plot(gens_b, means_b, label=f"B={budget}  n={len(runs_c)}", color=c)
            ax.fill_between(gens_b, means_b - stds_b, means_b + stds_b, alpha=0.12, color=c)
        reeval_tag = "reeval=on" if reeval else "reeval=off"
        ax.set_title(f"{STRATEGY_LABELS.get(strategy, strategy)}\n{task}  {reeval_tag}",
                     fontsize=8)
        ax.set_xlabel("Generation", fontsize=7)
        ax.set_ylabel("Running-best fitness", fontsize=7)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(fig_budget_keys):]:
        ax.set_visible(False)

    fig.suptitle("Effect of brain budget  (darker=larger budget)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "budget_comparison.png", dpi=150)
    plt.close(fig)
    print("  budget_comparison.png")

# ── Plot 4 — Symmetry over generations (per condition) ───────────────────────

fig, axes = plt.subplots(len(tasks_present), max(1, len(conditions) // max(len(tasks_present), 1)),
                         figsize=(max(6, n_conditions * 4), 5 * len(tasks_present)),
                         squeeze=False)

task_cond_map: dict[str, list[tuple]] = defaultdict(list)
for c in conditions:
    task_cond_map[c[0]].append(c)

for row, task in enumerate(tasks_present):
    task_conds_list = task_cond_map[task]
    for col, cond in enumerate(task_conds_list):
        if col >= axes.shape[1]:
            break
        ax = axes[row][col]
        runs_c = by_condition[cond]
        color  = cond_color(cond)
        _, strategy, budget, reeval = cond
        reeval_tag = "reeval" if reeval else "no-reeval"

        all_syms_by_gen: dict[int, list[float]] = defaultdict(list)
        for run in runs_c:
            for r in run["records"]:
                if r.get("yz_symmetry") is not None:
                    all_syms_by_gen[r["gen"]].append(r["yz_symmetry"])

        gens_s = sorted(g for g in all_syms_by_gen if all_syms_by_gen[g])
        if gens_s:
            gens_arr  = np.array(gens_s)
            means_arr = np.array([np.mean(all_syms_by_gen[g]) for g in gens_s])
            stds_arr  = np.array([np.std(all_syms_by_gen[g])  for g in gens_s])
            maxs_arr  = np.array([np.max(all_syms_by_gen[g])  for g in gens_s])
            ax.plot(gens_arr, means_arr, color=color, linewidth=1.5, label="mean")
            ax.fill_between(gens_arr, means_arr - stds_arr, means_arr + stds_arr,
                            alpha=0.2, color=color, label="±1 std")
            ax.plot(gens_arr, maxs_arr, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.7, label="max")

        ax.set_ylim(0, 1)
        ax.set_xlabel("Generation", fontsize=7)
        ax.set_title(f"{STRATEGY_LABELS.get(strategy, strategy)}\nB={budget}  {reeval_tag}\n{task}",
                     fontsize=7)
    if task_cond_map[task]:
        axes[row][0].set_ylabel("yz_symmetry")

    for col in range(len(task_cond_map[task]), axes.shape[1]):
        axes[row][col].set_visible(False)

fig.suptitle("Symmetry distribution over generations (pooled across reps)")
fig.tight_layout()
fig.savefig(OUT_DIR / "symmetry_over_gens.png", dpi=150)
plt.close(fig)
print("  symmetry_over_gens.png")

# ── Plot 5 — Timing summary ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax_wall, ax_per_ind, ax_cumul = axes

for cond in conditions:
    runs_c = by_condition[cond]
    _, strategy, budget, reeval = cond
    color = cond_color(cond)
    ls    = cond_linestyle(cond)
    alpha = cond_alpha(cond)
    reeval_tag = "reeval" if reeval else "no-reeval"
    label = f"{STRATEGY_LABELS.get(strategy, strategy)} B={budget} {reeval_tag}"

    for run in runs_c:
        timings = run["timing"]
        if not timings:
            continue
        gens_t     = [t["gen"] for t in timings]
        wall_times = [t["wall_time_s"] for t in timings]
        mean_evals = [t["mean_eval_time_s"] for t in timings]
        cumulative = np.cumsum(wall_times)
        ax_wall.plot(gens_t, wall_times, color=color, linestyle=ls, alpha=alpha * 0.6,
                     linewidth=0.8)
        ax_per_ind.plot(gens_t, mean_evals, color=color, linestyle=ls, alpha=alpha * 0.6,
                        linewidth=0.8)
        ax_cumul.plot(gens_t, cumulative / 60.0, color=color, linestyle=ls, alpha=alpha * 0.6,
                      linewidth=0.8)

    # Mean across reps
    all_wall: dict[int, list] = defaultdict(list)
    all_eval: dict[int, list] = defaultdict(list)
    for run in runs_c:
        for t in run["timing"]:
            all_wall[t["gen"]].append(t["wall_time_s"])
            all_eval[t["gen"]].append(t["mean_eval_time_s"])
    if all_wall:
        gt  = np.array(sorted(all_wall.keys()))
        mw  = np.array([np.mean(all_wall[g]) for g in gt])
        me  = np.array([np.mean(all_eval[g]) for g in gt])
        ax_wall.plot(gt, mw, color=color, linestyle=ls, alpha=alpha, linewidth=2,
                     label=label)
        ax_per_ind.plot(gt, me, color=color, linestyle=ls, alpha=alpha, linewidth=2,
                        label=label)
        ax_cumul.plot(gt, np.cumsum(mw) / 60.0, color=color, linestyle=ls, alpha=alpha,
                      linewidth=2, label=label)

for ax, title, ylabel in [
    (ax_wall,    "Wall time per generation",     "Wall time (s)"),
    (ax_per_ind, "Mean time per individual",     "Mean eval time (s/ind)"),
    (ax_cumul,   "Cumulative runtime",           "Cumulative wall time (min)"),
]:
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=5, ncol=2)

fig.suptitle("Timing summary")
fig.tight_layout()
fig.savefig(OUT_DIR / "timing_summary.png", dpi=150)
plt.close(fig)
print("  timing_summary.png")

# ── Timing text table ─────────────────────────────────────────────────────────

lines = [
    f"{'Condition':<55}  {'Reps':>4}  {'Gens':>4}  {'s/ind':>7}  "
    f"{'s/gen':>7}  {'Total min':>10}",
    "-" * 100,
]
for cond in conditions:
    runs_c = by_condition[cond]
    _, strategy, budget, reeval = cond
    reeval_tag = "reeval" if reeval else "no-reeval"
    label = f"{STRATEGY_LABELS.get(strategy, strategy)} B={budget} {reeval_tag}"
    all_timing = [t for run in runs_c for t in run["timing"]]
    if not all_timing:
        continue
    n_reps   = len(runs_c)
    n_gens   = len(set(t["gen"] for t in all_timing))
    mean_ind = float(np.mean([t["mean_eval_time_s"] for t in all_timing]))
    mean_gen = float(np.mean([t["wall_time_s"] for t in all_timing]))
    total_m  = sum(t["wall_time_s"] for t in all_timing) / 60.0
    lines.append(f"{label:<55}  {n_reps:>4}  {n_gens:>4}  {mean_ind:>7.1f}  "
                 f"{mean_gen:>7.1f}  {total_m:>10.1f}")

table_str = "\n".join(lines)
print("\n" + table_str)
(OUT_DIR / "timing_table.txt").write_text(table_str + "\n")
print("\n  timing_table.txt")

# ── Plot 6 — Pareto fronts (NSGA-II only) ────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from shared import pareto_rank, genome_input_dim  # type: ignore[import]

nsga2_conds = [c for c in conditions if c[1].startswith("nsga2-")]
if nsga2_conds:
    ncols = min(4, len(nsga2_conds))
    nrows = (len(nsga2_conds) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, cond in zip(axes_flat, nsga2_conds):
        _, strategy, budget, reeval = cond
        runs_c = by_condition[cond]
        records_all = [r for run in runs_c for r in run["records"]
                       if r.get("fitness") is not None and np.isfinite(r["fitness"])]
        if not records_all:
            ax.set_visible(False)
            continue

        gens   = sorted(set(r["gen"] for r in records_all))
        n_gens = len(gens)
        cmap   = plt.get_cmap("plasma")
        gen_idx = {g: i for i, g in enumerate(gens)}

        x_all = [-r["fitness"] for r in records_all]
        if strategy == "nsga2-efficiency":
            y_all = [-r.get("control_cost", 0.0) for r in records_all]
            objs  = [(r["fitness"], r.get("control_cost", 0.0)) for r in records_all]
        else:
            y_all = [r.get("yz_symmetry", 0.0) for r in records_all]
            objs  = [(r["fitness"], -r.get("yz_symmetry", 0.0)) for r in records_all]

        colors_pts = [cmap(gen_idx[r["gen"]] / max(n_gens - 1, 1)) for r in records_all]
        ax.scatter(x_all, y_all, c=colors_pts, alpha=0.4, s=12, zorder=2)

        ranks = pareto_rank(objs)
        front_idx = [i for i, rk in enumerate(ranks) if rk == 0]
        if front_idx:
            fx = [x_all[i] for i in front_idx]
            fy = [y_all[i] for i in front_idx]
            sorted_front = sorted(zip(fx, fy))
            sfx, sfy = zip(*sorted_front)
            ax.scatter(sfx, sfy, color="crimson", s=50, zorder=5, edgecolors="darkred",
                       linewidths=0.5, label=f"Global front (n={len(front_idx)})")
            ax.step(sfx, sfy, where="post", color="crimson", linewidth=1.5, zorder=4)

        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gens[0], vmax=gens[-1]))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Generation")

        reeval_tag = "reeval" if reeval else "no-reeval"
        if strategy == "nsga2-efficiency":
            ax.set_xlabel("Locomotion →")
            ax.set_ylabel("Efficiency →  (less cost = up)")
        else:
            ax.set_xlabel("Locomotion →")
            ax.set_ylabel("Symmetry →  (more = up)")
        ax.set_title(f"{STRATEGY_LABELS.get(strategy, strategy)}\nB={budget}  {reeval_tag}  n={len(runs_c)} reps",
                     fontsize=8)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(nsga2_conds):]:
        ax.set_visible(False)

    fig.suptitle("NSGA-II: all individuals + global Pareto front  (top-right = better)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pareto_fronts.png", dpi=150)
    plt.close(fig)
    print("  pareto_fronts.png")

# ── Videos — top-N per condition ─────────────────────────────────────────────

print("\n--- Rendering videos ---")

try:
    import mujoco
    import cv2
    HAS_MUJOCO = True
    HAS_CV2    = True
except ImportError as e:
    HAS_MUJOCO = "mujoco" not in str(e)
    HAS_CV2    = "cv2" not in str(e)
    if not HAS_MUJOCO:
        print("  [skip] mujoco not importable")
    elif not HAS_CV2:
        print("  [skip] cv2 not available")

if HAS_MUJOCO and HAS_CV2:
    sys.path.insert(0, str(Path(__file__).parent))
    from shared import (  # type: ignore[import]
        Network, fill_parameters, build_world_for_body,
        analyze_sections, isolate_green,
    )
    from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state  # type: ignore[import]

    def find_checkpoint(checkpoints_dir: Path, genome_hash_val: Optional[str]) -> Optional[Path]:
        if not checkpoints_dir.exists():
            return None
        for cp in sorted(checkpoints_dir.iterdir()):
            if not cp.is_dir():
                continue
            genome_path = cp / "best_genome.json"
            if not genome_path.exists():
                continue
            if genome_hash_val is not None:
                with genome_path.open() as fh:
                    stored = json.load(fh)
                try:
                    from shared import genome_hash as gh_fn  # type: ignore[import]
                    if gh_fn(stored) == genome_hash_val:
                        return cp
                except Exception:
                    pass
        return None

    def render_video(rec: dict, cp: Path, cfg: dict, out_dir: Path,
                     strategy: str, budget: int, task: str,
                     role: str = "") -> None:
        gen      = rec["gen"]
        fitness  = rec["fitness"]
        symmetry = rec.get("yz_symmetry", 0.0) or 0.0
        duration = float(cfg.get("dur", 10.0))

        with (cp / "best_genome.json").open() as fh:
            genome_dict = json.load(fh)
        weights = np.load(cp / "best_weights.npy")

        add_food = task == "food"
        model, data, target_mocap_id, food_cam_name = build_world_for_body(
            genome_dict, reach_radius=0.20, arena_radius=3.0, add_food_target=add_food
        )

        input_dim = genome_input_dim(model, data, use_vision=add_food)
        network   = Network(input_size=input_dim, output_size=model.nu)
        fill_parameters(network, weights)

        render_h, render_w = 480, 640
        renderer = mujoco.Renderer(model, height=render_h, width=render_w)
        vision_renderer: Optional[mujoco.Renderer] = None
        if add_food:
            vision_renderer = mujoco.Renderer(model, height=48 * 2, width=64 * 2)

        core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
        if core_id < 0:
            core_id = 3
        track_cam = mujoco.MjvCamera()
        track_cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        track_cam.trackbodyid = core_id
        track_cam.distance    = 2.0
        track_cam.azimuth     = 45.0
        track_cam.elevation   = -25.0

        frames: list[np.ndarray] = []
        mujoco.mj_resetData(model, data)

        num_wps = int(cfg.get("num_waypoints", 10))
        waypoints = [
            np.array([np.cos(2 * np.pi * i / num_wps) * float(cfg.get("arena_radius", 3.0)) * 0.6,
                      np.sin(2 * np.pi * i / num_wps) * float(cfg.get("arena_radius", 3.0)) * 0.6,
                      0.0])
            for i in range(num_wps)
        ]
        current_wp_idx = 0
        reach_radius = float(cfg.get("reach_radius", 0.20))
        if add_food and target_mocap_id is not None:
            data.mocap_pos[target_mocap_id] = waypoints[0]

        step = 0
        current_action = np.zeros(model.nu)
        while data.time < duration:
            if step % 50 == 0:
                robot_state = get_robot_state(data)
                if add_food and vision_renderer is not None:
                    vision_renderer.update_scene(data, camera=food_cam_name)
                    img    = vision_renderer.render()
                    vision = analyze_sections(isolate_green(img))
                    phase    = [2.0 * np.sin(data.time * 2.0 * np.pi),
                                2.0 * np.cos(data.time * 2.0 * np.pi)]
                    progress = [current_wp_idx / max(num_wps - 1, 1)]
                    state = np.concatenate([robot_state, vision, phase, progress]).astype(np.float32)
                else:
                    state = robot_state.astype(np.float32)
                current_action = network.forward(model, data, state)
                if add_food and target_mocap_id is not None and current_wp_idx < num_wps:
                    dist = float(np.linalg.norm(np.array(data.qpos[:2]) - waypoints[current_wp_idx][:2]))
                    if dist <= reach_radius:
                        current_wp_idx = min(current_wp_idx + 1, num_wps - 1)
                        data.mocap_pos[target_mocap_id] = waypoints[current_wp_idx]
            data.ctrl[:] = current_action
            mujoco.mj_step(model, data)
            if step % 17 == 0:
                renderer.update_scene(data, camera=track_cam)
                frame = renderer.render().copy()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"{strategy} B={budget}  Gen {gen}  Fit {fitness:.3f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame_bgr, f"Sym {symmetry:.3f}",
                            (render_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                if role:
                    cv2.putText(frame_bgr, role,
                                (10, render_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 255), 2)
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            step += 1

        renderer.close()
        if vision_renderer is not None:
            vision_renderer.close()
        if not frames:
            return

        role_slug = f"_{role.lower().replace(' ', '_')}" if role else ""
        reeval_slug = "_reeval" if cfg.get("repeat_evals") else ""
        fname = (f"video_{task}_{strategy}_B{budget}{reeval_slug}{role_slug}"
                 f"_gen{gen:03d}_fit{fitness:.3f}.mp4")
        out_path = out_dir / fname
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 30, (render_w, render_h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  {out_path.name}")

    for cond in conditions:
        _, strategy, budget, reeval = cond
        runs_c = by_condition[cond]

        all_finite: list[dict] = []
        for run in runs_c:
            for r in run["records"]:
                if r.get("fitness") is not None and np.isfinite(r["fitness"]):
                    r["_run"] = run
                    all_finite.append(r)

        checkpoints_by_run = {id(run): run["data_dir"] / "checkpoints" for run in runs_c}

        def _cp(rec: dict) -> Optional[Path]:
            return find_checkpoint(checkpoints_by_run[id(rec["_run"])], rec.get("genome_hash"))

        task_name = runs_c[0]["task"] if runs_c else "locomotion"
        cfg_sample = runs_c[0]["cfg"] if runs_c else {}

        if strategy.startswith("nsga2-"):
            if strategy == "nsga2-efficiency":
                objs = [(r["fitness"], r.get("control_cost", 0.0)) for r in all_finite]
                sec_key  = lambda r: r.get("control_cost", 0.0)
            else:
                objs = [(r["fitness"], -r.get("yz_symmetry", 0.0)) for r in all_finite]
                sec_key  = lambda r: -r.get("yz_symmetry", 0.0)

            ranks = pareto_rank(objs)
            front_recs = [r for r, rk in zip(all_finite, ranks) if rk == 0]
            front_with_cp = [(r, _cp(r)) for r in front_recs if _cp(r) is not None]
            front_with_cp.sort(key=lambda x: x[0]["fitness"])

            selected_roles: list[tuple] = []
            if front_with_cp:
                best_loco = min(front_with_cp, key=lambda x: x[0]["fitness"])
                selected_roles.append((best_loco[0], best_loco[1], "best locomotion"))
                best_sec = min(front_with_cp, key=lambda x: sec_key(x[0]))
                if best_sec[0].get("genome_hash") != best_loco[0].get("genome_hash"):
                    role_name = "best efficiency" if strategy == "nsga2-efficiency" else "best symmetry"
                    selected_roles.append((best_sec[0], best_sec[1], role_name))
                if len(front_with_cp) > 2:
                    mid_val = (sec_key(best_loco[0]) + sec_key(best_sec[0])) / 2.0
                    used = {r["genome_hash"] for r, _, _ in selected_roles}
                    candidates = [(r, cp) for r, cp in front_with_cp
                                  if r.get("genome_hash") not in used]
                    if candidates:
                        mid = min(candidates, key=lambda x: abs(sec_key(x[0]) - mid_val))
                        selected_roles.append((mid[0], mid[1], "middle"))

            for rec, cp, role in selected_roles:
                render_video(rec, cp, cfg_sample, OUT_DIR,
                             strategy, budget, task_name, role=role)
        else:
            all_finite.sort(key=lambda r: r["fitness"])
            selected: list[tuple] = []
            seen: set = set()
            for rec in all_finite:
                if len(selected) >= args.top_n:
                    break
                gh = rec.get("genome_hash")
                if gh in seen:
                    continue
                cp = _cp(rec)
                if cp is None:
                    continue
                seen.add(gh)
                selected.append((rec, cp))
            for rec, cp in selected:
                render_video(rec, cp, cfg_sample, OUT_DIR, strategy, budget, task_name)

print(f"\nAll outputs written to {OUT_DIR}")
