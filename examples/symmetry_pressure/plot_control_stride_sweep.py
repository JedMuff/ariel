"""
Fitness comparison for the control-stride sweep (slurm/run_control_stride_sweep.sh):
forward task, 3 genome types x strides {20, 50, 100}, 3 reps (seeds 42-44).
The stride-9 baseline is the forward runs of the symmetry-pressure sweep with
the same seeds (rep0-2), which ran at shared.py's default CONTROL_STEP_FREQ.

Produces, under <stride-root>/analysis/:
  fitness_curves.png
    2 x 3 grid: rows = per-generation best / mean fitness, cols = genome type;
    one line per stride, mean ± 1 SEM across reps (same aggregation as
    plot_task_genome_summary.py).
  fitness_by_stride.png
    4 x n_strides grid: cols = stride; rows = mean fitness per genome type,
    mean tree_symmetric - tree, best fitness per genome type, best
    tree_symmetric - tree.
  final_best.png
    best fitness at the last generation, one dot per rep + group mean, per
    genome type and stride.
and prints the final-generation numbers as a table.

Usage:
    python plot_control_stride_sweep.py
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sweep_common import GENOME_TYPES, load_fitness_by_gen, parse_run_tag, run_config, run_data_path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK = "forward"
BASELINE_STRIDE = 9  # shared.CONTROL_STEP_FREQ, used by the symmetry-pressure sweep
BASELINE_REPS = {0, 1, 2}  # seeds 42-44, matching the stride sweep

parser = argparse.ArgumentParser()
parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
parser.add_argument("--baseline-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_symmetry_pressure_sweep")
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()

# Ordinal blue ramp (light = fast control, dark = slow control).
STRIDE_COLOR = {9: "#a6c8f2", 20: "#6aa6ec", 25: "#3987e5", 33: "#2470cc", 50: "#1c5cab", 100: "#0d366b"}
STRIDE_MARKER = {9: "o", 20: "s", 25: "v", 33: "P", 50: "^", 100: "D"}
GENOME_LABEL = {"tree": "tree", "tree_symmetric": "tree (symmetric)", "cppn": "cppn"}
# Same genome colors as plot_task_genome_summary.py.
GENOME_COLOR = {"tree": "#2a78d6", "tree_symmetric": "#eb6834", "cppn": "#1baf7a"}
DIFF_COLOR = "#3d3d3a"
CI_ALPHA = 0.15

_STRIDE_TAG_RE = re.compile(
    r"^ctrlstride_forward_(?P<genome>tree_symmetric|tree|cppn)_s(?P<stride>\d+)_rep(?P<rep>\d+)_(?P<job>\d+)_\d+$"
)


def stride_label(stride: int) -> str:
    base = " (baseline)" if stride == BASELINE_STRIDE else ""
    return f"stride {stride} (~{500 / stride:.0f} Hz){base}"


# ── Discovery ─────────────────────────────────────────────────────────────

def discover_runs() -> dict[tuple[str, int], list[Path]]:
    """(genome, stride) -> run dirs with a run_data.jsonl.

    A (genome, stride, rep) rerun under a later array job supersedes the
    earlier one; dirs without run_data.jsonl (failed jobs) are ignored. The
    stride-9 baseline ran with the ANN brain, so it is only added when the
    stride runs are ANN too."""
    runs: dict[tuple[str, int], list[Path]] = defaultdict(list)
    latest: dict[tuple[str, int, int], tuple[int, Path]] = {}
    brains = set()

    for d in sorted(args.stride_root.glob("ctrlstride_*")):
        m = _STRIDE_TAG_RE.match(d.name)
        if not (m and d.is_dir() and run_data_path(d, TASK).exists()):
            continue
        stride = int(m["stride"])
        cfg = run_config(d, TASK)
        assert cfg.get("control_step_freq") == stride, \
            f"{d.name}: run_config stride {cfg.get('control_step_freq')} != tag stride {stride}"
        brains.add(cfg.get("brain", "ann"))
        key = (m["genome"], stride, int(m["rep"]))
        job = int(m["job"])
        if key in latest:
            print(f"[info] {d.name if job < latest[key][0] else latest[key][1].name} superseded by a later job")
        if key not in latest or job > latest[key][0]:
            latest[key] = (job, d)
    for (genome, stride, _), (_, d) in sorted(latest.items()):
        runs[(genome, stride)].append(d)

    if brains != {"ann"}:
        print(f"[info] stride runs use brain {sorted(brains)}; skipping the ANN stride-{BASELINE_STRIDE} baseline")
        return runs

    for d in sorted(args.baseline_root.glob("sympress_forward_*")):
        info = parse_run_tag(d.name)
        if info and d.is_dir() and info["rep"] in BASELINE_REPS:
            runs[(info["genome"], BASELINE_STRIDE)].append(d)

    return runs


def group_stats(per_rep_series: list[dict[int, float]]) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± 1 SEM across reps, generation-aligned (as in plot_task_genome_summary.py)."""
    by_gen: dict[int, list[float]] = defaultdict(list)
    for series in per_rep_series:
        for g, v in series.items():
            by_gen[g].append(v)
    gens = sorted(by_gen)
    means = np.array([np.mean(by_gen[g]) for g in gens])
    sems = np.array([stats.sem(by_gen[g]) if len(by_gen[g]) > 1 else 0.0 for g in gens])
    return gens, means, means - sems, means + sems


def per_rep_series(run_dirs: list[Path]) -> tuple[list[dict[int, float]], list[dict[int, float]], list[str]]:
    """Per-rep {gen: mean fitness}, {gen: best fitness}, and names of runs skipped for missing data."""
    mean_series, best_series, missing = [], [], []
    for run_dir in run_dirs:
        by_gen = load_fitness_by_gen(run_dir, TASK)
        if not by_gen:
            missing.append(run_dir.name)
            continue
        mean_series.append({g: float(np.mean(vs)) for g, vs in by_gen.items()})
        best_series.append({g: float(np.min(vs)) for g, vs in by_gen.items()})
    return mean_series, best_series, missing


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_curves(data: dict, strides: list[int], out_path: Path) -> None:
    fig, axes = plt.subplots(2, len(GENOME_TYPES), figsize=(15, 8.5), sharex=True, sharey="row")
    for col, genome in enumerate(GENOME_TYPES):
        for row, kind in enumerate(["best", "mean"]):
            ax = axes[row, col]
            for stride in strides:
                if (genome, stride) not in data:
                    continue
                series = data[(genome, stride)][kind]
                gens, m, lo, hi = group_stats(series)
                color = STRIDE_COLOR[stride]
                ax.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA, linewidth=0)
                ax.plot(gens, m, color=color, linewidth=2, label=stride_label(stride),
                        marker=STRIDE_MARKER[stride], markevery=5, markersize=5)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(GENOME_LABEL[genome])
            if row == 1:
                ax.set_xlabel("Generation")
            if col == 0:
                ax.set_ylabel(f"{kind.capitalize()} fitness (lower is better)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Control-stride sweep, forward task — mean ± 1 SEM across reps\n"
                 "(stride = physics steps between controller updates; physics at 500 Hz)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_by_stride(data: dict, strides: list[int], out_path: Path) -> None:
    """Columns = stride. Per fitness kind (mean, best): a row with one line per
    genome type, then a row with tree_symmetric - tree (negative = symmetric
    better), whose band is the unpaired SEM sqrt(sem_sym^2 + sem_tree^2)."""
    rows = [("mean", False), ("mean", True), ("best", False), ("best", True)]
    fig, axes = plt.subplots(len(rows), len(strides), figsize=(4.2 * len(strides), 13),
                             sharex=True, sharey="row")
    for col, stride in enumerate(strides):
        for row, (kind, is_diff) in enumerate(rows):
            ax = axes[row, col]
            ax.grid(alpha=0.3)
            if is_diff:
                sym, tree = data.get(("tree_symmetric", stride)), data.get(("tree", stride))
                if sym and tree:
                    g_s, m_s, lo_s, _ = group_stats(sym[kind])
                    g_t, m_t, lo_t, _ = group_stats(tree[kind])
                    gens = sorted(set(g_s) & set(g_t))
                    i_s = [g_s.index(g) for g in gens]
                    i_t = [g_t.index(g) for g in gens]
                    diff = m_s[i_s] - m_t[i_t]
                    se = np.hypot((m_s - lo_s)[i_s], (m_t - lo_t)[i_t])
                    ax.fill_between(gens, diff - se, diff + se, color=DIFF_COLOR, alpha=CI_ALPHA, linewidth=0)
                    ax.plot(gens, diff, color=DIFF_COLOR, linewidth=2, label="tree (symmetric) − tree")
                ax.axhline(0, color="#888888", linewidth=1, zorder=1)
            else:
                for genome in GENOME_TYPES:
                    if (genome, stride) not in data:
                        continue
                    gens, m, lo, hi = group_stats(data[(genome, stride)][kind])
                    color = GENOME_COLOR[genome]
                    ax.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA, linewidth=0)
                    ax.plot(gens, m, color=color, linewidth=2, label=GENOME_LABEL[genome])
            if row == 0:
                ax.set_title(stride_label(stride))
            if row == len(rows) - 1:
                ax.set_xlabel("Generation")
            if col == 0:
                ax.set_ylabel(f"{kind.capitalize()} fitness\n"
                              + ("sym − tree (< 0: sym better)" if is_diff else "(lower is better)"))
    handles, labels = [], []
    for ax in (axes[0, 0], axes[1, 0]):
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labels += lab
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Control-stride sweep, forward task — genome types per stride\n"
                 "(mean ± 1 SEM across reps; difference rows: ± unpaired SEM)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def final_best(series: list[dict[int, float]]) -> list[float]:
    return [s[max(s)] for s in series]


def plot_final(data: dict, strides: list[int], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(GENOME_TYPES), figsize=(13, 4.5), sharey=True)
    for ax, genome in zip(axes, GENOME_TYPES):
        for i, stride in enumerate(strides):
            if (genome, stride) not in data:
                continue
            vals = final_best(data[(genome, stride)]["best"])
            color = STRIDE_COLOR[stride]
            jitter = np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(i + jitter, vals, color=color, marker=STRIDE_MARKER[stride], s=40, zorder=3,
                       edgecolor="white", linewidth=1)
            ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=color, linewidth=2.5, zorder=2)
        ax.set_xticks(range(len(strides)), [f"{s}\n~{500 / s:.0f} Hz" for s in strides])
        ax.set_xlabel("Control stride")
        ax.set_title(GENOME_LABEL[genome])
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Final-generation best fitness\n(lower is better)")
    fig.suptitle("Final best fitness per rep (dots) and mean across reps (bar)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def print_table(data: dict, strides: list[int]) -> None:
    print(f"\n{'genome':<16}{'stride':>7}{'n':>3}  {'final best (mean ± sd)':>24}  {'final pop mean (mean ± sd)':>28}")
    for genome in GENOME_TYPES:
        for stride in strides:
            if (genome, stride) not in data:
                continue
            best = final_best(data[(genome, stride)]["best"])
            mean = final_best(data[(genome, stride)]["mean"])
            print(f"{genome:<16}{stride:>7}{len(best):>3}  "
                  f"{np.mean(best):>12.3f} ± {np.std(best, ddof=1) if len(best) > 1 else 0:<9.3f}  "
                  f"{np.mean(mean):>16.3f} ± {np.std(mean, ddof=1) if len(mean) > 1 else 0:<9.3f}")


def main() -> None:
    runs = discover_runs()
    data = {}
    for key, run_dirs in runs.items():
        mean_series, best_series, missing = per_rep_series(run_dirs)
        for name in missing:
            print(f"[warn] no run_data.jsonl for {name}, skipping")
        if mean_series:
            data[key] = {"mean": mean_series, "best": best_series}

    strides = sorted({s for _, s in data})
    out_dir = args.stride_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves_path = out_dir / "fitness_curves.png"
    plot_curves(data, strides, curves_path)
    print(f"Saved -> {curves_path}")

    by_stride_path = out_dir / "fitness_by_stride.png"
    plot_by_stride(data, strides, by_stride_path)
    print(f"Saved -> {by_stride_path}")

    final_path = out_dir / "final_best.png"
    plot_final(data, strides, final_path)
    print(f"Saved -> {final_path}")

    print_table(data, strides)


if __name__ == "__main__":
    main()
