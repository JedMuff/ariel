"""
Per-skill score curves for the multi-skill tasks (multidirection, turn_avg),
comparing genome types.

Each skill's score is the negated per-skill fitness in run_data.jsonl (score
minus penalties; higher is better). For every task, produces
analysis/<task>/skill_curves.png with one column per skill and two rows:
  top    : population mean of the skill's score per generation
  bottom : the skill's score in each generation's best individual (by task fitness)
Every line is the mean across reps with a ±1 SEM band (as in
plot_task_genome_summary.py). multidirection also gets
analysis/multidirection/skill_curves_ranked.png: each individual's 5
direction scores sorted from best to worst before averaging, since which
direction a body is good at is arbitrary relative to its shape.

Usage:
    python plot_skill_curves.py --sweep-root ../../__data__/ariel_symmetry_pressure_sweep_dann
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sweep_common import parse_run_tag, run_data_path

REPO_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--sweep-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_symmetry_pressure_sweep")
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()

TASK_SKILLS = {
    "multidirection": ["dir0", "dir1", "dir2", "dir3", "dir4"],
    "turn_avg": ["fwd", "left", "right"],
}
SKILL_LABEL = {
    "dir0": "dir0 (0°)", "dir1": "dir1 (72°)", "dir2": "dir2 (144°)",
    "dir3": "dir3 (216°)", "dir4": "dir4 (288°)",
    "fwd": "forward", "left": "turn left", "right": "turn right",
}
GENOME_TYPES = ["tree", "tree_symmetric"]
GENOME_COLOR = {"tree": "#2a78d6", "tree_symmetric": "#eb6834"}
GENOME_LABEL = {"tree": "tree", "tree_symmetric": "tree (symmetric)"}
CI_ALPHA = 0.15


def load_records(run_dir: Path, task: str) -> dict[int, list[dict]]:
    by_gen: dict[int, list[dict]] = defaultdict(list)
    path = run_data_path(run_dir, task)
    if path.exists():
        for line in path.open():
            rec = json.loads(line)
            if rec.get("per_skill"):
                by_gen[rec["gen"]].append(rec)
    return by_gen


def skill_scores(rec: dict, skills: list[str]) -> np.ndarray:
    return np.array([-rec["per_skill"][s]["fitness"] for s in skills])


def per_rep_curves(by_gen: dict[int, list[dict]], skills: list[str]) -> dict[str, dict[int, np.ndarray]]:
    """gen -> per-skill vector, for the population mean, the best individual,
    and the population mean of each individual's descending-sorted scores."""
    out = {"mean": {}, "best": {}, "ranked": {}}
    for g, recs in by_gen.items():
        scores = np.array([skill_scores(r, skills) for r in recs])
        out["mean"][g] = scores.mean(axis=0)
        out["best"][g] = skill_scores(min(recs, key=lambda r: r["fitness"]), skills)
        out["ranked"][g] = (-np.sort(-scores, axis=1)).mean(axis=0)
    return out


def group_stats(series: list[dict[int, np.ndarray]], k: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Mean and SEM across reps of skill k, generation-aligned."""
    by_gen: dict[int, list[float]] = defaultdict(list)
    for s in series:
        for g, v in s.items():
            by_gen[g].append(v[k])
    gens = sorted(by_gen)
    means = np.array([np.mean(by_gen[g]) for g in gens])
    sems = np.array([stats.sem(by_gen[g]) if len(by_gen[g]) > 1 else 0.0 for g in gens])
    return gens, means, sems


def plot_grid(task: str, rows: list[tuple[str, str]], col_labels: list[str],
              curves: dict[str, list[dict]], out_path: Path, title: str) -> None:
    n_cols = len(col_labels)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.4 * n_cols, 3.0 * len(rows)),
                             sharex=True, sharey="row", squeeze=False)
    for r, (kind, row_label) in enumerate(rows):
        for k, col_label in enumerate(col_labels):
            ax = axes[r, k]
            for genome in GENOME_TYPES:
                series = [c[kind] for c in curves.get(genome, [])]
                if not series:
                    continue
                gens, m, sem = group_stats(series, k)
                ax.fill_between(gens, m - sem, m + sem, color=GENOME_COLOR[genome], alpha=CI_ALPHA, linewidth=0)
                ax.plot(gens, m, color=GENOME_COLOR[genome], linewidth=2, label=GENOME_LABEL[genome])
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(col_label)
            if k == 0:
                ax.set_ylabel(f"{row_label}\nskill score")
            if r == len(rows) - 1:
                ax.set_xlabel("Generation")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{title} — task={task}\n(mean across reps, ± 1 SEM)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[{task}] Saved -> {out_path}")


def main() -> None:
    for task, skills in TASK_SKILLS.items():
        curves: dict[str, list[dict]] = defaultdict(list)
        for d in sorted(args.sweep_root.glob(f"sympress_{task}_*")):
            info = parse_run_tag(d.name)
            if info is None or info["genome"] not in GENOME_TYPES:
                continue
            by_gen = load_records(d, task)
            if by_gen:
                curves[info["genome"]].append(per_rep_curves(by_gen, skills))
        if not curves:
            print(f"[{task}] no runs found, skipping")
            continue

        out_dir = args.sweep_root / "analysis" / task
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_grid(task, [("mean", "Population mean"), ("best", "Best individual")],
                  [SKILL_LABEL[s] for s in skills], curves, out_dir / "skill_curves.png",
                  "Per-skill score over generations")
        if task == "multidirection":
            plot_grid(task, [("ranked", "Population mean")],
                      ["best direction", "2nd", "3rd", "4th", "worst direction"], curves,
                      out_dir / "skill_curves_ranked.png",
                      "Direction scores ranked per individual (best to worst)")


if __name__ == "__main__":
    main()
