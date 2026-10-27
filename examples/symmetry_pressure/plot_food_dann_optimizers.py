"""
Compare optimisers for gecko_food_dann.py runs.

Each run directory (one per optimiser setup and seed, named
<optimizer>_p<pop>_r<repeats>_s<seed>) holds __data__/gecko_food_dann/ with
learning_curve.jsonl and final_eval.json. Left: validation score of the search
mean (food + progress over the 20 validation food sets) against generation,
thin lines per seed and the seed mean bold. Right: each seed's held-out score
of its selected controller, with the setup mean. Every setup spends the same
200 episodes per generation, so generations are equal budgets.

Usage:
  python plot_food_dann_optimizers.py <runs_dir> [--out food_dann_optimizers.png]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reference categorical palette, slots 1-3 in fixed order (validated light mode).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"cma": "CMA-ES", "sepcma": "sep-CMA-ES", "oes": "OpenAI-ES"}
ORDER = ["cma", "sepcma", "oes"]

parser = argparse.ArgumentParser()
parser.add_argument("runs_dir", type=Path)
parser.add_argument("--out", type=Path, default=Path("food_dann_optimizers.png"))
args = parser.parse_args()

runs: dict[str, list[dict]] = defaultdict(list)
for d in sorted(args.runs_dir.iterdir()):
    m = re.match(r"(cma|sepcma|oes)_p(\d+)_r(\d+)_s(\d+)$", d.name)
    data = d / "__data__" / "gecko_food_dann"
    if not m or not (data / "final_eval.json").exists():
        continue
    rows = [json.loads(line) for line in (data / "learning_curve.jsonl").open()]
    val = [(r["gen"], -r["val_fitness"]) for r in rows if "val_fitness" in r]
    final = json.load((data / "final_eval.json").open())["summary"]
    runs[m[1]].append({"pop": int(m[2]), "repeats": int(m[3]), "seed": int(m[4]), "val": val,
                       "final": -final["fitness"]["mean"], "food": final["waypoints_reached"]["mean"]})

setups = [k for k in ORDER if k in runs]
fig, (ax_curve, ax_final) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
for ax in (ax_curve, ax_final):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

end_labels: list[list] = []  # [y, x, text] of each setup's seed-mean line end
for i, key in enumerate(setups):
    color, rs = COLORS[i], runs[key]
    label = f"{LABELS[key]} ({rs[0]['pop']}×{rs[0]['repeats']})"
    for r in rs:
        g, v = zip(*r["val"], strict=True)
        ax_curve.plot(g, v, color=color, linewidth=1, alpha=0.35)
    n = min(len(r["val"]) for r in rs)
    gens = [g for g, _ in rs[0]["val"][:n]]
    mean = np.mean([[v for _, v in r["val"][:n]] for r in rs], axis=0)
    ax_curve.plot(gens, mean, color=color, linewidth=2, label=label)
    end_labels.append([float(mean[-1]), gens[-1], LABELS[key]])

    finals = [r["final"] for r in rs]
    x = np.full(len(finals), i) + np.linspace(-0.08, 0.08, len(finals))
    ax_final.scatter(x, finals, s=36, color=color, edgecolor="white", linewidth=1.5, zorder=3)
    ax_final.hlines(np.mean(finals), i - 0.25, i + 0.25, color=color, linewidth=2, zorder=2)
    ax_final.annotate(f"{np.mean(finals):.2f}", (i + 0.27, np.mean(finals)), va="center", fontsize=9, color=TEXT)

# Direct labels at the line ends, pushed apart so they never overlap.
lo, hi = ax_curve.get_ylim()
gap = 0.06 * (hi - lo)
end_labels.sort()
for j in range(1, len(end_labels)):
    end_labels[j][0] = max(end_labels[j][0], end_labels[j - 1][0] + gap)
for y, x, text in end_labels:
    ax_curve.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT)

ax_curve.set_xlabel("generation (200 episodes each)", color=MUTED, fontsize=9)
ax_curve.set_ylabel("validation score (food + progress)", color=MUTED, fontsize=9)
ax_curve.set_title("Search mean on 20 validation food sets", loc="left", fontsize=11, color=TEXT)
ax_curve.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=TEXT)
ax_curve.set_xlim(left=0)
ax_final.set_xticks(range(len(setups)), [LABELS[k] for k in setups])
ax_final.set_ylabel("held-out score (food + progress)", color=MUTED, fontsize=9)
ax_final.set_title("Selected controller, 10 held-out sets", loc="left", fontsize=11, color=TEXT)
ax_final.set_xlim(-0.5, len(setups) - 0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=150, facecolor="white")
print(f"saved {args.out}")

print(f"{'setup':24s} {'seed':>4s} {'best val':>9s} {'held-out':>9s} {'food':>5s}")
for key in setups:
    for r in sorted(runs[key], key=lambda r: r["seed"]):
        print(f"{LABELS[key] + f' ({r['pop']}x{r['repeats']})':24s} {r['seed']:4d} "
              f"{max(v for _, v in r['val']):9.3f} {r['final']:9.3f} {r['food']:5.2f}")
