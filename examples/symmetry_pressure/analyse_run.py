"""
Analyse a completed symmetry-pressure run and produce plots.

Usage:
    python examples/symmetry_pressure/analyse_run.py --run-dir __data__/gecko_locomotion/
    python examples/symmetry_pressure/analyse_run.py --run-dir __data__/gecko_locomotion/ --strategy nsga2-symmetry
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Analyse a symmetry-pressure run")
parser.add_argument("--run-dir", required=True, type=Path,
                    help="Directory containing run_data.jsonl (e.g. __data__/gecko_locomotion/)")
parser.add_argument("--strategy", type=str, default=None,
                    help="Override strategy-type (reads run_config.json if omitted)")
args = parser.parse_args()

RUN_DIR  = args.run_dir.resolve()
JSONL    = RUN_DIR / "run_data.jsonl"
OUT_DIR  = RUN_DIR / "analysis"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ── Load config ───────────────────────────────────────────────────────────────

config: dict[str, Any] = {}
config_path = RUN_DIR / "run_config.json"
if config_path.exists():
    with config_path.open() as fh:
        config = json.load(fh)

strategy: str = args.strategy or config.get("strategy_type", "plus")
brain_budget: int = config.get("brain_budget", 0)

# ── Load JSONL ────────────────────────────────────────────────────────────────

records: list[dict[str, Any]] = []
with JSONL.open() as fh:
    for line in fh:
        line = line.strip()
        if line:
            records.append(json.loads(line))

if not records:
    raise SystemExit(f"No records in {JSONL}")

# Group records by generation
by_gen: dict[int, list[dict]] = defaultdict(list)
for r in records:
    by_gen[r["gen"]].append(r)

generations = sorted(by_gen.keys())

print(f"Loaded {len(records)} records across {len(generations)} generations")
print(f"Strategy: {strategy}  brain_budget: {brain_budget}")

# ── Plot 1 — Outer Fitness Curve ──────────────────────────────────────────────

gen_mean:    list[float] = []
gen_std:     list[float] = []
running_best: list[float] = []
best_so_far = float("inf")

for g in generations:
    fits = [r["fitness"] for r in by_gen[g]
            if r["fitness"] is not None and np.isfinite(r["fitness"])]
    if fits:
        gen_mean.append(float(np.mean(fits)))
        gen_std.append(float(np.std(fits)))
        best_so_far = min(best_so_far, min(fits))
    else:
        gen_mean.append(float("nan"))
        gen_std.append(0.0)
    running_best.append(best_so_far)

fig, ax = plt.subplots(figsize=(9, 5))
gen_arr = np.array(generations)
mean_arr = np.array(gen_mean)
std_arr  = np.array(gen_std)
ax.plot(gen_arr, mean_arr, label="Mean fitness", color="steelblue")
ax.fill_between(gen_arr, mean_arr - std_arr, mean_arr + std_arr,
                alpha=0.25, color="steelblue")
ax.plot(gen_arr, running_best, label="Running best", color="crimson", linestyle="--")
ax.set_xlabel("Outer generation")
ax.set_ylabel("Fitness (lower is better)")
ax.set_title(f"Outer Fitness Curve  [{strategy}]")
ax.legend()
ax.annotate("lower is better", xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=8, color="grey")
fig.tight_layout()
fig.savefig(OUT_DIR / "fitness_curve.png", dpi=150)
plt.close(fig)
print("  fitness_curve.png")

# ── Plot 2 — Inner Learning Curve ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
cmap = plt.get_cmap("Blues")
n_gens = len(generations)
plotted = 0
for gi, g in enumerate(generations):
    recs_g = by_gen[g]
    best_rec = min(
        (r for r in recs_g if r["fitness"] is not None and np.isfinite(r["fitness"])),
        key=lambda r: r["fitness"],
        default=None,
    )
    if best_rec is None:
        continue
    lc = best_rec.get("learning_curve", [])
    if not lc:
        continue
    color = cmap(0.2 + 0.8 * gi / max(n_gens - 1, 1))
    ax.plot(lc, color=color, alpha=0.85, linewidth=0.8)
    plotted += 1

sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=generations[0], vmax=generations[-1]))
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Outer generation")
ax.set_xlabel("Inner CMA-ES generation")
ax.set_ylabel("Best fitness so far")
ax.set_title(f"Inner Learning Curves (best body per gen)  [{strategy}]")
fig.tight_layout()
fig.savefig(OUT_DIR / "inner_learning_curve.png", dpi=150)
plt.close(fig)
print("  inner_learning_curve.png")

# ── Plot 3 — Morphological Diversity (yz_symmetry proxy) ─────────────────────

gen_sym_dist_mean: list[float] = []
gen_sym_dist_std:  list[float] = []

for g in generations:
    syms = [r["yz_symmetry"] for r in by_gen[g]
            if r.get("yz_symmetry") is not None]
    if len(syms) < 2:
        gen_sym_dist_mean.append(float("nan"))
        gen_sym_dist_std.append(float("nan"))
        continue
    arr = np.array(syms)
    n = len(arr)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(abs(arr[i] - arr[j]))
    gen_sym_dist_mean.append(float(np.mean(dists)))
    gen_sym_dist_std.append(float(np.std(dists)))

fig, ax = plt.subplots(figsize=(9, 5))
dm = np.array(gen_sym_dist_mean)
ds = np.array(gen_sym_dist_std)
ax.plot(gen_arr, dm, color="darkorange", label="Mean pairwise |Δsymmetry|")
ax.fill_between(gen_arr, np.maximum(dm - ds, 0), dm + ds,
                alpha=0.25, color="darkorange")
ax.set_xlabel("Outer generation")
ax.set_ylabel("Mean pairwise |Δyz_symmetry|")
ax.set_title(f"Diversity (symmetry-proxy)  [{strategy}]")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "diversity_curve.png", dpi=150)
plt.close(fig)
print("  diversity_curve.png")

# ── Plot 4 — Symmetry Distribution Over Generations ──────────────────────────

sym_by_gen: list[list[float]] = []
gen_labels: list[str] = []
for g in generations:
    syms = [r["yz_symmetry"] for r in by_gen[g]
            if r.get("yz_symmetry") is not None]
    if syms:
        sym_by_gen.append(syms)
        gen_labels.append(str(g))

fig, ax = plt.subplots(figsize=(max(8, len(sym_by_gen) * 0.5), 5))
positions = list(range(1, len(sym_by_gen) + 1))
parts = ax.violinplot(sym_by_gen, positions=positions, showmedians=True)
for pc in parts.get("bodies", []):
    pc.set_facecolor("mediumpurple")
    pc.set_alpha(0.6)
ax.set_xticks(positions)
ax.set_xticklabels(gen_labels, rotation=45 if len(gen_labels) > 10 else 0, fontsize=7)
ax.set_xlabel("Outer generation")
ax.set_ylabel("yz_symmetry")
ax.set_title(f"Symmetry Distribution  [{strategy}]")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT_DIR / "symmetry_over_gens.png", dpi=150)
plt.close(fig)
print("  symmetry_over_gens.png")

# ── Plot 5 — Parent Lineage Graph ─────────────────────────────────────────────

G = nx.DiGraph()
fitness_map:  dict[Any, float] = {}
symmetry_map: dict[Any, float] = {}

for r in records:
    nid = r["ind_id"]
    if nid is None:
        continue
    G.add_node(nid, gen=r["gen"])
    fitness_map[nid]  = r["fitness"] if (r["fitness"] is not None and np.isfinite(r["fitness"])) else float("nan")
    symmetry_map[nid] = r.get("yz_symmetry", 0.0) or 0.0
    for pid in (r.get("parent_ids") or []):
        if pid is not None:
            G.add_edge(pid, nid)

nodes = list(G.nodes())
if nodes:
    gen_of: dict[Any, int] = {n: G.nodes[n].get("gen", 0) for n in nodes}
    rank_within: dict[int, list[Any]] = defaultdict(list)
    for n in nodes:
        rank_within[gen_of[n]].append(n)
    for g_nodes in rank_within.values():
        g_nodes.sort(key=lambda n: fitness_map.get(n, float("nan")) if not np.isnan(fitness_map.get(n, float("nan"))) else 1e9)

    pos: dict[Any, tuple[float, float]] = {}
    for n in nodes:
        g = gen_of[n]
        g_list = rank_within[g]
        y_idx = g_list.index(n)
        n_in_gen = len(g_list)
        pos[n] = (g, y_idx - (n_in_gen - 1) / 2.0)

    all_fits = [v for v in fitness_map.values() if not np.isnan(v)]
    vmin = min(all_fits) if all_fits else 0.0
    vmax = max(all_fits) if all_fits else 1.0
    node_colors = [fitness_map.get(n, vmax) for n in nodes]
    node_sizes  = [30 + 200 * symmetry_map.get(n, 0.0) for n in nodes]

    fig, ax = plt.subplots(figsize=(max(10, len(generations) * 0.8), 7))
    nx.draw_networkx_edges(G, pos=pos, ax=ax, arrows=True,
                           arrowsize=6, width=0.4, alpha=0.4,
                           edge_color="grey")
    sc = nx.draw_networkx_nodes(G, pos=pos, ax=ax,
                                node_color=node_colors, node_size=node_sizes,
                                cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.9)
    plt.colorbar(sc, ax=ax, label="Fitness (lower/darker=best)")
    ax.set_xlabel("Outer generation")
    ax.set_title(f"Lineage Graph  [{strategy}]  (node size ∝ symmetry)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lineage.png", dpi=150)
    plt.close(fig)
    print("  lineage.png")

# ── Plot 6 — Pareto Front (NSGA-II strategies only) ───────────────────────────

if strategy in ("nsga2-efficiency", "nsga2-symmetry"):
    from shared import pareto_rank  # type: ignore[import]

    for g in generations:
        recs_g = [r for r in by_gen[g]
                  if r["fitness"] is not None and np.isfinite(r["fitness"])]
        if not recs_g:
            continue

        if strategy == "nsga2-efficiency":
            objs = [(r["fitness"], r.get("control_cost", 0.0)) for r in recs_g]
            x_vals = [o[0] for o in objs]
            y_vals = [o[1] for o in objs]
            x_label = "Fitness (lower-left = better)"
            y_label = "Control cost (lower = less energy)"
        else:
            objs = [(r["fitness"], -r.get("yz_symmetry", 0.0)) for r in recs_g]
            x_vals = [r["fitness"] for r in recs_g]
            y_vals = [r.get("yz_symmetry", 0.0) for r in recs_g]
            x_label = "Fitness (lower-left = better)"
            y_label = "yz_symmetry (higher = more symmetric)"

        ranks = pareto_rank(objs)
        front_idx = [i for i, rk in enumerate(ranks) if rk == 0]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x_vals, y_vals, c="steelblue", alpha=0.6, s=30, label="All individuals")

        if front_idx:
            if strategy == "nsga2-efficiency":
                front_pts = sorted([(x_vals[i], y_vals[i]) for i in front_idx], key=lambda p: p[0])
            else:
                front_pts = sorted([(x_vals[i], y_vals[i]) for i in front_idx], key=lambda p: p[0])
            fx, fy = zip(*front_pts)
            ax.scatter(fx, fy, c="crimson", s=60, zorder=5, label=f"Pareto front (n={len(front_idx)})")
            ax.plot(fx, fy, c="crimson", linewidth=1.0)
            ax.annotate(f"n={len(front_idx)} on front", xy=(0.98, 0.97),
                        xycoords="axes fraction", ha="right", fontsize=9)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Pareto Front — gen {g}  [{strategy}]")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"pareto_front_gen{g:03d}.png", dpi=150)
        plt.close(fig)

    print(f"  pareto_front_gen*.png ({len(generations)} files)")

print(f"\nAll plots written to {OUT_DIR}")
