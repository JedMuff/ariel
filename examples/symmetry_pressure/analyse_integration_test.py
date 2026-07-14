"""
Analyse all integration-test runs and produce combined plots + per-run plots.

Usage:
    python examples/symmetry_pressure/analyse_integration_test.py \
        --results-dir data/symmetry_integration_test \
        [--top-n 3]

Outputs (inside <results-dir>/analysis/):
    fitness_curves.png          — fitness vs generation, all strategies overlaid
    diversity_curves.png        — pairwise |Δsymmetry| diversity, all strategies
    symmetry_over_gens.png      — violin per generation per strategy
    lineage_<strategy>.png      — parent lineage graph per run
    timing_summary.png          — wall time per gen + per-individual estimates
    timing_table.txt            — text table of time estimates
    video_gen*_body*_fit*.mp4   — top-N videos per run (requires MuJoCo + cv2)
"""

import argparse
import json
import os
import sys
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

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=Path, required=True,
                    help="Root dir containing inttest_* subdirs")
parser.add_argument("--top-n", type=int, default=3,
                    help="Number of top individuals to render as video (per run)")
parser.add_argument("--run-ids", nargs="+", metavar="ID",
                    help="Only include runs whose directory name contains one of these substrings")
parser.add_argument("--new-only", action="store_true",
                    help="Only include runs that have the c_hinge field (new fitness schema)")
args = parser.parse_args()

RESULTS_DIR = args.results_dir.resolve()
TOP_N       = args.top_n
OUT_DIR     = RESULTS_DIR / "analysis"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ── Discover runs ──────────────────────────────────────────────────────────────

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


def load_run(run_dir: Path) -> Optional[dict]:
    data_dir = run_dir / "__data__" / "gecko_locomotion"
    jsonl    = data_dir / "run_data.jsonl"
    timing   = data_dir / "timing.jsonl"
    config   = data_dir / "run_config.json"
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

    strategy   = cfg.get("strategy_type", run_dir.name)
    new_schema = bool(records and "c_hinge" in records[0])
    return {
        "run_dir":    run_dir,
        "data_dir":   data_dir,
        "strategy":   strategy,
        "cfg":        cfg,
        "records":    records,
        "timing":     timing_records,
        "new_schema": new_schema,
    }


runs: list[dict] = []
for candidate in sorted(RESULTS_DIR.iterdir()):
    if not candidate.is_dir():
        continue
    if args.run_ids and not any(rid in candidate.name for rid in args.run_ids):
        continue
    run = load_run(candidate)
    if run is None:
        continue
    if args.new_only and not run["new_schema"]:
        print(f"  [skip] {candidate.name} — old schema (no c_hinge)")
        continue
    runs.append(run)
    schema_tag = " [new]" if run["new_schema"] else " [old]"
    print(f"Loaded {run['strategy']:22s}  "
          f"{len(run['records'])} records, {len(run['timing'])} gen timings  "
          f"({candidate.name}){schema_tag}")

if not runs:
    raise SystemExit(f"No valid runs found under {RESULTS_DIR}")


def by_gen(records: list[dict]) -> dict[int, list[dict]]:
    d: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        d[r["gen"]].append(r)
    return d


# ── Plot 1 — Fitness curves (all strategies on one axes) ─────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax_mean, ax_best = axes

for run in runs:
    strat  = run["strategy"]
    color  = STRATEGY_COLORS.get(strat, "grey")
    label  = STRATEGY_LABELS.get(strat, strat)
    bg     = by_gen(run["records"])
    gens   = sorted(bg.keys())
    means, stds, bests = [], [], []
    best_so_far = float("inf")
    for g in gens:
        fits = [r["fitness"] for r in bg[g]
                if r["fitness"] is not None and np.isfinite(r["fitness"])]
        if fits:
            means.append(float(np.mean(fits)))
            stds.append(float(np.std(fits)))
            best_so_far = min(best_so_far, min(fits))
        else:
            means.append(float("nan"))
            stds.append(0.0)
        bests.append(best_so_far)

    ga = np.array(gens)
    ma = np.array(means)
    sa = np.array(stds)

    ax_mean.plot(ga, ma, label=label, color=color)
    ax_mean.fill_between(ga, ma - sa, ma + sa, alpha=0.15, color=color)
    ax_best.plot(ga, bests, label=label, color=color)

for ax, title in zip(axes, ["Mean ± std fitness", "Running-best fitness"]):
    ax.set_xlabel("Outer generation")
    ax.set_ylabel("Fitness (lower is better)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.annotate("lower is better", xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=8, color="grey")

fig.suptitle("Integration test — fitness curves")
fig.tight_layout()
fig.savefig(OUT_DIR / "fitness_curves.png", dpi=150)
plt.close(fig)
print("  fitness_curves.png")

# ── Plot 2 — Diversity (pairwise |Δsymmetry|) ────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

for run in runs:
    strat = run["strategy"]
    color = STRATEGY_COLORS.get(strat, "grey")
    label = STRATEGY_LABELS.get(strat, strat)
    bg    = by_gen(run["records"])
    gens  = sorted(bg.keys())
    dm, ds = [], []
    for g in gens:
        syms = [r["yz_symmetry"] for r in bg[g] if r.get("yz_symmetry") is not None]
        if len(syms) < 2:
            dm.append(float("nan"))
            ds.append(0.0)
            continue
        arr = np.array(syms)
        dists = [abs(arr[i] - arr[j]) for i in range(len(arr)) for j in range(i+1, len(arr))]
        dm.append(float(np.mean(dists)))
        ds.append(float(np.std(dists)))

    ga = np.array(gens)
    dma = np.array(dm)
    dsa = np.array(ds)
    ax.plot(ga, dma, label=label, color=color)
    ax.fill_between(ga, np.maximum(dma - dsa, 0), dma + dsa, alpha=0.15, color=color)

ax.set_xlabel("Outer generation")
ax.set_ylabel("Mean pairwise |Δyz_symmetry|")
ax.set_title("Morphological diversity (symmetry proxy)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "diversity_curves.png", dpi=150)
plt.close(fig)
print("  diversity_curves.png")

# ── Plot 3 — Symmetry distribution violin (per strategy, faceted) ─────────────

n_runs = len(runs)
fig, axes = plt.subplots(1, n_runs, figsize=(max(6, n_runs * 6), 5), sharey=True)
if n_runs == 1:
    axes = [axes]

for ax, run in zip(axes, runs):
    strat  = run["strategy"]
    color  = STRATEGY_COLORS.get(strat, "grey")
    label  = STRATEGY_LABELS.get(strat, strat)
    bg     = by_gen(run["records"])
    gens   = sorted(bg.keys())
    sym_by_gen, gen_labels = [], []
    for g in gens:
        syms = [r["yz_symmetry"] for r in bg[g] if r.get("yz_symmetry") is not None]
        if syms:
            sym_by_gen.append(syms)
            gen_labels.append(str(g))

    if sym_by_gen:
        positions = list(range(1, len(sym_by_gen) + 1))
        parts = ax.violinplot(sym_by_gen, positions=positions, showmedians=True)
        for pc in parts.get("bodies", []):
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
        ax.set_xticks(positions)
        ax.set_xticklabels(gen_labels,
                           rotation=45 if len(gen_labels) > 10 else 0, fontsize=7)

    ax.set_xlabel("Outer generation")
    ax.set_title(label)
    ax.set_ylim(0, 1)

axes[0].set_ylabel("yz_symmetry")
fig.suptitle("Symmetry distribution over generations")
fig.tight_layout()
fig.savefig(OUT_DIR / "symmetry_over_gens.png", dpi=150)
plt.close(fig)
print("  symmetry_over_gens.png")

# ── Plot 4 — Lineage graph (one per run) ──────────────────────────────────────

for run in runs:
    strat   = run["strategy"]
    records = run["records"]

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
    if not nodes:
        continue

    gen_of: dict[Any, int] = {n: G.nodes[n].get("gen", 0) for n in nodes}
    rank_within: dict[int, list[Any]] = defaultdict(list)
    for n in nodes:
        rank_within[gen_of[n]].append(n)
    for g_nodes in rank_within.values():
        g_nodes.sort(key=lambda n: fitness_map.get(n, float("nan"))
                     if not np.isnan(fitness_map.get(n, float("nan"))) else 1e9)

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

    generations = sorted(gen_of.values())
    fig, ax = plt.subplots(figsize=(max(10, len(set(generations)) * 0.8), 7))
    nx.draw_networkx_edges(G, pos=pos, ax=ax, arrows=True,
                           arrowsize=6, width=0.4, alpha=0.4, edge_color="grey")
    sc = nx.draw_networkx_nodes(G, pos=pos, ax=ax,
                                node_color=node_colors, node_size=node_sizes,
                                cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.9)
    plt.colorbar(sc, ax=ax, label="Fitness (lower/darker=best)")
    ax.set_xlabel("Outer generation")
    ax.set_title(f"Lineage graph  [{STRATEGY_LABELS.get(strat, strat)}]"
                 "  (node size ∝ symmetry)")
    ax.axis("off")
    fig.tight_layout()
    fname = f"lineage_{strat}.png"
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  {fname}")

# ── Plot 5 — Pareto fronts (NSGA-II strategies) ───────────────────────────────

nsga2_runs = [r for r in runs if r["strategy"].startswith("nsga2-")]

sys.path.insert(0, str(Path(__file__).parent))
from shared import pareto_rank  # type: ignore[import]

if nsga2_runs:

    # One panel per NSGA-II strategy, both in one figure
    fig, axes = plt.subplots(1, len(nsga2_runs),
                             figsize=(8 * len(nsga2_runs), 6),
                             squeeze=False)

    for ax, run in zip(axes[0], nsga2_runs):
        strat   = run["strategy"]
        records_all = [r for r in run["records"]
                       if r["fitness"] is not None and np.isfinite(r["fitness"])]
        gens    = sorted(set(r["gen"] for r in records_all))
        n_gens  = len(gens)
        cmap    = plt.get_cmap("plasma")
        gen_idx = {g: i for i, g in enumerate(gens)}

        # Scatter all points coloured by generation
        x_all = [-r["fitness"] for r in records_all]          # flip: right = better locomotion
        if strat == "nsga2-efficiency":
            y_all = [-r.get("control_cost", 0.0) for r in records_all]  # flip: up = less energy
        else:
            y_all = [r.get("yz_symmetry", 0.0) for r in records_all]    # up = more symmetric

        colors_pts = [cmap(gen_idx[r["gen"]] / max(n_gens - 1, 1)) for r in records_all]
        ax.scatter(x_all, y_all, c=colors_pts, alpha=0.55, s=22, zorder=2)

        # Global Pareto front across ALL generations
        # Objectives are (minimise -fitness, minimise cost/-symmetry) = (minimise neg_fitness, ...)
        # i.e. both objectives already "lower is better" in the minimisation sense before flipping
        if strat == "nsga2-efficiency":
            objs_all = [(r["fitness"], r.get("control_cost", 0.0)) for r in records_all]
        else:
            objs_all = [(r["fitness"], -r.get("yz_symmetry", 0.0)) for r in records_all]

        ranks_all = pareto_rank(objs_all)
        front_idx = [i for i, rk in enumerate(ranks_all) if rk == 0]

        if front_idx:
            fx = [x_all[i] for i in front_idx]
            fy = [y_all[i] for i in front_idx]
            # Sort by x for a clean staircase line
            sorted_front = sorted(zip(fx, fy), key=lambda p: p[0])
            sfx, sfy = zip(*sorted_front)
            ax.scatter(sfx, sfy, color="crimson", s=80, zorder=5,
                       label=f"Global Pareto front (n={len(front_idx)})", edgecolors="darkred", linewidths=0.5)
            # Staircase step line — better is top-right so step "post" works
            ax.step(sfx, sfy, where="post", color="crimson", linewidth=1.5, zorder=4)

        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gens[0], vmax=gens[-1]))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Generation")

        if strat == "nsga2-efficiency":
            ax.set_xlabel("Locomotion fitness →  (better right)")
            ax.set_ylabel("Energy efficiency →  (lower cost = up)")
        else:
            ax.set_xlabel("Locomotion fitness →  (better right)")
            ax.set_ylabel("Bilateral symmetry →  (more symmetric = up)")

        ax.set_title(f"Pareto front — {STRATEGY_LABELS.get(strat, strat)}\n"
                     "all individuals (colour = generation); global front in red")
        ax.legend(fontsize=8)

    fig.suptitle("NSGA-II: all individuals + global Pareto front  (top-right = better)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pareto_fronts.png", dpi=150)
    plt.close(fig)
    print("  pareto_fronts.png")

# ── Plot 7 — Timing summary ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax_wall, ax_per_ind, ax_cumul = axes

for run in runs:
    strat   = run["strategy"]
    color   = STRATEGY_COLORS.get(strat, "grey")
    label   = STRATEGY_LABELS.get(strat, strat)
    timings = run["timing"]
    if not timings:
        continue

    gens_t      = [t["gen"] for t in timings]
    wall_times  = [t["wall_time_s"] for t in timings]
    mean_evals  = [t["mean_eval_time_s"] for t in timings]
    n_inds      = [t["n_individuals"] for t in timings]
    cumulative  = np.cumsum(wall_times)

    ax_wall.plot(gens_t, wall_times, label=label, color=color, marker="o", markersize=3)
    ax_per_ind.plot(gens_t, mean_evals, label=label, color=color, marker="o", markersize=3)
    ax_cumul.plot(gens_t, cumulative / 60.0, label=label, color=color)

ax_wall.set_xlabel("Generation")
ax_wall.set_ylabel("Wall time (s)")
ax_wall.set_title("Wall time per generation")
ax_wall.legend(fontsize=8)

ax_per_ind.set_xlabel("Generation")
ax_per_ind.set_ylabel("Mean eval time (s/individual)")
ax_per_ind.set_title("Mean time per individual")
ax_per_ind.legend(fontsize=8)

ax_cumul.set_xlabel("Generation")
ax_cumul.set_ylabel("Cumulative wall time (min)")
ax_cumul.set_title("Cumulative runtime")
ax_cumul.legend(fontsize=8)

fig.suptitle("Timing summary")
fig.tight_layout()
fig.savefig(OUT_DIR / "timing_summary.png", dpi=150)
plt.close(fig)
print("  timing_summary.png")

# ── Timing text table ────────────────────────────────────────────────────────

lines = []
lines.append(f"{'Strategy':<22}  {'Gens':>4}  {'Inds/gen':>8}  "
             f"{'Mean s/ind':>10}  {'Mean s/gen':>10}  {'Total min':>10}")
lines.append("-" * 80)

for run in runs:
    strat   = run["strategy"]
    timings = run["timing"]
    if not timings:
        continue
    n_gens   = len(timings)
    mean_n   = float(np.mean([t["n_individuals"] for t in timings]))
    mean_ind = float(np.mean([t["mean_eval_time_s"] for t in timings]))
    mean_gen = float(np.mean([t["wall_time_s"] for t in timings]))
    total_m  = sum(t["wall_time_s"] for t in timings) / 60.0
    lines.append(f"{STRATEGY_LABELS.get(strat, strat):<22}  {n_gens:>4}  "
                 f"{mean_n:>8.1f}  {mean_ind:>10.1f}  {mean_gen:>10.1f}  {total_m:>10.1f}")

table_str = "\n".join(lines)
print("\n" + table_str)
(OUT_DIR / "timing_table.txt").write_text(table_str + "\n")
print("\n  timing_table.txt")

# ── Videos — top-N per run ────────────────────────────────────────────────────

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
            # Import genome_hash lazily to avoid hard dep at top
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from shared import genome_hash as gh_fn  # type: ignore[import]
                if gh_fn(stored) == genome_hash_val:
                    return cp
            except Exception:
                pass
    return None


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
        print("  [skip] mujoco not importable — skipping video render")
    elif not HAS_CV2:
        print("  [skip] cv2 not available — skipping video render")

if HAS_MUJOCO and HAS_CV2:
    sys.path.insert(0, str(Path(__file__).parent))
    from shared import (  # type: ignore[import]
        Network, fill_parameters, build_world_for_body,
    )
    from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state  # type: ignore[import]

    def render_video(rec: dict, cp: Path, cfg: dict, out_dir: Path, strat: str,
                     role: str = "") -> None:
        gen      = rec["gen"]
        fitness  = rec["fitness"]
        symmetry = rec.get("yz_symmetry", 0.0) or 0.0
        duration = float(cfg.get("dur", 10.0))

        with (cp / "best_genome.json").open() as fh:
            genome_dict = json.load(fh)
        weights = np.load(cp / "best_weights.npy")

        model, data, _target_mocap_id, _cam_name = build_world_for_body(
            genome_dict, reach_radius=0.20, arena_radius=3.0, add_food_target=False
        )

        input_dim = len(get_robot_state(data))
        network   = Network(input_size=input_dim, output_size=model.nu)
        fill_parameters(network, weights)

        render_h, render_w = 480, 640
        renderer = mujoco.Renderer(model, height=render_h, width=render_w)

        # Isometric tracking camera locked onto the robot core body
        core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
        if core_id < 0:
            core_id = 3  # fallback: robot1_core is almost always body 3
        track_cam = mujoco.MjvCamera()
        track_cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        track_cam.trackbodyid = core_id
        track_cam.distance    = 2.0   # zoomed out enough to see the whole body + some ground
        track_cam.azimuth     = 45.0  # isometric azimuth
        track_cam.elevation   = -25.0 # isometric elevation

        frames: list[np.ndarray] = []
        mujoco.mj_resetData(model, data)

        step = 0
        current_action = np.zeros(model.nu)
        while data.time < duration:
            if step % 50 == 0:
                state = get_robot_state(data).astype(np.float32)
                current_action = network.forward(model, data, state)
            data.ctrl[:] = current_action
            mujoco.mj_step(model, data)

            if step % 17 == 0:
                renderer.update_scene(data, camera=track_cam)
                frame = renderer.render().copy()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"{strat}  Gen {gen}  Fit {fitness:.3f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_bgr, f"Sym {symmetry:.3f}",
                            (render_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if role:
                    cv2.putText(frame_bgr, role,
                                (10, render_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 2)
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            step += 1

        renderer.close()

        if not frames:
            print(f"  No frames for gen={gen} fit={fitness:.3f}")
            return

        role_slug = f"_{role.lower().replace(' ', '_')}" if role else ""
        fname = f"video_{strat}{role_slug}_gen{gen:03d}_body{rec.get('ind_id') or 0:02d}_fit{fitness:.3f}.mp4"
        out_path = out_dir / fname
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 30, (render_w, render_h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  {out_path.name}")

    for run in runs:
        strat       = run["strategy"]
        cfg         = run["cfg"]
        records_run = run["records"]
        checkpoints = run["data_dir"] / "checkpoints"

        finite = [r for r in records_run
                  if r.get("fitness") is not None and np.isfinite(r["fitness"])]

        def _cp(rec: dict) -> Optional[Path]:
            return find_checkpoint(checkpoints, rec.get("genome_hash"))

        if strat.startswith("nsga2-"):
            # Build global Pareto front, then pick:
            #   1. best locomotion  (lowest fitness on front)
            #   2. best 2nd objective (lowest control_cost or highest symmetry on front)
            #   3. middle-front individual (closest to midpoint of front by 2nd objective)
            if strat == "nsga2-efficiency":
                objs = [(r["fitness"], r.get("control_cost", 0.0)) for r in finite]
                sec_key   = lambda r: r.get("control_cost", 0.0)   # minimise
                sec_best  = min
            else:
                objs = [(r["fitness"], -r.get("yz_symmetry", 0.0)) for r in finite]
                sec_key   = lambda r: -r.get("yz_symmetry", 0.0)  # minimise (i.e. maximise symmetry)
                sec_best  = min

            ranks = pareto_rank(objs)
            front_recs = [r for r, rk in zip(finite, ranks) if rk == 0]

            # Filter to those with checkpoints, then pick roles
            front_with_cp = [(r, _cp(r)) for r in front_recs if _cp(r) is not None]
            front_with_cp.sort(key=lambda x: x[0]["fitness"])  # sort by locomotion

            selected_roles: list[tuple[dict, Path, str]] = []
            if front_with_cp:
                # Best locomotion (lowest fitness = leftmost when negated)
                best_loco = min(front_with_cp, key=lambda x: x[0]["fitness"])
                selected_roles.append((best_loco[0], best_loco[1], "best locomotion"))

                # Best 2nd objective
                best_sec = min(front_with_cp, key=lambda x: sec_key(x[0]))
                if best_sec[0]["genome_hash"] != best_loco[0]["genome_hash"]:
                    if strat == "nsga2-efficiency":
                        selected_roles.append((best_sec[0], best_sec[1], "best efficiency"))
                    else:
                        selected_roles.append((best_sec[0], best_sec[1], "best symmetry"))

                # Middle: closest 2nd-objective value to midpoint between best_loco and best_sec
                if len(front_with_cp) > 2:
                    mid_val = (sec_key(best_loco[0]) + sec_key(best_sec[0])) / 2.0
                    used_hashes = {r["genome_hash"] for r, _, _ in selected_roles}
                    candidates = [(r, cp) for r, cp in front_with_cp
                                  if r["genome_hash"] not in used_hashes]
                    if candidates:
                        mid = min(candidates, key=lambda x: abs(sec_key(x[0]) - mid_val))
                        selected_roles.append((mid[0], mid[1], "middle"))

            print(f"\n  {strat}: {len(selected_roles)} individuals selected from global front "
                  f"({len(front_recs)} on front, {len(front_with_cp)} with checkpoints)")
            for rec, cp, role in selected_roles:
                sec_val = rec.get("control_cost", rec.get("yz_symmetry", 0.0))
                print(f"    [{role}] gen={rec['gen']}  fit={rec['fitness']:.3f}  "
                      f"2nd={sec_val:.3f}")
                render_video(rec, cp, cfg, OUT_DIR, strat, role=role)

        else:
            # Non-NSGA-II: just top-N by fitness as before
            finite.sort(key=lambda r: r["fitness"])
            selected: list[tuple[dict, Path]] = []
            seen: set = set()
            for rec in finite:
                if len(selected) >= TOP_N:
                    break
                gh = rec.get("genome_hash")
                if gh in seen:
                    continue
                cp = _cp(rec)
                if cp is None:
                    continue
                seen.add(gh)
                selected.append((rec, cp))

            print(f"\n  {strat}: {len(selected)} individuals with checkpoints")
            for rec, cp in selected:
                render_video(rec, cp, cfg, OUT_DIR, strat)

print(f"\nAll outputs written to {OUT_DIR}")
