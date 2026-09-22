"""Manual test/visualization for sim_ariel.tree_edit_distance.

Generates a random TreeGenome, mutates it to produce a second genome,
computes the tree edit distance between the two, and plots both as
top-down trees side by side with the distance in the figure title.

Usage:
    uv run examples/d_social_learning/test_tree_edit_distance.py \
        [--seed N] [--out __data__/tree_edit_distance_test.png] [--timeout SECONDS]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).parent
for _p in [str(_THIS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from sim_ariel.morphology_ops import mutate, random_individual
from sim_ariel.tree_edit_distance import _DEFAULT_TIME_BUDGET_S, tree_edit_distance_breakdown

NODE_COLORS = {
    "CORE": "#c44e52",
    "BRICK": "#4c72b0",
    "HINGE": "#55a868",
}


def _tree_layout(graph: nx.DiGraph, root) -> dict:
    """Top-down layered layout: y = -depth, x spread by DFS leaf order.

    Not a real tidy-tree algorithm (no collision avoidance beyond leaf
    spacing), but good enough for these small (<=13 node) trees.
    """
    pos = {}
    next_x = [0.0]

    def place(node, depth):
        children = list(graph.successors(node))
        if not children:
            pos[node] = (next_x[0], -depth)
            next_x[0] += 1.0
            return
        for child in children:
            place(child, depth + 1)
        xs = [pos[c][0] for c in children]
        pos[node] = (sum(xs) / len(xs), -depth)

    place(root, 0)
    return pos


def _rotation_short(rotation: str) -> str:
    return rotation.removeprefix("DEG_")


def _draw_tree(
    ax: plt.Axes, graph: nx.DiGraph, title: str, highlight_ids: set | None = None,
) -> None:
    """highlight_ids: nodes drawn with a black ring -- the nodes the computed
    edit path actually deletes/inserts/relabels, per tree_edit_distance_breakdown
    (not a naive same-id comparison -- the edit path's node correspondence
    isn't necessarily the identity mapping)."""
    highlight_ids = highlight_ids or set()
    root = next(n for n in graph.nodes if graph.in_degree(n) == 0)
    pos = _tree_layout(graph, root)

    node_colors = [NODE_COLORS.get(graph.nodes[n]["type"], "#999999") for n in graph.nodes]
    edgecolors = ["black" if n in highlight_ids else "none" for n in graph.nodes]
    linewidths = [2.5 if n in highlight_ids else 0 for n in graph.nodes]
    labels = {
        n: f"#{n}\n{graph.nodes[n]['type'][0]}{_rotation_short(graph.nodes[n]['rotation'])}"
        for n in graph.nodes
    }

    nx.draw(
        graph, pos, ax=ax, with_labels=True, labels=labels,
        node_color=node_colors, edgecolors=edgecolors, linewidths=linewidths,
        node_size=1100, font_size=7, font_weight="bold",
        font_color="white", arrows=True, arrowsize=12, edge_color="#888888",
    )
    edge_labels = {(u, v): d["face"] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=6)
    ax.set_title(f"{title}\n{graph.number_of_nodes()} nodes")
    ax.set_axis_off()


def _print_breakdown(breakdown: dict) -> None:
    print("\nEdit operations:")
    for label, ops in (("Node", breakdown["node_ops"]), ("Edge", breakdown["edge_ops"])):
        for op in ops:
            if op["op"] == "match":
                continue  # unchanged -- not part of the cost, skip for signal-to-noise
            print(f"  [{label}] {op['op']:<6} a={op['a']!r:>8} b={op['b']!r:>8}  cost={op['cost']}")

    totals: dict[str, float] = {}
    for ops in (breakdown["node_ops"], breakdown["edge_ops"]):
        for op in ops:
            if op["op"] == "match":
                continue
            totals[op["op"]] = totals.get(op["op"], 0.0) + op["cost"]
    print("Totals by operation:", {k: round(v, 2) for k, v in totals.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--out", type=str, default="__data__/tree_edit_distance_test.png")
    parser.add_argument(
        "--timeout", type=float, default=_DEFAULT_TIME_BUDGET_S,
        help=f"Wall-clock budget in seconds passed to tree_edit_distance's time_budget_s "
             f"(default {_DEFAULT_TIME_BUDGET_S}, matching the real experiment.py default). "
             "Raise it to see the distance converge closer to the true minimum on harder "
             "(more dissimilar) pairs, at the cost of a slower call.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    genome_a = random_individual()
    genome_b = mutate(genome_a)
    t0 = time.monotonic()
    breakdown = tree_edit_distance_breakdown(genome_a, genome_b, time_budget_s=args.timeout)
    elapsed = time.monotonic() - t0
    distance = breakdown["cost"]

    graph_a = TreeGenome.from_dict(genome_a).to_networkx()
    graph_b = TreeGenome.from_dict(genome_b).to_networkx()

    if distance is None:
        print(f"No complete edit path found within timeout={args.timeout}s -- try a larger --timeout.")
        highlight_a: set = set()
        highlight_b: set = set()
    else:
        highlight_a = {op["a"] for op in breakdown["node_ops"] if op["op"] in ("delete", "subst")}
        highlight_b = {op["b"] for op in breakdown["node_ops"] if op["op"] in ("insert", "subst")}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 6))
    _draw_tree(ax_a, graph_a, "Genome A (original)", highlight_ids=highlight_a)
    _draw_tree(ax_b, graph_b, "Genome B (mutated)", highlight_ids=highlight_b)
    distance_str = f"{distance:.2f}" if distance is not None else "N/A (timed out)"
    fig.suptitle(
        f"Tree edit distance = {distance_str}  (timeout={args.timeout}s, took {elapsed:.3f}s)",
        fontsize=14, fontweight="bold",
    )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name,
                    markerfacecolor=color, markersize=10)
        for name, color in NODE_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)

    print(f"Genome A: {len(genome_a['nodes'])} nodes")
    print(f"Genome B: {len(genome_b['nodes'])} nodes")
    print(f"Timeout: {args.timeout}s (took {elapsed:.3f}s)")
    print(f"Tree edit distance: {distance}")
    if distance is not None:
        _print_breakdown(breakdown)
    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    main()
