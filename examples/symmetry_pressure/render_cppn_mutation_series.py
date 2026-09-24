"""
Render random CPPN genomes and the same genomes after repeated mutation.

Genomes are drawn with genome_adapter._random_cppn_genome (6 decoder inputs +
bias) and kept only if valid (>= shared.MIN_HINGES hinges), as the initial
population does. Each is then mutated cumulatively with
genome_adapter._mutate_cppn_genome (one `mutate_one` operator per mutation,
weights `_CPPN_MUTATION_PROBS`), with no selection or validity retries, so
the images show pure mutation drift. Bodies are decoded with the chosen
MorphologyDecoderBestFirst flags (default: local+dist / per-module) and
rendered at spawn pose.

Output (default ../../__data__/cppn_mutation_series_<variant>_seed<seed>/):
    genome_<i>.png      2 rows (top-down, isometric) x one column per step
    overview_iso.png    one row per genome, one column per step
    genomes.json        {genome index: {step: genotype dict}}

Usage:
    python render_cppn_mutation_series.py
    python render_cppn_mutation_series.py --n 10 --steps 0 5 10 25 50
    python render_cppn_mutation_series.py --no-local-competition
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent))

import genome_adapter
import shared
from ariel.body_phenotypes.robogen_lite.config import ModuleType
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import MorphologyDecoderBestFirst
from graph_render import ISO, TOP, render_graph

console = Console()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n", type=int, default=20, help="number of random valid genomes")
parser.add_argument("--steps", type=int, nargs="+", default=[0, 10, 20, 30, 40, 50],
                    help="mutation counts to render (cumulative)")
parser.add_argument("--max-modules", type=int, default=25)
parser.add_argument("--no-distance-input", dest="distance_input", action="store_false")
parser.add_argument("--no-local-inputs", dest="local_inputs", action="store_false")
parser.add_argument("--no-local-competition", dest="local_competition", action="store_false")
parser.add_argument("--thumb-px", type=int, default=220)
parser.add_argument("--out", type=Path, default=None)
args = parser.parse_args()

if MorphologyDecoderBestFirst.num_inputs(args.distance_input, args.local_inputs) != genome_adapter.NUM_CPPN_INPUTS:
    # _random_cppn_genome/_mutate_cppn_genome use the adapter's shared
    # IdManager, which is sized for NUM_CPPN_INPUTS decoder inputs.
    sys.exit("This input encoding needs a different genome input count; only 6-input encodings are supported.")

VARIANT = (("local" if args.local_inputs else "xyz") + ("+dist" if args.distance_input else "")
           + ("_per-module" if args.local_competition else "_global"))


def decode(genome: Genome) -> nx.DiGraph:
    return MorphologyDecoderBestFirst(
        genome, args.max_modules,
        distance_input=args.distance_input,
        local_inputs=args.local_inputs,
        local_competition=args.local_competition,
    ).decode()


def describe(genome: Genome, graph: nx.DiGraph, step: int) -> str:
    hinges = sum(1 for _, d in graph.nodes(data=True) if d["type"] == ModuleType.HINGE.name)
    hidden = sum(1 for n in genome.nodes.values() if n.typ == "hidden")
    flag = "" if hinges >= shared.MIN_HINGES else "  INVALID"
    return (f"{step} mutations{flag}\n"
            f"modules {graph.number_of_nodes()}  hinges {hinges}\n"
            f"hidden {hidden}  conns {len(genome.connections)}")


def main() -> None:
    out = args.out or (Path(__file__).resolve().parents[2] / "__data__"
                       / f"cppn_mutation_series_{VARIANT.replace('+', '_')}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    steps = sorted(set(args.steps))
    series: list[list[tuple[int, Genome, nx.DiGraph]]] = []
    drawn = 0
    while len(series) < args.n:
        genome = genome_adapter._random_cppn_genome()
        drawn += 1
        graph = decode(genome)
        if sum(1 for _, d in graph.nodes(data=True) if d["type"] == ModuleType.HINGE.name) < shared.MIN_HINGES:
            continue
        frames, done = [], 0
        for step in steps:
            for _ in range(step - done):
                genome = genome_adapter._mutate_cppn_genome(genome)
            done = step
            frames.append((step, genome.copy(), decode(genome)))
        series.append(frames)
    console.log(f"{args.n} valid genomes from {drawn} random draws ({args.n / drawn:.0%} valid)")

    cols = len(steps)
    for i, frames in enumerate(series):
        fig, axes = plt.subplots(2, cols, figsize=(cols * 2.3, 2 * 2.5 + 0.5))
        for c, (step, genome, graph) in enumerate(frames):
            for r, (az, el) in enumerate([TOP, ISO]):
                ax = axes[r, c]
                ax.imshow(render_graph(graph, az, el, args.thumb_px))
                ax.axis("off")
                if r == 0:
                    ax.set_title(describe(genome, graph, step), fontsize=7)
        fig.suptitle(f"{VARIANT}  genome {i}  (seed {args.seed})   rows: top-down, isometric", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"genome_{i:02d}.png", dpi=100)
        plt.close(fig)
        console.log(f"genome {i}: " + ", ".join(f"{s}:{g.number_of_nodes()}" for s, _, g in frames))

    fig, axes = plt.subplots(len(series), cols, figsize=(cols * 1.8, len(series) * 1.9 + 0.5), squeeze=False)
    for i, frames in enumerate(series):
        for c, (step, genome, graph) in enumerate(frames):
            ax = axes[i, c]
            ax.imshow(render_graph(graph, *ISO, args.thumb_px // 2))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{step} mutations", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"genome {i}", fontsize=8)
    fig.suptitle(f"{VARIANT}  (seed {args.seed})  isometric", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "overview_iso.png", dpi=100)
    plt.close(fig)

    (out / "genomes.json").write_text(json.dumps(
        {i: {step: genome.to_dict() for step, genome, _ in frames} for i, frames in enumerate(series)}
    ))
    console.log(f"[green]Wrote {out}[/green]")


if __name__ == "__main__":
    main()
