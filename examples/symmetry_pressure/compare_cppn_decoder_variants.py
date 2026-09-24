"""
Compare CPPN decoder variants on random initial genomes.

Crosses the three `MorphologyDecoderBestFirst` flags
    distance_input     (+dist: distance-from-core inputs)
    local_inputs       (local: face direction + outwardness instead of xyz)
    local_competition  (per-module competition, breadth-first)
For each input encoding, --n random genomes are drawn exactly as
genome_adapter._random_cppn_genome draws them (the input count differs per
encoding, so genomes differ between encodings); both competition modes decode
the same genomes.

Per variant it reports, over all --n genomes:
    valid       fraction with >= shared.MIN_HINGES hinges (what the initial
                population keeps)
    modules     median module count, and fraction at --max-modules
and over the valid ones only (the bodies evolution starts from):
    depth       median longest core->leaf path
    leaves      median number of free ends
    chains      fraction with <= 2 leaves and >= 10 modules
    one_sided   median |centroid - core| / rms radius of module centres:
                0 = spread evenly around the core, ~1 = all on one side
    elongation  median sqrt(l1/l2) of the module-centre covariance
                (1 = round, large = stretched along one axis)
and renders the first 25 valid bodies of each variant, top-down and
isometric, at spawn pose, as 5x5 contact sheets.

Output (default ../../__data__/cppn_decoder_variants_seed<seed>/):
    metrics.csv, metrics.md, <variant>_top.png, <variant>_iso.png

Usage:
    python compare_cppn_decoder_variants.py
    python compare_cppn_decoder_variants.py --seed 7 --n 100
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")

import argparse
import csv
import itertools
import random
import time
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
from graph_render import render_graph
from ariel.body_phenotypes.robogen_lite.collision_utils import IDENTITY, BodyCollisionChecker
from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE, ModuleType
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import MorphologyDecoderBestFirst

console = Console()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n", type=int, default=200, help="random genomes per input encoding")
parser.add_argument("--max-modules", type=int, default=25)
parser.add_argument("--sheet-size", type=int, default=25, help="bodies per contact sheet")
parser.add_argument("--thumb-px", type=int, default=200)
parser.add_argument("--out", type=Path, default=None)
args = parser.parse_args()


def variant_name(distance: bool, local: bool, competition: bool) -> str:
    inputs = ("local" if local else "xyz") + ("+dist" if distance else "")
    return f"{inputs} / {'per-module' if competition else 'global'}"


def module_centres(graph: nx.DiGraph) -> np.ndarray:
    checker = BodyCollisionChecker()
    core = graph.nodes[IDX_OF_CORE]
    checker.add_module(IDX_OF_CORE, IDENTITY, core["type"], core["rotation"])
    for u, v in nx.bfs_edges(graph, IDX_OF_CORE):
        frame = checker.child_frame(u, graph.edges[u, v]["face"])
        checker.add_module(v, frame, graph.nodes[v]["type"], graph.nodes[v]["rotation"])
    return np.array([checker.module_centre(m) for m in graph.nodes])


def body_metrics(graph: nx.DiGraph) -> dict:
    n = graph.number_of_nodes()
    hinges = sum(1 for _, d in graph.nodes(data=True) if d["type"] == ModuleType.HINGE.name)
    depth = max(nx.shortest_path_length(graph, IDX_OF_CORE).values())
    leaves = sum(1 for m in graph if graph.out_degree(m) == 0 and m != IDX_OF_CORE)
    one_sided = elongation = np.nan
    if n >= 3:
        pts = module_centres(graph)
        core = pts[list(graph.nodes).index(IDX_OF_CORE)]
        rel = pts - core
        rms = np.sqrt(np.mean(np.sum(rel**2, axis=1)))
        one_sided = float(np.linalg.norm(rel.mean(axis=0)) / rms) if rms > 0 else np.nan
        eig = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
        elongation = float(np.sqrt(eig[0] / eig[1])) if eig[1] > 1e-12 else np.inf
    return {
        "modules": n, "hinges": hinges, "valid": hinges >= shared.MIN_HINGES,
        "depth": depth, "leaves": leaves, "chain": leaves <= 2 and n >= 10,
        "one_sided": one_sided, "elongation": elongation,
    }


def contact_sheet(items: list[tuple[int, nx.DiGraph]], title: str, azimuth: float, elevation: float, path: Path) -> None:
    cols = 5
    rows = max(1, int(np.ceil(len(items) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.3 + 0.4))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, (idx, graph) in zip(np.atleast_1d(axes).flat, items):
        ax.imshow(render_graph(graph, azimuth, elevation, args.thumb_px))
        ax.set_title(f"#{idx}  n={graph.number_of_nodes()}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    out = args.out or Path(__file__).resolve().parents[2] / "__data__" / f"cppn_decoder_variants_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for distance, local in itertools.product([False, True], repeat=2):
        n_inputs = MorphologyDecoderBestFirst.num_inputs(distance, local)
        n_outputs = genome_adapter.NUM_CPPN_OUTPUTS
        random.seed(args.seed)
        np.random.seed(args.seed)
        id_manager = IdManager(
            node_start=n_inputs + 1 + n_outputs - 1,
            innov_start=(n_inputs + 1) * n_outputs - 1,
        )
        genomes = [genome_adapter._random_cppn_genome(n_inputs, id_manager) for _ in range(args.n)]

        for competition in [False, True]:
            name = variant_name(distance, local, competition)
            metrics, graphs = [], []
            t0 = time.perf_counter()
            for g in genomes:
                graph = MorphologyDecoderBestFirst(
                    g, args.max_modules,
                    distance_input=distance, local_inputs=local, local_competition=competition,
                ).decode()
                graphs.append(graph)
            ms = 1000 * (time.perf_counter() - t0) / len(genomes)
            metrics = [body_metrics(gr) for gr in graphs]

            valid_metrics = [m for m in metrics if m["valid"]] or [
                {k: np.nan for k in metrics[0]}
            ]

            def med(key, ms_=valid_metrics):
                return float(np.nanmedian([m[key] for m in ms_]))

            def frac(key, ms_=valid_metrics):
                return float(np.mean([bool(m[key]) for m in ms_]))

            row = {
                "variant": name, "valid": frac("valid", metrics),
                "modules": med("modules", metrics),
                "at_max": float(np.mean([m["modules"] == args.max_modules for m in metrics])),
                "depth": med("depth"), "leaves": med("leaves"), "chains": frac("chain"),
                "one_sided": med("one_sided"), "elongation": med("elongation"), "decode_ms": ms,
            }
            rows.append(row)
            console.log(row)

            valid = [(i, gr) for i, (gr, m) in enumerate(zip(graphs, metrics)) if m["valid"]][: args.sheet_size]
            stem = name.replace(" / ", "_").replace("+", "_")
            summary = (f"{name}   valid {row['valid']:.0%}  chains {row['chains']:.0%}  "
                       f"one-sided {row['one_sided']:.2f}  elongation {row['elongation']:.1f}")
            if valid:
                contact_sheet(valid, f"{summary}\ntop-down", 90.0, -90.0, out / f"{stem}_top.png")
                contact_sheet(valid, f"{summary}\nisometric", 45.0, -35.0, out / f"{stem}_iso.png")

    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = ("| variant | valid | modules (med, all) | at max (all) | depth (med, valid) | leaves (med, valid) "
              "| chains (valid) | one-sided (valid) | elongation (valid) | decode ms |")
    lines = [header, "|" + "---|" * 10]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['valid']:.0%} | {r['modules']:.0f} | {r['at_max']:.0%} | {r['depth']:.0f} | "
            f"{r['leaves']:.0f} | {r['chains']:.0%} | {r['one_sided']:.2f} | {r['elongation']:.1f} | {r['decode_ms']:.1f} |"
        )
    (out / "metrics.md").write_text("\n".join(lines) + "\n")
    console.print("\n".join(lines))
    console.log(f"[green]Wrote {out}[/green]")


if __name__ == "__main__":
    main()
