"""Prepare step for the interactive ancestry-tree viewer.

Reads one sweep run directory, builds the ancestry graph (see
`ancestry_tree_common.build_ancestry_graph`), renders one genome-diagram +
one phenotype-thumbnail image per unique genome (deduped by `genome_hash`,
since elite individuals are re-evaluated -- and get an identical genome --
across many generations), and writes a self-contained manifest.json + images/
bundle that `ancestry_tree_view.py` turns into a single interactive HTML page.

Usage:
    python ancestry_tree_prepare.py __data__/sympress_forward_tree_rep0_37568_0

    # Fast smoke test:
    python ancestry_tree_prepare.py <run_dir> --limit-generations 3 --jobs 2
"""

from __future__ import annotations

# MUST be first: mujoco's headless GL backend is resolved the first time
# `mujoco` is imported anywhere in this process (verified -- setting
# MUJOCO_GL afterwards has no effect). `shared`/`genome_adapter`, imported
# transitively below, both import mujoco at module level, so this has to run
# before any of today's imports pull them in. See phenotype_thumbnail.py's
# module docstring and examples/re_book/6_replay_best.py:33-34.
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")

import argparse
import datetime
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ancestry_tree_common as atc
import genome_diagram  # no mujoco import -- safe at module scope in the parent process
from rich.console import Console

console = Console()


def _render_one(task: dict) -> tuple[str, bool, bool]:
    """Worker: render genome + phenotype thumbnails for one unique genome_hash.

    Runs in a ProcessPoolExecutor worker (forked from a parent that already
    resolved MUJOCO_GL correctly before its own `import mujoco`). Returns
    (genome_hash, genome_ok, phenotype_ok).

    `task["diagram_genome"]` (used only for the schematic genome diagram) may
    differ from `task["genome_dict"]` (the real, valid genome, used for the
    phenotype render) when parent-diff highlighting is active for a tree
    genome: it's `build_tree_diff_best`'s union genome, which includes grey
    "deleted" ghost nodes that never correspond to an actually-evaluated
    body and so must never be handed to the MuJoCo phenotype renderer.
    """
    import phenotype_thumbnail

    genome_ok = True
    if not task["genome_out"].exists():
        try:
            genome_diagram.render_genome_thumbnail(
                task["diagram_genome"], task["genome_type"], task["genome_out"],
                size_px=task["size_px"], xlim=task["xlim"], ylim=task["ylim"],
                figsize_px=task["figsize_px"],
                node_status=task["node_status"], edge_status=task["edge_status"],
            )
        except Exception:
            genome_ok = False

    phenotype_ok = True
    if not task["phenotype_out"].exists():
        phenotype_ok = phenotype_thumbnail.render_phenotype_thumbnail(
            task["genome_dict"], task["genome_type"], task["phenotype_out"],
            size_px=task["size_px"], max_modules=task["max_modules"],
        )

    return task["genome_hash"], genome_ok, phenotype_ok


def _resolve_parent_genomes(
    node: "atc.TreeNode", graph: "atc.AncestryGraph", cache: dict[Path, dict]
) -> list[dict]:
    """Load up to 2 parent genome dicts for an origin node's reproduction
    edges (`node.parents`), for genome-diagram diff highlighting. Returns
    `[]` for gen-0 founders and unresolvable ("missing") parent edges --
    both normal; the caller renders with no diff overlay in that case.
    """
    genomes = []
    for p in node.parents:
        if p.get("kind") != "reproduction" or p.get("parent_node_id") is None:
            continue
        parent_node = graph.nodes.get(p["parent_node_id"])
        if parent_node is None or parent_node.checkpoint_dir is None:
            continue
        genome_path = parent_node.checkpoint_dir / "best_genome.json"
        if not genome_path.exists():
            continue
        if genome_path not in cache:
            cache[genome_path] = json.loads(genome_path.read_text())
        genomes.append(cache[genome_path])
    return genomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--task", type=str, default=None, help="Override task auto-detection")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <run_dir>/analysis/ancestry_tree/")
    parser.add_argument("--image-size", type=int, default=220)
    parser.add_argument("--max-modules", type=int, default=None,
                        help="Default: read from run_config.json. Overriding changes what a "
                             "CPPN genome decodes to relative to what was actually evolved.")
    parser.add_argument("--limit-generations", type=int, default=None,
                        help="Only process generations [0, N] -- fast smoke test.")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-render even if images already exist")
    parser.add_argument("--no-genome-images", action="store_true")
    parser.add_argument("--no-phenotype-images", action="store_true")
    args = parser.parse_args()

    layout = atc.discover_run_layout(args.run_dir, task_override=args.task)
    console.rule(f"[bold magenta]Ancestry Tree Prepare[/bold magenta] — {layout.run_dir.name}")
    console.log(f"task={layout.task} genome_type={layout.genome_type} data_dir={layout.data_dir}")

    max_modules = args.max_modules
    if max_modules is None:
        max_modules = layout.run_config.get("max_modules", 25)
    elif max_modules != layout.run_config.get("max_modules", max_modules):
        console.log(
            f"[yellow]WARNING: --max-modules={max_modules} overrides run_config.json's "
            f"max_modules={layout.run_config.get('max_modules')}. A CPPN genome can decode "
            f"to a different body under a different max_modules.[/yellow]"
        )

    out_dir = args.out_dir or (layout.run_dir / "analysis" / "ancestry_tree")
    images_dir = out_dir / "images"
    genome_dir = images_dir / "genome"
    phenotype_dir = images_dir / "phenotype"
    genome_dir.mkdir(parents=True, exist_ok=True)
    phenotype_dir.mkdir(parents=True, exist_ok=True)

    rows = atc.load_run_data(layout.run_data_path)
    if args.limit_generations is not None:
        rows = [r for r in rows if r["gen"] <= args.limit_generations]
    console.log(f"Loaded {len(rows)} run_data.jsonl rows")

    pop = layout.run_config.get("pop")
    lam = layout.run_config.get("lam")
    if pop is None or lam is None:
        console.log(
            "[yellow]No pop/lam in run_config.json -- skipping reconstruction of "
            "invalid (never-evaluated) individuals.[/yellow]"
        )

    ckpt_index = atc.index_checkpoints(layout.checkpoints_dir)
    graph = atc.build_ancestry_graph(rows, ckpt_index, pop=pop, lam=lam)
    console.log(
        f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"{graph.num_generations} generations"
    )
    console.log(
        f"Checkpoints matched: {graph.checkpoint_matches}/{len(rows)} "
        f"(fitness mismatches: {graph.checkpoint_fitness_mismatches}, "
        f"orphan individuals: {graph.orphan_individuals})"
    )
    if graph.invalid_reconstructed:
        console.log(
            f"Reconstructed {graph.invalid_reconstructed} invalid (never-evaluated) "
            f"individuals from ind_id gaps -- shown as red placeholder dots"
        )
    if graph.passthrough_created:
        console.log(
            f"Bridged {graph.passthrough_created} \"still alive, not re-evaluated\" gaps "
            f"with passthrough placeholders (lanes: {graph.num_lanes}) -- guarantees every "
            f"edge spans exactly one generation"
        )

    # Dedup rendering by genome_hash: first node with a given hash + checkpoint wins.
    # (Also always that individual's *origin* node -- nodes are inserted
    # origin-gen-first per individual in build_ancestry_graph, and a
    # non-origin real occurrence shares its origin's hash by construction --
    # so `node.parents` below is exactly the reproduction-edge data we need.)
    render_jobs: dict[str, tuple[dict, Path, atc.TreeNode]] = {}
    for node in graph.nodes.values():
        if node.checkpoint_dir is None or node.genome_hash is None:
            continue
        if node.genome_hash in render_jobs:
            continue
        genome_path = node.checkpoint_dir / "best_genome.json"
        if not genome_path.exists():
            continue
        render_jobs[node.genome_hash] = (json.loads(genome_path.read_text()), node.checkpoint_dir, node)

    console.log(f"Unique genomes to render: {len(render_jobs)} (dedup ratio: "
                f"{len(render_jobs) / max(graph.checkpoint_matches, 1):.2f})")

    # Parent-diff highlighting: new/changed nodes highlighted, deleted tree
    # nodes shown as grey ghosts (see genome_diagram.build_tree_diff_best /
    # build_cppn_diff). No overlay for gen-0 founders or an unresolvable
    # ("missing") parent edge -- both normal, rendered with no diff.
    _parent_genome_cache: dict[Path, dict] = {}
    diagram_genomes: dict[str, dict] = {}  # genome actually drawn (union, for a diffed tree genome)
    node_statuses: dict[str, dict] = {}
    edge_statuses: dict[str, dict] = {}
    n_diffed = 0
    for genome_hash, (genome_dict, _ckpt_dir, node) in render_jobs.items():
        diagram_genomes[genome_hash] = genome_dict
        parent_genomes = _resolve_parent_genomes(node, graph, _parent_genome_cache)
        if not parent_genomes:
            continue
        if layout.genome_type in ("tree", "tree_symmetric"):
            union_genome, node_status, edge_status = genome_diagram.build_tree_diff_best(
                genome_dict, parent_genomes
            )
            diagram_genomes[genome_hash] = union_genome
        elif layout.genome_type == "cppn":
            node_status, edge_status = genome_diagram.build_cppn_diff(genome_dict, parent_genomes)
        else:
            continue
        node_statuses[genome_hash] = node_status
        edge_statuses[genome_hash] = edge_status
        n_diffed += 1
    console.log(f"Parent-diff highlighting resolved for {n_diffed}/{len(render_jobs)} genomes "
                f"({len(render_jobs) - n_diffed} are founders or have no resolvable parent)")

    # One shared (xlim, ylim, figsize_px) for every genome diagram in this
    # run, so they all render at an identical scale with identical output
    # pixel dimensions -- otherwise each PNG is independently tight-cropped
    # to its own genome's extent, and node/edge sizes look inconsistent once
    # images of different native sizes get scaled to fit the same on-page
    # box (see `genome_diagram.render_genome_thumbnail`'s docstring). Uses
    # `diagram_genomes` (not the raw child genomes) so a diffed tree
    # genome's grey ghost subtree is accounted for in the shared canvas too.
    xlim = ylim = figsize_px = None
    if render_jobs:
        xlim, ylim, figsize_px = genome_diagram.compute_shared_window(
            list(diagram_genomes.values()), layout.genome_type, args.image_size,
        )
        console.log(f"Shared genome-diagram window: xlim={xlim} ylim={ylim} "
                    f"figsize_px=({figsize_px[0]:.0f}, {figsize_px[1]:.0f})")

    tasks = []
    for genome_hash, (genome_dict, _ckpt_dir, _node) in render_jobs.items():
        genome_out = genome_dir / f"{genome_hash}.png"
        phenotype_out = phenotype_dir / f"{genome_hash}.png"
        if args.force:
            for p in (genome_out, phenotype_out):
                p.unlink(missing_ok=True)
        tasks.append({
            "genome_hash": genome_hash,
            "genome_dict": genome_dict,
            "diagram_genome": diagram_genomes[genome_hash],
            "genome_type": layout.genome_type,
            "genome_out": genome_out,
            "phenotype_out": phenotype_out,
            "size_px": args.image_size,
            "max_modules": max_modules,
            "xlim": xlim, "ylim": ylim, "figsize_px": figsize_px,
            "node_status": node_statuses.get(genome_hash),
            "edge_status": edge_statuses.get(genome_hash),
        })

    rendered_ok: dict[str, tuple[bool, bool]] = {}
    if tasks:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for genome_hash, genome_ok, phenotype_ok in pool.map(_render_one, tasks):
                if args.no_genome_images:
                    genome_ok = False
                if args.no_phenotype_images:
                    phenotype_ok = False
                rendered_ok[genome_hash] = (genome_ok, phenotype_ok)

    n_genome_fail = sum(1 for g, p in rendered_ok.values() if not g)
    n_phenotype_fail = sum(1 for g, p in rendered_ok.values() if not p)
    console.log(f"Genome renders failed: {n_genome_fail}/{len(tasks)}  "
                f"Phenotype renders failed: {n_phenotype_fail}/{len(tasks)}")

    # Assign image paths back onto nodes.
    n_with_images = 0
    for node in graph.nodes.values():
        gh = node.genome_hash
        if gh is None or gh not in rendered_ok:
            continue
        genome_ok, phenotype_ok = rendered_ok[gh]
        genome_ok = genome_ok and not args.no_genome_images
        phenotype_ok = phenotype_ok and not args.no_phenotype_images
        if genome_ok:
            node.genome_image = f"images/genome/{gh}.png"
        if phenotype_ok:
            node.phenotype_image = f"images/phenotype/{gh}.png"
        node.has_images = bool(genome_ok or phenotype_ok)
        if node.has_images:
            n_with_images += 1

    finite_fitness = [n.fitness for n in graph.nodes.values() if n.fitness is not None]
    run_meta = {
        "run_dir": str(layout.run_dir),
        "task": layout.task,
        "genome_type": layout.genome_type,
        "strategy_type": layout.run_config.get("strategy_type"),
        "repeat_evals": layout.run_config.get("repeat_evals"),
        "pop": layout.run_config.get("pop"),
        "lam": layout.run_config.get("lam"),
        "budget": layout.run_config.get("budget"),
        "max_modules": max_modules,
        "seed": layout.run_config.get("seed"),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prepare_args": {
            "image_size": args.image_size,
            "limit_generations": args.limit_generations,
            "max_modules_override": args.max_modules,
        },
        "num_generations": graph.num_generations,
        "num_lanes": graph.num_lanes,
        "num_nodes": len(graph.nodes),
        "num_nodes_with_images": n_with_images,
        "num_invalid_nodes": graph.invalid_reconstructed,
        "num_passthrough_nodes": graph.passthrough_created,
        "num_unique_genome_hashes": len(render_jobs),
        "fitness_min": min(finite_fitness) if finite_fitness else None,
        "fitness_max": max(finite_fitness) if finite_fitness else None,
        "fitness_lower_is_better": True,
        "image_size_px": args.image_size,
    }

    manifest_path = out_dir / "manifest.json"
    atc.write_manifest(manifest_path, run_meta, list(graph.nodes.values()), graph.edges)
    console.log(f"[bold green]Manifest written -> {manifest_path}[/bold green]")
    console.log(f"Nodes with images: {n_with_images}/{len(graph.nodes)}")
    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
