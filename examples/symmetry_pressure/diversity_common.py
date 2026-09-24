"""
Population genome diversity (mean pairwise tree edit distance) for the
symmetry-pressure sweep, shared by plot_run_curves.py and the amalgamated
per-task plots.

cppn genomes encode a CPPN network (nodes/connections), not the robot body
tree, so they're decoded to their phenotype graph first via
genome_adapter._cppn_decode — that way diversity is measured in phenotype
space for every genome type, and is directly comparable across the
tree/tree_symmetric/cppn sweep.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from genome_adapter import _cppn_decode
from sweep_common import checkpoints_dir


# ── Tree edit distance (Zhang-Shasha) ────────────────────────────────────────

def _genome_to_ordered_tree(genome: dict) -> dict:
    """Convert genome dict to a nested ordered labeled tree.

    Children are sorted by attachment face name (alphabetically) to give a
    deterministic ordering.  Node labels are the first character of the
    module type: C, B, or H.
    """
    nodes = genome["nodes"]
    edges = genome["edges"]

    children: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in edges:
        children[str(e["parent"])].append((e["face"], str(e["child"])))
    for pid in children:
        children[pid].sort(key=lambda x: x[0])

    core_id = next(nid for nid, n in nodes.items() if n["type"] == "CORE")

    def build(nid: str) -> dict:
        return {
            "label": nodes[nid]["type"][0],
            "children": [build(cid) for _, cid in children.get(nid, [])],
        }

    return build(core_id)


def _graph_to_ordered_tree(graph, core_id: int = 0) -> dict:
    """Same shape as _genome_to_ordered_tree, but for a decoded CPPN
    phenotype (an nx.DiGraph from genome_adapter._cppn_decode), so cppn
    diversity is measured in phenotype space like tree/tree_symmetric."""
    children: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for parent, child, data in graph.edges(data=True):
        children[parent].append((data["face"], child))
    for pid in children:
        children[pid].sort(key=lambda x: x[0])

    def build(nid: int) -> dict:
        return {
            "label": graph.nodes[nid]["type"][0],
            "children": [build(cid) for _, cid in children.get(nid, [])],
        }

    return build(core_id)


def _prepare(tree: dict) -> tuple[list[str], list[int], list[int]]:
    labels: list[str] = []
    lml: list[int] = []
    lml_rightmost: dict[int, int] = {}

    def visit(node: dict) -> int:
        child_idxs = [visit(c) for c in node["children"]]
        i = len(labels)
        labels.append(node["label"])
        lml.append(lml[child_idxs[0]] if child_idxs else i)
        lml_rightmost[lml[i]] = i
        return i

    visit(tree)
    return labels, lml, sorted(lml_rightmost.values())


def _zhang_shasha(
    t1: tuple[list[str], list[int], list[int]],
    t2: tuple[list[str], list[int], list[int]],
) -> int:
    labs1, lml1, kr1 = t1
    labs2, lml2, kr2 = t2
    n1, n2 = len(labs1), len(labs2)

    if n1 == 0:
        return n2
    if n2 == 0:
        return n1

    td = [[0] * n2 for _ in range(n1)]

    for k1 in kr1:
        for k2 in kr2:
            l1, l2 = lml1[k1], lml2[k2]
            s1, s2 = k1 - l1 + 2, k2 - l2 + 2
            fd = [[0] * s2 for _ in range(s1)]

            for i in range(1, s1):
                fd[i][0] = fd[i - 1][0] + 1
            for j in range(1, s2):
                fd[0][j] = fd[0][j - 1] + 1

            for i in range(1, s1):
                for j in range(1, s2):
                    ni, nj = l1 + i - 1, l2 + j - 1
                    c = 0 if labs1[ni] == labs2[nj] else 1

                    if lml1[ni] == l1 and lml2[nj] == l2:
                        fd[i][j] = min(
                            fd[i - 1][j] + 1,
                            fd[i][j - 1] + 1,
                            fd[i - 1][j - 1] + c,
                        )
                        td[ni][nj] = fd[i][j]
                    else:
                        pi = lml1[ni] - l1
                        pj = lml2[nj] - l2
                        fd[i][j] = min(
                            fd[i - 1][j] + 1,
                            fd[i][j - 1] + 1,
                            fd[pi][pj] + td[ni][nj],
                        )

    return td[n1 - 1][n2 - 1]


def pairwise_ted_stats(genomes: list[dict], genome_type: str, max_modules: int) -> tuple[float, float]:
    """(mean, std) of pairwise tree edit distance over all pairs in a
    generation's population.
    """
    if len(genomes) < 2:
        return 0.0, 0.0
    if genome_type == "cppn":
        trees = []
        for g in genomes:
            graph = _cppn_decode(g, max_modules)
            if graph.number_of_nodes() == 0:
                continue
            trees.append(_graph_to_ordered_tree(graph))
    else:
        trees = [_genome_to_ordered_tree(g) for g in genomes]
    if len(trees) < 2:
        return 0.0, 0.0
    prepared = [_prepare(t) for t in trees]
    dists = []
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            dists.append(_zhang_shasha(prepared[i], prepared[j]))
    if not dists:
        return 0.0, 0.0
    return float(np.mean(dists)), float(np.std(dists))


def load_diversity_by_gen(
    run_dir: Path, task: str, genome_type: str, max_modules: int
) -> dict[int, tuple[float, float]]:
    """gen -> (mean, std) pairwise tree edit distance."""
    base = checkpoints_dir(run_dir, task)
    if not base.exists():
        return {}
    by_gen: dict[int, list[dict]] = defaultdict(list)
    for ckpt in sorted(base.iterdir()):
        meta_path, genome_path = ckpt / "meta.json", ckpt / "best_genome.json"
        if not meta_path.exists() or not genome_path.exists():
            continue
        gen = json.loads(meta_path.read_text())["gen"]
        by_gen[gen].append(json.loads(genome_path.read_text()))
    return {
        gen: pairwise_ted_stats(genomes, genome_type, max_modules)
        for gen, genomes in sorted(by_gen.items())
    }
