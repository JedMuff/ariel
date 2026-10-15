"""Tree edit distance between two TreeGenome genomes.

Trees are unordered (children are distinguished only by which ``face`` they
attach to on their parent, not by sequence). Two distances share one cost
model (node type/rotation mismatch, node indel, edge face mismatch, edge
indel -- all 1.0):

- ``tree_edit_distance`` (novelty metric STRUCT): networkx graph edit
  distance under a wall-clock budget. Most pairs of real morphologies hit the
  budget, so the result is an approximate, load-dependent upper bound.
- ``top_down_tree_distance`` (novelty metric TOPDOWN): exact minimum cost
  over *top-down* mappings (a node can only be matched if its parent is
  matched to the other node's parent), by dynamic programming with an optimal
  child-subtree assignment at every node pair. Deterministic and ~3000x
  faster; it can't exploit mappings that skip levels (e.g. after a hoist
  mutation), so it is >= the true graph edit distance.
"""

from functools import lru_cache

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from ariel.ec.genotypes.tree.tree_genome import TreeGenome

_NODE_TYPE_MISMATCH_COST = 1.0
_NODE_ROTATION_MISMATCH_COST = 1.0
_NODE_INDEL_COST = 1.0
_EDGE_FACE_MISMATCH_COST = 1.0
_EDGE_INDEL_COST = 1.0
_DEFAULT_TIME_BUDGET_S = 1.0


def _node_subst_cost(n1: dict, n2: dict) -> float:
    if n1["type"] != n2["type"]:
        return _NODE_TYPE_MISMATCH_COST
    if n1["rotation"] != n2["rotation"]:
        return _NODE_ROTATION_MISMATCH_COST
    return 0.0


def _edge_subst_cost(e1: dict, e2: dict) -> float:
    return _EDGE_FACE_MISMATCH_COST if e1["face"] != e2["face"] else 0.0


def tree_edit_distance(
    genome_a: dict, genome_b: dict, time_budget_s: float = _DEFAULT_TIME_BUDGET_S,
) -> float:
    """Approximate graph edit distance between two genome dicts.

    Parameters
    ----------
    genome_a, genome_b:
        Dicts as returned by ``Individual.genotype_["morph"]`` (output of
        ``TreeGenome.to_dict()``).
    time_budget_s:
        Wall-clock budget passed through to ``nx.graph_edit_distance``'s
        native ``timeout``, which is checked deep inside its recursive
        branch-and-bound search (not just between top-level solutions), so
        it reliably bounds runtime even when the search never completes a
        single full edit path in time. Needed because exact GED is
        worst-case exponential: for two genuinely dissimilar trees (not a
        mutation pair), this search can otherwise run for minutes on even
        these tiny (<=13-node) trees. An earlier version used
        ``nx.optimize_graph_edit_distance`` and only checked the budget
        between yields -- that doesn't help when the first yield itself
        never arrives in time, and taking just its first (unbounded) yield
        measured up to ~3x too high vs. the true minimum on a real mutation
        pair. This is bounded either way and converges to the exact minimum
        well within the default budget for a related (e.g.
        parent/mutated-child) pair.

    Returns
    -------
    float
        Raw/unnormalized cost, same convention as
        ``sim_evogym.evogym_body_descriptors.aligned_hamming_distance``.
    """
    graph_a = TreeGenome.from_dict(genome_a).to_networkx()
    graph_b = TreeGenome.from_dict(genome_b).to_networkx()

    distance = nx.graph_edit_distance(
        graph_a,
        graph_b,
        node_subst_cost=_node_subst_cost,
        node_del_cost=lambda n: _NODE_INDEL_COST,
        node_ins_cost=lambda n: _NODE_INDEL_COST,
        edge_subst_cost=_edge_subst_cost,
        edge_del_cost=lambda e: _EDGE_INDEL_COST,
        edge_ins_cost=lambda e: _EDGE_INDEL_COST,
        timeout=time_budget_s,
    )
    if distance is None:
        # Confirmed reachable: the timeout can fire before the search
        # completes even one full edit path (observed with a very small
        # budget; plausible under load with the default too). Fall back to
        # the trivial "delete everything from A, insert everything into B"
        # upper bound rather than propagating None into callers that sort
        # or average these distances.
        distance = (
            graph_a.number_of_nodes() * _NODE_INDEL_COST
            + graph_b.number_of_nodes() * _NODE_INDEL_COST
            + graph_a.number_of_edges() * _EDGE_INDEL_COST
            + graph_b.number_of_edges() * _EDGE_INDEL_COST
        )
    return distance


def _rooted_tree(genome: dict) -> tuple[dict, dict, int]:
    nodes = {int(k): (v["type"], v["rotation"]) for k, v in genome["nodes"].items()}
    children: dict[int, list[tuple[int, str]]] = {k: [] for k in nodes}
    child_ids = set()
    for e in genome["edges"]:
        children[e["parent"]].append((e["child"], e["face"]))
        child_ids.add(e["child"])
    root = next(k for k in nodes if k not in child_ids)
    return nodes, children, root


def top_down_tree_distance(genome_a: dict, genome_b: dict) -> float:
    """Exact top-down tree edit distance under ``tree_edit_distance``'s costs.

    Roots are always matched. Matching node u to node v costs their node
    substitution cost plus an optimal assignment of u's child subtrees to v's:
    a matched child pair costs its own top-down distance plus the face
    mismatch of the two connecting edges; an unmatched child subtree is
    deleted/inserted whole, at node + parent-edge indel cost per node.

    Parameters
    ----------
    genome_a, genome_b:
        Dicts as returned by ``Individual.genotype_["morph"]``.

    Returns
    -------
    float
        Raw/unnormalized cost (0.0 for identical trees).
    """
    nodes_a, kids_a, root_a = _rooted_tree(genome_a)
    nodes_b, kids_b, root_b = _rooted_tree(genome_b)
    indel_unit = _NODE_INDEL_COST + _EDGE_INDEL_COST

    @lru_cache(maxsize=None)
    def size_a(u: int) -> int:
        return 1 + sum(size_a(c) for c, _ in kids_a[u])

    @lru_cache(maxsize=None)
    def size_b(v: int) -> int:
        return 1 + sum(size_b(c) for c, _ in kids_b[v])

    @lru_cache(maxsize=None)
    def dist(u: int, v: int) -> float:
        (type_u, rot_u), (type_v, rot_v) = nodes_a[u], nodes_b[v]
        if type_u != type_v:
            cost = _NODE_TYPE_MISMATCH_COST
        elif rot_u != rot_v:
            cost = _NODE_ROTATION_MISMATCH_COST
        else:
            cost = 0.0
        ca, cb = kids_a[u], kids_b[v]
        if not ca and not cb:
            return cost
        # Square assignment matrix: rows = u's children + one "insert" slot
        # per child of v, cols = v's children + one "delete" slot per child
        # of u (each child may only use its own indel slot).
        n, m = len(ca), len(cb)
        mat = np.full((n + m, n + m), np.inf)
        mat[n:, m:] = 0.0
        for i, (a, face_a) in enumerate(ca):
            for j, (b, face_b) in enumerate(cb):
                face_cost = _EDGE_FACE_MISMATCH_COST if face_a != face_b else 0.0
                mat[i, j] = dist(a, b) + face_cost
            mat[i, m + i] = indel_unit * size_a(a)
        for j, (b, _) in enumerate(cb):
            mat[n + j, j] = indel_unit * size_b(b)
        rows, cols = linear_sum_assignment(mat)
        return cost + float(mat[rows, cols].sum())

    return dist(root_a, root_b)


def tree_edit_distance_breakdown(
    genome_a: dict, genome_b: dict, time_budget_s: float = _DEFAULT_TIME_BUDGET_S,
) -> dict:
    """Like ``tree_edit_distance``, but returns the actual edit operations
    behind the cost instead of just the number.

    Diagnostic/exploratory use only (e.g. test_tree_edit_distance.py) --
    ``nx.optimize_edit_paths`` (the "advanced interface" under
    ``nx.graph_edit_distance``) does strictly more work to reconstruct the
    node/edge correspondence, so the hot path (``tree_edit_distance``, called
    O(n^2) times per generation) doesn't use this.

    Returns
    -------
    dict with keys:
        "cost": float total cost, or None if no complete edit path was found
            within the budget (mirrors ``nx.graph_edit_distance`` -- no
            worst-case-bound fallback here since this is for inspection, not
            for feeding into arithmetic).
        "node_ops": list of dicts, one per node in the correspondence:
            {"op": "match"|"subst"|"delete"|"insert", "a": id or None,
             "b": id or None, "cost": float}
            "a"/"b" are node ids in genome_a/genome_b respectively; the
            missing side is None for a delete (a-only) or insert (b-only).
        "edge_ops": list of dicts, same shape, one per edge in the
            correspondence, "a"/"b" holding (parent, child) id tuples.
    """
    graph_a = TreeGenome.from_dict(genome_a).to_networkx()
    graph_b = TreeGenome.from_dict(genome_b).to_networkx()

    best = None
    for node_path, edge_path, cost in nx.optimize_edit_paths(
        graph_a,
        graph_b,
        node_subst_cost=_node_subst_cost,
        node_del_cost=lambda n: _NODE_INDEL_COST,
        node_ins_cost=lambda n: _NODE_INDEL_COST,
        edge_subst_cost=_edge_subst_cost,
        edge_del_cost=lambda e: _EDGE_INDEL_COST,
        edge_ins_cost=lambda e: _EDGE_INDEL_COST,
        timeout=time_budget_s,
    ):
        best = (node_path, edge_path, cost)

    if best is None:
        return {"cost": None, "node_ops": [], "edge_ops": []}

    node_path, edge_path, cost = best

    node_ops = []
    for u, v in node_path:
        if u is None:
            node_ops.append({"op": "insert", "a": None, "b": v, "cost": _NODE_INDEL_COST})
        elif v is None:
            node_ops.append({"op": "delete", "a": u, "b": None, "cost": _NODE_INDEL_COST})
        else:
            op_cost = _node_subst_cost(graph_a.nodes[u], graph_b.nodes[v])
            node_ops.append({"op": "match" if op_cost == 0 else "subst", "a": u, "b": v, "cost": op_cost})

    edge_ops = []
    for e1, e2 in edge_path:
        if e1 is None:
            edge_ops.append({"op": "insert", "a": None, "b": e2, "cost": _EDGE_INDEL_COST})
        elif e2 is None:
            edge_ops.append({"op": "delete", "a": e1, "b": None, "cost": _EDGE_INDEL_COST})
        else:
            op_cost = _edge_subst_cost(graph_a.edges[e1], graph_b.edges[e2])
            edge_ops.append({"op": "match" if op_cost == 0 else "subst", "a": e1, "b": e2, "cost": op_cost})

    return {"cost": cost, "node_ops": node_ops, "edge_ops": edge_ops}
