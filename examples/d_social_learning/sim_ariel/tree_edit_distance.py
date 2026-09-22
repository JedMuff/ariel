"""Tree edit distance between two TreeGenome genomes.

Trees are unordered (children are distinguished only by which ``face`` they
attach to on their parent, not by sequence), so this uses graph edit
distance rather than an ordered-tree algorithm like Zhang-Shasha.
"""

import networkx as nx

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
