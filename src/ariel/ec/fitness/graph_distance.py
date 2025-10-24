"""Fitness functions for robot morphology evolution.

This module provides shared fitness evaluation functions that can be used
across different evolutionary algorithms. The primary fitness metric is
an attribute-aware graph edit distance that considers:
- Node attributes: module type and rotation
- Edge attributes: attachment face/connection point

This provides a more accurate morphological distance than structure-only
comparison.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from networkx import DiGraph


def node_match(node1: dict[str, Any], node2: dict[str, Any]) -> bool:
    """Compare two nodes based on their attributes.

    Nodes are considered matching if they have the same module type
    and rotation. This is stricter than structure-only comparison.

    Parameters
    ----------
    node1 : dict[str, Any]
        First node's attribute dictionary
    node2 : dict[str, Any]
        Second node's attribute dictionary

    Returns
    -------
    bool
        True if nodes match (same type and rotation), False otherwise
    """
    # Both type and rotation must match for nodes to be considered equal
    type_match = node1.get("type") == node2.get("type")
    rotation_match = node1.get("rotation") == node2.get("rotation")

    return type_match and rotation_match


def edge_match(edge1: dict[str, Any], edge2: dict[str, Any]) -> bool:
    """Compare two edges based on their attributes.

    Edges are considered matching if they use the same attachment face.
    This ensures that connections through different faces (FRONT, BACK,
    LEFT, RIGHT, TOP, BOTTOM) are treated as different.

    Parameters
    ----------
    edge1 : dict[str, Any]
        First edge's attribute dictionary
    edge2 : dict[str, Any]
        Second edge's attribute dictionary

    Returns
    -------
    bool
        True if edges match (same face), False otherwise
    """
    # Attachment face must match for edges to be considered equal
    return edge1.get("face") == edge2.get("face")


def graph_distance(
    graph1: DiGraph[Any],
    graph2: DiGraph[Any],
    timeout: float = 0.1,
    use_attributes: bool = True,
) -> float:
    """Calculate distance between two robot morphology graphs.

    This function computes the graph edit distance (GED) between two
    robot morphologies. By default, it considers node and edge attributes
    to provide a more accurate morphological distance.

    Parameters
    ----------
    graph1 : DiGraph
        First robot morphology graph
    graph2 : DiGraph
        Second robot morphology graph
    timeout : float, optional
        Maximum time in seconds for GED computation, by default 0.1
    use_attributes : bool, optional
        Whether to consider node/edge attributes in distance calculation,
        by default True. Set to False for structure-only comparison.

    Returns
    -------
    float
        Distance metric (lower is better). Returns a high penalty value
        (1000.0) if computation times out or fails.

    Notes
    -----
    The graph edit distance is the minimum cost sequence of operations
    (node/edge insertions, deletions, substitutions) needed to transform
    one graph into another. When use_attributes=True:
    - Node substitutions require matching type AND rotation
    - Edge substitutions require matching attachment face

    This makes the distance metric more semantically meaningful for
    robot morphologies.

    Examples
    --------
    >>> # Two graphs with same structure but different module types
    >>> distance = graph_distance(evolved_graph, target_graph)
    >>> # Distance will be higher with use_attributes=True since types differ
    """
    try:
        if use_attributes:
            # Attribute-aware comparison
            ged = nx.graph_edit_distance(
                graph1,
                graph2,
                node_match=node_match,
                edge_match=edge_match,
                timeout=timeout,
            )
        else:
            # Structure-only comparison (original behavior)
            ged = nx.graph_edit_distance(
                graph1,
                graph2,
                timeout=timeout,
            )

        # Return the computed distance, or high penalty if timeout
        return float(ged) if ged is not None else 1000.0

    except Exception:
        # Return high penalty for any errors (invalid graphs, etc.)
        return 1000.0


def calculate_fitness(
    candidate_graph: DiGraph[Any],
    target_graph: DiGraph[Any],
    timeout: float = 0.1,
    use_attributes: bool = True,
) -> float:
    """Calculate fitness as distance to target morphology.

    This is a convenience wrapper around graph_distance() that provides
    a consistent fitness evaluation interface for evolutionary algorithms.
    Lower fitness values indicate better matches to the target.

    Parameters
    ----------
    candidate_graph : DiGraph
        The candidate robot morphology to evaluate
    target_graph : DiGraph
        The target robot morphology to match
    timeout : float, optional
        Maximum time in seconds for distance computation, by default 0.1
    use_attributes : bool, optional
        Whether to consider node/edge attributes, by default True

    Returns
    -------
    float
        Fitness value (lower is better)

    Examples
    --------
    >>> fitness = calculate_fitness(evolved_robot, target_robot)
    >>> print(f"Fitness: {fitness:.2f}")
    """
    return graph_distance(
        candidate_graph,
        target_graph,
        timeout=timeout,
        use_attributes=use_attributes,
    )


def evaluate_tree_fitness_worker(
    args: tuple[DiGraph[Any], DiGraph[Any], float, bool]
) -> float:
    """Worker function for parallel fitness evaluation.

    This top-level function is designed to be used with multiprocessing.Pool
    for parallel evaluation of multiple individuals. It must be a module-level
    function (not a method) to be picklable.

    Parameters
    ----------
    args : tuple[DiGraph, DiGraph, float, bool]
        Tuple containing (candidate_graph, target_graph, timeout, use_attributes)

    Returns
    -------
    float
        Fitness value (lower is better)

    Examples
    --------
    >>> from multiprocessing import Pool
    >>> args_list = [(tree1, target, 0.1, True), (tree2, target, 0.1, True)]
    >>> with Pool(4) as pool:
    ...     fitnesses = pool.map(evaluate_tree_fitness_worker, args_list)
    """
    candidate_graph, target_graph, timeout, use_attributes = args
    return graph_distance(
        candidate_graph,
        target_graph,
        timeout=timeout,
        use_attributes=use_attributes,
    )
