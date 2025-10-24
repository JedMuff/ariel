"""Fitness functions for robot morphology evolution.

This module provides shared fitness evaluation functions that can be used
across different evolutionary algorithms. Available fitness functions include:

1. Graph Edit Distance (GED) based:
   - Attribute-aware graph edit distance considering module type and rotation
   - Edge attributes: attachment face/connection point
   - Novelty-adjusted fitness for diversity maintenance

2. Morphological Measures based:
   - Branching: ratio of filled cores/bricks to max potential
   - Limbs: ratio of single-neighbor modules to max potential
   - Length of limbs: ratio of double-neighbor modules to max potential
   - Coverage: proportion of bounding box filled with modules
   - Symmetry: maximum symmetry across all planes
   - Size: number of modules in the robot
   - Combined: weighted combination of multiple measures
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import networkx as nx
from networkx import DiGraph

# Ensure myscripts is in path for multiprocessing workers
_MYSCRIPTS_PATH = Path(__file__).parent
if str(_MYSCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(_MYSCRIPTS_PATH))

# Import at module level so it's available in worker processes
try:
    from morphological_measures import Body, MorphologicalMeasures
    _MORPHOLOGICAL_AVAILABLE = True
except ImportError:
    _MORPHOLOGICAL_AVAILABLE = False
    Body = None
    MorphologicalMeasures = None


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


def calculate_novelty_adjusted_fitness(
    candidate_graph: DiGraph[Any],
    target_graph: DiGraph[Any],
    novelty_archive: Any,
    timeout: float = 0.1,
    use_attributes: bool = True,
    k_neighbors: int = 5,
    min_novelty: float = 1e-6,
) -> float:
    """Calculate fitness as edit distance divided by novelty.

    This fitness function encourages both proximity to the target and
    behavioral/structural diversity by dividing the edit distance by
    the novelty score from the archive. Higher novelty (more diverse
    individuals) will have lower fitness values for the same edit distance.

    Parameters
    ----------
    candidate_graph : DiGraph
        The candidate robot morphology to evaluate
    target_graph : DiGraph
        The target robot morphology to match
    novelty_archive : PoissonArchive
        The novelty archive for computing diversity scores
    timeout : float, optional
        Maximum time in seconds for distance computation, by default 0.1
    use_attributes : bool, optional
        Whether to consider node/edge attributes, by default True
    k_neighbors : int, optional
        Number of nearest neighbors to use for novelty calculation, by default 5
    min_novelty : float, optional
        Minimum novelty value to prevent division by very small numbers,
        by default 1e-6

    Returns
    -------
    float
        Fitness value (lower is better). When novelty is high (diverse),
        the fitness is reduced. When novelty is low (similar to archive),
        the fitness is increased.

    Notes
    -----
    - If the archive is empty, novelty returns inf, so we handle this by
      returning just the edit distance (equivalent to standard fitness).
    - A min_novelty parameter prevents division by very small novelty values
      which could cause numerical instability.

    Examples
    --------
    >>> from myscripts.novelty import PoissonArchive
    >>> archive = PoissonArchive(min_distance=3.0, distance_fn=graph_distance)
    >>> fitness = calculate_novelty_adjusted_fitness(
    ...     evolved_robot, target_robot, archive, k_neighbors=5
    ... )
    """
    # Calculate standard edit distance
    edit_distance = graph_distance(
        candidate_graph,
        target_graph,
        timeout=timeout,
        use_attributes=use_attributes,
    )

    # Calculate novelty score
    novelty_score = novelty_archive.novelty(candidate_graph, k=k_neighbors)

    # Handle edge cases
    if novelty_score == float("inf"):
        # Archive is empty, return standard edit distance
        return edit_distance

    # Apply minimum novelty threshold to prevent division issues
    novelty_score = max(novelty_score, min_novelty)

    # Return distance divided by novelty
    # Higher novelty = lower fitness (rewards diversity)
    return edit_distance / novelty_score


def evaluate_novelty_fitness_worker(
    args: tuple[DiGraph[Any], DiGraph[Any], Any, float, bool, int, float]
) -> float:
    """Worker function for parallel novelty-adjusted fitness evaluation.

    This top-level function is designed to be used with multiprocessing.Pool
    for parallel evaluation of multiple individuals with novelty adjustment.
    It must be a module-level function (not a method) to be picklable.

    Parameters
    ----------
    args : tuple[DiGraph, DiGraph, PoissonArchive, float, bool, int, float]
        Tuple containing (candidate_graph, target_graph, novelty_archive,
        timeout, use_attributes, k_neighbors, min_novelty)

    Returns
    -------
    float
        Novelty-adjusted fitness value (lower is better)

    Notes
    -----
    The novelty_archive should be treated as read-only during parallel
    evaluation. Archive updates should happen sequentially after fitness
    evaluation is complete.

    Examples
    --------
    >>> from multiprocessing import Pool
    >>> from myscripts.novelty import PoissonArchive
    >>> archive = PoissonArchive(min_distance=3.0, distance_fn=graph_distance)
    >>> args_list = [
    ...     (tree1, target, archive, 0.1, True, 5, 1e-6),
    ...     (tree2, target, archive, 0.1, True, 5, 1e-6)
    ... ]
    >>> with Pool(4) as pool:
    ...     fitnesses = pool.map(evaluate_novelty_fitness_worker, args_list)
    """
    (
        candidate_graph,
        target_graph,
        novelty_archive,
        timeout,
        use_attributes,
        k_neighbors,
        min_novelty,
    ) = args

    return calculate_novelty_adjusted_fitness(
        candidate_graph,
        target_graph,
        novelty_archive,
        timeout=timeout,
        use_attributes=use_attributes,
        k_neighbors=k_neighbors,
        min_novelty=min_novelty,
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


# ============================================================================
# Morphological Measure-based Fitness Functions
# ============================================================================

def morphological_fitness(
    genome: DiGraph[Any],
    measure_name: str = "branching",
    maximize: bool = True,
    log_dir: str | None = None,
) -> float:
    """Calculate fitness based on a morphological measure.

    This function creates a Body from the genotype graph, computes all
    morphological measures, and returns the specified measure as fitness.

    Parameters
    ----------
    genome : DiGraph
        The robot morphology graph (genotype).
    measure_name : str, optional
        Name of the morphological measure to use as fitness, by default "branching".
        Available measures:
        - "branching": Ratio of filled cores/bricks to max potential
        - "limbs": Ratio of single-neighbor modules to max potential
        - "length_of_limbs": Ratio of double-neighbor modules to max potential
        - "coverage": Proportion of bounding box filled with modules
        - "symmetry": Maximum symmetry across all planes
        - "xy_symmetry": Symmetry in xy-plane
        - "xz_symmetry": Symmetry in xz-plane
        - "yz_symmetry": Symmetry in yz-plane
        - "proportion_2d": Width/height ratio (2D robots only)
        - "num_modules": Total number of modules
        - "num_bricks": Number of brick modules
        - "num_active_hinges": Number of hinge modules
        - "bounding_box_volume": Volume of bounding box
    maximize : bool, optional
        Whether to maximize the measure (True) or minimize it (False),
        by default True. When False, returns negative of the measure.
    log_dir : str | None, optional
        Directory for logging individual data, by default None.
        Currently unused but maintained for API compatibility.

    Returns
    -------
    float
        Fitness value based on the specified morphological measure.
        Higher values are better when maximize=True, lower when maximize=False.

    Raises
    ------
    ValueError
        If the genome is empty, has no core, or measure_name is invalid.
    AttributeError
        If the specified measure doesn't exist.

    Examples
    --------
    >>> import networkx as nx
    >>> # Create a simple robot graph
    >>> graph = nx.DiGraph()
    >>> # ... populate graph ...
    >>> fitness = morphological_fitness(graph, measure_name="branching")
    >>> print(f"Branching fitness: {fitness:.3f}")
    """
    try:
        if not _MORPHOLOGICAL_AVAILABLE:
            raise ImportError("Morphological measures module not available")

        # Create Body wrapper from genome
        body = Body(genome)

        # Compute morphological measures
        measures = MorphologicalMeasures(body)

        # Get the requested measure
        if not hasattr(measures, measure_name):
            available = [
                "branching", "limbs", "length_of_limbs", "coverage", "symmetry",
                "xy_symmetry", "xz_symmetry", "yz_symmetry", "proportion_2d",
                "num_modules", "num_bricks", "num_active_hinges",
                "bounding_box_volume", "bounding_box_width", "bounding_box_height",
                "bounding_box_depth",
            ]
            msg = (
                f"Unknown measure '{measure_name}'. "
                f"Available measures: {', '.join(available)}"
            )
            raise AttributeError(msg)

        fitness_value = getattr(measures, measure_name)

        # Handle maximize/minimize
        if not maximize:
            fitness_value = -fitness_value

        return float(fitness_value)

    except Exception as e:
        # Return penalty for invalid genomes
        print(f"Error computing morphological fitness: {e}")
        return -1000.0 if maximize else 1000.0


def branching_fitness(
    genome: DiGraph[Any],
    log_dir: str | None = None,
) -> float:
    """Fitness based on branching (filled cores and bricks).

    Higher branching means more modules have all their attachment points filled,
    indicating a more compact, well-connected structure.

    Parameters
    ----------
    genome : DiGraph
        The robot morphology graph.
    log_dir : str | None, optional
        Directory for logging (unused, for API compatibility).

    Returns
    -------
    float
        Branching fitness value (higher is better, range 0.0-1.0).
    """
    return morphological_fitness(genome, measure_name="branching", maximize=True, log_dir=log_dir)


def limbs_fitness(
    genome: DiGraph[Any],
    log_dir: str | None = None,
) -> float:
    """Fitness based on number of limbs (modules with single neighbor).

    Higher limbs fitness indicates more extremities, suggesting a more
    spread-out, branching structure.

    Parameters
    ----------
    genome : DiGraph
        The robot morphology graph.
    log_dir : str | None, optional
        Directory for logging (unused, for API compatibility).

    Returns
    -------
    float
        Limbs fitness value (higher is better, range 0.0-1.0).
    """
    return morphological_fitness(genome, measure_name="limbs", maximize=True, log_dir=log_dir)


def length_of_limbs_fitness(
    genome: DiGraph[Any],
    log_dir: str | None = None,
) -> float:
    """Fitness based on length of limbs (modules with two neighbors).

    Higher values indicate longer, more extended limbs rather than
    compact structures.

    Parameters
    ----------
    genome : DiGraph
        The robot morphology graph.
    log_dir : str | None, optional
        Directory for logging (unused, for API compatibility).

    Returns
    -------
    float
        Length of limbs fitness (higher is better, range 0.0-1.0).
    """
    return morphological_fitness(
        genome, measure_name="length_of_limbs", maximize=True, log_dir=log_dir
    )


def coverage_fitness(
    genome: DiGraph[Any],
    log_dir: str | None = None,
) -> float:
    """Fitness based on bounding box coverage.

    Higher coverage means the robot fills more of its bounding box,
    indicating a more compact, space-filling structure.

    Parameters
    ----------
    genome : DiGraph
        The robot morphology graph.
    log_dir : str | None, optional
        Directory for logging (unused, for API compatibility).

    Returns
    -------
    float
        Coverage fitness value (higher is better, range 0.0-1.0).
    """
    return morphological_fitness(genome, measure_name="coverage", maximize=True, log_dir=log_dir)


def symmetry_fitness(
    genome: DiGraph[Any],
    log_dir: str | None = None,
) -> float:
    """Fitness based on maximum symmetry across all planes.

    Higher symmetry indicates the robot is more symmetric, which can
    be beneficial for locomotion stability.

    Parameters
    ----------
    genome : DiGraph
        The robot morphology graph.
    log_dir : str | None, optional
        Directory for logging (unused, for API compatibility).

    Returns
    -------
    float
        Symmetry fitness value (higher is better, range 0.0-1.0).
    """
    return morphological_fitness(genome, measure_name="symmetry", maximize=True, log_dir=log_dir)


def size_fitness(
    genome: DiGraph[Any],
    maximize: bool = True,
    log_dir: str | None = None,
) -> float:
    """Fitness based on number of modules.

    Can be used to encourage or discourage large robots.

    Parameters
    ----------
    genome : DiGraph
        The robot morphology graph.
    maximize : bool, optional
        Whether to maximize (True) or minimize (False) size, by default True.
    log_dir : str | None, optional
        Directory for logging (unused, for API compatibility).

    Returns
    -------
    float
        Size fitness value (number of modules).
    """
    return morphological_fitness(
        genome, measure_name="num_modules", maximize=maximize, log_dir=log_dir
    )


def combined_morphological_fitness(
    genome: DiGraph[Any],
    weights: dict[str, float] | None = None,
    log_dir: str | None = None,
) -> float:
    """Calculate weighted combination of multiple morphological measures.

    This allows for multi-objective fitness based on several morphological
    characteristics.

    Parameters
    ----------
    genome : DiGraph
        The robot morphology graph.
    weights : dict[str, float] | None, optional
        Dictionary mapping measure names to weights. If None, uses equal
        weights for branching, limbs, and symmetry. Example:
        {"branching": 0.5, "symmetry": 0.3, "coverage": 0.2}
    log_dir : str | None, optional
        Directory for logging (unused, for API compatibility).

    Returns
    -------
    float
        Weighted sum of morphological measures.

    Examples
    --------
    >>> weights = {"branching": 0.4, "symmetry": 0.4, "coverage": 0.2}
    >>> fitness = combined_morphological_fitness(graph, weights=weights)
    """
    if weights is None:
        # Default: equal weights for key measures
        weights = {
            "branching": 1.0 / 3.0,
            "limbs": 1.0 / 3.0,
            "symmetry": 1.0 / 3.0,
        }

    try:
        if not _MORPHOLOGICAL_AVAILABLE:
            raise ImportError("Morphological measures module not available")

        body = Body(genome)
        measures = MorphologicalMeasures(body)

        total_fitness = 0.0
        total_weight = sum(weights.values())

        for measure_name, weight in weights.items():
            if hasattr(measures, measure_name):
                value = getattr(measures, measure_name)
                total_fitness += value * weight
            else:
                print(f"Warning: Unknown measure '{measure_name}', skipping")

        # Normalize by total weight
        return total_fitness / total_weight if total_weight > 0 else 0.0

    except Exception as e:
        print(f"Error computing combined fitness: {e}")
        return -1000.0


def morphological_fitness_worker(
    args: tuple[DiGraph[Any], str, bool],
) -> float:
    """Worker function for parallel morphological fitness evaluation.

    This top-level function is designed to be used with multiprocessing.Pool.
    It must be a module-level function to be picklable.

    Parameters
    ----------
    args : tuple[DiGraph, str, bool]
        Tuple containing (genome, measure_name, maximize).

    Returns
    -------
    float
        Fitness value.

    Examples
    --------
    >>> from multiprocessing import Pool
    >>> args_list = [(graph1, "branching", True), (graph2, "branching", True)]
    >>> with Pool(4) as pool:
    ...     fitnesses = pool.map(morphological_fitness_worker, args_list)
    """
    genome, measure_name, maximize = args
    return morphological_fitness(genome, measure_name=measure_name, maximize=maximize)
