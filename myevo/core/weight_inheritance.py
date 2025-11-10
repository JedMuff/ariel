"""Weight inheritance utilities for evolutionary strategies.

This module provides tools for inheriting neural network controller weights
from parents to offspring during evolution. Supports both Lamarckian evolution
(inheriting optimized weights) and non-Lamarckian evolution (inheriting initial weights).
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from networkx import DiGraph


def adapt_weights_to_morphology(
    parent_weights: np.ndarray,
    parent_layer_sizes: list[int],
    offspring_layer_sizes: list[int],
    rng: np.random.Generator,
    sigma: float = 1.0,
) -> np.ndarray:
    """Adapt parent weights to fit offspring morphology.

    When morphology changes (joints added/removed), the neural network
    architecture changes. This function intelligently adapts parent weights:
    - Hidden layer weights are fully preserved (sizes don't change)
    - Input/output layer weights are adapted (grown/shrunk as needed)
    - New weights are randomly initialized

    Parameters
    ----------
    parent_weights : np.ndarray
        Flat array of parent's learned weights
    parent_layer_sizes : list[int]
        Parent's layer sizes [input, hidden1, ..., output]
    offspring_layer_sizes : list[int]
        Offspring's layer sizes [input, hidden1, ..., output]
    rng : np.random.Generator
        Random number generator for new weights
    sigma : float
        Range for random initialization [-sigma, sigma]

    Returns
    -------
    np.ndarray
        Adapted flat weight array for offspring

    Examples
    --------
    >>> # Parent: [59, 32, 16, 32, 23] -> offspring: [50, 32, 16, 32, 20]
    >>> # Hidden layers [32, 16, 32] preserved, I/O adapted
    >>> adapted = adapt_weights_to_morphology(
    ...     parent_weights=parent_weights,
    ...     parent_layer_sizes=[59, 32, 16, 32, 23],
    ...     offspring_layer_sizes=[50, 32, 16, 32, 20],
    ...     rng=np.random.default_rng(42),
    ...     sigma=1.0,
    ... )
    """
    # Step 1: Unpack parent weights into layer-by-layer structure
    parent_layers = []
    idx = 0

    for i in range(len(parent_layer_sizes) - 1):
        in_size = parent_layer_sizes[i]
        out_size = parent_layer_sizes[i + 1]

        # Extract weight matrix
        w_size = in_size * out_size
        w = parent_weights[idx:idx + w_size].reshape(in_size, out_size)
        idx += w_size

        # Extract bias vector
        b = parent_weights[idx:idx + out_size]
        idx += out_size

        parent_layers.append({'w': w, 'b': b})

    # Step 2: Adapt each layer to offspring architecture
    offspring_layers = []

    for i in range(len(offspring_layer_sizes) - 1):
        parent_in_size = parent_layer_sizes[i]
        parent_out_size = parent_layer_sizes[i + 1]
        offspring_in_size = offspring_layer_sizes[i]
        offspring_out_size = offspring_layer_sizes[i + 1]

        parent_w = parent_layers[i]['w']
        parent_b = parent_layers[i]['b']

        # Adapt weight matrix
        if parent_in_size == offspring_in_size and parent_out_size == offspring_out_size:
            # Same size - copy directly
            offspring_w = parent_w.copy()
        elif parent_in_size <= offspring_in_size and parent_out_size <= offspring_out_size:
            # Offspring larger - copy parent + random init for new weights
            offspring_w = rng.uniform(-sigma, sigma, (offspring_in_size, offspring_out_size))
            offspring_w[:parent_in_size, :parent_out_size] = parent_w
        elif parent_in_size >= offspring_in_size and parent_out_size >= offspring_out_size:
            # Offspring smaller - truncate parent weights
            offspring_w = parent_w[:offspring_in_size, :offspring_out_size].copy()
        else:
            # Mixed case - one dimension grows, other shrinks
            offspring_w = rng.uniform(-sigma, sigma, (offspring_in_size, offspring_out_size))
            rows_to_copy = min(parent_in_size, offspring_in_size)
            cols_to_copy = min(parent_out_size, offspring_out_size)
            offspring_w[:rows_to_copy, :cols_to_copy] = parent_w[:rows_to_copy, :cols_to_copy]

        # Adapt bias vector
        if parent_out_size == offspring_out_size:
            # Same size - copy directly
            offspring_b = parent_b.copy()
        elif parent_out_size < offspring_out_size:
            # Offspring larger - copy parent + random init for new biases
            offspring_b = rng.uniform(-sigma, sigma, offspring_out_size)
            offspring_b[:parent_out_size] = parent_b
        else:
            # Offspring smaller - truncate parent biases
            offspring_b = parent_b[:offspring_out_size].copy()

        offspring_layers.append({'w': offspring_w, 'b': offspring_b})

    # Step 3: Repack into flat array
    flat_weights = []
    for layer in offspring_layers:
        flat_weights.append(layer['w'].flatten())
        flat_weights.append(layer['b'])

    return np.concatenate(flat_weights)


def tree_distance(tree1: DiGraph, tree2: DiGraph, timeout: float = 1.0) -> float:
    """Compute tree edit distance between two tree genotypes.

    Uses NetworkX's graph edit distance with node and edge attribute matching.
    This provides a structural similarity measure for determining which parent
    an offspring most closely resembles.

    Parameters
    ----------
    tree1 : DiGraph
        First tree genotype.
    tree2 : DiGraph
        Second tree genotype.
    timeout : float, optional
        Maximum time for computation, by default 1.0 seconds.

    Returns
    -------
    float
        Tree edit distance (lower = more similar).
    """
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

    # Compute graph edit distance with attribute matching
    # If this times out or fails, let the exception propagate
    ged = nx.graph_edit_distance(
        tree1,
        tree2,
        node_match=node_match,
        edge_match=edge_match,
        timeout=timeout,
    )
    # If timeout occurred, ged will be None
    if ged is None:
        raise TimeoutError(
            f"Graph edit distance computation timed out after {timeout}s. "
            "Trees may be too large or complex."
        )
    return float(ged)


class ParentWeightManager:
    """Manages neural network weights for evolutionary weight inheritance.

    This class handles storage and retrieval of controller weights from evaluated
    individuals, enabling offspring to inherit weights from their parents. It supports
    different inheritance modes for crossover offspring.

    Used by evolutionary strategies to implement both Lamarckian evolution
    (inherit optimized weights) and non-Lamarckian evolution (inherit initial weights).
    """

    def __init__(self, crossover_mode: str = "closest_parent", sigma: float = 1.0):
        """Initialize the weight manager.

        Parameters
        ----------
        crossover_mode : str, optional
            How to combine parent weights for crossover offspring, by default "closest_parent".
            Options:
            - "average": Average both parents' weights
            - "parent1": Use only first parent's weights
            - "random": Randomly choose one parent's weights
            - "closest_parent": Use weights from parent with smallest tree edit distance
        sigma : float, optional
            Range for random weight initialization when adapting morphology, by default 1.0.
        """
        self.learned_weights: dict[int, tuple[np.ndarray, list[int]]] = {}
        self.all_evaluated_individuals: list[Any] = []
        self.crossover_mode = crossover_mode
        self.sigma = sigma
        self.rng = np.random.default_rng()

    def store_weights(self, individual_id: int, weights: np.ndarray, layer_sizes: list[int]) -> None:
        """Store weights for an evaluated individual.

        Parameters
        ----------
        individual_id : int
            ID of the individual (use individual.id).
        weights : np.ndarray
            Controller weights.
        layer_sizes : list[int]
            Neural network layer sizes [input, hidden1, ..., output].
        """
        self.learned_weights[individual_id] = (weights.copy(), layer_sizes.copy())

    def add_evaluated_individual(self, individual: Any) -> None:
        """Add an evaluated individual to the tracking list.

        This is needed to look up parent individuals later for weight inheritance.

        Parameters
        ----------
        individual : Any
            Evaluated Individual object.
        """
        self.all_evaluated_individuals.append(individual)

    def get_parent_weights(
        self,
        individual: Any,
        offspring_tree: DiGraph | None = None,
        offspring_layer_sizes: list[int] | None = None,
    ) -> np.ndarray | None:
        """Get inherited weights from parent(s) for an individual.

        Automatically adapts parent weights to fit offspring morphology if needed.

        Parameters
        ----------
        individual : Any
            The offspring individual to get parent weights for.
        offspring_tree : DiGraph | None, optional
            The offspring's tree genotype (needed for closest_parent mode),
            by default None.
        offspring_layer_sizes : list[int] | None, optional
            The offspring's neural network layer sizes for weight adaptation,
            by default None.

        Returns
        -------
        np.ndarray | None
            Inherited (and adapted) weights from parent(s), or None if not available.
        """
        # Check if individual has parent information
        if "parent1_id" not in individual.tags:
            return None

        parent1_id = individual.tags["parent1_id"]

        # Find parent Individual objects
        from ariel.ec import TreeGenotype

        parent1_ind = None
        parent2_ind = None

        for ind in self.all_evaluated_individuals:
            if ind.id == parent1_id:
                parent1_ind = ind
            if "parent2_id" in individual.tags and ind.id == individual.tags["parent2_id"]:
                parent2_ind = ind
            if parent1_ind is not None and (parent2_ind is not None or "parent2_id" not in individual.tags):
                break

        # Check if this is a crossover offspring (has two parents)
        if "parent2_id" in individual.tags and parent2_ind is not None:
            return self._combine_parent_weights(
                parent1_ind=parent1_ind,
                parent2_ind=parent2_ind,
                offspring_tree=offspring_tree,
                offspring_layer_sizes=offspring_layer_sizes,
            )
        else:
            # Mutation offspring - inherit from single parent
            if parent1_ind is not None:
                parent_data = self.learned_weights.get(parent1_ind.id)
                if parent_data is None:
                    return None

                parent_weights, parent_layer_sizes = parent_data

                # Adapt weights if offspring has different morphology
                if offspring_layer_sizes is not None and parent_layer_sizes != offspring_layer_sizes:
                    return adapt_weights_to_morphology(
                        parent_weights=parent_weights,
                        parent_layer_sizes=parent_layer_sizes,
                        offspring_layer_sizes=offspring_layer_sizes,
                        rng=self.rng,
                        sigma=self.sigma,
                    )
                else:
                    return parent_weights.copy()
            return None

    def _combine_parent_weights(
        self,
        parent1_ind: Any | None,
        parent2_ind: Any | None,
        offspring_tree: DiGraph | None = None,
        offspring_layer_sizes: list[int] | None = None,
    ) -> np.ndarray | None:
        """Combine weights from two parents for crossover offspring.

        Automatically adapts both parents' weights to offspring morphology before combining.

        Parameters
        ----------
        parent1_ind : Any | None
            First parent Individual.
        parent2_ind : Any | None
            Second parent Individual.
        offspring_tree : DiGraph | None, optional
            Offspring's tree genotype (for closest_parent mode), by default None.
        offspring_layer_sizes : list[int] | None, optional
            Offspring's neural network layer sizes for weight adaptation, by default None.

        Returns
        -------
        np.ndarray | None
            Combined (and adapted) weights, or None if parents don't have weights.
        """
        # Get weights and layer sizes from both parents if available
        parent1_data = self.learned_weights.get(parent1_ind.id) if parent1_ind is not None else None
        parent2_data = self.learned_weights.get(parent2_ind.id) if parent2_ind is not None else None

        # Handle cases where one or both parents don't have weights
        if parent1_data is None and parent2_data is None:
            return None

        # Helper function to adapt parent weights to offspring
        def adapt_parent(parent_data: tuple[np.ndarray, list[int]] | None) -> np.ndarray | None:
            if parent_data is None:
                return None
            parent_weights, parent_layer_sizes = parent_data
            if offspring_layer_sizes is not None and parent_layer_sizes != offspring_layer_sizes:
                return adapt_weights_to_morphology(
                    parent_weights=parent_weights,
                    parent_layer_sizes=parent_layer_sizes,
                    offspring_layer_sizes=offspring_layer_sizes,
                    rng=self.rng,
                    sigma=self.sigma,
                )
            else:
                return parent_weights.copy()

        # Adapt both parents to offspring morphology
        weights1 = adapt_parent(parent1_data)
        weights2 = adapt_parent(parent2_data)

        # Handle cases where one parent doesn't have weights
        if weights1 is None:
            return weights2
        if weights2 is None:
            return weights1

        # Both parents have weights - combine based on crossover mode
        if self.crossover_mode == "parent1":
            return weights1
        elif self.crossover_mode == "average":
            return (weights1 + weights2) / 2
        elif self.crossover_mode == "random":
            return weights1 if self.rng.random() < 0.5 else weights2
        elif self.crossover_mode == "closest_parent":
            if offspring_tree is None:
                # Fallback to average if no offspring tree provided
                return (weights1 + weights2) / 2

            # Extract trees from parents
            from ariel.ec import TreeGenotype
            parent1_tree = parent1_ind.genotype.tree if isinstance(parent1_ind.genotype, TreeGenotype) else parent1_ind.genotype
            parent2_tree = parent2_ind.genotype.tree if isinstance(parent2_ind.genotype, TreeGenotype) else parent2_ind.genotype

            # Compute distance to each parent
            dist1 = tree_distance(offspring_tree, parent1_tree)
            dist2 = tree_distance(offspring_tree, parent2_tree)

            # Return weights from closest parent
            return weights1 if dist1 <= dist2 else weights2
        else:
            # Default to averaging
            return (weights1 + weights2) / 2

    def clear_old_weights(self, keep_ids: set[int]) -> None:
        """Clear weights for individuals no longer needed.

        This helps manage memory by removing weights for old individuals.

        Parameters
        ----------
        keep_ids : set[int]
            Set of individual IDs to keep.
        """
        # Remove weights for IDs not in keep set
        old_ids = set(self.learned_weights.keys()) - keep_ids
        for individual_id in old_ids:
            del self.learned_weights[individual_id]

    def get_weights(self, individual_id: int) -> tuple[np.ndarray, list[int]] | None:
        """Retrieve stored weights and layer sizes for an individual.

        Parameters
        ----------
        individual_id : int
            ID of the individual.

        Returns
        -------
        tuple[np.ndarray, list[int]] | None
            Tuple of (weights, layer_sizes), or None if not found.
        """
        data = self.learned_weights.get(individual_id)
        if data is not None:
            weights, layer_sizes = data
            return (weights.copy(), layer_sizes.copy())
        return None

    def has_weights(self, individual_id: int) -> bool:
        """Check if weights are stored for an individual.

        Parameters
        ----------
        individual_id : int
            ID of the individual.

        Returns
        -------
        bool
            True if weights are stored, False otherwise.
        """
        return individual_id in self.learned_weights

    def __len__(self) -> int:
        """Return number of individuals with stored weights."""
        return len(self.learned_weights)
