"""Lamarckian evolution utilities for weight inheritance.

This module provides tools for Lamarckian evolution, where offspring can inherit
learned controller weights from their parents. This can significantly speed up
evolution by preserving learned behaviors across generations.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from networkx import DiGraph


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
    try:
        ged = nx.graph_edit_distance(
            tree1,
            tree2,
            node_match=node_match,
            edge_match=edge_match,
            timeout=timeout,
        )
        return float(ged) if ged is not None else 1000.0
    except Exception:
        return 1000.0


class ParentWeightManager:
    """Manages learned controller weights for Lamarckian evolution.

    This class handles storage and retrieval of learned controller weights,
    enabling offspring to inherit weights from their parents. It supports
    different inheritance modes for crossover offspring.
    """

    def __init__(self, crossover_mode: str = "average"):
        """Initialize the weight manager.

        Parameters
        ----------
        crossover_mode : str, optional
            How to combine parent weights for crossover offspring, by default "average".
            Options:
            - "average": Average both parents' weights
            - "parent1": Use only first parent's weights
            - "random": Randomly choose one parent's weights
            - "closest_parent": Use weights from parent with smallest tree edit distance
        """
        self.learned_weights: dict[int, np.ndarray] = {}
        self.all_evaluated_individuals: list[Any] = []
        self.crossover_mode = crossover_mode
        self.rng = np.random.default_rng()

    def store_weights(self, tree_id: int, weights: np.ndarray) -> None:
        """Store learned weights for a genotype.

        Parameters
        ----------
        tree_id : int
            ID of the tree genotype (use id(tree)).
        weights : np.ndarray
            Learned controller weights.
        """
        self.learned_weights[tree_id] = weights.copy()

    def add_evaluated_individual(self, individual: Any) -> None:
        """Add an evaluated individual to the tracking list.

        This is needed to look up parent genotypes later for weight inheritance.

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
    ) -> np.ndarray | None:
        """Get inherited weights from parent(s) for an individual.

        Parameters
        ----------
        individual : Any
            The offspring individual to get parent weights for.
        offspring_tree : DiGraph | None, optional
            The offspring's tree genotype (needed for closest_parent mode),
            by default None.

        Returns
        -------
        np.ndarray | None
            Inherited weights from parent(s), or None if not available.
        """
        # Check if individual has parent information
        if "parent1_id" not in individual.tags:
            return None

        parent1_id = individual.tags["parent1_id"]

        # Find parent Individual objects and their genotypes
        from ariel.ec import TreeGenotype

        parent1_tree = None
        parent2_tree = None

        for ind in self.all_evaluated_individuals:
            if ind.id == parent1_id:
                parent1_tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
            if "parent2_id" in individual.tags and ind.id == individual.tags["parent2_id"]:
                parent2_tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
            if parent1_tree is not None and (parent2_tree is not None or "parent2_id" not in individual.tags):
                break

        # Check if this is a crossover offspring (has two parents)
        if "parent2_id" in individual.tags and parent2_tree is not None:
            return self._combine_parent_weights(
                parent1_tree=parent1_tree,
                parent2_tree=parent2_tree,
                offspring_tree=offspring_tree,
            )
        else:
            # Mutation offspring - inherit from single parent
            if parent1_tree is not None:
                parent_weights = self.learned_weights.get(id(parent1_tree))
                return parent_weights.copy() if parent_weights is not None else None
            return None

    def _combine_parent_weights(
        self,
        parent1_tree: DiGraph | None,
        parent2_tree: DiGraph | None,
        offspring_tree: DiGraph | None = None,
    ) -> np.ndarray | None:
        """Combine weights from two parents for crossover offspring.

        Parameters
        ----------
        parent1_tree : DiGraph | None
            First parent's tree genotype.
        parent2_tree : DiGraph | None
            Second parent's tree genotype.
        offspring_tree : DiGraph | None, optional
            Offspring's tree genotype (for closest_parent mode), by default None.

        Returns
        -------
        np.ndarray | None
            Combined weights, or None if parents don't have weights.
        """
        # Get weights from both parents if available
        weights1 = self.learned_weights.get(id(parent1_tree)) if parent1_tree is not None else None
        weights2 = self.learned_weights.get(id(parent2_tree)) if parent2_tree is not None else None

        # Handle cases where one or both parents don't have weights
        if weights1 is None:
            return weights2.copy() if weights2 is not None else None
        if weights2 is None:
            return weights1.copy() if weights1 is not None else None

        # Both parents have weights - combine based on crossover mode
        if self.crossover_mode == "parent1":
            return weights1.copy()
        elif self.crossover_mode == "average":
            return (weights1 + weights2) / 2
        elif self.crossover_mode == "random":
            return weights1.copy() if self.rng.random() < 0.5 else weights2.copy()
        elif self.crossover_mode == "closest_parent":
            if offspring_tree is None:
                # Fallback to average if no offspring tree provided
                return (weights1 + weights2) / 2

            # Compute distance to each parent
            dist1 = tree_distance(offspring_tree, parent1_tree)
            dist2 = tree_distance(offspring_tree, parent2_tree)

            # Return weights from closest parent
            return weights1.copy() if dist1 <= dist2 else weights2.copy()
        else:
            # Default to averaging
            return (weights1 + weights2) / 2

    def clear_old_weights(self, keep_ids: set[int]) -> None:
        """Clear weights for genotypes no longer in population.

        This helps manage memory by removing weights for old genotypes.

        Parameters
        ----------
        keep_ids : set[int]
            Set of tree IDs to keep (active genotypes).
        """
        # Remove weights for IDs not in keep set
        old_ids = set(self.learned_weights.keys()) - keep_ids
        for tree_id in old_ids:
            del self.learned_weights[tree_id]

    def get_weights(self, tree_id: int) -> np.ndarray | None:
        """Retrieve stored weights for a genotype.

        Parameters
        ----------
        tree_id : int
            ID of the tree genotype.

        Returns
        -------
        np.ndarray | None
            Stored weights, or None if not found.
        """
        weights = self.learned_weights.get(tree_id)
        return weights.copy() if weights is not None else None

    def has_weights(self, tree_id: int) -> bool:
        """Check if weights are stored for a genotype.

        Parameters
        ----------
        tree_id : int
            ID of the tree genotype.

        Returns
        -------
        bool
            True if weights are stored, False otherwise.
        """
        return tree_id in self.learned_weights

    def __len__(self) -> int:
        """Return number of genotypes with stored weights."""
        return len(self.learned_weights)
