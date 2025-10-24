"""Tree-based genotype representation using NetworkX graphs.

This module provides a tree genotype system for evolutionary robotics that:
- Uses NetworkX directed graphs for efficient tree representation
- Aligns with the ARIEL body phenotype space (6 faces, 8 rotations at 45° intervals)
- Supports mutation and crossover operations
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

import networkx as nx
import numpy as np

from ariel.body_phenotypes.robogen_lite.collision_utils import (
    check_tree_self_collision,
)
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.ec.genotypes.base import Genotype


class TreeGenotype(Genotype):
    """Tree-based genotype using NetworkX graphs with ARIEL phenotype space.

    Attributes
    ----------
    max_part_limit : int
        Maximum number of parts allowed in the robot body.
    max_actuators : int
        Maximum number of actuator modules (hinges) allowed.
    module_types : list[ModuleType]
        Available module types for tree construction.
    default_depth : int
        Default depth for random tree generation.
    """

    def __init__(
        self,
        max_part_limit: int = 25,
        max_actuators: int = 12,
        default_depth: int = 4,
        mutation_strength: int | None = 1,
        mutation_reps: int | None = 1,
        mutate_attributes_prob: float = 0.0,
        enable_collision_repair: bool = True,
        max_repair_iterations: int = 100,
    ):
        """Initialize the tree genotype system.

        Parameters
        ----------
        max_part_limit : int, optional
            Maximum number of parts in the robot, by default 25.
        max_actuators : int, optional
            Maximum number of actuators, by default 12.
        default_depth : int, optional
            Default depth for random tree generation, by default 4.
        enable_collision_repair : bool, optional
            Whether to enable collision repair after genetic operations, by default True.
        max_repair_iterations : int, optional
            Maximum iterations for collision repair, by default 100.
        """
        self.max_part_limit = max_part_limit
        self.max_actuators = max_actuators
        self.default_depth = default_depth
        self.enable_collision_repair = enable_collision_repair
        self.max_repair_iterations = max_repair_iterations

        # Module types available for tree construction (excluding NONE)
        self.module_types = [ModuleType.BRICK, ModuleType.HINGE, None]

        # Node index counter for graph construction
        self.next_node_id = 1  # 0 is reserved for CORE

        if mutation_strength is not None:
            self.mutation_strength = mutation_strength
        else:
            self.mutation_strength = random.randint(0, 3)

        if mutation_reps is not None:
            self.mutation_reps = mutation_reps
        else:
            self.mutation_reps = random.randint(1, 3)
        self.mutate_attributes_prob = mutate_attributes_prob

        self.tree = self.random_tree(self.default_depth)
        

    def random_tree(self, depth: int, origin: bool = True) -> nx.DiGraph:
        """Generate a random tree represented as a NetworkX directed graph.

        Parameters
        ----------
        depth : int
            Maximum depth of the tree.
        origin : bool, optional
            If True, start with CORE module, by default True.

        Returns
        -------
        nx.DiGraph
            A directed graph representing the robot tree structure.
        """
        graph = nx.DiGraph()

        if origin:
            # Add CORE node as root
            graph.add_node(
                IDX_OF_CORE,
                type=ModuleType.CORE.name,
                rotation=ModuleRotationsIdx.DEG_0.name,
            )
            self.next_node_id = 1

            # Build tree from CORE
            self._build_random_subtree(graph, IDX_OF_CORE, depth, ModuleType.CORE)

        # Apply collision repair
        graph = self.repair_tree(graph)

        return graph

    def _build_random_subtree(
        self,
        graph: nx.DiGraph,
        parent_id: int,
        depth: int,
        parent_type: ModuleType,
    ) -> None:
        """Recursively build a random subtree from a parent node.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph being constructed.
        parent_id : int
            The node ID of the parent.
        depth : int
            Remaining depth for tree construction.
        parent_type : ModuleType
            The type of the parent module.
        """
        if depth == 0:
            return

        # Get available attachment faces for this module type
        available_faces = ALLOWED_FACES.get(parent_type, [])

        # For each available face, potentially add a child
        for face in available_faces:
            # Randomly decide whether to add a child at this face
            module_type = random.choice(self.module_types)

            if module_type is None:
                continue

            # Create child node
            child_id = self.next_node_id
            self.next_node_id += 1

            # Select random rotation for the child
            allowed_rotations = ALLOWED_ROTATIONS.get(module_type, [ModuleRotationsIdx.DEG_0])
            rotation = random.choice(allowed_rotations)

            # Add node to graph
            graph.add_node(
                child_id,
                type=module_type.name,
                rotation=rotation.name,
            )

            # Add edge from parent to child with face information
            graph.add_edge(parent_id, child_id, face=face.name)

            # Recursively build subtree
            self._build_random_subtree(graph, child_id, depth - 1, module_type)

    def repair_tree(self, tree: nx.DiGraph) -> nx.DiGraph:
        """
        Repair a tree by removing modules that cause self-collisions.

        This method builds a new collision-free tree using BFS traversal.
        Starting from the core, it incrementally adds nodes and checks for
        collisions after each addition. Nodes that cause collisions (and
        their descendants) are excluded from the final tree.

        Parameters
        ----------
        tree : nx.DiGraph
            The tree to repair

        Returns
        -------
        nx.DiGraph
            The repaired tree with collision-free structure
        """
        if not self.enable_collision_repair:
            return tree

        from collections import deque

        # Start with just the CORE node
        repaired_tree = nx.DiGraph()
        repaired_tree.add_node(
            IDX_OF_CORE,
            **tree.nodes[IDX_OF_CORE]
        )

        # BFS queue - contains nodes to process (nodes that are in repaired tree)
        queue = deque([IDX_OF_CORE])

        # Process nodes in BFS order
        while queue:
            parent_id = queue.popleft()

            # Get all children of this parent in the original tree
            children = list(tree.successors(parent_id))

            for child_id in children:
                # Get child node attributes and edge attributes from original tree
                child_attrs = tree.nodes[child_id]
                edge_attrs = tree.edges[parent_id, child_id]

                # Add child node to repaired tree
                repaired_tree.add_node(child_id, **child_attrs)

                # Add edge from parent to child
                repaired_tree.add_edge(parent_id, child_id, **edge_attrs)

                # Check if adding this child causes a collision
                if check_tree_self_collision(repaired_tree, penetration_threshold=-0.001):
                    # Collision detected - remove this child
                    # (Don't add to queue, so its descendants won't be processed)
                    repaired_tree.remove_node(child_id)
                else:
                    # No collision - keep the child and process its children later
                    queue.append(child_id)

        return repaired_tree

    def random_population(self, pop_size: int, **kwargs: Any) -> list[nx.DiGraph]:
        """Generate a random population of tree genotypes.

        Implements the abstract method from Genotype base class.

        Parameters
        ----------
        pop_size : int
            Number of individuals in the population.
        **kwargs : Any
            Additional parameters:
            - depth (int): Maximum depth of each tree, defaults to self.default_depth

        Returns
        -------
        list[nx.DiGraph]
            List of NetworkX graphs representing the population.
        """
        depth = kwargs.get("depth", self.default_depth)
        population = []
        for _ in range(pop_size):
            tree = self.random_tree(depth)
            population.append(tree)
        return population

    def mutate_tree(self, tree: nx.DiGraph) -> nx.DiGraph:
        """Mutate a tree by replacing a random subtree.

        Parameters
        ----------
        tree : nx.DiGraph
            The tree to mutate.

        Returns
        -------
        nx.DiGraph
            The mutated tree.
        """
        mutated_tree = deepcopy(tree)

        for _ in range(self.mutation_reps):
            # Get all non-root nodes
            nodes = [n for n in mutated_tree.nodes() if n != IDX_OF_CORE]

            if not nodes:
                # If tree only has root, add a small random subtree
                depth = 1
                self._build_random_subtree(
                    mutated_tree,
                    IDX_OF_CORE,
                    depth,
                    ModuleType.CORE,
                )
                return mutated_tree

            # Select random node to replace
            target_node = random.choice(nodes)

            # Find parent and edge information
            predecessors = list(mutated_tree.predecessors(target_node))
            if not predecessors:
                return mutated_tree

            parent = predecessors[0]
            parent_type = ModuleType[mutated_tree.nodes[parent]["type"]]
            edge_data = mutated_tree.edges[parent, target_node]
            face = ModuleFaces[edge_data["face"]]

            # Remove old subtree
            descendants = list(nx.descendants(mutated_tree, target_node))
            mutated_tree.remove_nodes_from([target_node] + descendants)

            # Create new random subtree at this position
            depth = self.mutation_strength #self.mutation_strength = random.randint(0, 1)
            module_type = random.choice(self.module_types)

            if module_type is not None:
                # Renumber node IDs to avoid conflicts
                self.next_node_id = max(mutated_tree.nodes()) + 1 if mutated_tree.nodes() else 1

                new_node_id = self.next_node_id
                self.next_node_id += 1

                # Select random rotation
                allowed_rotations = ALLOWED_ROTATIONS.get(module_type, [ModuleRotationsIdx.DEG_0])
                rotation = random.choice(allowed_rotations)

                # Add new node
                mutated_tree.add_node(
                    new_node_id,
                    type=module_type.name,
                    rotation=rotation.name,
                )

                # Reconnect to parent
                mutated_tree.add_edge(parent, new_node_id, face=face.name)

                # Build random subtree from new node
                self._build_random_subtree(mutated_tree, new_node_id, depth, module_type)

        # Apply collision repair
        mutated_tree = self.repair_tree(mutated_tree)

        return mutated_tree

    def mutation(self, population: np.ndarray) -> np.ndarray:
        """Apply mutation to an entire population.

        Parameters
        ----------
        population : np.ndarray
            Array of tree genotypes.

        Returns
        -------
        np.ndarray
            Array of mutated tree genotypes.
        """
        new_population = []
        for individual in population:
            mutated_tree = self.mutate_tree(individual.tree)
            mutated_individual = TreeGenotype(
                max_part_limit=self.max_part_limit,
                max_actuators=self.max_actuators,
                default_depth=self.default_depth,
                mutation_strength=self.mutation_strength,
                mutation_reps=self.mutation_reps,
                enable_collision_repair=self.enable_collision_repair,
                max_repair_iterations=self.max_repair_iterations,
            )
            mutated_individual.tree = mutated_tree
            
            # mutate mutation strength and reps for next generation
            random_number_between_0_and_1 = random.random()
            if random_number_between_0_and_1 < self.mutate_attributes_prob:
                mutated_individual.mutation_strength = mutated_individual.mutation_strength + random.randint(-1, 1)
                mutated_individual.mutation_strength = max(0, mutated_individual.mutation_strength)
                mutated_individual.mutation_reps = mutated_individual.mutation_reps + random.randint(-1, 2)
                mutated_individual.mutation_reps = max(1, mutated_individual.mutation_reps)

            new_population.append(mutated_individual)
            
        return np.array(new_population, dtype=object)

    def crossover_tree(
        self,
        parent1: nx.DiGraph,
        parent2: nx.DiGraph,
    ) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Perform subtree crossover between two parent trees.

        Parameters
        ----------
        parent1 : nx.DiGraph
            First parent tree.
        parent2 : nx.DiGraph
            Second parent tree.

        Returns
        -------
        tuple[nx.DiGraph, nx.DiGraph]
            Two offspring trees.
        """
        # Handle None cases
        if parent1 is None or len(parent1.nodes()) == 0:
            return deepcopy(parent2), deepcopy(parent1) if parent1 is not None else nx.DiGraph()
        if parent2 is None or len(parent2.nodes()) == 0:
            return deepcopy(parent1), deepcopy(parent2) if parent2 is not None else nx.DiGraph()

        offspring1 = deepcopy(parent1)
        offspring2 = deepcopy(parent2)

        # Get non-root nodes from each parent
        nodes1 = [n for n in offspring1.nodes() if n != IDX_OF_CORE]
        nodes2 = [n for n in offspring2.nodes() if n != IDX_OF_CORE]

        if not nodes1 or not nodes2:
            return offspring1, offspring2

        # Select random crossover points
        crossover_node1 = random.choice(nodes1)
        crossover_node2 = random.choice(nodes2)

        # Get parent information for crossover nodes
        parents1 = list(offspring1.predecessors(crossover_node1))
        parents2 = list(offspring2.predecessors(crossover_node2))

        if not parents1 or not parents2:
            return offspring1, offspring2

        parent_node1 = parents1[0]
        parent_node2 = parents2[0]

        # Get edge data (face information)
        edge_data1 = offspring1.edges[parent_node1, crossover_node1]
        edge_data2 = offspring2.edges[parent_node2, crossover_node2]

        # Extract subtrees
        subtree1_nodes = [crossover_node1] + list(nx.descendants(offspring1, crossover_node1))
        subtree2_nodes = [crossover_node2] + list(nx.descendants(offspring2, crossover_node2))

        subtree1 = offspring1.subgraph(subtree1_nodes).copy()
        subtree2 = offspring2.subgraph(subtree2_nodes).copy()

        # Remove subtrees from offspring
        offspring1.remove_nodes_from(subtree1_nodes)
        offspring2.remove_nodes_from(subtree2_nodes)

        # Renumber subtree2 nodes to avoid conflicts with offspring1
        node_mapping2 = {}
        max_id1 = max(offspring1.nodes()) if offspring1.nodes() else 0
        for i, node in enumerate(subtree2.nodes()):
            node_mapping2[node] = max_id1 + i + 1
        subtree2 = nx.relabel_nodes(subtree2, node_mapping2)

        # Renumber subtree1 nodes to avoid conflicts with offspring2
        node_mapping1 = {}
        max_id2 = max(offspring2.nodes()) if offspring2.nodes() else 0
        for i, node in enumerate(subtree1.nodes()):
            node_mapping1[node] = max_id2 + i + 1
        subtree1 = nx.relabel_nodes(subtree1, node_mapping1)

        # Attach swapped subtrees
        # Add subtree2 to offspring1
        offspring1 = nx.compose(offspring1, subtree2)
        new_root2 = node_mapping2[crossover_node2]
        offspring1.add_edge(parent_node1, new_root2, **edge_data1)

        # Add subtree1 to offspring2
        offspring2 = nx.compose(offspring2, subtree1)
        new_root1 = node_mapping1[crossover_node1]
        offspring2.add_edge(parent_node2, new_root1, **edge_data2)

        # Apply collision repair to both offspring
        offspring1 = self.repair_tree(offspring1)
        offspring2 = self.repair_tree(offspring2)

        return offspring1, offspring2

    def crossover(
        self,
        population: np.ndarray,
        mating_pool1: np.ndarray,
        mating_pool2: np.ndarray,
    ) -> np.ndarray:
        """Apply crossover to pairs from mating pools.

        Parameters
        ----------
        population : np.ndarray
            The population array of TreeGenotype objects.
        mating_pool1 : np.ndarray
            Indices of first parents.
        mating_pool2 : np.ndarray
            Indices of second parents.

        Returns
        -------
        np.ndarray
            Array of offspring TreeGenotype objects from crossover.
        """
        new_population = []
        for parent1_idx, parent2_idx in zip(mating_pool1, mating_pool2):
            parent1_genotype = population[parent1_idx]
            parent2_genotype = population[parent2_idx]

            # Extract trees from TreeGenotype objects
            parent1_tree = parent1_genotype.tree
            parent2_tree = parent2_genotype.tree

            # Perform crossover on trees
            child1_tree, _ = self.crossover_tree(parent1_tree, parent2_tree)

            # Create new TreeGenotype object for offspring
            child1_genotype = TreeGenotype(
                max_part_limit=self.max_part_limit,
                max_actuators=self.max_actuators,
                default_depth=self.default_depth,
                mutation_strength=parent1_genotype.mutation_strength,
                mutation_reps=parent1_genotype.mutation_reps,
                mutate_attributes_prob=self.mutate_attributes_prob,
                enable_collision_repair=self.enable_collision_repair,
                max_repair_iterations=self.max_repair_iterations,
            )
            child1_genotype.tree = child1_tree

            # Inherit mutation parameters from first parent (could also be averaged or randomized)
            # Already set in constructor above

            new_population.append(child1_genotype)
        return np.array(new_population, dtype=object)

    def get_tree_size(self, tree: nx.DiGraph) -> int:
        """Get the number of modules in a tree.

        Parameters
        ----------
        tree : nx.DiGraph
            The tree graph.

        Returns
        -------
        int
            Number of nodes in the tree.
        """
        return len(tree.nodes())

    def get_tree_depth(self, tree: nx.DiGraph) -> int:
        """Get the maximum depth of a tree.

        Parameters
        ----------
        tree : nx.DiGraph
            The tree graph.

        Returns
        -------
        int
            Maximum depth of the tree.
        """
        if not tree or len(tree.nodes()) == 0:
            return 0

        # Calculate longest path from root
        try:
            return nx.dag_longest_path_length(tree)
        except nx.NetworkXError:
            return 0

    # ========== Genotype Base Class Interface ==========

    def random_genome(self) -> nx.DiGraph:
        """Generate a single random genome.

        Implements the abstract method from Genotype base class.
        Delegates to random_tree() with default depth.

        Returns
        -------
        nx.DiGraph
            A randomly generated tree genome.
        """
        return self.random_tree(depth=self.default_depth)

    def mutate(self, genome: nx.DiGraph) -> nx.DiGraph:
        """Mutate a single genome.

        Implements the abstract method from Genotype base class.
        Delegates to mutate_tree().

        Parameters
        ----------
        genome : nx.DiGraph
            The tree genome to mutate.

        Returns
        -------
        nx.DiGraph
            The mutated tree genome.
        """
        return self.mutate_tree(genome)

    def crossover(
        self,
        parent1: nx.DiGraph,
        parent2: nx.DiGraph,
    ) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Perform crossover between two parent genomes.

        Implements the abstract method from Genotype base class.
        Delegates to crossover_tree().

        Parameters
        ----------
        parent1 : nx.DiGraph
            First parent tree genome.
        parent2 : nx.DiGraph
            Second parent tree genome.

        Returns
        -------
        tuple[nx.DiGraph, nx.DiGraph]
            Two offspring tree genomes.
        """
        return self.crossover_tree(parent1, parent2)
