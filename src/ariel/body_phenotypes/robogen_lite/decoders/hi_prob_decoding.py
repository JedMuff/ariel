"""Highest-probability-decoding algorithm for ARIEL-robots.

Note
-----
* Graphs are represented as directed graphs (DiGraph) using NetworkX.
* Graphs are saved as JSON [1]_.

References
----------
.. [1] `NetworkX JSON Graph <https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.tree_data.html#networkx.readwrite.json_graph.tree_data>`_
Todo
----
- [ ] for loops to be replaced with vectorized operations
- [ ] DiGraph positioning use cartesian coordinates instead of spring layout
"""

# Evaluate type annotations in a deferred manner (ruff: UP037)
from __future__ import annotations

# Standard library
from typing import Any

# Third-party libraries
import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx import DiGraph

# Local libraries
from ariel import log
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    NUM_OF_FACES,
    ModuleFaces,
    ModuleInstance,
    ModuleRotationsIdx,
    ModuleType,
)

# Global constants
SEED = 42

# Global functions
RNG = np.random.default_rng(SEED)

# Numpy print options
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


class HighProbabilityDecoder:
    """Implements the high-probability-decoding algorithm."""

    def __init__(self, num_modules: int) -> None:
        """
        Initialize the high-probability-decoding algorithm.

        Parameters
        ----------
        num_modules : int
            Number of modules to be decoded.
        """
        self.num_modules = num_modules

        # Data structure to hold the decoded graph (not networkx graph)
        self._graph: dict[int, ModuleInstance] = {}

        # NetworkX graph
        self.graph: DiGraph[Any] = nx.DiGraph()

    def probability_matrices_to_graph(
        self,
        type_probability_space: npt.NDArray[np.float32],
        connection_probability_space: npt.NDArray[np.float32],
        rotation_probability_space: npt.NDArray[np.float32],
    ) -> DiGraph[Any]:
        """
        Convert probability matrices to a graph.

        Parameters
        ----------
        type_probability_space
            Probability space for module types.
        connection_probability_space
            Probability space for connections between modules.
        rotation_probability_space
            Probability space for module rotations.

        Returns
        -------
        DiGraph
            A graph representing the decoded modules and their connections.
        """
        # Reset the graph
        self._graph: dict[int, ModuleInstance] = {}
        self.graph: DiGraph[Any] = nx.DiGraph()

        # Store the probability spaces
        self.type_p_space = type_probability_space
        self.conn_p_space = connection_probability_space
        self.rot_p_space = rotation_probability_space

        # Apply constraints
        self.apply_connection_constraints()

        # Initialize module types and rotations
        self.set_module_types_and_rotations()

        # Decode probability spaces into a graph
        self.decode_probability_to_graph()

        # Create the final graph from the simple graph
        self.generate_networkx_graph()
        return self.graph

    def generate_networkx_graph(self) -> None:
        """Generate a NetworkX graph from the decoded graph."""
        for node in self.nodes:
            self.graph.add_node(
                node,
                type=self.type_dict[node].name,
                rotation=self.rot_dict[node].name,
            )
        for edges in self.edges:
            parent, child, face = edges
            self.graph.add_edge(
                parent,
                child,
                face=ModuleFaces(face).name,
            )

    def decode_probability_to_graph(
        self,
    ) -> None:
        """
        Decode the probability spaces into a graph.

        Raises
        ------
        ValueError
            If an attempt is made to use the core module as a child.
        ValueError
            If an attempt is made to instantiate a NONE module as a parent.
        ValueError
            If an attempt is made to instantiate a NONE module as a child.
        """
        # Create a dictionary to track instantiated modules
        pre_nodes = dict.fromkeys(range(self.num_modules), 0)

        # The core module is always instantiated
        pre_nodes[IDX_OF_CORE] = 1

        # Remove 'None' modules from instantiated modules
        for type_idx, module_type in self.type_dict.items():
            if module_type == ModuleType.NONE:
                del pre_nodes[type_idx]

        # List to hold edges (parent, child, face)
        self.edges = []

        # Available faces for connections
        available_faces = np.zeros_like(self.conn_p_space)
        available_faces[IDX_OF_CORE, :, :] = 1.0
        for _ in range(len(pre_nodes)):
            # Get the current state of the connection probability space
            current_state = self.conn_p_space * available_faces

            # Find the maximum value in the connection probability space
            max_index = np.unravel_index(
                np.argmax(current_state),
                current_state.shape,
            )

            # Convert to list for easier manipulation
            max_index = [int(i) for i in max_index]
            from_module, to_module, conn_face = max_index

            # Check if the maximum value is zero (no more connections)
            value_at_max = current_state[from_module, to_module, conn_face]
            if value_at_max == 0.0:
                msg = "No more connections can be made."
                log.debug(msg)
                break

            # Ensure the core module is never a child
            if to_module == IDX_OF_CORE:
                msg = "Cannot connect to the core module as a child.\n"
                msg += "This indicates an error in decoding."
                raise ValueError(msg)

            # Ensure no NONE modules are instantiated
            if self.type_dict[to_module] == ModuleType.NONE:
                msg = "Cannot instantiate a NONE module.\n"
                msg += "This indicates an error in decoding."
                raise ValueError(msg)

            if self.type_dict[from_module] == ModuleType.NONE:
                msg = "Cannot instantiate a NONE module.\n"
                msg += "This indicates an error in decoding."
                raise ValueError(msg)

            # Get module types and rotations
            self.edges.append(
                (from_module, to_module, conn_face),
            )

            # Update instantiated modules
            pre_nodes[to_module] = 1

            # Disable taken face
            self.conn_p_space[from_module, :, conn_face] = 0.0

            # Child has only one parent
            self.conn_p_space[:, to_module, :] = 0.0

            # Update available faces
            available_faces[to_module, :, :] = 1.0

        # Nodes and edges of the final graph
        self.nodes = {i for i in pre_nodes if pre_nodes[i] == 1}

    def set_module_types_and_rotations(self) -> None:
        """Set the module types and rotations using probability spaces."""
        # Module type from argmax of type probability space
        type_from_argmax = np.argmax(self.type_p_space, axis=1)
        self.type_dict = {
            i: ModuleType(int(type_from_argmax[i]))
            for i in range(self.num_modules)
        }

        # Constrain rotations and connections based on module types
        all_possible_faces = set(ModuleFaces)
        all_possible_rotations = set(ModuleRotationsIdx)
        for module_idx, module_type in self.type_dict.items():
            # Constrain connections based on module type
            allowed_faces = set(ALLOWED_FACES[module_type])
            disallowed_faces = all_possible_faces - allowed_faces
            for face in disallowed_faces:
                # Disable as parent
                self.conn_p_space[module_idx, :, face.value] = 0.0
                # Disable as child
                self.conn_p_space[:, module_idx, face.value] = 0.0

            # Constrain rotations based on module type
            allowed_rotations = set(ALLOWED_ROTATIONS[module_type])
            disallowed_rotations = all_possible_rotations - allowed_rotations
            for rotation in disallowed_rotations:
                self.rot_p_space[module_idx, rotation.value] = 0.0

        # Rotation type form argmax of rotation probability space
        rot_from_argmax = np.argmax(self.rot_p_space, axis=1)
        self.rot_dict = {
            i: ModuleRotationsIdx(int(rot_from_argmax[i]))
            for i in range(self.num_modules)
        }

    def apply_connection_constraints(
        self,
    ) -> None:
        """Apply connection constraints to probability spaces."""
        # Self connection not allowed
        for i in range(NUM_OF_FACES):
            np.fill_diagonal(self.conn_p_space[:, :, i], 0.0)

        # Core is unique
        self.type_p_space[:, int(ModuleType.CORE.value)] = 0.0
        self.type_p_space[IDX_OF_CORE, int(ModuleType.CORE.value)] = 1.0

        # Core is always a parent, never a child
        self.conn_p_space[:, IDX_OF_CORE, :] = 0.0
