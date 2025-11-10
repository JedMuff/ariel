"""MorphologicalMeasures class adapted for Ariel framework.

This module provides morphological analysis for modular robots in the Ariel framework.
The measures are based on the paper:

    Miras, K., Haasdijk, E., Glette, K., Eiben, A.E. (2018).
    Search Space Analysis of Evolvable Robot Morphologies.
    In: Sim, K., Kaufmann, P. (eds) Applications of Evolutionary Computation.
    EvoApplications 2018. Lecture Notes in Computer Science(), vol 10784. Springer, Cham.
    https://doi.org/10.1007/978-3-319-77538-8_47

Usage Example:
    ```python
    import networkx as nx
    from morphological_measures import Body, MorphologicalMeasures

    # Create a robot body graph (from your Ariel genotype)
    graph = nx.DiGraph()
    # ... populate graph with nodes and edges ...

    # Create Body wrapper
    body = Body(graph)

    # Compute morphological measures
    measures = MorphologicalMeasures(body)

    # Access various metrics
    print(f"Number of modules: {measures.num_modules}")
    print(f"Branching: {measures.branching}")
    print(f"Limbs: {measures.limbs}")
    print(f"Length of limbs: {measures.length_of_limbs}")
    print(f"Coverage: {measures.coverage}")
    print(f"Symmetry: {measures.symmetry}")
    ```

Notes:
    - Only works for robots with right angle module rotations (90 degrees)
    - Some measures only work for 2D robots (check the is_2d property)
    - The Body class wraps Ariel's graph representation to provide the interface
      expected by MorphologicalMeasures
"""

from __future__ import annotations

from collections import deque
from itertools import product
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE, ModuleFaces, ModuleType
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.modules.module import Module

TModule = TypeVar("TModule", bound=np.generic)

# Type aliases for clarity (matching revolve2 naming conventions)
ActiveHinge = HingeModule  # Map ActiveHinge to HingeModule
Brick = BrickModule
Core = CoreModule

# Vector3 type for grid positions (replacing pyrr.Vector3)
Vector3 = NDArray[np.int_]


class Body:
    """
    Wrapper class to adapt Ariel's graph representation to the interface expected by MorphologicalMeasures.

    This class provides methods to work with the robot body graph, including
    grid conversion and module queries.
    """

    def __init__(self, graph: DiGraph[Any], max_part_limit: int | None = None) -> None:
        """
        Initialize the Body from a NetworkX graph.

        Parameters
        ----------
        graph : DiGraph
            Graph representation of the robot body from Ariel framework.
        max_part_limit : int | None, optional
            Maximum number of parts allowed in the robot body.
            Used for normalized size calculation.
        """
        self.graph = graph
        self.max_part_limit = max_part_limit
        self._core: CoreModule | None = None
        self._modules: dict[int, Module] = {}
        self._build_modules()

    def _build_modules(self) -> None:
        """Build module instances from the graph."""
        for node in self.graph.nodes:
            module_type = self.graph.nodes[node]["type"]

            match module_type:
                case ModuleType.CORE.name:
                    self._core = CoreModule(index=IDX_OF_CORE)
                    self._modules[node] = self._core
                case ModuleType.HINGE.name:
                    self._modules[node] = HingeModule(index=node)
                case ModuleType.BRICK.name:
                    self._modules[node] = BrickModule(index=node)
                case ModuleType.NONE.name:
                    pass  # Skip None modules
                case _:
                    pass  # Unknown module type

    @property
    def core(self) -> CoreModule:
        """Get the core module."""
        if self._core is None:
            msg = "No core module found in graph"
            raise ValueError(msg)
        return self._core

    def find_modules_of_type(
        self, module_type: type[Module], exclude: list[type[Module]] | None = None
    ) -> list[Module]:
        """
        Find all modules of a specific type in the body.

        Parameters
        ----------
        module_type : type[Module]
            The type of module to search for.
        exclude : list[type[Module]] | None
            Module types to exclude from the search.

        Returns
        -------
        list[Module]
            List of modules matching the type criteria.
        """
        exclude = exclude or []
        result = []

        for module in self._modules.values():
            # Check if module is of the requested type
            if isinstance(module, module_type):
                # Check if module should be excluded
                if not any(isinstance(module, exc_type) for exc_type in exclude):
                    result.append(module)

        return result

    def to_grid(self) -> tuple[NDArray[Any], Vector3]:
        """
        Convert the graph representation to a 3D grid with module positions.

        Returns
        -------
        tuple[NDArray, Vector3]
            A tuple of (grid, core_position) where:
            - grid is a 3D numpy array with modules at their positions
            - core_position is the (x, y, z) position of the core in the grid
        """
        # Start with the core at origin
        if self._core is None:
            msg = "No core module found"
            raise ValueError(msg)

        # Track positions of all modules (node_id -> (x, y, z))
        positions: dict[int, tuple[int, int, int]] = {IDX_OF_CORE: (0, 0, 0)}

        # BFS to determine all module positions based on graph edges
        queue = deque([IDX_OF_CORE])
        visited = {IDX_OF_CORE}

        # Face direction vectors (in x, y, z)
        # These represent how positions change when moving in each face direction
        face_vectors = {
            ModuleFaces.FRONT: (0, 1, 0),   # Forward along y-axis
            ModuleFaces.BACK: (0, -1, 0),   # Backward along y-axis
            ModuleFaces.RIGHT: (1, 0, 0),   # Right along x-axis
            ModuleFaces.LEFT: (-1, 0, 0),   # Left along x-axis
            ModuleFaces.TOP: (0, 0, 1),     # Up along z-axis
            ModuleFaces.BOTTOM: (0, 0, -1), # Down along z-axis
        }

        while queue:
            current_node = queue.popleft()
            current_pos = positions[current_node]

            # Check all outgoing edges from this node
            for edge in self.graph.out_edges(current_node, data=True):
                from_node, to_node, edge_data = edge

                if to_node in visited or to_node not in self._modules:
                    continue

                # Get the face this connection is on
                face_name = edge_data.get("face", "FRONT")
                face = ModuleFaces[face_name]

                # Calculate new position based on face direction
                direction = face_vectors[face]
                new_pos = (
                    current_pos[0] + direction[0],
                    current_pos[1] + direction[1],
                    current_pos[2] + direction[2],
                )

                positions[to_node] = new_pos
                visited.add(to_node)
                queue.append(to_node)

        # Find bounding box
        if not positions:
            # Empty body, just core
            grid = np.empty((1, 1, 1), dtype=object)
            grid[0, 0, 0] = self._core
            return grid, np.array([0, 0, 0], dtype=np.int_)

        xs, ys, zs = zip(*positions.values())
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        # Create grid with appropriate size
        grid_shape = (
            max_x - min_x + 1,
            max_y - min_y + 1,
            max_z - min_z + 1,
        )
        grid = np.empty(grid_shape, dtype=object)
        grid.fill(None)

        # Place modules in grid
        core_grid_pos = None
        for node_id, pos in positions.items():
            grid_pos = (
                pos[0] - min_x,
                pos[1] - min_y,
                pos[2] - min_z,
            )
            grid[grid_pos] = self._modules[node_id]

            if node_id == IDX_OF_CORE:
                core_grid_pos = np.array(grid_pos, dtype=np.int_)

        if core_grid_pos is None:
            msg = "Core position not found in grid"
            raise ValueError(msg)

        return grid, core_grid_pos

    def get_module_children(self, module: Module) -> dict[ModuleFaces, Module | None]:
        """
        Get the children of a module based on graph edges.

        Parameters
        ----------
        module : Module
            The module to get children for.

        Returns
        -------
        dict[ModuleFaces, Module | None]
            Dictionary mapping faces to child modules (or None if no child).
        """
        children: dict[ModuleFaces, Module | None] = {}

        # Get all valid attachment faces for this module type
        if isinstance(module, CoreModule):
            valid_faces = list(ModuleFaces)
        elif isinstance(module, BrickModule):
            valid_faces = [
                ModuleFaces.FRONT,
                ModuleFaces.LEFT,
                ModuleFaces.RIGHT,
                ModuleFaces.TOP,
                ModuleFaces.BOTTOM,
            ]
        elif isinstance(module, HingeModule):
            valid_faces = [ModuleFaces.FRONT]
        else:
            valid_faces = []

        # Initialize all faces to None
        for face in valid_faces:
            children[face] = None

        # Find module's node ID in graph
        module_node = None
        for node_id, mod in self._modules.items():
            if mod is module:
                module_node = node_id
                break

        if module_node is None:
            return children

        # Get children from graph edges
        for edge in self.graph.out_edges(module_node, data=True):
            _, to_node, edge_data = edge
            if to_node in self._modules:
                face_name = edge_data.get("face", "FRONT")
                face = ModuleFaces[face_name]
                children[face] = self._modules[to_node]

        return children

    def get_module_attachment_points(self, module: Module) -> dict[int, ModuleFaces]:
        """
        Get the attachment points (faces) available on a module.

        Parameters
        ----------
        module : Module
            The module to get attachment points for.

        Returns
        -------
        dict[int, ModuleFaces]
            Dictionary mapping indices to faces.
        """
        if isinstance(module, CoreModule):
            faces = list(ModuleFaces)
        elif isinstance(module, BrickModule):
            faces = [
                ModuleFaces.FRONT,
                ModuleFaces.LEFT,
                ModuleFaces.RIGHT,
                ModuleFaces.TOP,
                ModuleFaces.BOTTOM,
            ]
        elif isinstance(module, HingeModule):
            faces = [ModuleFaces.FRONT]
        else:
            faces = []

        return {i: face for i, face in enumerate(faces)}


class MorphologicalMeasures(Generic[TModule]):
    """
    Modular robot morphological measures.

    Only works for robot with only right angle module rotations (90 degrees).
    Some measures only work for 2d robots, which is noted in their docstring.

    The measures are based on the following paper:
    Miras, K., Haasdijk, E., Glette, K., Eiben, A.E. (2018).
    Search Space Analysis of Evolvable Robot Morphologies.
    In: Sim, K., Kaufmann, P. (eds) Applications of Evolutionary Computation.
    EvoApplications 2018. Lecture Notes in Computer Science(), vol 10784. Springer, Cham.
    https://doi.org/10.1007/978-3-319-77538-8_47
    """

    """Represents the modules of a body in a 3D tensor."""
    grid: NDArray[TModule]
    symmetry_grid: NDArray[TModule]
    """Position of the core in 'body_as_grid'."""
    core_grid_position: Vector3[np.int_]

    """If the robot is two dimensional, i.e. all module rotations are 0 degrees."""
    is_2d: bool

    core: Core
    modules: list[Module]
    bricks: list[Brick]
    active_hinges: list[ActiveHinge]

    """If all slots of the core are filled with other modules."""
    core_is_filled: bool

    """Bricks which have all slots filled with other modules."""
    filled_bricks: list[Brick]

    """Active hinges which have all slots filled with other modules."""
    filled_active_hinges: list[ActiveHinge]

    """
    Modules that only connect to one other module.
    
    This includes children and parents.
    """
    single_neighbour_modules: list[Module]

    """
    Bricks that are only connected to one other module.

    Both children and parent are counted.
    """
    single_neighbour_bricks: list[Brick]

    """
    Bricks that are connected to exactly two other modules.

    Both children and parent are counted.
    """
    double_neighbour_bricks: list[Brick]

    """
    Active hinges that are connected to exactly two other modules.

    Both children and parent are counted.
    """
    double_neighbour_active_hinges: list[ActiveHinge]

    """
    X/Y-plane symmetry according to the paper but in 3D.

    X-axis is defined as forward/backward for the core module
    Y-axis is defined as left/right for the core module.
    """
    xy_symmetry: float

    """
    X/Z-plane symmetry according to the paper but in 3D.

    X-axis is defined as forward/backward for the core module
    Z-axis is defined as up/down for the core module.
    """
    xz_symmetry: float

    """
    Y/Z-plane symmetry according to the paper but in 3D.

    Y-axis is defined as left/right for the core module.
    Z-axis is defined as up/down for the core module.
    """
    yz_symmetry: float

    def __init__(self, body: Body) -> None:
        """
        Initialize this object.

        :param body: The body to measure.
        """
        self.body = body  # Store body reference
        self.grid, self.core_grid_position = body.to_grid()
        self.core = body.core
        self.is_2d = self.__calculate_is_2d()
        self.modules = body.find_modules_of_type(Module, exclude=[Core])
        self.bricks = body.find_modules_of_type(Brick)
        self.active_hinges = body.find_modules_of_type(ActiveHinge)
        self.core_is_filled = self.__calculate_core_is_filled()
        self.filled_bricks = self.__calculate_filled_bricks()
        self.filled_active_hinges = self.__calculate_filled_active_hinges()
        self.single_neighbour_bricks = self.__calculate_single_neighbour_bricks()
        self.single_neighbour_modules = self.__calculate_single_neighbour_modules()
        self.double_neighbour_bricks = self.__calculate_double_neighbour_bricks()
        self.double_neighbour_active_hinges = (
            self.__calculate_double_neighbour_active_hinges()
        )
        self.effective_joints_modules = self.__calculate_effective_joints()

        self.__pad_grid()
        self.xy_symmetry = self.__calculate_xy_symmetry()
        self.xz_symmetry = self.__calculate_xz_symmetry()
        self.yz_symmetry = self.__calculate_yz_symmetry()

    def __calculate_is_2d(self) -> bool:
        """
        Check if all modules in the body are in a 2D plane (all z-positions are the same).

        Returns
        -------
        bool
            True if the robot is 2D, False otherwise.
        """
        # Check if all modules have the same z-coordinate in the grid
        z_positions = set()
        for x, y, z in product(
            range(self.grid.shape[0]),
            range(self.grid.shape[1]),
            range(self.grid.shape[2]),
        ):
            if self.grid[x, y, z] is not None:
                z_positions.add(z)

        # If all modules share the same z position, the robot is 2D
        return len(z_positions) <= 1

    def __calculate_core_is_filled(self) -> bool:
        children = self.body.get_module_children(self.core)
        attachment_points = self.body.get_module_attachment_points(self.core)
        return all(
            children.get(face) is not None for face in attachment_points.values()
        )

    def __calculate_filled_bricks(self) -> list[Brick]:
        result = []
        for brick in self.bricks:
            children = self.body.get_module_children(brick)
            attachment_points = self.body.get_module_attachment_points(brick)
            if all(children.get(face) is not None for face in attachment_points.values()):
                result.append(brick)
        return result

    def __calculate_filled_active_hinges(self) -> list[ActiveHinge]:
        result = []
        for active_hinge in self.active_hinges:
            children = self.body.get_module_children(active_hinge)
            attachment_points = self.body.get_module_attachment_points(active_hinge)
            if all(children.get(face) is not None for face in attachment_points.values()):
                result.append(active_hinge)
        return result

    def __calculate_single_neighbour_bricks(self) -> list[Brick]:
        result = []
        for brick in self.bricks:
            children = self.body.get_module_children(brick)
            attachment_points = self.body.get_module_attachment_points(brick)
            if all(children.get(face) is None for face in attachment_points.values()):
                result.append(brick)
        return result

    def __calculate_single_neighbour_modules(self) -> list[Module]:
        result = []
        for module in self.modules:
            children = self.body.get_module_children(module)
            attachment_points = self.body.get_module_attachment_points(module)
            if all(children.get(face) is None for face in attachment_points.values()):
                result.append(module)
        return result

    def __calculate_double_neighbour_bricks(self) -> list[Brick]:
        result = []
        for brick in self.bricks:
            children = self.body.get_module_children(brick)
            attachment_points = self.body.get_module_attachment_points(brick)
            num_children = sum(
                1 for face in attachment_points.values() if children.get(face) is not None
            )
            if num_children == 1:
                result.append(brick)
        return result

    def __calculate_double_neighbour_active_hinges(self) -> list[ActiveHinge]:
        result = []
        for active_hinge in self.active_hinges:
            children = self.body.get_module_children(active_hinge)
            attachment_points = self.body.get_module_attachment_points(active_hinge)
            num_children = sum(
                1 for face in attachment_points.values() if children.get(face) is not None
            )
            if num_children == 1:
                result.append(active_hinge)
        return result

    def __calculate_effective_joints(self) -> list[Module]:
        """
        Calculate modules with both opposite faces attached.

        Effective joints are modules (core or bricks) that have both opposite
        faces attached to other modules.

        Returns
        -------
        list[Module]
            List of modules that qualify as effective joints.
        """
        result = []

        # Define opposite face pairs
        opposite_pairs = [
            (ModuleFaces.FRONT, ModuleFaces.BACK),
            (ModuleFaces.LEFT, ModuleFaces.RIGHT),
            (ModuleFaces.TOP, ModuleFaces.BOTTOM),
        ]

        # Check core
        children = self.body.get_module_children(self.core)
        for face1, face2 in opposite_pairs:
            if children.get(face1) is not None and children.get(face2) is not None:
                result.append(self.core)
                break  # Only count the core once

        # Check bricks (bricks don't have BACK face, only FRONT)
        # So for bricks, opposite pairs are: (LEFT, RIGHT) and (TOP, BOTTOM)
        brick_opposite_pairs = [
            (ModuleFaces.LEFT, ModuleFaces.RIGHT),
            (ModuleFaces.TOP, ModuleFaces.BOTTOM),
        ]

        for brick in self.bricks:
            children = self.body.get_module_children(brick)
            for face1, face2 in brick_opposite_pairs:
                if children.get(face1) is not None and children.get(face2) is not None:
                    result.append(brick)
                    break  # Only count each brick once

        return result

    def __pad_grid(self) -> None:
        x, y, z = self.grid.shape
        xoffs, yoffs, zoffs = self.core_grid_position
        self.symmetry_grid = np.empty(
            shape=(x + xoffs, y + yoffs, z + zoffs), dtype=Module
        )
        self.symmetry_grid.fill(None)
        self.symmetry_grid[:x, :y, :z] = self.grid

    def __calculate_xy_symmetry(self) -> float:
        num_along_plane = 0
        num_symmetrical = 0
        for x, y, z in product(
            range(self.bounding_box_depth),
            range(self.bounding_box_width),
            range(1, (self.bounding_box_height - 1) // 2),
        ):
            if self.symmetry_grid[x, y, self.core_grid_position[2]] is not None:
                num_along_plane += 1
            if self.symmetry_grid[
                x, y, self.core_grid_position[2] + z
            ] is not None and type(
                self.symmetry_grid[x, y, self.core_grid_position[2] + z]
            ) is type(
                self.symmetry_grid[x, y, self.core_grid_position[2] - z]
            ):
                num_symmetrical += 2

        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else difference

    def __calculate_xz_symmetry(self) -> float:
        num_along_plane = 0
        num_symmetrical = 0
        for x, y, z in product(
            range(self.bounding_box_depth),
            range(1, (self.bounding_box_width - 1) // 2),
            range(self.bounding_box_height),
        ):
            if self.symmetry_grid[x, self.core_grid_position[1], z] is not None:
                num_along_plane += 1
            if self.symmetry_grid[
                x, self.core_grid_position[1] + y, z
            ] is not None and type(
                self.symmetry_grid[x, self.core_grid_position[1] + y, z]
            ) is type(
                self.symmetry_grid[x, self.core_grid_position[1] - y, z]
            ):
                num_symmetrical += 2
        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else difference

    def __calculate_yz_symmetry(self) -> float:
        num_along_plane = 0
        num_symmetrical = 0
        for x, y, z in product(
            range(1, (self.bounding_box_depth - 1) // 2),
            range(self.bounding_box_width),
            range(self.bounding_box_height),
        ):
            if self.symmetry_grid[self.core_grid_position[0], y, z] is not None:
                num_along_plane += 1
            if self.symmetry_grid[
                self.core_grid_position[0] + x, y, z
            ] is not None and type(
                self.symmetry_grid[self.core_grid_position[0] + x, y, z]
            ) is type(
                self.symmetry_grid[self.core_grid_position[0] - x, y, z]
            ):
                num_symmetrical += 2
        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else difference

    @property
    def bounding_box_depth(self) -> int:
        """
        Get the depth of the bounding box around the body.

        Forward/backward axis for the core module.

        :returns: The depth.
        """
        return self.grid.shape[0]

    @property
    def bounding_box_width(self) -> int:
        """
        Get the width of the bounding box around the body.

        Right/left axis for the core module.

        :returns: The width.
        """
        return self.grid.shape[1]

    @property
    def bounding_box_height(self) -> int:
        """
        Get the height of the bounding box around the body.

        Up/down axis for the core module.

        :returns: The height.
        """
        return self.grid.shape[2]

    @property
    def num_modules(self) -> int:
        """
        Get the number of modules.

        :returns: The number of modules.
        """
        return 1 + len(self.modules)

    @property
    def num_bricks(self) -> int:
        """
        Get the number of bricks.

        :returns: The number of bricks.
        """
        return len(self.bricks)

    @property
    def num_active_hinges(self) -> int:
        """
        Get the number of active hinges.

        :returns: The number of active hinges.
        """
        return len(self.active_hinges)

    @property
    def num_filled_bricks(self) -> int:
        """
        Get the number of bricks which have all slots filled with other modules.

        :returns: The number of bricks.
        """
        return len(self.filled_bricks)

    @property
    def num_filled_active_hinges(self) -> int:
        """
        Get the number of bricks which have all slots filled with other modules.

        :returns: The number of bricks.
        """
        return len(self.filled_active_hinges)

    @property
    def num_filled_modules(self) -> int:
        """
        Get the number of modules which have all slots filled with other modules, including the core.

        :returns: The number of modules.
        """
        return (
            self.num_filled_bricks
            + self.num_active_hinges
            + (1 if self.core_is_filled else 0)
        )

    @property
    def max_potentionally_filled_core_and_bricks(self) -> int:
        """
        Get the maximum number of core and bricks that could potentially be filled with this set of modules if rearranged in an optimal way.

        This calculates 'b_max' from the paper.

        :returns: The calculated number.
        """
        # Snake-like is an optimal arrangement.
        #
        #   H H H H
        #   | | | |
        # H-C-B-B-B-H
        #   | | | |
        #   H H H H
        #
        # Every extra brick(B) requires 3 modules:
        # The bricks itself and two other modules for its sides(here displayed as H).
        # However, the core and final brick require three each to fill, which is cheaper than another brick.
        #
        # Expected sequence:
        # | num modules | 1 2 3 4 5 6 7 8 9 10 11 12 14
        # | return val  | 0 0 0 0 1 1 1 2 2 2  3  3  3

        pot_max_filled = max(0, (self.num_modules - 2) // 3)

        # Enough bricks must be available for this strategy.
        # We can count the core as the first brick.
        pot_max_filled = min(pot_max_filled, 1 + self.num_bricks)

        return pot_max_filled

    @property
    def filled_core_and_bricks_proportion(self) -> float:
        """
        Get the ratio between filled cores and bricks and how many that potentially could have been if this set of modules was rearranged in an optimal way.

        This calculates 'branching' from the paper.

        :returns: The proportion.
        """
        if self.max_potentionally_filled_core_and_bricks == 0:
            return 0.0

        return (
            len(self.filled_bricks) + (1 if self.core_is_filled else 0)
        ) / self.max_potentionally_filled_core_and_bricks

    @property
    def num_single_neighbour_modules(self) -> int:
        """
        Get the number of bricks that are only connected to one other module.

        Both children and parent are counted.

        :returns: The number of bricks.
        """
        return len(self.single_neighbour_modules)

    @property
    def max_potential_single_neighbour_modules(self) -> int:
        """
        Get the maximum number of bricks that could potentially have only one neighbour if this set of modules was rearranged in an optimal way.

        This calculates "l_max" from the paper.

        :returns: The calculated number.
        """
        # Snake-like is an optimal arrangement.
        #
        #   B B B B B
        #   | | | | |
        # B-C-B-B-B-B-B
        #   | | | | |
        #   B B B B B
        #
        # Expected sequence:
        # | num bricks | 0 1 2 3 4 5 6 7 8 9
        # | return val | 0 1 2 3 4 4 5 6 6 7

        return self.num_modules - 1 - max(0, (self.num_modules - 3) // 3)

    @property
    def num_double_neighbour_bricks(self) -> int:
        """
        Get the number of bricks that are connected to exactly two other modules.

        Both children and parent are counted.

        :returns: The number of bricks.
        """
        return len(self.double_neighbour_bricks)

    @property
    def num_double_neighbour_active_hinges(self) -> int:
        """
        Get the number of active hinges that are connected to exactly two other modules.

        Both children and parent are counted.

        :returns: The number of active hinges.
        """
        return len(self.double_neighbour_active_hinges)

    @property
    def potential_double_neighbour_bricks_and_active_hinges(self) -> int:
        """
        Get the maximum number of bricks and active hinges that could potentially have exactly two neighbours if this set of modules was rearranged in an optimal way.

        This calculates e_max from the paper.

        :returns: The calculated number.
        """
        #
        # C-M-M-M-M-M
        #
        # Snake in direction is optimal, no matter whether modules are bricks or active hinges.
        #
        # Simply add up the number of bricks and active hinges and subtract 1 for the final module.

        return max(0, self.num_bricks + self.num_active_hinges - 1)

    @property
    def double_neighbour_brick_and_active_hinge_proportion(self) -> float:
        """
        Get the ratio between the number of bricks and active hinges with exactly two neighbours and how many that could potentially have been if this set of modules was rearranged in an optimal way.

        This calculate length of limbs proportion(extensiveness) from the paper.

        :returns: The proportion.
        """
        if self.potential_double_neighbour_bricks_and_active_hinges == 0:
            return 0.0

        return (
            self.num_double_neighbour_bricks + self.num_double_neighbour_active_hinges
        ) / self.potential_double_neighbour_bricks_and_active_hinges

    @property
    def bounding_box_volume(self) -> int:
        """
        Get the volume of the bounding box.

        This calculates m_area from the paper.

        :returns: The volume.
        """
        return (
            self.bounding_box_width * self.bounding_box_height * self.bounding_box_depth
        )

    @property
    def bounding_box_volume_coverage(self) -> float:
        """
        Get the proportion of the bounding box that is filled with modules.

        This calculates 'coverage' from the paper.

        :returns: The proportion.
        """
        return self.num_modules / self.bounding_box_volume

    @property
    def branching(self) -> float:
        """
        Get the 'branching' measurement from the paper.

        Alias for filled_core_and_bricks_proportion.

        :returns: Branching measurement.
        """
        return self.filled_core_and_bricks_proportion

    @property
    def limbs(self) -> float:
        """
        Get the 'limbs' measurement from the paper.

        Alias for single_neighbour_brick_proportion.

        :returns: Limbs measurement.
        """
        if self.max_potential_single_neighbour_modules == 0:
            return 0.0
        return (
            self.num_single_neighbour_modules
            / self.max_potential_single_neighbour_modules
        )

    @property
    def length_of_limbs(self) -> float:
        """
        Get the 'length of limbs' measurement from the paper.

        Alias for double_neighbour_brick_and_active_hinge_proportion.

        :returns: Length of limbs measurement.
        """
        return self.double_neighbour_brick_and_active_hinge_proportion

    @property
    def coverage(self) -> float:
        """
        Get the 'coverage' measurement from the paper.

        Alias for bounding_box_volume_coverage.

        :returns: Coverage measurement.
        """
        return self.bounding_box_volume_coverage

    @property
    def proportion_2d(self) -> float:
        """
        Get the 'proportion' measurement from the paper.

        Only for 2d robots.

        :returns: Proportion measurement.
        """
        assert self.is_2d

        return min(self.bounding_box_depth, self.bounding_box_width) / max(
            self.bounding_box_depth, self.bounding_box_width
        )

    @property
    def proportion_3d(self) -> float:
        """
        Get the 'proportion' measurement from the paper.

        Only for 3d robots.

        :returns: Proportion measurement.
        """
        # Find the minmum and maxmum dimensions
        min_dim = min(
            self.bounding_box_depth,
            self.bounding_box_width,
            self.bounding_box_height,
        )
        max_dim = max(
            self.bounding_box_depth,
            self.bounding_box_width,
            self.bounding_box_height,
        )
        return min_dim / max_dim

    @property
    def symmetry(self) -> float:
        """
        Get the 'symmetry' measurement from the paper, but extended to 3d.

        :returns: Symmetry measurement.
        """
        return max(self.xy_symmetry, self.xz_symmetry, self.yz_symmetry)

    @property
    def num_effective_joints(self) -> int:
        """
        Get the number of effective joints.

        Effective joints are modules (core or bricks) that have both opposite
        faces attached to other modules.

        :returns: The number of effective joints.
        """
        return len(self.effective_joints_modules)

    @property
    def max_effective_joints(self) -> int:
        """
        Get the maximum number of effective joints possible.

        This calculates j_max = floor((m - 1) / 2) from the paper, where m is
        the total number of modules.

        :returns: The maximum number of effective joints.
        """
        return (self.num_modules - 1) // 2

    @property
    def joints(self) -> float:
        """
        Get the 'joints' measurement (effective joints proportion).

        This is calculated as j / j_max, where:
        - j is the number of effective joints (modules with both opposite faces attached)
        - j_max = floor((m - 1) / 2) is the maximum possible effective joints

        :returns: Joints measurement (0.0-1.0).
        """
        if self.max_effective_joints == 0:
            return 0.0
        return self.num_effective_joints / self.max_effective_joints

    @property
    def size(self) -> float:
        """
        Get the normalized size measurement.

        This is calculated as the number of modules divided by the maximum
        number of modules permitted (max_part_limit).

        :returns: Size measurement (0.0-1.0), or num_modules if max_part_limit is not set.
        """
        if self.body.max_part_limit is None or self.body.max_part_limit == 0:
            # If no limit is set, return the raw number of modules
            return float(self.num_modules)
        return self.num_modules / self.body.max_part_limit