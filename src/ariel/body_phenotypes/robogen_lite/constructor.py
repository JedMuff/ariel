"""TODO(jmdm): description of script."""

# Handle forward references in type hints
from __future__ import annotations

# Standard library
from typing import TYPE_CHECKING, Any

# Third-party libraries
import mujoco as mj
import networkx as nx

# Local libraries
from ariel.body_phenotypes.robogen_lite.collision_utils import check_self_collision
from ariel.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsTheta,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

# Type checking
if TYPE_CHECKING:
    from networkx import DiGraph

    from ariel.body_phenotypes.robogen_lite.modules.module import Module


def _has_self_collision(spec: mj.MjSpec) -> bool:
    """
    Check if a robot spec has any self-collisions using FCL.

    This function checks for geometric overlaps between robot geoms using
    the FCL (Flexible Collision Library) for efficient collision detection.
    Automatically excludes collisions at attachment points (parent-child).

    Parameters
    ----------
    spec : mj.MjSpec
        The MuJoCo specification to check

    Returns
    -------
    bool
        True if self-collision detected, False otherwise
    """
    return check_self_collision(spec)


def _build_spec_from_graph(
    graph: DiGraph[Any],
    blacklisted_modules: set[int],
) -> CoreModule:
    """
    Build a robot spec from graph, skipping blacklisted modules.

    Parameters
    ----------
    graph : DiGraph
        The robot structure graph
    blacklisted_modules : set[int]
        Set of module IDs to skip during construction

    Returns
    -------
    CoreModule
        The constructed core module
    """
    # Create all module objects
    modules: dict[int, Module] = {}
    for node in graph.nodes:
        # Skip blacklisted modules
        if node in blacklisted_modules:
            continue

        module_type = graph.nodes[node]["type"]
        module_rotation = graph.nodes[node]["rotation"]

        # Create the module based on its type
        match module_type:
            case ModuleType.CORE.name:
                module = CoreModule(index=IDX_OF_CORE)
            case ModuleType.HINGE.name:
                module = HingeModule(index=node)
            case ModuleType.BRICK.name:
                module = BrickModule(index=node)
            case ModuleType.NONE.name:
                module = None
            case _:
                msg = f"Unknown module type: {module_type}"
                raise ValueError(msg)

        if module:
            rotation_angle = ModuleRotationsTheta[module_rotation].value
            module.rotate(rotation_angle)
            modules[node] = module
        else:
            modules[node] = None

    # Get core module
    core_module = modules.get(IDX_OF_CORE)
    if not isinstance(core_module, CoreModule):
        msg = "The core module is not of type CoreModule."
        raise ValueError(msg)

    # Process edges in BFS order
    edges_to_process = list(nx.bfs_edges(graph, IDX_OF_CORE))

    # Attach bodies
    for from_node, to_node in edges_to_process:
        # Skip if either node is blacklisted or doesn't exist in modules
        if (from_node not in modules or to_node not in modules or
            not modules[from_node] or not modules[to_node]):
            continue

        # Get edge data
        edge_data = graph.edges[from_node, to_node]
        face = edge_data["face"]

        # Attach the module
        modules[from_node].sites[ModuleFaces[face]].attach_body(
            body=modules[to_node].body,
            prefix=f"{modules[from_node].index}-{modules[to_node].index}-{ModuleFaces[face].value}-",
        )

    return core_module


def construct_mjspec_from_graph(
    graph: DiGraph[Any],
    check_collisions: bool = True,
    max_iterations: int = 100,
) -> CoreModule:
    """
    Construct a MuJoCo specification from a graph representation with collision checking.

    Builds the robot iteratively, detecting self-collisions and removing problematic
    modules until a collision-free robot is produced.

    Parameters
    ----------
    graph : Graph
        A graph representation of the robot's structure.
    check_collisions : bool, optional
        If True, check for self-collisions during construction, by default True.
    max_iterations : int, optional
        Maximum number of rebuild iterations, by default 100.

    Returns
    -------
    CoreModule
        The core module of the robot, which contains all other modules.

    Raises
    ------
    ValueError
        If the graph contains unknown module types or max iterations exceeded.
    """
    if not check_collisions:
        # Build without collision checking (original behavior)
        return _build_spec_from_graph(graph, set())

    blacklisted_modules: set[int] = set()
    edges_to_process = list(nx.bfs_edges(graph, IDX_OF_CORE))

    for iteration in range(max_iterations):
        # Build spec with current blacklist
        core_module = _build_spec_from_graph(graph, blacklisted_modules)

        # Check each edge attachment incrementally
        collision_found = False
        for edge_idx, (from_node, to_node) in enumerate(edges_to_process):
            # Skip already blacklisted
            if to_node in blacklisted_modules:
                continue

            # Build spec up to and including this edge
            test_blacklist = blacklisted_modules.copy()
            # Blacklist all nodes from edges that come after this one in BFS order
            for later_idx in range(edge_idx + 1, len(edges_to_process)):
                _, later_to = edges_to_process[later_idx]
                test_blacklist.add(later_to)

            # Build with this configuration
            test_core = _build_spec_from_graph(graph, test_blacklist)

            # Check for collision
            if _has_self_collision(test_core.spec):
                # This module causes collision - blacklist it and its children
                blacklisted_modules.add(to_node)
                blacklisted_modules.update(nx.descendants(graph, to_node))
                collision_found = True
                break

        if not collision_found:
            # No collisions found, return the final build
            return core_module

    # Max iterations exceeded
    msg = f"Max iterations ({max_iterations}) exceeded in collision-free construction"
    raise ValueError(msg)
