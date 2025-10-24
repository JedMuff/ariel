"""Collision detection utilities using FCL (Flexible Collision Library)."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# Third-party libraries
import fcl
import mujoco as mj
import numpy as np
import quaternion as qnp
from numpy.typing import NDArray

if TYPE_CHECKING:
    import networkx as nx

    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType


# ========== Module Geometry Definitions ==========

@dataclass
class ModuleGeometry:
    """Geometry definition for a module component."""

    geom_type: int  # mujoco.mjtGeom type
    size: NDArray[np.float64]  # Size parameters (half-extents for box)
    local_pos: NDArray[np.float64]  # Local position relative to module origin
    local_quat: NDArray[np.float64]  # Local quaternion (wxyz format)


@dataclass
class AttachmentSite:
    """Attachment site definition for module faces."""

    local_pos: NDArray[np.float64]  # Local position
    local_quat: NDArray[np.float64]  # Local quaternion (wxyz format)


# Module dimensions (from module definition files)
CORE_DIMENSIONS = np.array([0.10, 0.10, 0.10])
BRICK_DIMENSIONS = np.array([0.05, 0.05, 0.05])
STATOR_DIMENSIONS = np.array([0.025, 0.03, 0.025])
ROTOR_DIMENSIONS = np.array([0.025, 0.02, 0.025])
SHRINK = 0.99  # For hinge stator to avoid z-fighting


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Convert euler angles to quaternion (wxyz format)."""
    q = qnp.from_euler_angles([roll, pitch, yaw])
    return np.array([q.w, q.x, q.y, q.z])


def get_module_geometries(module_type: str) -> list[ModuleGeometry]:
    """
    Get geometry definitions for a module type.

    Parameters
    ----------
    module_type : str
        Module type name (CORE, BRICK, HINGE)

    Returns
    -------
    list[ModuleGeometry]
        List of geometry components for this module
    """
    from ariel.body_phenotypes.robogen_lite.config import ModuleType

    if module_type == ModuleType.CORE.name:
        return [
            ModuleGeometry(
                geom_type=mj.mjtGeom.mjGEOM_BOX,
                size=CORE_DIMENSIONS,
                local_pos=np.array([0, CORE_DIMENSIONS[0], 0]),
                local_quat=np.array([1, 0, 0, 0]),  # Identity
            )
        ]
    elif module_type == ModuleType.BRICK.name:
        return [
            ModuleGeometry(
                geom_type=mj.mjtGeom.mjGEOM_BOX,
                size=BRICK_DIMENSIONS,
                local_pos=np.array([0, BRICK_DIMENSIONS[0], 0]),
                local_quat=np.array([1, 0, 0, 0]),  # Identity
            )
        ]
    elif module_type == ModuleType.HINGE.name:
        return [
            # Stator
            ModuleGeometry(
                geom_type=mj.mjtGeom.mjGEOM_BOX,
                size=STATOR_DIMENSIONS * SHRINK,
                local_pos=np.array([0, STATOR_DIMENSIONS[1], 0]),
                local_quat=np.array([1, 0, 0, 0]),  # Identity
            ),
            # Rotor
            ModuleGeometry(
                geom_type=mj.mjtGeom.mjGEOM_BOX,
                size=ROTOR_DIMENSIONS,
                local_pos=np.array([0, STATOR_DIMENSIONS[1] * 2 + ROTOR_DIMENSIONS[1], 0]),
                local_quat=np.array([1, 0, 0, 0]),  # Identity
            ),
        ]
    else:
        return []


def get_attachment_sites(module_type: str) -> dict[str, AttachmentSite]:
    """
    Get attachment site definitions for a module type.

    These transforms are extracted from the actual MuJoCo module definitions
    and represent the position and orientation of attachment sites in the
    module's local coordinate frame.

    Parameters
    ----------
    module_type : str
        Module type name (CORE, BRICK, HINGE)

    Returns
    -------
    dict[str, AttachmentSite]
        Dict mapping face name to attachment site
    """
    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType

    if module_type == ModuleType.CORE.name:
        # Attachment site transforms extracted from CoreModule
        # Quaternions are in wxyz format
        return {
            ModuleFaces.FRONT.name: AttachmentSite(
                local_pos=np.array([0.0, 0.2, -0.05]),
                local_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            ModuleFaces.BACK.name: AttachmentSite(
                local_pos=np.array([0.0, 0.0, -0.05]),
                local_quat=np.array([0.0, 0.0, 0.0, -1.0]),
            ),
            ModuleFaces.LEFT.name: AttachmentSite(
                local_pos=np.array([-0.1, 0.1, -0.05]),
                local_quat=np.array([-0.707107, 0.0, 0.0, -0.707107]),
            ),
            ModuleFaces.RIGHT.name: AttachmentSite(
                local_pos=np.array([0.1, 0.1, -0.05]),
                local_quat=np.array([0.707107, 0.0, 0.0, -0.707107]),
            ),
            ModuleFaces.TOP.name: AttachmentSite(
                local_pos=np.array([0.0, 0.1, 0.1]),
                local_quat=np.array([-0.707107, -0.707107, 0.0, 0.0]),
            ),
            ModuleFaces.BOTTOM.name: AttachmentSite(
                local_pos=np.array([0.0, 0.1, -0.1]),
                local_quat=np.array([0.0, 0.0, -0.707107, 0.707107]),
            ),
        }
    elif module_type == ModuleType.BRICK.name:
        # Attachment site transforms extracted from BrickModule
        # Quaternions are in wxyz format
        return {
            ModuleFaces.FRONT.name: AttachmentSite(
                local_pos=np.array([0.0, 0.1, 0.0]),
                local_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            ModuleFaces.LEFT.name: AttachmentSite(
                local_pos=np.array([-0.05, 0.05, 0.0]),
                local_quat=np.array([-0.707107, 0.0, 0.0, -0.707107]),
            ),
            ModuleFaces.RIGHT.name: AttachmentSite(
                local_pos=np.array([0.05, 0.05, 0.0]),
                local_quat=np.array([0.707107, 0.0, 0.0, -0.707107]),
            ),
            ModuleFaces.TOP.name: AttachmentSite(
                local_pos=np.array([0.0, 0.05, 0.05]),
                local_quat=np.array([-0.707107, -0.707107, 0.0, 0.0]),
            ),
            ModuleFaces.BOTTOM.name: AttachmentSite(
                local_pos=np.array([0.0, 0.05, -0.05]),
                local_quat=np.array([0.0, 0.0, -0.707107, 0.707107]),
            ),
        }
    elif module_type == ModuleType.HINGE.name:
        # Attachment site transforms extracted from HingeModule
        # Quaternions are in wxyz format
        return {
            ModuleFaces.FRONT.name: AttachmentSite(
                local_pos=np.array([0.0, 0.1, 0.0]),
                local_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
        }
    else:
        return {}


@dataclass
class GeomData:
    """Data class to store geometry information."""

    name: str
    geom_type: int  # mujoco.mjtGeom type
    size: NDArray[np.float64]  # Size parameters
    pos: NDArray[np.float64]  # Position in world frame
    quat: NDArray[np.float64]  # Quaternion (wxyz format) in world frame
    body_name: str


def _quat_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Multiply two quaternions (wxyz format).

    Parameters
    ----------
    q1 : NDArray[np.float64]
        First quaternion [w, x, y, z]
    q2 : NDArray[np.float64]
        Second quaternion [w, x, y, z]

    Returns
    -------
    NDArray[np.float64]
        Product quaternion [w, x, y, z]
    """
    # Convert to numpy-quaternion format and multiply
    quat1 = np.quaternion(*q1)
    quat2 = np.quaternion(*q2)
    result = quat1 * quat2
    return np.array([result.w, result.x, result.y, result.z])


def _quat_rotate_vector(
    quat: NDArray[np.float64],
    vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Rotate a vector by a quaternion.

    Parameters
    ----------
    quat : NDArray[np.float64]
        Quaternion [w, x, y, z]
    vec : NDArray[np.float64]
        Vector [x, y, z]

    Returns
    -------
    NDArray[np.float64]
        Rotated vector [x, y, z]
    """
    q = np.quaternion(*quat)
    v = np.quaternion(0, *vec)
    rotated = q * v * q.conjugate()
    return np.array([rotated.x, rotated.y, rotated.z])


def _compose_transforms(
    pos1: NDArray[np.float64],
    quat1: NDArray[np.float64],
    pos2: NDArray[np.float64],
    quat2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compose two transforms (position + quaternion).

    Parameters
    ----------
    pos1 : NDArray[np.float64]
        First position [x, y, z]
    quat1 : NDArray[np.float64]
        First quaternion [w, x, y, z]
    pos2 : NDArray[np.float64]
        Second position [x, y, z]
    quat2 : NDArray[np.float64]
        Second quaternion [w, x, y, z]

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Composed (position, quaternion)
    """
    # Rotate second position by first quaternion and add to first position
    rotated_pos2 = _quat_rotate_vector(quat1, pos2)
    composed_pos = pos1 + rotated_pos2

    # Multiply quaternions
    composed_quat = _quat_multiply(quat1, quat2)

    return composed_pos, composed_quat


def _get_module_rotation_quat(rotation_name: str) -> NDArray[np.float64]:
    """
    Get quaternion for module rotation around Y-axis.

    The attachment site defines the base orientation (where the child points).
    This quaternion represents ONLY the rotational angle variation around the Y-axis
    in the attached frame.

    Parameters
    ----------
    rotation_name : str
        Rotation name (e.g., DEG_0, DEG_45, etc.)

    Returns
    -------
    NDArray[np.float64]
        Quaternion [w, x, y, z] for rotation around Y-axis
    """
    from ariel.body_phenotypes.robogen_lite.config import ModuleRotationsTheta

    # Get rotation angle in degrees
    angle_deg = ModuleRotationsTheta[rotation_name].value

    # Simple rotation around Y-axis by the specified angle
    # DEG_0 → identity, DEG_45 → 45° around Y, DEG_315 → 315° around Y, etc.
    angle_rad = np.deg2rad(angle_deg)
    rotation_vector = np.array([0, angle_rad, 0])
    quat = qnp.from_rotation_vector(rotation_vector)
    return np.array([quat.w, quat.x, quat.y, quat.z])


def compute_world_transforms_from_tree(
    tree: nx.DiGraph,
    blacklisted_nodes: set[int] | None = None,
) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Compute world transforms for all nodes in a tree.

    Parameters
    ----------
    tree : nx.DiGraph
        Tree graph with node attributes (type, rotation) and edge attributes (face)
    blacklisted_nodes : set[int] | None
        Set of node IDs to skip

    Returns
    -------
    dict[int, tuple[NDArray, NDArray]]
        Dict mapping node_id to (world_pos, world_quat)
    """
    from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE

    if blacklisted_nodes is None:
        blacklisted_nodes = set()

    world_transforms = {}

    # Start with core at origin
    core_pos = np.array([0.0, 0.0, 0.0])
    core_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    world_transforms[IDX_OF_CORE] = (core_pos, core_quat)

    # Traverse tree in BFS order
    import networkx as nx

    for parent_id, child_id in nx.bfs_edges(tree, IDX_OF_CORE):
        # Skip blacklisted nodes
        if child_id in blacklisted_nodes:
            continue
        if parent_id not in world_transforms:
            continue  # Parent was blacklisted

        # Get parent world transform
        parent_pos, parent_quat = world_transforms[parent_id]

        # Get edge data (attachment face)
        edge_data = tree.edges[parent_id, child_id]
        face_name = edge_data["face"]

        # Get parent module type and attachment site
        parent_type = tree.nodes[parent_id]["type"]
        attachment_sites = get_attachment_sites(parent_type)
        site = attachment_sites[face_name]

        # Get child rotation
        child_rotation = tree.nodes[child_id]["rotation"]
        child_rotation_quat = _get_module_rotation_quat(child_rotation)

        # Compose transforms:
        # 1. Start with parent world transform
        # 2. Apply attachment site transform
        # 3. Apply child rotation (around Y-axis in the attached frame)
        #
        # The child module's origin is placed at the attachment site position
        # The child module is oriented by the attachment site quaternion
        # Then the child's rotation is applied (rotation around Y in attached frame)

        # First compose: parent + attachment site
        site_world_pos, site_world_quat = _compose_transforms(
            parent_pos, parent_quat, site.local_pos, site.local_quat
        )

        # Then apply child rotation: site_quat * rotation_quat
        # The rotation is around Y-axis in the site's frame
        child_world_pos = site_world_pos  # Position stays the same
        child_world_quat = _quat_multiply(site_world_quat, child_rotation_quat)

        world_transforms[child_id] = (child_world_pos, child_world_quat)

    return world_transforms


def create_fcl_objects_from_tree(
    tree: nx.DiGraph,
    blacklisted_nodes: set[int] | None = None,
) -> list[tuple[fcl.CollisionObject, int]]:
    """
    Create FCL collision objects directly from tree structure.

    Parameters
    ----------
    tree : nx.DiGraph
        Tree graph with node attributes (type, rotation)
    blacklisted_nodes : set[int] | None
        Set of node IDs to skip

    Returns
    -------
    list[tuple[fcl.CollisionObject, int]]
        List of (collision_object, node_id) pairs
    """
    if blacklisted_nodes is None:
        blacklisted_nodes = set()

    # Compute world transforms for all nodes
    world_transforms = compute_world_transforms_from_tree(tree, blacklisted_nodes)

    collision_objects = []

    # Create FCL objects for each node
    for node_id, (world_pos, world_quat) in world_transforms.items():
        if node_id in blacklisted_nodes:
            continue

        # Get module type
        module_type = tree.nodes[node_id]["type"]

        # Get geometries for this module
        geometries = get_module_geometries(module_type)

        # Create FCL object for each geometry component
        for geom in geometries:
            # Compose world transform with local geometry transform
            geom_world_pos, geom_world_quat = _compose_transforms(
                world_pos, world_quat, geom.local_pos, geom.local_quat
            )

            # Create FCL geometry
            if geom.geom_type == mj.mjtGeom.mjGEOM_BOX:
                # Box size is half-extents in MuJoCo
                fcl_geom = fcl.Box(geom.size[0] * 2, geom.size[1] * 2, geom.size[2] * 2)
            else:
                msg = f"Unsupported geometry type: {geom.geom_type}"
                raise NotImplementedError(msg)

            # Create FCL transform
            q = np.quaternion(*geom_world_quat)
            rot_matrix = qnp.as_rotation_matrix(q)
            transform = fcl.Transform(rot_matrix, geom_world_pos)

            # Create collision object
            collision_objects.append((fcl.CollisionObject(fcl_geom, transform), node_id))

    return collision_objects


def get_tree_collision_details(
    tree: nx.DiGraph,
    blacklisted_nodes: set[int] | None = None,
) -> list[tuple[int, int, float]]:
    """
    Get detailed collision information for a tree.

    Parameters
    ----------
    tree : nx.DiGraph
        Tree graph
    blacklisted_nodes : set[int] | None
        Set of node IDs to skip

    Returns
    -------
    list[tuple[int, int, float]]
        List of (node1, node2, penetration_depth) for all colliding pairs.
        Negative penetration means the objects are penetrating.
    """
    collisions = []

    try:
        # Create FCL collision objects
        collision_objects = create_fcl_objects_from_tree(tree, blacklisted_nodes)

        # Check all pairs for collision
        for i in range(len(collision_objects)):
            for j in range(i + 1, len(collision_objects)):
                obj1, node1 = collision_objects[i]
                obj2, node2 = collision_objects[j]

                # Skip if nodes are directly connected (parent-child)
                import networkx as nx

                if tree.has_edge(node1, node2) or tree.has_edge(node2, node1):
                    continue

                # Check for hinge stator-rotor (same node, different geometries)
                if node1 == node2:
                    continue

                # Perform distance query to get penetration depth
                request = fcl.DistanceRequest()
                result = fcl.DistanceResult()
                distance = fcl.distance(obj1, obj2, request, result)

                # Store all collision pairs (not just those exceeding threshold)
                collisions.append((node1, node2, distance))

        return collisions
    except Exception:
        return []


def check_tree_self_collision(
    tree: nx.DiGraph,
    blacklisted_nodes: set[int] | None = None,
    penetration_threshold: float = -0.001,
) -> bool:
    """
    Check if a tree has self-collisions using direct FCL approach.

    Uses a penetration threshold to distinguish between touching (acceptable)
    and actual collision (penetration). This matches MuJoCo's collision detection.

    Parameters
    ----------
    tree : nx.DiGraph
        Tree graph
    blacklisted_nodes : set[int] | None
        Set of node IDs to skip
    penetration_threshold : float
        Minimum penetration depth (in meters) to consider as collision.
        Negative values mean penetration. Default is -0.001 (1mm penetration).

    Returns
    -------
    bool
        True if collision detected, False otherwise
    """
    try:
        # Create FCL collision objects
        collision_objects = create_fcl_objects_from_tree(tree, blacklisted_nodes)

        # Check all pairs for collision
        for i in range(len(collision_objects)):
            for j in range(i + 1, len(collision_objects)):
                obj1, node1 = collision_objects[i]
                obj2, node2 = collision_objects[j]

                # Skip if nodes are directly connected (parent-child)
                import networkx as nx

                if tree.has_edge(node1, node2) or tree.has_edge(node2, node1):
                    continue

                # Check for hinge stator-rotor (same node, different geometries)
                if node1 == node2:
                    continue

                # Perform distance query to get penetration depth
                request = fcl.DistanceRequest()
                result = fcl.DistanceResult()
                distance = fcl.distance(obj1, obj2, request, result)

                # Negative distance means penetration
                # Only report collision if penetration exceeds threshold
                if distance < penetration_threshold:
                    return True

        return False
    except Exception:
        # If anything fails, consider it a collision
        return True


def _extract_geoms_from_spec(spec: mj.MjSpec) -> list[GeomData]:
    """
    Extract all geometries from a MuJoCo spec with their world transforms.

    This function compiles the spec and uses MuJoCo's forward kinematics
    to get accurate world transforms.

    Parameters
    ----------
    spec : mj.MjSpec
        MuJoCo specification to extract geometries from

    Returns
    -------
    list[GeomData]
        List of geometry data with world transforms
    """
    # Compile the model to get accurate transforms
    model = spec.compile()
    data = mj.MjData(model)

    # Run forward kinematics to compute all transforms
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    geoms: list[GeomData] = []

    # Iterate through all geometries in the compiled model
    for i in range(model.ngeom):
        # Get geometry name
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        if geom_name is None:
            geom_name = f"geom_{i}"

        # Get body name
        body_id = model.geom_bodyid[i]
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            body_name = f"body_{body_id}"

        # Get world position from MuJoCo's computed transforms
        world_pos = data.geom_xpos[i].copy()

        # Get world rotation matrix and convert to quaternion
        world_mat = data.geom_xmat[i].reshape(3, 3).copy()
        world_quat_np = qnp.from_rotation_matrix(world_mat)
        world_quat = np.array([world_quat_np.w, world_quat_np.x, world_quat_np.y, world_quat_np.z])

        # Get geometry type and size
        geom_type = model.geom_type[i]
        geom_size = model.geom_size[i].copy()

        geoms.append(
            GeomData(
                name=geom_name,
                geom_type=geom_type,
                size=geom_size,
                pos=world_pos,
                quat=world_quat,
                body_name=body_name,
            ),
        )

    return geoms


def _create_fcl_collision_object(geom: GeomData) -> fcl.CollisionObject:
    """
    Create an FCL collision object from geometry data.

    Parameters
    ----------
    geom : GeomData
        Geometry data to convert

    Returns
    -------
    fcl.CollisionObject
        FCL collision object

    Raises
    ------
    NotImplementedError
        If the geometry type is not supported
    """
    # Create FCL geometry based on type
    if geom.geom_type == mj.mjtGeom.mjGEOM_BOX:
        # Box size is half-extents in MuJoCo
        fcl_geom = fcl.Box(geom.size[0] * 2, geom.size[1] * 2, geom.size[2] * 2)
    elif geom.geom_type == mj.mjtGeom.mjGEOM_SPHERE:
        fcl_geom = fcl.Sphere(geom.size[0])
    elif geom.geom_type == mj.mjtGeom.mjGEOM_CAPSULE:
        fcl_geom = fcl.Capsule(geom.size[0], geom.size[1] * 2)
    elif geom.geom_type == mj.mjtGeom.mjGEOM_CYLINDER:
        fcl_geom = fcl.Cylinder(geom.size[0], geom.size[1] * 2)
    else:
        msg = f"Unsupported geometry type: {geom.geom_type}"
        raise NotImplementedError(msg)

    # Create transform
    # Convert quaternion to rotation matrix
    q = np.quaternion(*geom.quat)
    rot_matrix = qnp.as_rotation_matrix(q)

    # Create FCL transform
    transform = fcl.Transform(rot_matrix, geom.pos)

    # Create collision object
    return fcl.CollisionObject(fcl_geom, transform)


def _get_collision_exclusions(spec: mj.MjSpec) -> set[tuple[str, str]]:
    """
    Get collision exclusion pairs from MuJoCo spec.

    Parameters
    ----------
    spec : mj.MjSpec
        MuJoCo specification

    Returns
    -------
    set[tuple[str, str]]
        Set of (body_name1, body_name2) pairs that should not be checked for collision
    """
    exclusions = set()
    for exclude in spec.excludes:
        # Get body names
        body1 = exclude.bodyname1
        body2 = exclude.bodyname2
        if body1 and body2:
            # Add both orderings
            exclusions.add((body1, body2))
            exclusions.add((body2, body1))
    return exclusions


def _are_bodies_connected(body1_name: str, body2_name: str) -> bool:
    """
    Check if two bodies are directly connected (parent-child relationship).

    Body names have format like "core", "brick", "hinge", "stator", "rotor",
    or prefixed versions like "0-1-2-brick" (parent-child-face-type).

    Parameters
    ----------
    body1_name : str
        Name of first body
    body2_name : str
        Name of second body

    Returns
    -------
    bool
        True if bodies are directly connected (one is parent of the other)
    """
    # Parse body names to extract module indices
    # Format: "prefix-childidx-faceidx-moduletype" or just "moduletype"

    def get_module_info(name: str) -> tuple[str, int | None, int | None]:
        """Extract (module_type, parent_idx, child_idx) from body name."""
        parts = name.split("-")
        if len(parts) >= 4:
            # Format: parent-child-face-type
            try:
                parent_idx = int(parts[0])
                child_idx = int(parts[1])
                module_type = parts[3]
                return (module_type, parent_idx, child_idx)
            except (ValueError, IndexError):
                pass

        # Simple name like "core", "brick", "hinge", "stator", "rotor"
        return (name, None, None)

    type1, parent1, child1 = get_module_info(body1_name)
    type2, parent2, child2 = get_module_info(body2_name)

    # Check if one body's child index matches the other's module
    # Core always has index 0
    if "core" in body1_name and child2 is not None and parent2 == 0:
        return True
    if "core" in body2_name and child1 is not None and parent1 == 0:
        return True

    # Check if they share a parent-child relationship
    if child1 is not None and child2 is not None:
        if parent1 == child2 or parent2 == child1:
            return True

    # Check for hinge stator-rotor pairs (always connected)
    if ("stator" in body1_name and "rotor" in body2_name) or \
       ("rotor" in body1_name and "stator" in body2_name):
        # Extract the module index prefix (before "stator"/"rotor")
        prefix1 = body1_name.replace("-stator", "").replace("-rotor", "")
        prefix2 = body2_name.replace("-stator", "").replace("-rotor", "")
        if prefix1 == prefix2:
            return True

    return False


def check_self_collision(spec: mj.MjSpec) -> bool:
    """
    Check if a MuJoCo spec has self-collisions using FCL.

    This function extracts all geometries from the spec and checks for
    collisions using the FCL (Flexible Collision Library). It automatically
    excludes collision checks between:
    - Bodies with explicit exclusion pairs in the MuJoCo spec
    - Directly connected bodies (parent-child relationships)

    Parameters
    ----------
    spec : mj.MjSpec
        MuJoCo specification to check

    Returns
    -------
    bool
        True if self-collision detected, False otherwise
    """
    try:
        # Extract all geometries with world transforms
        geoms = _extract_geoms_from_spec(spec)

        # Get collision exclusions from MuJoCo spec
        spec_exclusions = _get_collision_exclusions(spec)

        # Create FCL collision objects
        collision_objects = [_create_fcl_collision_object(geom) for geom in geoms]

        # Check for collisions between all pairs
        for i in range(len(collision_objects)):
            for j in range(i + 1, len(collision_objects)):
                body1 = geoms[i].body_name
                body2 = geoms[j].body_name

                # Skip if this pair is excluded in MuJoCo spec
                if (body1, body2) in spec_exclusions:
                    continue

                # Skip if bodies are directly connected (parent-child)
                if _are_bodies_connected(body1, body2):
                    continue

                # Perform collision check
                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()
                num_contacts = fcl.collide(
                    collision_objects[i],
                    collision_objects[j],
                    request,
                    result,
                )

                # If any contact detected, return collision
                if num_contacts > 0:
                    return True

        return False
    except Exception:
        # If anything fails during collision checking, consider it a collision
        return True
