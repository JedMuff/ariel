from collections import deque

import networkx as nx
import numpy as np
import numpy.typing as npt
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    NUM_OF_TYPES_OF_MODULES,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.collision_utils import (
    IDENTITY,
    BodyCollisionChecker,
)
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome

console = Console()

# CPPN position inputs (metres) are divided by this: one brick's edge length,
# so neighbouring modules differ by about 1, as on the legacy integer grid.
FCL_POSITION_SCALE = 0.1


def softmax(raw_scores: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    e_x = np.exp(raw_scores - np.max(raw_scores))
    return e_x / e_x.sum()


def _fix_terminal_hinges(robot_graph: nx.DiGraph) -> None:
    """Convert any leaf HINGE node into a BRICK.

    A hinge with no child can rest directly on the ground and exploit
    contact-driven propulsion instead of using its joint. Hinge chains
    (hinge -> hinge -> ... -> brick) are unaffected since only leaf nodes
    are considered.
    """
    for node_id, data in robot_graph.nodes(data=True):
        if node_id == IDX_OF_CORE:
            continue
        if data["type"] == ModuleType.HINGE.name and robot_graph.out_degree(node_id) == 0:
            data["type"] = ModuleType.BRICK.name


class MorphologyDecoderBestFirst:
    """Decodes a CPPN using a true greedy, best-first search strategy."""

    def __init__(
        self,
        cppn_genome: Genome,
        max_modules: int = 20,
        allow_none: bool = True,
        legacy: bool = False,
        distance_input: bool = False,
        local_inputs: bool = False,
        local_competition: bool = False,
        local_threshold: float = 0.5,
    ):
        """
        Parameters
        ----------
        allow_none
            If True, a NONE type output leaves that face empty instead of
            being masked out, so the CPPN can choose not to place a part
            and the body can stop growing before max_modules.
        legacy
            Use the original integer-grid decoder instead of the FCL one.
            Only for re-decoding genomes evolved under it. The grid ignores
            module rotation and size, so it neither prevents real overlaps
            nor places modules where the CPPN inputs say; its face scan also
            stops at the first disallowed or occupied face (so bricks only
            grow from FRONT) and only lets a hinge attach to a parent's FRONT
            face.
        distance_input
            Also feed the CPPN the parent centre's and attachment point's
            distance from the core centre. These don't depend on direction,
            so a random genome's preference forms rings around the core.
        local_inputs
            Replace the absolute parent/attachment positions with bounded,
            local ones: the face's unit direction (parent centre ->
            attachment point) and its outwardness (attachment point's
            distance from the core centre minus the parent's). Unlike
            absolute positions they don't grow along a chain, so the chain
            tip gets no advantage from being far out.
        local_competition
            Instead of one global best-first search, visit modules
            breadth-first from the core; each module attaches children on
            its faces whose connection score exceeds ``local_threshold``,
            best first, until max_modules. Faces only compete with the other
            faces of their own module.
        local_threshold
            Connection score (sigmoid output) a face needs under
            ``local_competition``.

        The number of CPPN inputs depends on the input flags; see
        `num_inputs`.
        """
        self.cppn_genome = cppn_genome
        self.max_modules = max_modules
        self.allow_none = allow_none
        self.legacy = legacy
        self.distance_input = distance_input
        self.local_inputs = local_inputs
        self.local_competition = local_competition
        self.local_threshold = local_threshold
        self.face_deltas = {
            ModuleFaces.FRONT: (1, 0, 0),
            ModuleFaces.BACK: (-1, 0, 0),
            ModuleFaces.TOP: (0, 1, 0),
            ModuleFaces.BOTTOM: (0, -1, 0),
            ModuleFaces.RIGHT: (0, 0, 1),
            ModuleFaces.LEFT: (0, 0, -1),
        }

    def _get_child_coords(self, parent_pos: tuple, face: ModuleFaces) -> tuple:
        delta = self.face_deltas[face]
        return (
            parent_pos[0] + delta[0],
            parent_pos[1] + delta[1],
            parent_pos[2] + delta[2],
        )

    @staticmethod
    def num_inputs(
        distance_input: bool = False, local_inputs: bool = False
    ) -> int:
        """CPPN inputs the FCL decoder passes for these flags (a genome's
        bias input, if any, comes on top)."""
        return (4 if local_inputs else 6) + (2 if distance_input else 0)

    def decode(self) -> nx.DiGraph:
        if self.legacy:
            return self._decode_grid()
        if self.local_competition:
            return self._decode_fcl_local()
        return self._decode_fcl()

    def _fcl_inputs(
        self,
        parent_centre: npt.NDArray[np.float64],
        attach_point: npt.NDArray[np.float64],
        core_centre: npt.NDArray[np.float64],
    ) -> list[float]:
        """CPPN inputs for one face, in units of FCL_POSITION_SCALE."""
        pc = parent_centre / FCL_POSITION_SCALE
        ap = attach_point / FCL_POSITION_SCALE
        cc = core_centre / FCL_POSITION_SCALE
        pc_dist = float(np.linalg.norm(pc - cc))
        ap_dist = float(np.linalg.norm(ap - cc))
        if self.local_inputs:
            direction = ap - pc
            direction /= np.linalg.norm(direction)
            inputs = [*direction, ap_dist - pc_dist]
        else:
            inputs = [*pc, *ap]
        if self.distance_input:
            inputs += [pc_dist, ap_dist]
        return inputs

    def _face_candidates(
        self,
        checker: BodyCollisionChecker,
        parent_id: int,
        parent_type: ModuleType,
    ) -> list[tuple]:
        """(face, score, child_type, child_rot, child_frame) for each allowed
        face of a placed module where the CPPN wants a module."""
        parent_centre = checker.module_centre(parent_id)
        core_centre = checker.module_centre(IDX_OF_CORE)
        out = []
        for face in ALLOWED_FACES[parent_type]:
            frame = checker.child_frame(parent_id, face.name)
            choice = self._cppn_choice(
                self._fcl_inputs(parent_centre, frame[1], core_centre)
            )
            if choice is not None:
                out.append((face, *choice, frame))
        return out

    def _new_body(self) -> tuple[nx.DiGraph, BodyCollisionChecker]:
        checker = BodyCollisionChecker()
        robot_graph = nx.DiGraph()
        core_type, core_rot = ModuleType.CORE, ModuleRotationsIdx.DEG_0
        robot_graph.add_node(
            IDX_OF_CORE, type=core_type.name, rotation=core_rot.name
        )
        checker.add_module(
            IDX_OF_CORE, IDENTITY, core_type.name, core_rot.name
        )
        return robot_graph, checker

    def _decode_fcl_local(self) -> nx.DiGraph:
        """Breadth-first growth where faces only compete within their own
        module (see ``local_competition``); overlaps are checked with FCL as
        in `_decode_fcl`."""
        robot_graph, checker = self._new_body()
        queue = deque([IDX_OF_CORE])
        next_module_id = 1
        while queue and len(robot_graph) < self.max_modules:
            parent_id = queue.popleft()
            parent_type = ModuleType[robot_graph.nodes[parent_id]["type"]]
            candidates = [
                c
                for c in self._face_candidates(checker, parent_id, parent_type)
                if c[1] > self.local_threshold
            ]
            # Best first; ties keep face order.
            candidates.sort(key=lambda c: c[1], reverse=True)
            for face, _, child_type, child_rot, frame in candidates:
                if len(robot_graph) >= self.max_modules:
                    break
                if checker.collides(
                    frame, child_type.name, child_rot.name, ignore={parent_id}
                ):
                    continue
                child_id = next_module_id
                next_module_id += 1
                robot_graph.add_node(
                    child_id, type=child_type.name, rotation=child_rot.name
                )
                robot_graph.add_edge(parent_id, child_id, face=face.name)
                checker.add_module(
                    child_id, frame, child_type.name, child_rot.name
                )
                queue.append(child_id)

        _fix_terminal_hinges(robot_graph)
        return robot_graph

    def _cppn_choice(
        self, cppn_inputs: list[float]
    ) -> tuple[float, ModuleType, ModuleRotationsIdx] | None:
        """(connection score, child type, child rotation), or None if the
        CPPN picks NONE or a rotation the child type doesn't allow."""
        raw_outputs = self.cppn_genome.activate(cppn_inputs)
        conn_score = raw_outputs[0]
        type_probs = softmax(
            np.array(raw_outputs[1 : 1 + NUM_OF_TYPES_OF_MODULES])
        )
        rot_scores = np.array(raw_outputs[1 + NUM_OF_TYPES_OF_MODULES :])

        if not self.allow_none:
            type_probs[ModuleType.NONE.value] = -1.0
        type_probs[ModuleType.CORE.value] = -1.0

        child_type = ModuleType(np.argmax(type_probs))
        if child_type == ModuleType.NONE:
            return None  # CPPN chose to leave this face empty
        child_rot = ModuleRotationsIdx(np.argmax(softmax(rot_scores)))
        if child_rot not in ALLOWED_ROTATIONS[child_type]:
            return None
        return conn_score, child_type, child_rot

    def _decode_fcl(self) -> nx.DiGraph:
        """Best-first growth on the real module geometry.

        Every open face of every placed module is a candidate. By default the
        CPPN sees the parent module's centre and the face's attachment point,
        both in the robot frame in units of FCL_POSITION_SCALE (see
        `_fcl_inputs` for the other input flags), and scores the
        connection. Each step places the highest-scoring candidate whose
        module would not overlap an already placed one (checked with FCL,
        see collision_utils). Growth ends at max_modules or when no
        candidate is left.

        A face's CPPN output never changes, so it is computed once when its
        module is placed. A candidate that collides is dropped for good: the
        body only grows, so it would keep colliding.
        """
        robot_graph, checker = self._new_body()

        # (parent_id, face) -> (score, child_type, child_rot, child_frame)
        candidates: dict[tuple[int, ModuleFaces], tuple] = {}

        def add_candidates(parent_id: int, parent_type: ModuleType) -> None:
            for face, *rest in self._face_candidates(checker, parent_id, parent_type):
                candidates[parent_id, face] = tuple(rest)

        add_candidates(IDX_OF_CORE, ModuleType.CORE)
        next_module_id = 1

        while len(robot_graph) < self.max_modules and candidates:
            # Highest score first; ties go to the earliest-added candidate.
            ranked = sorted(
                candidates.items(), key=lambda kv: kv[1][0], reverse=True
            )
            placed = False
            for (parent_id, face), (_, child_type, child_rot, frame) in ranked:
                del candidates[parent_id, face]
                if checker.collides(
                    frame, child_type.name, child_rot.name, ignore={parent_id}
                ):
                    continue

                child_id = next_module_id
                next_module_id += 1
                robot_graph.add_node(
                    child_id, type=child_type.name, rotation=child_rot.name
                )
                robot_graph.add_edge(parent_id, child_id, face=face.name)
                checker.add_module(
                    child_id, frame, child_type.name, child_rot.name
                )
                add_candidates(child_id, child_type)
                placed = True
                break
            if not placed:
                break

        _fix_terminal_hinges(robot_graph)
        return robot_graph

    def _decode_grid(self) -> nx.DiGraph:
        robot_graph = nx.DiGraph()
        occupied_coords = {}
        module_data = {}

        core_id, core_pos, core_type, core_rot = (
            IDX_OF_CORE,
            (0, 0, 0),
            ModuleType.CORE,
            ModuleRotationsIdx.DEG_0,
        )
        robot_graph.add_node(
            core_id, type=core_type.name, rotation=core_rot.name
        )
        occupied_coords[core_pos] = core_id
        module_data[core_id] = {
            "pos": core_pos,
            "type": core_type,
            "rot": core_rot,
        }

        # The frontier now contains ALL modules with potential open faces.
        frontier = [core_id]
        next_module_id = 1

        # Check for the max module count.
        while len(robot_graph) < self.max_modules:
            potential_connections = []

            # At each step, we check the ENTIRE frontier of all existing modules.
            for parent_id in frontier:
                parent_pos = module_data[parent_id]["pos"]
                parent_type = module_data[parent_id]["type"]
                for face in ModuleFaces:
                    if face not in ALLOWED_FACES[parent_type]:
                        break

                    child_pos = self._get_child_coords(parent_pos, face)

                    if child_pos in occupied_coords:
                        break

                    cppn_inputs = list(parent_pos) + list(child_pos)
                    raw_outputs = self.cppn_genome.activate(cppn_inputs)

                    conn_score = raw_outputs[0]
                    type_scores = np.array(
                        raw_outputs[1 : 1 + NUM_OF_TYPES_OF_MODULES]
                    )
                    rot_scores = np.array(
                        raw_outputs[1 + NUM_OF_TYPES_OF_MODULES :]
                    )

                    type_probs = softmax(type_scores)

                    if not self.allow_none:
                        type_probs[
                            ModuleType.NONE.value
                        ] = -1.0  # Ignore NONE if that's the output
                    type_probs[
                        ModuleType.CORE.value
                    ] = -1.0  # Ignore CORE if that's the output

                    child_type = ModuleType(np.argmax(type_probs))
                    if child_type == ModuleType.NONE:
                        continue  # CPPN chose to leave this face empty
                    child_rot = ModuleRotationsIdx(
                        np.argmax(softmax(rot_scores))
                    )

                    if (
                        face in ALLOWED_FACES[child_type]
                        and child_rot in ALLOWED_ROTATIONS[child_type]
                    ):
                        potential_connections.append({
                            "score": conn_score,
                            "parent_id": parent_id,
                            "child_pos": child_pos,
                            "child_type": child_type,
                            "child_rot": child_rot,
                            "face": face,
                        })

            if not potential_connections:
                # With allow_none this is the normal way growth ends.
                if not self.allow_none:
                    console.log(
                        "[yellow]Decoder stalled: No valid connections found anywhere on the robot.[/yellow]"
                    )
                break

            best_conn = max(potential_connections, key=lambda x: x["score"])

            child_id = next_module_id
            robot_graph.add_node(
                child_id,
                type=best_conn["child_type"].name,
                rotation=best_conn["child_rot"].name,
            )
            robot_graph.add_edge(
                best_conn["parent_id"], child_id, face=best_conn["face"].name
            )

            occupied_coords[best_conn["child_pos"]] = child_id
            module_data[child_id] = {
                "pos": best_conn["child_pos"],
                "type": best_conn["child_type"],
                "rot": best_conn["child_rot"],
            }

            # I no longer remove the parent, I just add the new child.
            # (I think this makes snakes less likely)
            frontier.append(child_id)
            next_module_id += 1

        _fix_terminal_hinges(robot_graph)
        return robot_graph
