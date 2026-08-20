"""Genome-representation adapter for the symmetry-pressure task scripts.

Lets `gecko_food_skills.py` / `gecko_skill_tasks.py` stay genome-agnostic:
create individuals, reproduce, decode to an MjSpec, hash, and score
morphological symmetry the same way regardless of whether the body is
represented as a `TreeGenome` (optionally symmetry-enforced) or a CPPN-NEAT
`Genome` decoded via `MorphologyDecoderBestFirst`.

The tree adapters are thin wrappers around the existing `shared.py` helpers.
The CPPN adapter is new: CPPN genomes have no wiring into the `ariel.ec`
EA/Individual/Population outer loop anywhere else in the codebase today.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import mujoco
import numpy as np

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleType,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome as CPPNGenome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    MorphologyDecoderBestFirst,
)
from ariel.ec import Individual
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from ariel.utils.morphological_descriptor import MorphologicalMeasures

import shared

GenomeType = Literal["tree", "tree_symmetric", "cppn"]

# CPPN inputs: parent (x, y, z) + child (x, y, z) grid coords (matches
# cppn_best_first.MorphologyDecoderBestFirst._get_child_coords usage).
# Outputs: 1 connection score + one score per module type + one per rotation.
NUM_CPPN_INPUTS  = 6
NUM_CPPN_OUTPUTS = 1 + NUM_OF_TYPES_OF_MODULES + NUM_OF_ROTATIONS

_CPPN_NODE_ADD_RATE = 0.2
_CPPN_CONN_ADD_RATE = 0.3
_CPPN_INIT_STRUCTURAL_MUTATIONS = 3
_MAX_OFFSPRING_ATTEMPTS = 50


@dataclass
class GenomeAdapter:
    genotype_key: str
    create_individual: Callable[[np.random.Generator, int, int], Individual]
    make_offspring: Callable[[list[Individual], int, np.random.Generator, int, int], list[Individual]]
    to_spec: Callable[[dict], Optional[mujoco.MjSpec]]
    hash: Callable[[dict], str]
    symmetry_score: Callable[[dict], float]
    module_count: Callable[[dict], int]


# ── Tree adapters (wrap shared.py, no behavior change) ────────────────────────


def _tree_module_count(genotype_dict: dict) -> int:
    try:
        from ariel.ec.genotypes.tree.tree_genome import TreeGenome
        return len(TreeGenome.from_dict(genotype_dict).nodes)
    except Exception:
        return 0


def _make_tree_adapter(symmetry_axis: Optional[MirrorAxis]) -> GenomeAdapter:
    def _create_individual(rng: np.random.Generator, num_modules: int, max_depth: int) -> Individual:
        return shared.create_individual(num_modules, max_depth, symmetry_axis=symmetry_axis)

    def _make_offspring(
        parents: list[Individual], lam: int, rng: np.random.Generator, num_modules: int, max_depth: int
    ) -> list[Individual]:
        return shared.make_offspring(parents, lam, rng, num_modules, max_depth, symmetry_axis=symmetry_axis)

    def _symmetry_score(genotype_dict: dict) -> float:
        axis_str = "y_zero" if symmetry_axis in (None, MirrorAxis.Y_ZERO) else "x_equals_y"
        return shared.bilateral_symmetry_score(genotype_dict, axis=axis_str)

    return GenomeAdapter(
        genotype_key="morph",
        create_individual=_create_individual,
        make_offspring=_make_offspring,
        to_spec=shared.genome_to_spec,
        hash=shared.genome_hash,
        symmetry_score=_symmetry_score,
        module_count=_tree_module_count,
    )


# ── CPPN adapter ────────────────────────────────────────────────────────────────

# One IdManager for the whole process: NEAT innovation/node IDs must stay
# globally unique across the entire run for crossover gene-alignment to be
# meaningful (standard NEAT historical-marking convention).
_id_manager = IdManager(
    node_start=NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS - 1,
    innov_start=(NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS) - 1,
)


def _cppn_decode(genotype_dict: dict, max_modules: int):
    genome = CPPNGenome.from_dict(genotype_dict)
    decoder = MorphologyDecoderBestFirst(cppn_genome=genome, max_modules=max_modules)
    return decoder.decode()


def _cppn_is_valid(graph) -> bool:
    if graph.number_of_nodes() == 0:
        return False
    num_hinges = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == ModuleType.HINGE.name)
    return num_hinges >= shared.MIN_HINGES


def _random_cppn_genome() -> CPPNGenome:
    genome = CPPNGenome.random(
        num_inputs=NUM_CPPN_INPUTS,
        num_outputs=NUM_CPPN_OUTPUTS,
        next_node_id=NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS,
        next_innov_id=0,
    )
    for _ in range(_CPPN_INIT_STRUCTURAL_MUTATIONS):
        genome.mutate(
            _CPPN_NODE_ADD_RATE, _CPPN_CONN_ADD_RATE,
            _id_manager.get_next_innov_id, _id_manager.get_next_node_id,
        )
    return genome


def _mutate_cppn_genome(genome: CPPNGenome) -> CPPNGenome:
    child = genome.copy()
    child.mutate(
        _CPPN_NODE_ADD_RATE, _CPPN_CONN_ADD_RATE,
        _id_manager.get_next_innov_id, _id_manager.get_next_node_id,
    )
    return child


def _cppn_create_individual(rng: np.random.Generator, num_modules: int, max_depth: int) -> Individual:  # noqa: ARG001
    while True:
        genome = _random_cppn_genome()
        graph = _cppn_decode(genome.to_dict(), num_modules)
        if _cppn_is_valid(graph):
            break
    ind = Individual()
    ind.id = shared._next_ind_id()
    ind.genotype = {"cppn": genome.to_dict()}
    ind.tags = {"ps": False, "valid": True, "best_brain": []}
    return ind


def _cppn_make_offspring(
    parents: list[Individual], lam: int, rng: np.random.Generator, num_modules: int, max_depth: int  # noqa: ARG001
) -> list[Individual]:
    offspring: list[Individual] = []
    while len(offspring) < lam:
        use_sexual = len(parents) >= 2 and rng.random() < 0.6
        if use_sexual:
            p1, p2 = random.sample(parents, 2)
            g1 = CPPNGenome.from_dict(p1.genotype["cppn"])
            g2 = CPPNGenome.from_dict(p2.genotype["cppn"])
            g1.fitness = -(p1.fitness_ or 0.0)
            g2.fitness = -(p2.fitness_ or 0.0)
            child_genome = g1.crossover(g2)
            parent_ids = [p1.id, p2.id]
        else:
            p = random.choice(parents)
            child_genome = CPPNGenome.from_dict(p.genotype["cppn"])
            parent_ids = [p.id]

        child_genome = _mutate_cppn_genome(child_genome)

        attempts = 0
        valid = False
        graph = _cppn_decode(child_genome.to_dict(), num_modules)
        while attempts < _MAX_OFFSPRING_ATTEMPTS:
            if _cppn_is_valid(graph):
                valid = True
                break
            child_genome = _mutate_cppn_genome(child_genome)
            graph = _cppn_decode(child_genome.to_dict(), num_modules)
            attempts += 1

        child = Individual()
        child.id = shared._next_ind_id()
        child.genotype = {"cppn": child_genome.to_dict(), "parent_ids": parent_ids}
        child.tags = {"ps": False, "valid": valid, "best_brain": []}
        child.requires_eval = True
        if not valid:
            child.fitness = float("inf")
            child.requires_eval = False
        offspring.append(child)
    return offspring


def cppn_genome_to_spec(genotype_dict: dict, max_modules: int) -> Optional[mujoco.MjSpec]:
    """Module-level (picklable) CPPN decode-to-spec, bound to `max_modules` via
    `functools.partial` by `_make_cppn_adapter` below.

    `to_spec` gets threaded into `shared.train_skill_for_body`, which passes
    it through `ProcessPoolExecutor(initargs=...)` to worker processes. A
    closure over `max_modules` would only survive that under the "fork" start
    method (by accident of inherited memory, not real pickling) and silently
    break under "spawn"/"forkserver" — so this must be a plain top-level
    function, bound via `functools.partial` (itself picklable since its
    underlying func + args are), not a nested closure.
    """
    try:
        graph = _cppn_decode(genotype_dict, max_modules)
        if graph.number_of_nodes() == 0:
            return None
        return construct_mjspec_from_graph(graph).spec
    except Exception:
        return None


def _make_cppn_adapter(max_modules: int) -> GenomeAdapter:
    # Decoding is generative (greedy best-first search bounded by max_modules),
    # so every decode of a given genome — at creation, reproduction, or later
    # body-construction time — must use the same max_modules or the "same"
    # genome could decode to a different-sized robot.
    # _symmetry_score / _module_count are only ever called in the parent
    # process (logging/checkpointing), so closures are fine here — unlike
    # to_spec above, they never cross a ProcessPoolExecutor boundary.
    def _symmetry_score(genotype_dict: dict) -> float:
        try:
            graph = _cppn_decode(genotype_dict, max_modules)
            return float(MorphologicalMeasures(graph).symmetry)
        except Exception:
            return 0.0

    def _module_count(genotype_dict: dict) -> int:
        try:
            return _cppn_decode(genotype_dict, max_modules).number_of_nodes()
        except Exception:
            return 0

    return GenomeAdapter(
        genotype_key="cppn",
        create_individual=_cppn_create_individual,
        make_offspring=_cppn_make_offspring,
        to_spec=functools.partial(cppn_genome_to_spec, max_modules=max_modules),
        hash=shared.genome_hash,
        symmetry_score=_symmetry_score,
        module_count=_module_count,
    )


# ── Factory ───────────────────────────────────────────────────────────────────


def genome_adapter_from_cli(name: GenomeType, max_modules: int) -> GenomeAdapter:
    if name == "tree":
        return _make_tree_adapter(None)
    if name == "tree_symmetric":
        return _make_tree_adapter(MirrorAxis.Y_ZERO)
    if name == "cppn":
        return _make_cppn_adapter(max_modules)
    raise ValueError(f"Unknown genome type: {name!r}")
