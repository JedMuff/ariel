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

# CPPN inputs: 6 values per face from the decoder; which ones depends on the
# genotype's decoder_version (see _CPPN_DECODER_FLAGS below and
# cppn_best_first.MorphologyDecoderBestFirst._fcl_inputs).
# New genomes also have a bias input (a constant in [-1, 1] stored in the
# genome and appended by CPPNGenome.activate), so they have one more input node
# than the decoder passes values.
# Outputs: 1 connection score + one score per module type + one per rotation.
NUM_CPPN_INPUTS  = 6
NUM_CPPN_GENOME_INPUTS = NUM_CPPN_INPUTS + 1
NUM_CPPN_OUTPUTS = 1 + NUM_OF_TYPES_OF_MODULES + NUM_OF_ROTATIONS

# Each mutation applies exactly one of these operators, drawn with these
# weights (CPPNGenome.mutate_one), so every mutation changes the genome.
_CPPN_MUTATION_PROBS = {
    "weights": 0.4,
    "biases": 0.2,
    "bias_input": 0.1,
    "add_connection": 0.2,
    "add_node": 0.1,
}
# Initial genomes get 0.._CPPN_INIT_MAX_HIDDEN hidden nodes (uniform), each
# followed by one extra random connection.
_CPPN_INIT_MAX_HIDDEN = 3
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
    node_start=NUM_CPPN_GENOME_INPUTS + NUM_CPPN_OUTPUTS - 1,
    innov_start=(NUM_CPPN_GENOME_INPUTS * NUM_CPPN_OUTPUTS) - 1,
)


# Every CPPN genotype records which decoder built its body, so checkpoints
# keep re-rendering/analysing as they evolved. CPPNGenome.from_dict ignores
# the extra key. Genotypes created from now on get _CPPN_DECODER_VERSION.
#   1 (key missing): legacy integer-grid decoder, NONE masked.
#   2: FCL collision checks on the real module geometry, NONE leaves a face
#      empty; global best-first growth on absolute xyz inputs.
#   3: as 2, but local + distance inputs (face direction, outwardness,
#      parent/attachment distance from the core) and per-module competition
#      (breadth-first; a face attaches if its score > 0.5). Chosen from the
#      decoder comparison in __data__/cppn_decoder_variants_seed42/FINDINGS.md.
#      Same 6 decoder inputs as 2, so genomes are created/mutated the same way.
_CPPN_DECODER_VERSION_KEY = "decoder_version"
_CPPN_DECODER_VERSION = 3
_CPPN_DECODER_FLAGS: dict[int, dict] = {
    1: {"legacy": True, "allow_none": False},
    2: {},
    3: {"distance_input": True, "local_inputs": True, "local_competition": True},
}


def _cppn_genotype(genome: CPPNGenome) -> dict:
    return {**genome.to_dict(), _CPPN_DECODER_VERSION_KEY: _CPPN_DECODER_VERSION}


def _cppn_decode(genotype_dict: dict, max_modules: int):
    genome = CPPNGenome.from_dict(genotype_dict)
    version = genotype_dict.get(_CPPN_DECODER_VERSION_KEY, 1)
    decoder = MorphologyDecoderBestFirst(
        cppn_genome=genome,
        max_modules=max_modules,
        **_CPPN_DECODER_FLAGS[version],
    )
    return decoder.decode()


def _cppn_is_valid(graph) -> bool:
    if graph.number_of_nodes() == 0:
        return False
    num_hinges = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == ModuleType.HINGE.name)
    return num_hinges >= shared.MIN_HINGES


def _random_cppn_genome(
    num_inputs: int = NUM_CPPN_INPUTS, id_manager: IdManager = _id_manager
) -> CPPNGenome:
    """`num_inputs` excludes the bias input. A non-default `num_inputs` needs
    its own `id_manager` starting after that genome's input/output node and
    initial connection ids."""
    genome = CPPNGenome.random(
        num_inputs=num_inputs,
        num_outputs=NUM_CPPN_OUTPUTS,
        next_node_id=num_inputs + 1 + NUM_CPPN_OUTPUTS,
        next_innov_id=0,
        bias_input=True,
    )
    # Hidden nodes split a random connection (random activation and bias),
    # then get one extra random connection each.
    for _ in range(random.randint(0, _CPPN_INIT_MAX_HIDDEN)):
        genome._mutate_add_node(id_manager.get_next_innov_id, id_manager.get_next_node_id)
        genome._mutate_add_connection(id_manager.get_next_innov_id)
    return genome


def _mutate_cppn_genome(genome: CPPNGenome) -> CPPNGenome:
    child = genome.copy()
    child.mutate_one(
        _CPPN_MUTATION_PROBS,
        _id_manager.get_next_innov_id, _id_manager.get_next_node_id,
    )
    return child


def _cppn_create_individual(rng: np.random.Generator, num_modules: int, max_depth: int) -> Individual:  # noqa: ARG001
    while True:
        genome = _random_cppn_genome()
        graph = _cppn_decode(_cppn_genotype(genome), num_modules)
        if _cppn_is_valid(graph):
            break
    ind = Individual()
    ind.id = shared._next_ind_id()
    ind.genotype = {"cppn": _cppn_genotype(genome)}
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
        graph = _cppn_decode(_cppn_genotype(child_genome), num_modules)
        while attempts < _MAX_OFFSPRING_ATTEMPTS:
            if _cppn_is_valid(graph):
                valid = True
                break
            child_genome = _mutate_cppn_genome(child_genome)
            graph = _cppn_decode(_cppn_genotype(child_genome), num_modules)
            attempts += 1

        child = Individual()
        child.id = shared._next_ind_id()
        child.genotype = {"cppn": _cppn_genotype(child_genome), "parent_ids": parent_ids}
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
