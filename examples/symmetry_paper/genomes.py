"""Body genomes of the paper, behind one adapter so evolve.py and
sample_bodies.py treat them alike:

  tree            unconstrained tree genome
  tree_symmetric  mirror-symmetric about the y = 0 plane (mirrored limbs)
  tree_tied       each placed module gets a randomly placed twin with
                  probability tie_prob, matching tree_symmetric's gene count
                  without mirroring (ariel.ec.genotypes.tree.tying)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

from ariel.ec import Individual
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

import sim

GenomeType = Literal["tree", "tree_symmetric", "tree_tied"]
GENOME_TYPES: tuple[str, ...] = ("tree", "tree_symmetric", "tree_tied")

# tree_tied: chance a placed module gets a randomly placed twin. 0.47 makes
# ~64% of modules tied (2p / (1 + p)), the mirrored share of tree_symmetric
# bodies in the zoo-module runs. Under random drift with mild size selection
# the tied share (0.82) matches tree_symmetric's mirrored share (0.80).
DEFAULT_TIE_PROB = 0.47


@dataclass
class GenomeAdapter:
    create_individual: Callable[[np.random.Generator, int, int], Individual]
    make_offspring: Callable[[list[Individual], int, np.random.Generator, int, int], list[Individual]]
    symmetry_score: Callable[[dict], float]


def module_count(genotype_dict: dict) -> int:
    try:
        return len(TreeGenome.from_dict(genotype_dict).nodes)
    except Exception:
        return 0


def _make_adapter(symmetry_axis: Optional[MirrorAxis], tie_prob: Optional[float] = None) -> GenomeAdapter:
    def _create_individual(rng: np.random.Generator, num_modules: int, max_depth: int) -> Individual:  # noqa: ARG001
        return sim.create_individual(num_modules, max_depth, symmetry_axis=symmetry_axis, tie_prob=tie_prob)

    def _make_offspring(
        parents: list[Individual], lam: int, rng: np.random.Generator, num_modules: int, max_depth: int
    ) -> list[Individual]:
        return sim.make_offspring(parents, lam, rng, num_modules, max_depth, symmetry_axis=symmetry_axis,
                                  tie_prob=tie_prob)

    def _symmetry_score(genotype_dict: dict) -> float:
        axis_str = "y_zero" if symmetry_axis in (None, MirrorAxis.Y_ZERO) else "x_equals_y"
        return sim.bilateral_symmetry_score(genotype_dict, axis=axis_str)

    return GenomeAdapter(create_individual=_create_individual, make_offspring=_make_offspring,
                         symmetry_score=_symmetry_score)


def genome_adapter(name: GenomeType, tie_prob: float = DEFAULT_TIE_PROB) -> GenomeAdapter:
    if name == "tree":
        return _make_adapter(None)
    if name == "tree_symmetric":
        return _make_adapter(MirrorAxis.Y_ZERO)
    if name == "tree_tied":
        return _make_adapter(None, tie_prob=tie_prob)
    raise ValueError(f"Unknown genome type: {name!r}")
