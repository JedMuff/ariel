"""TreeGenome → 8-d morphological descriptor vector."""

import numpy as np

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.utils.morphological_descriptor import MorphologicalMeasures


def tree_descriptor(genome_dict: dict) -> np.ndarray:
    """Return 8-d descriptor d = [B, L, E, C, J, S, D, P] from a stored genome dict.

    All values are in [0, 1] by construction of MorphologicalMeasures.

    Parameters
    ----------
    genome_dict:
        Dict as returned by ``Individual.genotype_["morph"]`` (output of
        ``TreeGenome.to_dict()``).

    Returns
    -------
    np.ndarray shape (8,)
    """
    genome = TreeGenome.from_dict(genome_dict)
    graph = genome.to_networkx()
    m = MorphologicalMeasures(graph)
    d = np.array([m.B, m.L, m.E, m.C, m.J, m.S, m.D, m.P], dtype=np.float64)
    assert np.all((d >= 0.0) & (d <= 1.0)), f"Descriptor out of [0,1]: {d}"
    return d
