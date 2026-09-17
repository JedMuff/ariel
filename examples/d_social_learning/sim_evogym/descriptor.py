"""EvoGym voxel-grid → 5-d morphological descriptor vector."""

import numpy as np

from evogym_body_descriptors import relative_activity, size, compactness, elongation, symmetry
from morphology_ops import body_from_list


def voxel_descriptor(genome) -> np.ndarray:
    """Return a 5-d descriptor for an EvoGym body grid.

    d = [relative_activity, size, compactness, elongation, symmetry]

    All values are in [0, 1] by construction.

    Parameters
    ----------
    body : np.ndarray shape (H, W) with dtype int
        Voxel grid where each cell is one of {0=empty,1=rigid,2=soft,3=h-act,4=v-act}.

    Returns
    -------
    np.ndarray shape (5,) in [0, 1]
    """
    body = body_from_list(genome)

    d = np.array(
        [
            relative_activity(body),
            size(body),
            compactness(body),
            elongation(body),
            symmetry(body),
        ],
        dtype=np.float64,
    )
    assert np.all((d >= 0.0) & (d <= 1.0)), f"Descriptor out of [0,1]: {d}"
    return d
