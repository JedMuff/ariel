"""EvoGym morphological descriptor — PLACEHOLDER.

Returns zeros(8) until a collaborator implements voxel-based descriptors.
When descriptors are zero, novelty = 0 and fitness = pure distance.
"""

import numpy as np


def voxel_descriptor(body: np.ndarray) -> np.ndarray:
    """Return an 8-d descriptor for an EvoGym body grid.

    Currently a stub returning zeros. Novelty will be 0 until implemented.

    Parameters
    ----------
    body : np.ndarray shape (H, W) with dtype int
        Voxel grid where each cell is one of {0=empty,1=rigid,2=soft,3=h-act,4=v-act}.

    Returns
    -------
    np.ndarray shape (8,) in [0, 1]
    """
    return np.zeros(8, dtype=np.float64)
