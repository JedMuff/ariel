"""kNN-based novelty computation from descriptor vectors."""

import numpy as np
from scipy.spatial import cKDTree

K_NOVELTY = 15


def compute_novelty(descriptors: list[np.ndarray], k: int = K_NOVELTY) -> np.ndarray:
    """Return per-individual novelty scores (mean distance to k nearest neighbours).

    Parameters
    ----------
    descriptors:
        List of N descriptor vectors, each shape (D,), all values in [0, 1].
    k:
        Number of nearest neighbours.

    Returns
    -------
    np.ndarray shape (N,)
        Novelty score for each individual. Returns zeros when N <= 1.
    """
    n = len(descriptors)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    mat = np.array(descriptors, dtype=np.float64)
    assert np.all((mat >= 0.0) & (mat <= 1.0)), "Descriptor values must be in [0, 1]"

    k_actual = min(k, n - 1)
    tree = cKDTree(mat)
    dists, _ = tree.query(mat, k=k_actual + 1)  # +1 because closest is self
    return dists[:, 1:].mean(axis=1)
