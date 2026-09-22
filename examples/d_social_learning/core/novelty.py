"""kNN-based novelty computation from descriptor vectors."""

from typing import Callable

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


def compute_novelty_ted(
    morphologies: list, distance_fn: Callable[[object, object], float], k: int = K_NOVELTY,
) -> np.ndarray:
    """Like ``compute_novelty``, but for a metric with no cheap spatial index.

    Builds the full pairwise distance matrix via ``distance_fn`` (e.g. tree
    edit distance) instead of a cKDTree, since such metrics generally aren't
    embeddable in a space a KD-tree can index.

    Parameters
    ----------
    morphologies:
        List of N raw morphology objects (whatever ``distance_fn`` accepts).
    distance_fn:
        Symmetric distance function between two morphologies.
    k:
        Number of nearest neighbours.

    Returns
    -------
    np.ndarray shape (N,)
        Novelty score for each individual. Returns zeros when N <= 1.
    """
    n = len(morphologies)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    k_actual = min(k, n - 1)
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_fn(morphologies[i], morphologies[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    sorted_dists = np.sort(dist_mat, axis=1)
    return sorted_dists[:, 1:k_actual + 1].mean(axis=1)
