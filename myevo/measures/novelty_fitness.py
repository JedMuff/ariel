"""Novelty fitness evaluation using morphological feature vectors.

This module provides efficient novelty calculation using KD-tree spatial indexing
for fast nearest neighbor search in morphological feature space.
"""

import numpy as np
from scipy.spatial import KDTree


def extract_morphological_vector(measures):
    """Extract a feature vector from morphological measures.

    Args:
        measures: MorphologicalMeasures object from myevo/measures/morphological_measures.py

    Returns:
        numpy array: [branching, limbs, length_of_limbs, coverage, joints, proportion, symmetry, size]
    """
    # Choose proportion based on whether the robot is 2D or 3D
    proportion = measures.proportion_2d if measures.is_2d else measures.proportion_3d

    return np.array([
        measures.branching,
        measures.limbs,
        measures.length_of_limbs,
        measures.coverage,
        measures.joints,
        proportion,
        measures.symmetry,
        measures.size
    ], dtype=np.float64)


class KDTreeArchive:
    """Archive optimized for Euclidean distance using KD-tree for fast nearest neighbor search.

    Works with morphological feature vectors and provides efficient novelty calculation
    for large archives using KD-tree spatial indexing.

    Combines Poisson disk sampling (optional min_distance) with KD-tree for
    efficient nearest neighbor queries. This is much faster than linear search for
    large archives when using Euclidean distance.
    """

    def __init__(self, min_distance=None, feature_extractor=None, adaptive=False):
        """Initialize the KDTreeArchive.

        Args:
            min_distance (float, optional): Minimum allowed distance between archived individuals.
                                           If None, all individuals are added.
            feature_extractor (callable, optional): Function to extract feature vector from individual.
                                                   Defaults to extract_morphological_vector.
            adaptive (bool): Whether to adaptively adjust the min_distance based on archive growth.
        """
        self.archive = []
        self.feature_vectors = []
        self.min_distance = min_distance
        self.feature_extractor = feature_extractor or extract_morphological_vector
        self.adaptive = adaptive
        self.kdtree = None
        self._insert_attempts = 0
        self._insert_successes = 0

    def _rebuild_kdtree(self):
        """Rebuild the KD-tree with current feature vectors."""
        if self.feature_vectors:
            self.kdtree = KDTree(np.array(self.feature_vectors))
        else:
            self.kdtree = None

    def add(self, individual):
        """Attempt to add an individual to the archive.

        If min_distance is None, always adds the individual.
        Otherwise, only adds if it's far enough from all existing members.

        Args:
            individual: The individual to add (must be compatible with feature_extractor).

        Returns:
            bool: True if added, False otherwise.
        """
        # Extract feature vector
        feature_vec = self.feature_extractor(individual)

        if not self.archive:
            self.archive.append(individual)
            self.feature_vectors.append(feature_vec)
            self._rebuild_kdtree()
            self._insert_successes += 1
            return True

        # If no minimum distance constraint, always add
        if self.min_distance is None:
            self.archive.append(individual)
            self.feature_vectors.append(feature_vec)
            self._rebuild_kdtree()
            self._insert_successes += 1
            return True

        # Check if too close to any existing member using KD-tree
        distances, _ = self.kdtree.query(feature_vec, k=1)
        min_dist = distances if np.isscalar(distances) else distances[0]

        if min_dist < self.min_distance:
            # Too close to an existing member — reject
            self._insert_attempts += 1
            return False

        # Far enough from all others — accept
        self.archive.append(individual)
        self.feature_vectors.append(feature_vec)
        self._rebuild_kdtree()
        self._insert_successes += 1
        self._insert_attempts += 1

        if self.adaptive:
            self._adapt_threshold()

        return True

    def novelty(self, individual, k=5):
        """Compute the novelty of an individual with respect to the archive using KD-tree.

        Novelty is defined as the average distance to the k nearest neighbors in the archive.

        Args:
            individual: The individual to evaluate.
            k (int): Number of nearest neighbors to consider.

        Returns:
            float: Novelty score (average of k nearest distances).
        """
        if not self.archive:
            return float("inf")

        # Extract feature vector
        feature_vec = self.feature_extractor(individual)

        # Query KD-tree for k nearest neighbors
        k_actual = min(k, len(self.archive))
        distances, _ = self.kdtree.query(feature_vec, k=k_actual)

        # Handle both single and multiple neighbors
        if np.isscalar(distances):
            return float(distances)

        return np.mean(distances)

    def _adapt_threshold(self):
        """Simple adaptive strategy: adjust threshold based on insertion ratio."""
        if self._insert_attempts < 10:
            return  # avoid early noise

        ratio = self._insert_successes / self._insert_attempts
        if ratio > 0.5:
            self.min_distance *= 1.05
        elif ratio < 0.1:
            self.min_distance *= 0.95

        self._insert_attempts = 0
        self._insert_successes = 0

    def get_archive(self):
        """Return the current list of archived individuals.

        Returns:
            list: All archived individuals.
        """
        return self.archive

    def __len__(self):
        """Return the number of individuals in the archive.

        Returns:
            int: Archive size.
        """
        return len(self.archive)
