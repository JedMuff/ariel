"""The vectorised aligned Hamming distance used for the final-results
donor-offspring distances must match the reference implementation
(sim_evogym/evogym_body_descriptors.aligned_hamming_distance)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3] / "examples" / "d_social_learning"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from final_results_data import aligned_hamming_batch  # noqa: E402
from sim_evogym.evogym_body_descriptors import aligned_hamming_distance  # noqa: E402


def random_bodies(rng, n, size=5):
    """Random voxel grids (materials 0-4), each with at least one voxel, some
    sparse so that they can be shifted around."""
    bodies = rng.integers(0, 5, size=(n, size, size)) * (rng.random((n, 1, 1)) < rng.random((n, size, size)))
    bodies[:, 2, 2] = np.where(bodies.reshape(n, -1).any(axis=1), bodies[:, 2, 2], 1)
    return bodies


def test_matches_reference():
    rng = np.random.default_rng(0)
    a, b = random_bodies(rng, 60), random_bodies(rng, 60)
    expected = [aligned_hamming_distance(x.astype(float), y.astype(float)) for x, y in zip(a, b)]
    np.testing.assert_array_equal(aligned_hamming_batch(a, b), expected)


def test_identical_and_shifted_bodies_have_zero_distance():
    body = np.zeros((5, 5), dtype=int)
    body[0, 0], body[0, 1], body[1, 0] = 1, 3, 4
    shifted = np.roll(body, (2, 3), axis=(0, 1))
    assert aligned_hamming_batch(np.stack([body, body]), np.stack([body, shifted])).tolist() == [0, 0]
