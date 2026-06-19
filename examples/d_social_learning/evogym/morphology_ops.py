"""Voxel grid initialisation and mutation for EvoGym domain.

Direct encoding: fixed 5×5 grid, cell types {0=empty,1=rigid,2=soft,3=h-act,4=v-act}.
Constraints enforced:
  - At least one actuator (type 3 or 4).
  - All non-empty voxels must be connected (4-connectivity).
"""

from __future__ import annotations

import random

import numpy as np

ROWS = 5
COLS = 5
TYPES = [0, 1, 2, 3, 4]
ACTUATOR_TYPES = {3, 4}
MIN_ACTUATORS = 1
MAX_RETRIES = 200


def random_body(rng: random.Random | None = None) -> np.ndarray:
    """Return a random valid 5×5 voxel body (dtype int)."""
    rng = rng or random
    for _ in range(MAX_RETRIES):
        body = np.array(
            [rng.choice(TYPES) for _ in range(ROWS * COLS)], dtype=int
        ).reshape(ROWS, COLS)
        if _is_valid(body):
            return body
    # fallback: minimal valid body (single actuator)
    body = np.zeros((ROWS, COLS), dtype=int)
    body[2, 2] = 3
    return body


def mutate_body(body: np.ndarray, rng: random.Random | None = None) -> np.ndarray:
    """Return a mutated copy of *body*.

    Randomly resamples one or more voxels; retries until constraints are met.
    """
    rng = rng or random
    original = body.copy()
    for _ in range(MAX_RETRIES):
        candidate = original.copy()
        n_mutations = rng.randint(1, max(1, (ROWS * COLS) // 5))
        positions = [(r, c) for r in range(ROWS) for c in range(COLS)]
        chosen = rng.sample(positions, min(n_mutations, len(positions)))
        for r, c in chosen:
            candidate[r, c] = rng.choice(TYPES)
        if _is_valid(candidate):
            return candidate
    return original


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _has_actuator(body: np.ndarray) -> bool:
    return int((body == 3).sum() + (body == 4).sum()) >= MIN_ACTUATORS


def _is_connected(body: np.ndarray) -> bool:
    nonzero = list(zip(*np.nonzero(body)))
    if len(nonzero) == 0:
        return False
    if len(nonzero) == 1:
        return True
    visited = set()
    queue = [nonzero[0]]
    visited.add(nonzero[0])
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and 0 <= nr < ROWS and 0 <= nc < COLS and body[nr, nc] != 0:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return len(visited) == len(nonzero)


def _is_valid(body: np.ndarray) -> bool:
    return _has_actuator(body) and _is_connected(body)


def body_to_list(body: np.ndarray) -> list[list[int]]:
    return body.tolist()


def body_from_list(lst: list[list[int]]) -> np.ndarray:
    return np.array(lst, dtype=int)
