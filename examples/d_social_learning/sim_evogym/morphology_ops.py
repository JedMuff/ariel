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
BODY_TYPES = [1, 2, 3, 4]
ACTUATOR_TYPES = [3, 4]
MIN_ACTUATORS = 1
MAX_RETRIES = 200


def random_individual(rng: random.Random | None = None) -> list:
    """Return a random valid 5×5 voxel body (dtype int)."""
    rng = rng or random
    body = np.full((5, 5), 0.0)

    body[rng.randint(0, 4)][rng.randint(0, 4)] = rng.choice(ACTUATOR_TYPES)
    for i in range(rng.randint(10, 20)):
        success = False
        while not success:
            new_grid = np.copy(body)
            x = rng.randint(0, 4)
            y = rng.randint(0, 4)
            if new_grid[x][y] != 0.0:
                continue

            new_grid[x][y] = float(rng.choice(BODY_TYPES))
            if not _is_valid(new_grid):
                continue

            body = new_grid
            success = True
    return body_to_list(body)


def mutate(body: list, rng: random.Random | None = None) -> list:
    """Return a mutated copy of *body*.

    Randomly resamples one or more voxels; retries until constraints are met.
    """
    rng = rng or random
    original = body_from_list(body.copy())
    for _ in range(MAX_RETRIES):
        candidate = original.copy()
        n_mutations = rng.randint(0, max(1, (ROWS * COLS) // 5))
        positions = [(r, c) for r in range(ROWS) for c in range(COLS)]
        chosen = rng.sample(positions, min(n_mutations, len(positions)))
        for r, c in chosen:
            candidate[r, c] = rng.choice(TYPES)
        if _is_valid(candidate):
            return body_to_list(candidate)
    return body_to_list(original)


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
