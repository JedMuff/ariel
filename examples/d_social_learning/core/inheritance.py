"""Weight-inheritance scheme functions.

Each function takes the current population state and the index of the target
individual, and returns (init_mean, donor_ids).

Population state is a list of dicts with keys:
  - "descriptor": np.ndarray shape (8,) in [0,1]
  - "theta": np.ndarray or None  (None = no prior brain)
  - "fitness": float or None
  - "db_id": int or None  (database primary key, for donor tracking)
  - "morphology": raw stored morphology
  - "distance_fns": {"STRUCT": fn, "TOPDOWN": fn or None} structural
    distances between two morphologies (TOPDOWN is ariel-only)

K_INHERIT controls how many neighbours are averaged in *_many schemes.
"""

import random

import numpy as np

K_INHERIT = 3


def darwinian(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    return np.random.uniform(-1.0, 1.0, size=n_params).astype(np.float64), []

def lamarckian(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    parent_id = pop_state[idx].get("parent_id")
    if parent_id is not None:
        parent_state = next((s for s in pop_state if s.get("db_id") == parent_id), None)
        if parent_state and parent_state["theta"] is not None:
            donor_ids = [parent_id]
            return np.asarray(parent_state["theta"], dtype=np.float64), donor_ids

    return darwinian(pop_state, idx, n_params)


def random_scheme(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    candidates = [i for i, s in enumerate(pop_state) if s["theta"] is not None]
    if not candidates:
        return darwinian(pop_state, idx, n_params)
    chosen = random.choice(candidates)
    db_id = pop_state[chosen].get("db_id")
    donor_ids = [db_id] if db_id is not None else []
    return np.asarray(pop_state[chosen]["theta"], dtype=np.float64), donor_ids


def random_many(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    candidates = [i for i, s in enumerate(pop_state) if s["theta"] is not None]
    if not candidates:
        return darwinian(pop_state, idx, n_params, k_inherit)
    k = min(k_inherit, len(candidates))
    chosen = random.sample(candidates, k)
    thetas = np.array([pop_state[i]["theta"] for i in chosen], dtype=np.float64)
    donor_ids = [pop_state[i].get("db_id") for i in chosen if pop_state[i].get("db_id") is not None]
    return thetas.mean(axis=0), donor_ids


def best(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    evaluated = [(i, s) for i, s in enumerate(pop_state) if s["theta"] is not None and s["distance"] is not None]
    if not evaluated:
        return darwinian(pop_state, idx, n_params, k_inherit)
    best_i, best_s = max(evaluated, key=lambda t: t[1]["distance"])
    db_id = best_s.get("db_id")
    donor_ids = [db_id] if db_id is not None else []
    return np.asarray(best_s["theta"], dtype=np.float64), donor_ids


def best_many(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    evaluated = [(i, s) for i, s in enumerate(pop_state) if s["theta"] is not None and s["distance"] is not None]
    if not evaluated:
        return darwinian(pop_state, idx, n_params, k_inherit)
    top = sorted(evaluated, key=lambda t: t[1]["distance"], reverse=True)[:k_inherit]
    thetas = np.array([s["theta"] for _, s in top], dtype=np.float64)
    donor_ids = [s.get("db_id") for _, s in top if s.get("db_id") is not None]
    return thetas.mean(axis=0), donor_ids


def similar_DESCR(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    """Donor = nearest neighbour by Euclidean distance in morphological-descriptor space."""
    target_desc = np.asarray(pop_state[idx]["descriptor"], dtype=np.float64)
    candidates = [
        (i, s) for i, s in enumerate(pop_state)
        if i != idx and s["theta"] is not None
    ]
    if not candidates:
        return darwinian(pop_state, idx, n_params, k_inherit)
    nearest_i, nearest_s = min(
        candidates,
        key=lambda t: np.linalg.norm(np.asarray(t[1]["descriptor"], dtype=np.float64) - target_desc),
    )
    db_id = nearest_s.get("db_id")
    donor_ids = [db_id] if db_id is not None else []
    return np.asarray(nearest_s["theta"], dtype=np.float64), donor_ids


def similar_many_DESCR(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    """Like ``similar_DESCR``, averaged over the K_INHERIT nearest donors."""
    target_desc = np.asarray(pop_state[idx]["descriptor"], dtype=np.float64)
    candidates = [
        (i, s) for i, s in enumerate(pop_state)
        if i != idx and s["theta"] is not None
    ]
    if not candidates:
        return darwinian(pop_state, idx, n_params, k_inherit)
    candidates.sort(key=lambda t: np.linalg.norm(np.asarray(t[1]["descriptor"], dtype=np.float64) - target_desc))
    top = candidates[:k_inherit]
    thetas = np.array([s["theta"] for _, s in top], dtype=np.float64)
    donor_ids = [s.get("db_id") for _, s in top if s.get("db_id") is not None]
    return thetas.mean(axis=0), donor_ids


def _nearest_by_structure(
    pop_state: list[dict], idx: int, metric: str, k: int,
) -> list[dict] | None:
    """The k candidates (other individuals with a brain) nearest to idx by
    ``pop_state[idx]["distance_fns"][metric]`` applied to the raw morphology
    -- *not* the descriptor vector. Morphology is passed through as-is (not
    cast to a numeric array): ariel's morphology is a genome dict, not an
    array. None if there are no candidates."""
    distance_fn = pop_state[idx]["distance_fns"][metric]
    target_morphology = pop_state[idx]["morphology"]
    candidates = [s for i, s in enumerate(pop_state) if i != idx and s["theta"] is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda s: distance_fn(s["morphology"], target_morphology))
    return candidates[:k]


def _inherit_from(donors: list[dict]) -> tuple[np.ndarray, list[int]]:
    thetas = np.array([s["theta"] for s in donors], dtype=np.float64)
    donor_ids = [s.get("db_id") for s in donors if s.get("db_id") is not None]
    return thetas.mean(axis=0), donor_ids


def similar_STRUCT(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    """Donor = nearest neighbour by the STRUCT distance (tree edit distance
    for ariel, aligned Hamming distance on the voxel grid for evogym)."""
    donors = _nearest_by_structure(pop_state, idx, "STRUCT", 1)
    if donors is None:
        return darwinian(pop_state, idx, n_params, k_inherit)
    return _inherit_from(donors)


def similar_many_STRUCT(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    """Like ``similar_STRUCT``, averaged over the K_INHERIT nearest donors."""
    donors = _nearest_by_structure(pop_state, idx, "STRUCT", k_inherit)
    if donors is None:
        return darwinian(pop_state, idx, n_params, k_inherit)
    return _inherit_from(donors)


def similar_TOPDOWN(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    """Donor = nearest neighbour by exact top-down tree edit distance (ariel
    only; see sim_ariel/tree_edit_distance.py's top_down_tree_distance)."""
    donors = _nearest_by_structure(pop_state, idx, "TOPDOWN", 1)
    if donors is None:
        return darwinian(pop_state, idx, n_params, k_inherit)
    return _inherit_from(donors)


def similar_many_TOPDOWN(
    pop_state: list[dict],
    idx: int,
    n_params: int,
    k_inherit: int = K_INHERIT,
) -> tuple[np.ndarray, list[int]]:
    """Like ``similar_TOPDOWN``, averaged over the K_INHERIT nearest donors."""
    donors = _nearest_by_structure(pop_state, idx, "TOPDOWN", k_inherit)
    if donors is None:
        return darwinian(pop_state, idx, n_params, k_inherit)
    return _inherit_from(donors)


SCHEMES = {
    "darwinian": darwinian,
    "lamarckian": lamarckian,
    "random": random_scheme,
    "random_many": random_many,
    "best": best,
    "best_many": best_many,
    "similar_DESCR": similar_DESCR,
    "similar_many_DESCR": similar_many_DESCR,
    "similar_STRUCT": similar_STRUCT,
    "similar_many_STRUCT": similar_many_STRUCT,
    "similar_TOPDOWN": similar_TOPDOWN,
    "similar_many_TOPDOWN": similar_many_TOPDOWN,
}
