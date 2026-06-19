"""Weight-inheritance scheme functions.

Each function takes the current population state and the index of the target
individual, and returns (init_mean, donor_ids).

Population state is a list of dicts with keys:
  - "descriptor": np.ndarray shape (8,) in [0,1]
  - "theta": np.ndarray or None  (None = no prior brain)
  - "fitness": float or None
  - "db_id": int or None  (database primary key, for donor tracking)

K_INHERIT controls how many neighbours are averaged in *_many schemes.
"""

import random

import numpy as np

K_INHERIT = 3


def darwinian(
    pop_state: list[dict],
    idx: int,
    n_params: int,
) -> tuple[np.ndarray, list[int]]:
    return np.zeros(n_params, dtype=np.float64), []


def lamarckian(
    pop_state: list[dict],
    idx: int,
    n_params: int,
) -> tuple[np.ndarray, list[int]]:
    theta = pop_state[idx]["theta"]
    db_id = pop_state[idx].get("db_id")
    if theta is None:
        return np.zeros(n_params, dtype=np.float64), []
    donor_ids = [db_id] if db_id is not None else []
    return np.asarray(theta, dtype=np.float64), donor_ids


def random_scheme(
    pop_state: list[dict],
    idx: int,
    n_params: int,
) -> tuple[np.ndarray, list[int]]:
    candidates = [i for i, s in enumerate(pop_state) if s["theta"] is not None]
    if not candidates:
        return np.zeros(n_params, dtype=np.float64), []
    chosen = random.choice(candidates)
    db_id = pop_state[chosen].get("db_id")
    donor_ids = [db_id] if db_id is not None else []
    return np.asarray(pop_state[chosen]["theta"], dtype=np.float64), donor_ids


def random_many(
    pop_state: list[dict],
    idx: int,
    n_params: int,
) -> tuple[np.ndarray, list[int]]:
    candidates = [i for i, s in enumerate(pop_state) if s["theta"] is not None]
    if not candidates:
        return np.zeros(n_params, dtype=np.float64), []
    k = min(K_INHERIT, len(candidates))
    chosen = random.sample(candidates, k)
    thetas = np.array([pop_state[i]["theta"] for i in chosen], dtype=np.float64)
    donor_ids = [pop_state[i].get("db_id") for i in chosen if pop_state[i].get("db_id") is not None]
    return thetas.mean(axis=0), donor_ids


def best(
    pop_state: list[dict],
    idx: int,
    n_params: int,
) -> tuple[np.ndarray, list[int]]:
    evaluated = [(i, s) for i, s in enumerate(pop_state) if s["theta"] is not None and s["fitness"] is not None]
    if not evaluated:
        return np.zeros(n_params, dtype=np.float64), []
    best_i, best_s = max(evaluated, key=lambda t: t[1]["fitness"])
    db_id = best_s.get("db_id")
    donor_ids = [db_id] if db_id is not None else []
    return np.asarray(best_s["theta"], dtype=np.float64), donor_ids


def best_many(
    pop_state: list[dict],
    idx: int,
    n_params: int,
) -> tuple[np.ndarray, list[int]]:
    evaluated = [(i, s) for i, s in enumerate(pop_state) if s["theta"] is not None and s["fitness"] is not None]
    if not evaluated:
        return np.zeros(n_params, dtype=np.float64), []
    top = sorted(evaluated, key=lambda t: t[1]["fitness"], reverse=True)[:K_INHERIT]
    thetas = np.array([s["theta"] for _, s in top], dtype=np.float64)
    donor_ids = [s.get("db_id") for _, s in top if s.get("db_id") is not None]
    return thetas.mean(axis=0), donor_ids


def similar(
    pop_state: list[dict],
    idx: int,
    n_params: int,
) -> tuple[np.ndarray, list[int]]:
    target_desc = np.asarray(pop_state[idx]["descriptor"], dtype=np.float64)
    candidates = [
        (i, s) for i, s in enumerate(pop_state)
        if i != idx and s["theta"] is not None
    ]
    if not candidates:
        return np.zeros(n_params, dtype=np.float64), []
    nearest_i, nearest_s = min(
        candidates,
        key=lambda t: np.linalg.norm(np.asarray(t[1]["descriptor"], dtype=np.float64) - target_desc),
    )
    db_id = nearest_s.get("db_id")
    donor_ids = [db_id] if db_id is not None else []
    return np.asarray(nearest_s["theta"], dtype=np.float64), donor_ids


def similar_many(
    pop_state: list[dict],
    idx: int,
    n_params: int,
) -> tuple[np.ndarray, list[int]]:
    target_desc = np.asarray(pop_state[idx]["descriptor"], dtype=np.float64)
    candidates = [
        (i, s) for i, s in enumerate(pop_state)
        if i != idx and s["theta"] is not None
    ]
    if not candidates:
        return np.zeros(n_params, dtype=np.float64), []
    candidates.sort(key=lambda t: np.linalg.norm(np.asarray(t[1]["descriptor"], dtype=np.float64) - target_desc))
    top = candidates[:K_INHERIT]
    thetas = np.array([s["theta"] for _, s in top], dtype=np.float64)
    donor_ids = [s.get("db_id") for _, s in top if s.get("db_id") is not None]
    return thetas.mean(axis=0), donor_ids


SCHEMES = {
    "darwinian": darwinian,
    "lamarckian": lamarckian,
    "random": random_scheme,
    "random_many": random_many,
    "best": best,
    "best_many": best_many,
    "similar": similar,
    "similar_many": similar_many,
}
