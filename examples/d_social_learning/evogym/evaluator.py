"""Picklable EvoGym evaluator: inner CMA-ES + DistributedMLP + evogym sim."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
from pathlib import Path

# d_social_learning/evogym/ and d_social_learning/ both contain package-name
# subdirectories (evogym/, ariel/) that would shadow the real installed packages
# if added to sys.path.  We add only src/ (for ariel) and this directory; the
# evogym_adapter module is loaded by absolute path via importlib to avoid
# adding the parent dir to sys.path.
_THIS_DIR = Path(__file__).parent       # d_social_learning/evogym/
_SOCIAL_DIR = _THIS_DIR.parent          # d_social_learning/
_REPO_ROOT = _SOCIAL_DIR.parent.parent  # repo root
_SRC_DIR = _REPO_ROOT / "src"           # src/ contains the ariel package

for _p in [str(_THIS_DIR), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util as _ilu

def _load_evogym_adapter():
    _spec = _ilu.spec_from_file_location("evogym_adapter", _SOCIAL_DIR / "evogym_adapter.py")
    _mod = _ilu.module_from_spec(_spec)
    sys.modules.setdefault("evogym_adapter", _mod)
    _spec.loader.exec_module(sys.modules["evogym_adapter"])
    return sys.modules["evogym_adapter"]

import numpy as np

ENV_NAME = "Walker-v0"
N_STEPS = 500
N_NEIGHBORS = 8  # EvoGym Moore neighbourhood


def evaluate_individual(args: tuple) -> dict:
    """Picklable worker: inner CMA-ES for one EvoGym individual.

    Parameters
    ----------
    args : tuple
        (body_list, init_mean_list, donor_ids, inner_gens, pop_size)

    Returns
    -------
    dict with keys:
        distance       : float  — best x-displacement achieved
        best_theta     : list[float]  — best θ weights
        init_fitness   : float  — episode fitness of inherited theta before any learning
        learning_curve : list[list[float]]  — per inner-gen, fitness of every candidate
        donor_ids      : list[int]  — db ids of individuals whose theta was inherited
    """
    from ariel.simulation.controllers.cmaes_learner import CMAESLearner
    from ariel.simulation.controllers.distributed_mlp import DistributedMLP
    _adapter = _load_evogym_adapter()
    body_to_adjacency = _adapter.body_to_adjacency
    get_node_inputs = _adapter.get_node_inputs
    scale_actions = _adapter.scale_actions
    import evogym.envs  # noqa: F401 — registers envs
    import gymnasium as gym

    body_list, init_mean_list, donor_ids, inner_gens, pop_size = args

    _empty = {"distance": 0.0, "best_theta": [], "init_fitness": 0.0, "learning_curve": [], "donor_ids": donor_ids}

    try:
        import numpy as _np
        body = _np.array(body_list, dtype=int)

        env = gym.make(ENV_NAME, body=body)
        sim = env.unwrapped  # access evogym methods bypassing gymnasium wrappers
        adjacency = body_to_adjacency(body)

        brain = DistributedMLP(n_neighbors=N_NEIGHBORS)
        init_mean = (
            _np.asarray(init_mean_list, dtype=_np.float64)
            if init_mean_list
            else None
        )

        def run_episode(theta: _np.ndarray) -> float:
            brain.set_theta(theta)
            obs, _info = env.reset()
            x_start = float(sim.object_pos_at_time(sim.get_time(), "robot")[0].mean())
            for step in range(N_STEPS):
                node_inputs, t_sig = get_node_inputs(sim, body, adjacency, step)
                raw = brain.forward_all(node_inputs, t_sig)
                action = scale_actions(raw)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            x_end = float(sim.object_pos_at_time(sim.get_time(), "robot")[0].mean())
            return x_end - x_start

        # Evaluate inherited theta before any learning
        if init_mean is not None:
            init_fitness = run_episode(init_mean)
        else:
            init_fitness = run_episode(_np.zeros(brain.n_params, dtype=_np.float64))

        learner = CMAESLearner(
            n_params=brain.n_params,
            init_mean=init_mean,
            sigma=0.5,
            pop_size=pop_size,
        )

        learning_curve: list[list[float]] = []
        for _ in range(inner_gens):
            candidates = learner.ask()
            fitnesses = [run_episode(theta) for theta in candidates]
            learner.tell(candidates, fitnesses)
            learning_curve.append(fitnesses)

        env.close()
        return {
            "distance": learner.best_fitness,
            "best_theta": learner.best_theta.tolist(),
            "init_fitness": init_fitness,
            "learning_curve": learning_curve,
            "donor_ids": donor_ids,
        }

    except Exception as exc:  # noqa: BLE001
        print(f"[evogym evaluator] worker error: {exc}")
        try:
            env.close()
        except Exception:
            pass
        return _empty
