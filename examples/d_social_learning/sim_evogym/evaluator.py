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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

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
import numpy as np
import evogym.envs
from evogym import EvoSim

ENV_NAME = "Walker-v0"
# ENV_NAME = "UpStepper-v0"
# ENV_NAME = "Carrier-v0"
N_STEPS = 499

def initialize_world(genome):
    EvoSim._has_displayed_version = True
    import gymnasium as gym
    import evogym_adapter as adapter
    body_to_adjacency = adapter.body_to_adjacency
    body = np.array(genome, dtype=int)

    env = gym.make(ENV_NAME, body=body)
    sim = env.unwrapped  # access evogym methods bypassing gymnasium wrappers
    adjacency = body_to_adjacency(body)

    return {
        "env": env,
        "sim": sim,
        "adjacency": adjacency,
        "body": body
    }

def kill_world(simulator_specifics: dict):
    simulator_specifics['env'].close()

def run_episode(theta: np.ndarray, brain, simulator_specifics: dict) -> dict:
    import evogym_adapter as adapter
    get_node_inputs = adapter.get_node_inputs
    scale_actions = adapter.scale_actions

    env = simulator_specifics["env"]
    sim = simulator_specifics["sim"]
    adjacency = simulator_specifics["adjacency"]
    body = simulator_specifics["body"]

    brain.set_theta(theta)
    obs, _info = env.reset()

    x_start = float(sim.object_pos_at_time(sim.get_time(), "robot")[0].mean())
    x_package_start = 0
    if ENV_NAME == "Carrier-v0":
        x_package_start = float(sim.object_pos_at_time(sim.get_time(), "package")[0].mean())

    raw = None
    for step in range(N_STEPS):
        if step % 5 == 0 or raw is None:
            node_inputs, t_sig = get_node_inputs(sim, body, adjacency, step)
            raw = brain.forward_all(node_inputs, t_sig)
        action = scale_actions(raw)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    x_end = float(sim.object_pos_at_time(sim.get_time(), "robot")[0].mean())
    result = x_end - x_start  # Normal fitness
    if ENV_NAME == "Carrier-v0":
        x_package_end = float(sim.object_pos_at_time(sim.get_time(), "package")[0].mean())
        result = x_package_end - x_package_start  # Fitness if Carrier environment and package still being carried
        if float(sim.object_pos_at_time(sim.get_time(), "package")[0].min()) < 1.5:
            result = -abs(x_package_end - x_end)  # Fitness if Carrier environment and package dropped

    return {"fitness": result}
