"""
Train a gecko to walk forward along the +x axis using CMA-ES.

Fitness = negative x-displacement over the episode (lower is better, so
maximising forward travel = minimising negative travel).

Weights are saved to __data__/gecko_forward/best_weights.npy.
Use gecko_forward_viewer.py (run under mjpython) to replay interactively.

Usage
-----
    uv run python examples/re_book/gecko_forward.py
    uv run python examples/re_book/gecko_forward.py --budget 100 --pop 32 --workers 8 --dur 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional, cast

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")

import mujoco
import nevergrad as ng
import numpy as np
import torch
from torch import nn
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Gecko forward-walking brain evolution")
parser.add_argument("--budget",      type=int,   default=80)
parser.add_argument("--pop",         type=int,   default=32)
parser.add_argument("--workers",     type=int,   default=max(1, (os.cpu_count() or 1) - 1))
parser.add_argument("--dur",         type=float, default=8.0,  help="Episode duration (s)")
parser.add_argument("--hidden",      type=int,   default=32)
parser.add_argument("--seed",        type=int,   default=42)
args = parser.parse_args()

BUDGET     = args.budget
POP_SIZE   = args.pop
NUM_WORKERS = max(1, args.workers)
DURATION   = args.dur
HIDDEN     = args.hidden
BASE_SEED  = args.seed

DATA = Path.cwd() / "__data__" / "gecko_forward"
DATA.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = DATA / "best_weights.npy"

CONTROL_HZ = 20   # policy runs at 20 Hz
SPAWN_POS  = [0.0, 0.0, 0.1]


# ── Network ───────────────────────────────────────────────────────────────────

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1    = nn.Linear(input_size, HIDDEN)
        self.fc2    = nn.Linear(HIDDEN, HIDDEN)
        self.fc_out = nn.Linear(HIDDEN, output_size)
        self.act    = nn.ELU()
        self.out_act = nn.Tanh()
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, state: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(state, dtype=torch.float32)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return (self.out_act(self.fc_out(x)) * (np.pi / 2)).numpy()


def fill_parameters(net: nn.Module, vector: np.ndarray) -> None:
    offset = 0
    for p in net.parameters():
        n = p.numel()
        p.data.view(-1)[:] = torch.as_tensor(vector[offset: offset + n])
        offset += n


# ── World builder ─────────────────────────────────────────────────────────────

def build_world() -> tuple[mujoco.MjModel, mujoco.MjData]:
    world = SimpleFlatWorld()
    body  = gecko()
    world.spawn(body.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    return model, data


# ── Episode (pure physics, no rendering) ─────────────────────────────────────

def run_episode(model: mujoco.MjModel, data: mujoco.MjData, net: Network) -> float:
    mujoco.mj_resetData(model, data)
    dt             = model.opt.timestep
    control_period = int(round(1.0 / (CONTROL_HZ * dt)))
    current_ctrl   = np.zeros(model.nu)
    step           = 0

    while data.time < DURATION:
        if step % control_period == 0:
            state        = _make_state(data)
            current_ctrl = net.forward(state)
        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)
        step += 1

    return float(data.qpos[1])   # y-displacement


def _make_state(data: mujoco.MjData) -> np.ndarray:
    robot = get_state_from_data(data)
    phase = [np.sin(data.time * 2.0 * np.pi), np.cos(data.time * 2.0 * np.pi)]
    return np.concatenate([robot, phase]).astype(np.float32)


# ── Worker pool ───────────────────────────────────────────────────────────────

_ctx: Optional[dict[str, Any]] = None


def _init_worker(seed: int) -> None:
    torch.set_num_threads(1)
    s = (seed + os.getpid()) % (2**32 - 1)
    np.random.seed(s)
    torch.manual_seed(s)


def _get_ctx() -> dict[str, Any]:
    global _ctx
    if _ctx is None:
        model, data = build_world()
        state_dim   = len(_make_state(data))
        net         = Network(input_size=state_dim, output_size=model.nu)
        _ctx = {"model": model, "data": data, "net": net}
    return cast(dict[str, Any], _ctx)


def _evaluate(weights: np.ndarray) -> float:
    ctx = _get_ctx()
    fill_parameters(ctx["net"], weights)
    x = run_episode(ctx["model"], ctx["data"], ctx["net"])
    return x   # negate: CMA-ES minimises


# ── CMA-ES ────────────────────────────────────────────────────────────────────

def evolve() -> np.ndarray:
    model, data = build_world()
    state_dim   = len(_make_state(data))
    net         = Network(input_size=state_dim, output_size=model.nu)
    num_params  = sum(p.numel() for p in net.parameters())

    min_lambda = 4 + int(3 * np.log(max(num_params, 2)))
    pop        = max(POP_SIZE, min_lambda)
    if pop % 2 != 0:
        pop += 1

    console.rule("[bold cyan]Gecko Forward Walk — CMA-ES[/bold cyan]")
    console.log(f"params={num_params}  pop={pop}  budget={BUDGET} gens  workers={NUM_WORKERS}  dur={DURATION}s")

    np.random.seed(BASE_SEED)
    init_guess = np.random.uniform(-0.5, 0.5, size=num_params)
    param      = ng.p.Array(init=init_guess).set_mutation(sigma=0.3)
    optimizer  = ng.optimizers.ParametrizedCMA(popsize=pop)(
        parametrization=param, budget=BUDGET * pop, num_workers=pop
    )

    best_fitness = float("inf")
    best_weights: Optional[np.ndarray] = None
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        mp_context=__import__("multiprocessing").get_context("spawn"),
        initializer=_init_worker,
        initargs=(BASE_SEED,),
    ) as pool:
        for gen in range(BUDGET):
            candidates = [optimizer.ask() for _ in range(pop)]
            fitnesses  = list(pool.map(_evaluate, [c.value for c in candidates]))

            for cand, fit in zip(candidates, fitnesses):
                optimizer.tell(cand, fit)

            best_idx = int(np.argmin(fitnesses))
            best     = float(fitnesses[best_idx])
            if best < best_fitness:
                best_fitness = best
                best_weights = candidates[best_idx].value.copy()

            console.log(
                f"gen {gen+1:3d}/{BUDGET}  "
                f"best_y={-best:.3f} m  "
                f"best_seen_y={-best_fitness:.3f} m  "
                f"({time.time()-t0:.0f}s)"
            )

    assert best_weights is not None
    np.save(WEIGHTS_PATH, best_weights)
    console.log(f"[green]Saved → {WEIGHTS_PATH}[/green]")
    return best_weights


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    evolve()


if __name__ == "__main__":
    main()
