"""
Interactive viewer for the gecko forward-walk brain.

Must be run under mjpython on macOS:

    mjpython examples/re_book/gecko_forward_viewer.py
    mjpython examples/re_book/gecko_forward_viewer.py --dur 20
    mjpython examples/re_book/gecko_forward_viewer.py --weights path/to/weights.npy
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from torch import nn

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld

CONTROL_HZ   = 20
SPAWN_POS    = [0.0, 0.0, 0.1]
DEFAULT_WEIGHTS = Path("__data__/gecko_forward/best_weights.npy")

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
parser.add_argument("--dur",     type=float, default=15.0, help="Seconds to run before stopping")
parser.add_argument("--hidden",  type=int,   default=32)
args = parser.parse_args()


# ── Network (must match gecko_forward.py) ─────────────────────────────────────

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1     = nn.Linear(input_size, args.hidden)
        self.fc2     = nn.Linear(args.hidden, args.hidden)
        self.fc_out  = nn.Linear(args.hidden, output_size)
        self.act     = nn.ELU()
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


def _make_state(data: mujoco.MjData) -> np.ndarray:
    robot = get_state_from_data(data)
    phase = [np.sin(data.time * 2.0 * np.pi), np.cos(data.time * 2.0 * np.pi)]
    return np.concatenate([robot, phase]).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

weights_path = Path(args.weights)
if not weights_path.exists():
    raise SystemExit(f"Weights not found: {weights_path}\nTrain first with: uv run python examples/re_book/gecko_forward.py")

world = SimpleFlatWorld()
body  = gecko()
world.spawn(body.spec, position=SPAWN_POS)
model = world.spec.compile()
data  = mujoco.MjData(model)

state_dim = len(_make_state(data))
net = Network(input_size=state_dim, output_size=model.nu)
fill_parameters(net, np.load(weights_path))
print(f"Loaded weights from {weights_path}")

mujoco.mj_resetData(model, data)
dt             = model.opt.timestep
control_period = int(round(1.0 / (CONTROL_HZ * dt)))
current_ctrl   = np.zeros(model.nu)
step           = 0

core_id = next(
    (i for i in range(model.nbody)
     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("_core")),
    1,
)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = core_id
    viewer.cam.distance    = 1.5
    viewer.cam.azimuth     = 90.0
    viewer.cam.elevation   = -15.0

    while viewer.is_running() and data.time < args.dur:
        t_start = time.time()

        if step % control_period == 0:
            current_ctrl = net.forward(_make_state(data))
        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)
        step += 1
        viewer.sync()

        elapsed = time.time() - t_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

print(f"Final y-position: {data.qpos[1]:.3f} m")
