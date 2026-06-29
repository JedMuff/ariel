"""
Retrain the amputee gecko (front-right arm removed) starting from the pruned
original weights, then record side-view and top-view GIF + MP4 of the best result.

The pruned weights are obtained by surgically slicing the fr-arm input columns
(state dims 9 & 10) and output rows (actuator dims 4 & 5) from the original
gecko_forward weights, then used as the CMA-ES initial guess.

Weights are saved to __data__/gecko_amputee_retrain/best_weights.npy.

Usage
-----
    # Full retrain then video:
    uv run python examples/re_book/gecko_amputee_retrain.py \\
        --weights __data__/gecko_forward/best_weights.npy

    # Lighter run for testing:
    uv run python examples/re_book/gecko_amputee_retrain.py \\
        --weights __data__/gecko_forward/best_weights.npy \\
        --budget 50 --pop 32 --dur 8

    # Skip video rendering (just train):
    uv run python examples/re_book/gecko_amputee_retrain.py \\
        --weights __data__/gecko_forward/best_weights.npy --no-video
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
from PIL import Image
from torch import nn
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

console = Console()

CONTROL_HZ = 20
SPAWN_POS  = [0.0, 0.0, 0.1]

SIDE_W, SIDE_H = 640, 360
TOP_W,  TOP_H  = 640, 360

_FR_STATE_INDICES = [9, 10]
_FR_CTRL_INDICES  = [4, 5]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden: int = 32) -> None:
        super().__init__()
        self.fc1     = nn.Linear(input_size, hidden)
        self.fc2     = nn.Linear(hidden, hidden)
        self.fc_out  = nn.Linear(hidden, output_size)
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


# ---------------------------------------------------------------------------
# Weight surgery
# ---------------------------------------------------------------------------

def prune_weights(
    orig_weights: np.ndarray,
    input_size_orig: int  = 13,
    output_size_orig: int = 8,
    hidden: int           = 32,
) -> np.ndarray:
    """Slice fr-arm input columns and output rows out of the flat weight vector."""
    orig_net = Network(input_size_orig, output_size_orig, hidden)
    fill_parameters(orig_net, orig_weights)

    keep_in  = [i for i in range(input_size_orig)  if i not in _FR_STATE_INDICES]
    keep_out = [i for i in range(output_size_orig) if i not in _FR_CTRL_INDICES]

    fc1_w = orig_net.fc1.weight.data[:, keep_in]
    fc1_b = orig_net.fc1.bias.data
    fc2_w = orig_net.fc2.weight.data
    fc2_b = orig_net.fc2.bias.data
    fco_w = orig_net.fc_out.weight.data[keep_out, :]
    fco_b = orig_net.fc_out.bias.data[keep_out]

    return np.concatenate([
        fc1_w.view(-1).numpy(), fc1_b.numpy(),
        fc2_w.view(-1).numpy(), fc2_b.numpy(),
        fco_w.view(-1).numpy(), fco_b.numpy(),
    ])


# ---------------------------------------------------------------------------
# Amputee gecko body builder (shared between train worker and video)
# ---------------------------------------------------------------------------

def build_amputee_world():
    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
    from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
    from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
    from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
    from ariel.simulation.environments import SimpleFlatWorld

    core = CoreModule(index=0)
    neck = HingeModule(index=1)
    abdomen = BrickModule(index=2)
    spine = HingeModule(index=3)
    butt = BrickModule(index=4)

    fl_leg = HingeModule(index=5)
    fl_leg.rotate(90)
    fl_leg2 = HingeModule(index=15)
    fl_leg2.rotate(90)
    fl_flipper = BrickModule(index=6)

    bl_leg = HingeModule(index=9)
    bl_leg.rotate(45)
    bl_flipper = BrickModule(index=10)
    br_leg = HingeModule(index=11)
    br_leg.rotate(-45)
    br_flipper = BrickModule(index=12)

    core.sites[ModuleFaces.FRONT].attach_body(body=neck.body,    prefix="neck")
    neck.sites[ModuleFaces.FRONT].attach_body(body=abdomen.body, prefix="abdomen")
    abdomen.sites[ModuleFaces.FRONT].attach_body(body=spine.body, prefix="spine")
    spine.sites[ModuleFaces.FRONT].attach_body(body=butt.body,   prefix="butt")
    core.sites[ModuleFaces.LEFT].attach_body(body=fl_leg.body,   prefix="fl_leg")
    fl_leg.sites[ModuleFaces.FRONT].attach_body(body=fl_leg2.body, prefix="fl_flipper")
    fl_leg2.sites[ModuleFaces.FRONT].attach_body(body=fl_flipper.body, prefix="fl_flipper2")
    butt.sites[ModuleFaces.LEFT].attach_body(body=bl_leg.body,   prefix="bl_leg")
    bl_leg.sites[ModuleFaces.FRONT].attach_body(body=bl_flipper.body, prefix="bl_flipper")
    butt.sites[ModuleFaces.RIGHT].attach_body(body=br_leg.body,  prefix="br_leg")
    br_leg.sites[ModuleFaces.FRONT].attach_body(body=br_flipper.body, prefix="br_flipper")

    world = SimpleFlatWorld(floor_size=(100, 100, 1))
    world.spawn(core.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    return model, data


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def _make_state(data: mujoco.MjData) -> np.ndarray:
    from ariel.simulation.controllers.utils.data_get import get_state_from_data
    robot = get_state_from_data(data)
    phase = [np.sin(data.time * 2.0 * np.pi), np.cos(data.time * 2.0 * np.pi)]
    return np.concatenate([robot, phase]).astype(np.float32)


def run_episode(model, data, net, duration: float) -> float:
    mujoco.mj_resetData(model, data)
    dt             = model.opt.timestep
    control_period = int(round(1.0 / (CONTROL_HZ * dt)))
    current_ctrl   = np.zeros(model.nu)
    step           = 0
    while data.time < duration:
        if step % control_period == 0:
            current_ctrl = net.forward(_make_state(data))
        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)
        step += 1
    return float(data.qpos[1])


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

_ctx: Optional[dict[str, Any]] = None


def _init_worker(seed: int) -> None:
    torch.set_num_threads(1)
    s = (seed + os.getpid()) % (2**32 - 1)
    np.random.seed(s)
    torch.manual_seed(s)


def _get_ctx(hidden: int, duration: float) -> dict[str, Any]:
    global _ctx
    if _ctx is None:
        model, data = build_amputee_world()
        state_dim   = len(_make_state(data))
        net         = Network(input_size=state_dim, output_size=model.nu, hidden=hidden)
        _ctx = {"model": model, "data": data, "net": net, "duration": duration}
    return cast(dict[str, Any], _ctx)


_WORKER_HIDDEN:   int   = 32
_WORKER_DURATION: float = 8.0


def _evaluate(weights: np.ndarray) -> float:
    ctx = _get_ctx(_WORKER_HIDDEN, _WORKER_DURATION)
    fill_parameters(ctx["net"], weights)
    return run_episode(ctx["model"], ctx["data"], ctx["net"], ctx["duration"])


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

def evolve(args) -> np.ndarray:
    global _WORKER_HIDDEN, _WORKER_DURATION
    _WORKER_HIDDEN   = args.hidden
    _WORKER_DURATION = args.dur

    model, data = build_amputee_world()
    state_dim   = len(_make_state(data))
    net         = Network(input_size=state_dim, output_size=model.nu, hidden=args.hidden)
    num_params  = sum(p.numel() for p in net.parameters())

    orig_weights = np.load(args.weights)
    init_weights = prune_weights(orig_weights,
                                  input_size_orig=args.input_size_orig,
                                  output_size_orig=args.output_size_orig,
                                  hidden=args.hidden)
    assert init_weights.shape[0] == num_params, \
        f"Pruned weight count {init_weights.shape[0]} != expected {num_params}"

    pop = max(args.pop, 4 + int(3 * np.log(max(num_params, 2))))
    if pop % 2 != 0:
        pop += 1

    console.rule("[bold cyan]Gecko Amputee Retrain — CMA-ES[/bold cyan]")
    console.log(f"params={num_params}  pop={pop}  budget={args.budget} gens  "
                f"workers={args.workers}  dur={args.dur}s")
    console.log(f"Warm-starting from pruned gecko_forward weights: {args.weights}")

    param     = ng.p.Array(init=init_weights).set_mutation(sigma=args.sigma)
    optimizer = ng.optimizers.ParametrizedCMA(popsize=pop)(
        parametrization=param, budget=args.budget * pop, num_workers=pop
    )

    best_fitness = float("inf")
    best_weights: Optional[np.ndarray] = None
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=__import__("multiprocessing").get_context("spawn"),
        initializer=_init_worker,
        initargs=(args.seed,),
    ) as pool:
        for gen in range(args.budget):
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
                f"gen {gen+1:3d}/{args.budget}  "
                f"best_y={-best:.3f} m  "
                f"best_seen_y={-best_fitness:.3f} m  "
                f"({time.time()-t0:.0f}s)"
            )

    assert best_weights is not None
    out_dir = Path(args.out_dir) / "gecko_amputee_retrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "best_weights.npy"
    np.save(weights_path, best_weights)
    console.log(f"[green]Saved → {weights_path}[/green]")
    return best_weights


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _side_camera(x, y, z, distance=1.8):
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat    = np.array([x, y, z], dtype=np.float64)
    cam.distance  = distance
    cam.azimuth   = 0.0
    cam.elevation = -10.0
    return cam


def _top_camera(x, y, z, height_m=1.2):
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat    = np.array([x, y, z], dtype=np.float64)
    cam.distance  = height_m
    cam.azimuth   = 0.0
    cam.elevation = -90.0
    return cam


# ---------------------------------------------------------------------------
# GIF / MP4 writers
# ---------------------------------------------------------------------------

def _save_gif(frames, path, fps):
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                loop=0, duration=int(1000/fps), optimize=False)
    print(f"Saved GIF → {path}  ({len(frames)} frames @ {fps} fps)")


def _save_mp4(frames, path, fps):
    import cv2
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        for codec in ("avc1", "XVID", "MJPG"):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
            if writer.isOpened():
                break
    if not writer.isOpened():
        print(f"Warning: could not open MP4 writer for {path} — skipping.")
        return
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved MP4 → {path}  ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------

def record_video(best_weights: np.ndarray, args) -> None:
    from ariel.simulation.controllers.utils.data_get import get_state_from_data

    model, data = build_amputee_world()
    state_dim   = len(_make_state(data))
    net = Network(input_size=state_dim, output_size=model.nu, hidden=args.hidden)
    fill_parameters(net, best_weights)

    def _run(duration):
        mujoco.mj_resetData(model, data)
        dt             = model.opt.timestep
        control_period = int(round(1.0 / (CONTROL_HZ * dt)))
        total_steps    = int(duration / dt)
        steps_per_frame = max(1, int(round(1.0 / (args.fps * dt))))
        current_ctrl   = np.zeros(model.nu)

        side_frames: list[np.ndarray] = []
        top_frames:  list[np.ndarray] = []

        side_renderer = mujoco.Renderer(model, height=SIDE_H, width=SIDE_W)
        top_renderer  = mujoco.Renderer(model, height=TOP_H,  width=TOP_W)
        scene_opt     = mujoco.MjvOption()

        for step in range(total_steps):
            if step % control_period == 0:
                current_ctrl = net.forward(_make_state(data))
            data.ctrl[:] = current_ctrl
            mujoco.mj_step(model, data)

            if step % steps_per_frame == 0:
                cx = args.side_x if args.side_x is not None else float(data.qpos[0])
                cy = args.side_y if args.side_y is not None else float(data.qpos[1])
                tx = args.top_x  if args.top_x  is not None else float(data.qpos[0])
                ty = args.top_y  if args.top_y  is not None else float(data.qpos[1])
                side_cam = _side_camera(cx, cy, args.side_z, distance=args.side_distance)
                top_cam  = _top_camera(tx, ty, args.top_z,  height_m=args.top_height)
                side_renderer.update_scene(data, camera=side_cam, scene_option=scene_opt)
                top_renderer.update_scene(data,  camera=top_cam,  scene_option=scene_opt)
                side_frames.append(side_renderer.render().copy())
                top_frames.append(top_renderer.render().copy())

        side_renderer.close()
        top_renderer.close()
        return side_frames, top_frames

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    side_dur = args.side_duration or args.video_duration
    top_dur  = args.top_duration  or args.video_duration

    print(f"Recording side view ({side_dur}s) @ {args.fps} fps …")
    side_frames, _ = _run(side_dur)
    print(f"Recording top view ({top_dur}s) @ {args.fps} fps …")
    _, top_frames  = _run(top_dur)

    if not args.no_gif:
        _save_gif(side_frames, out / "gecko_amputee_retrain_side.gif", args.fps)
        _save_gif(top_frames,  out / "gecko_amputee_retrain_top.gif",  args.fps)

    if not args.no_mp4:
        _save_mp4(side_frames, out / "gecko_amputee_retrain_side.mp4", args.fps)
        _save_mp4(top_frames,  out / "gecko_amputee_retrain_top.mp4",  args.fps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights",          required=True,   help="Path to original gecko_forward best_weights.npy")
    parser.add_argument("--input-size-orig",  type=int, default=13)
    parser.add_argument("--output-size-orig", type=int, default=8)
    parser.add_argument("--budget",           type=int,   default=200)
    parser.add_argument("--pop",              type=int,   default=32)
    parser.add_argument("--workers",          type=int,   default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--dur",              type=float, default=16.0,  help="Episode duration (s) for training")
    parser.add_argument("--hidden",           type=int,   default=32)
    parser.add_argument("--sigma",            type=float, default=0.1,  help="CMA-ES initial step size (smaller = stay near warm start)")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--out-dir",          default="__data__")
    parser.add_argument("--no-train",         action="store_true", help="Skip training; load weights from out-dir/gecko_amputee_retrain/best_weights.npy")
    parser.add_argument("--no-video",         action="store_true", help="Skip video rendering after training")
    parser.add_argument("--no-gif",           action="store_true")
    parser.add_argument("--no-mp4",           action="store_true")
    parser.add_argument("--video-duration",   type=float, default=16.0)
    parser.add_argument("--side-duration",    type=float, default=None)
    parser.add_argument("--top-duration",     type=float, default=18.0)
    parser.add_argument("--fps",              type=int,   default=30)
    parser.add_argument("--side-distance",    type=float, default=1.8)
    parser.add_argument("--side-x",           type=float, default=0.0)
    parser.add_argument("--side-y",           type=float, default=-1.6)
    parser.add_argument("--side-z",           type=float, default=0.12)
    parser.add_argument("--top-height",       type=float, default=1.2)
    parser.add_argument("--top-x",            type=float, default=0.0)
    parser.add_argument("--top-y",            type=float, default=-1.6)
    parser.add_argument("--top-z",            type=float, default=1.0)
    args = parser.parse_args()

    if args.no_train:
        weights_path = Path(args.out_dir) / "gecko_amputee_retrain" / "best_weights.npy"
        console.log(f"Skipping training — loading weights from {weights_path}")
        best_weights = np.load(weights_path)
    else:
        best_weights = evolve(args)

    if not args.no_video:
        record_video(best_weights, args)


if __name__ == "__main__":
    main()
