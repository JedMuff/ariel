"""Optimise one brain on one fixed zoo body with CMA-ES (kgd's zoo protocol).

Replicates apets-ariel's src/aapets/zoo/evolve.py so the numbers are comparable:
  - bodies: kgd's 23 canonical bodies with kgd's hinge actuator (vendored in
    canonical_bodies.py / zoo_modules.py)
  - episode: 15 s from rest, no settling phase; the controller is queried at
    20 Hz and its output written straight to data.ctrl (no smoothing, no
    penalties); physics at MuJoCo's default 0.002 s
  - fitness: signed x-displacement of the core divided by episode time
    (XSpeedMonitor)
  - CMA-ES: x0 = 0.5, sigma0 = 0.5, default popsize, tolfun=0,
    tolflatfitness=10, budget in evaluations (kgd's run_all.sh uses 10000)
The same CMA-ES settings are used for every brain, except that the ANN starts
at x0 = 0 (DEFAULT_INITIAL_MEAN): at 0.5 its ~1.5k weights saturate every tanh
output, all candidates hold the same pose, fitness is flat and tolflatfitness
stops the run after 10 generations.

Known deviation: bodies are spawned in the local SimpleFlatWorld, not kgd's
make_world, so floor/contact parameters may differ slightly.

Usage:
  python zoo_benchmark.py --body spider --brain matsuoka --seed 42 --budget 10000 \\
      --workers 16 --out-dir runs/spider_matsuoka_42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cma
import mujoco
import numpy as np
import torch

import canonical_bodies
from brains import BRAIN_KINDS, make_brain
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld

CORE_BODY = "robot1_core"

# CMA-ES x0 per brain (see module docstring for why the ANN differs).
DEFAULT_INITIAL_MEAN = {"ann": 0.0, "sine": 0.5, "revolve_cpg": 0.5, "matsuoka": 0.5}


# ── World / episode ───────────────────────────────────────────────────────────


def build_world(body: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = canonical_bodies.get(body).spec
    world = SimpleFlatWorld()
    try:
        world.spawn(spec, correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(canonical_bodies.get(body).spec, correct_collision_with_floor=False)
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def make_brain_for(kind: str, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
    mujoco.mj_resetData(model, data)
    return make_brain(kind, len(get_state_from_data(data)), model.nu, model.opt.timestep)


def run_episode(
    model: mujoco.MjModel, data: mujoco.MjData, brain: Any, duration: float, control_freq: float,
) -> float:
    """Roll out one episode and return the core's x-speed (m/s)."""
    steps_per_ctrl = round(1.0 / (control_freq * model.opt.timestep))
    n_ctrl = round(duration * control_freq)
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CORE_BODY)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    brain.reset()
    x0 = float(data.xpos[core_id, 0])

    for _ in range(n_ctrl):
        data.ctrl[:] = brain.act(float(data.time), get_state_from_data(data))
        mujoco.mj_step(model, data, nstep=steps_per_ctrl)

    if data.time <= 0:
        return 0.0
    return (float(data.xpos[core_id, 0]) - x0) / float(data.time)


# ── Parallel evaluation ───────────────────────────────────────────────────────

_worker_ctx: Optional[dict[str, Any]] = None


def _worker_init(body: str, brain_kind: str, duration: float, control_freq: float) -> None:
    global _worker_ctx  # noqa: PLW0603
    torch.set_num_threads(1)
    model, data = build_world(body)
    _worker_ctx = {
        "model": model, "data": data, "brain": make_brain_for(brain_kind, model, data),
        "duration": duration, "control_freq": control_freq,
    }


def _worker_eval(params: list[float]) -> float:
    assert _worker_ctx is not None
    ctx = _worker_ctx
    ctx["brain"].set_params(np.asarray(params))
    return run_episode(ctx["model"], ctx["data"], ctx["brain"], ctx["duration"], ctx["control_freq"])


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--body", required=True, choices=sorted(canonical_bodies.get_all()))
    parser.add_argument("--brain", required=True, choices=BRAIN_KINDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=10000, help="CMA-ES evaluations")
    parser.add_argument("--duration", type=float, default=15.0, help="Episode length (s)")
    parser.add_argument("--control-freq", type=float, default=20.0, help="Controller rate (Hz)")
    parser.add_argument("--initial-mean", type=float, default=None,
                        help="CMA-ES x0 (default: per brain, see DEFAULT_INITIAL_MEAN)")
    parser.add_argument("--initial-std", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.initial_mean is None:
        args.initial_mean = DEFAULT_INITIAL_MEAN[args.brain]

    t_start = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, data = build_world(args.body)
    brain = make_brain_for(args.brain, model, data)
    n_params = brain.num_params

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(n_hinges=model.nu, n_params=n_params, physics_dt=model.opt.timestep)
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    es = cma.CMAEvolutionStrategy(
        n_params * [args.initial_mean], args.initial_std,
        {"seed": args.seed, "tolfun": 0, "tolflatfitness": 10,
         "maxfevals": args.budget, "verbose": -9},
    )

    best_speed, best_params = -np.inf, np.full(n_params, args.initial_mean)
    history_path = args.out_dir / "history.csv"
    with (
        history_path.open("w", newline="") as hf,
        ProcessPoolExecutor(
            max_workers=min(args.workers, es.popsize),
            initializer=_worker_init,
            initargs=(args.body, args.brain, args.duration, args.control_freq),
        ) as pool,
    ):
        writer = csv.writer(hf)
        writer.writerow(["gen", "evals", "gen_best", "gen_mean", "best", "sigma", "wall_s"])
        gen = 0
        while not es.stop():
            solutions = es.ask()
            speeds = list(pool.map(_worker_eval, [s.tolist() for s in solutions]))
            es.tell(solutions, [-s for s in speeds])

            i = int(np.argmax(speeds))
            if speeds[i] > best_speed:
                best_speed, best_params = speeds[i], np.asarray(solutions[i])

            gen += 1
            wall = time.perf_counter() - t_start
            writer.writerow([gen, es.countevals, max(speeds), float(np.mean(speeds)),
                             best_speed, es.sigma, round(wall, 1)])
            hf.flush()
            if gen % 10 == 0 or es.stop():
                print(f"gen {gen:4d}  evals {es.countevals:6d}  best {best_speed:+.4f} m/s  "
                      f"gen mean {np.mean(speeds):+.4f}  sigma {es.sigma:.3f}  {wall:.0f}s", flush=True)

    np.save(args.out_dir / "champion.npy", best_params)

    _worker_init(args.body, args.brain, args.duration, args.control_freq)
    rerun_speed = _worker_eval(best_params.tolist())

    summary = {
        "body": args.body, "brain": args.brain, "seed": args.seed,
        "n_hinges": model.nu, "n_params": n_params,
        "budget": args.budget, "evals": es.countevals, "generations": gen,
        "stop": {k: str(v) for k, v in es.stop().items()},
        "best_xspeed": best_speed, "rerun_xspeed": rerun_speed,
        "wall_s": time.perf_counter() - t_start,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not np.isclose(rerun_speed, best_speed):
        print(f"WARNING: champion re-evaluation differs ({rerun_speed} vs {best_speed})")


if __name__ == "__main__":
    main()
