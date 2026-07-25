"""NSGA-II multi-objective tuning of gecko brain/CMA-ES hyperparameters.

Objectives (per trial): minimise wall-clock training time, maximise best
locomotion fitness achieved. Search space: CMA-ES sigma/pop_size/inner_gens
and DistributedMLP hidden-layer width. Always uses inner-parallelization
(one trial's CMA-ES generations evaluated in parallel across --workers);
trials themselves run sequentially so each gets the full worker pool.

Study state is persisted to a sqlite database (resumable by passing the same
--study-name / --storage back in). Each trial's learned weights, learning
curve, and hyperparameters are additionally saved to an .npz file immediately
after that trial completes, since Optuna's storage does not keep arbitrary
arrays.

--study-name defaults to a timestamp (gecko_nsga2_<YYYYmmdd_HHMMSS>), and
--storage / --weights-dir default to paths derived from --study-name, so a
plain rerun always starts a fresh study instead of resuming/overwriting a
previous one. Pass --study-name explicitly (optionally with --storage) to
resume a specific prior study.

Usage:
    uv run examples/d_social_learning/ariel/tune_gecko_brain.py \
        --n-trials 100 --workers 16 [--study-name gecko_nsga2_20260725_120000] \
        [--storage sqlite:///__data__/tuning/gecko_nsga2_20260725_120000.db] \
        [--weights-dir __data__/tuning/gecko_nsga2_20260725_120000_weights] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np
import optuna
from rich.console import Console

import gecko_evaluator as ge

console = Console()


def make_objective(workers: int, duration: float, ctrl_every: int, weights_dir: Path):
    def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
        sigma = trial.suggest_float("sigma", 0.1, 1.0, log=True)
        pop_size = trial.suggest_int("pop_size", 4, 32)
        inner_gens = trial.suggest_int("inner_gens", 5, 50)
        hidden = trial.suggest_int("hidden", 8, 64, step=8)

        from ariel.simulation.controllers.distributed_mlp import DistributedMLP
        n_params = DistributedMLP(n_neighbors=ge.N_NEIGHBORS, hidden=hidden).n_params

        result = ge.run_cma_training(
            n_params=n_params,
            run_episode_fn=None,
            sigma=sigma,
            pop_size=pop_size,
            inner_gens=inner_gens,
            parallel_inner=True,
            inner_workers=workers,
            hidden=hidden,
            duration=duration,
            ctrl_every=ctrl_every,
        )

        weights_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            weights_dir / f"trial_{trial.number}.npz",
            theta=result.best_theta,
            fitness=result.best_fitness,
            time_s=result.wall_time_s,
            sigma=sigma,
            pop_size=pop_size,
            inner_gens=inner_gens,
            hidden=hidden,
        )

        console.log(
            f"trial {trial.number}: time={result.wall_time_s:.1f}s "
            f"fitness={result.best_fitness:.4f} "
            f"(sigma={sigma:.3f} pop_size={pop_size} inner_gens={inner_gens} hidden={hidden})"
        )

        return result.wall_time_s, result.best_fitness

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--duration", type=float, default=ge.DURATION)
    parser.add_argument("--ctrl-every", type=int, default=ge.CTRL_EVERY)
    default_study_name = f"gecko_nsga2_{datetime.now():%Y%m%d_%H%M%S}"
    parser.add_argument("--study-name", type=str, default=default_study_name)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--weights-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Default storage/weights-dir are derived from --study-name (whether it
    # was passed explicitly or defaulted to the timestamp above), so a run
    # never collides with or silently resumes/overwrites a previous one.
    storage = args.storage or f"sqlite:///__data__/tuning/{args.study_name}.db"
    weights_dir = args.weights_dir or Path(f"__data__/tuning/{args.study_name}_weights")

    storage_path = storage.removeprefix("sqlite:///")
    if storage_path:
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold cyan]Tune gecko brain | trials={args.n_trials} workers={args.workers} study={args.study_name}")

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(args.workers, args.duration, args.ctrl_every, weights_dir)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)

    console.log(f"Completed {len(study.trials)} total trials.")
    console.log(f"Pareto front: {len(study.best_trials)} trials.")
    for t in study.best_trials:
        console.log(f"  trial {t.number}: time={t.values[0]:.1f}s fitness={t.values[1]:.4f} params={t.params}")

    console.rule("[bold green]Done")


if __name__ == "__main__":
    main()
