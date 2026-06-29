"""
Symmetry-pressure investigation — forward locomotion task.

Outer loop:  (mu+lambda | mu,lambda | nsga2-efficiency | nsga2-symmetry) ES
             over TreeGenome bodies using the ariel.ec engine.
Inner loop:  CMA-ES (nevergrad) over neural-network brain weights.
Task:        Forward locomotion along X-axis. Proprioception only (no camera).
Fitness:     -(x_final - x_initial)  [lower = further forward]

Data saved per individual during evolution (run_data.jsonl):
  gen, ind_id, parent_ids, fitness, learning_curve, trajectory,
  food_positions (null), control_cost, yz_symmetry
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np
import torch
from rich.console import Console
from rich.traceback import install

from ariel.ec import EA, EAOperation, EASettings, Individual, Population
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.utils.renderers import VideoRecorder

from shared import (
    Network,
    action_control_cost,
    bilateral_symmetry_score,
    build_world_for_body,
    create_individual,
    ensure_ctx_for_body,
    fill_parameters,
    genome_hash,
    genome_input_dim,
    init_worker,
    make_offspring,
    mutate_morph,
    nsga2_survivor_selection,
    train_body_serial,
)

install()
warnings.filterwarnings("ignore", message="TPA: apparent inconsistency",
                        category=UserWarning, module="cma")

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Symmetry-pressure: forward locomotion")
parser.add_argument("--budget",        type=int,   default=40)
parser.add_argument("--pop",           type=int,   default=20,  help="mu")
parser.add_argument("--lam",           type=int,   default=20,  help="lambda")
parser.add_argument("--brain-budget",  type=int,   default=500, help="CMA inner generations")
parser.add_argument("--brain-pop",     type=int,   default=20)
parser.add_argument("--brain-workers", type=int,   default=max(1, os.cpu_count() or 1))
parser.add_argument("--dur",           type=float, default=15.0, help="Episode duration (s)")
parser.add_argument("--max-modules",   type=int,   default=25)
parser.add_argument("--max-depth",     type=int,   default=25)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--strategy-type",
                    choices=["plus", "comma", "nsga2-efficiency", "nsga2-symmetry"],
                    default="plus",
                    help="Survivor selection strategy")
parser.add_argument("--num-evals",     type=int,   default=1,
                    help="Evaluations per individual per generation (>1 = re-evaluation)")
parser.add_argument("--no-video",      action="store_true")
args = parser.parse_args()

BUDGET        = args.budget
MU            = args.pop
LAM           = args.lam
BRAIN_BUDGET  = args.brain_budget
BRAIN_POP     = args.brain_pop
BRAIN_WORKERS = max(1, args.brain_workers)
DURATION      = args.dur
NUM_MODULES   = args.max_modules
MAX_DEPTH     = args.max_depth
BASE_SEED     = args.seed
STRATEGY      = args.strategy_type
NUM_EVALS     = args.num_evals

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)
CHECKPOINTS = DATA / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True, parents=True)
JSONL_PATH = DATA / "run_data.jsonl"

RNG = np.random.default_rng(BASE_SEED)

# ── Episode runner (locomotion) ───────────────────────────────────────────────


def run_episode_locomotion(
    ctx: dict[str, Any],
    waypoints: Any,  # unused for locomotion; kept for uniform interface
    weights: np.ndarray,
) -> dict[str, Any]:
    model: mujoco.MjModel = ctx["model"]
    data: mujoco.MjData   = ctx["data"]
    network: Network      = ctx["network"]

    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    x0 = float(data.qpos[0])
    trajectory: list[list[float]] = []
    ctrl_history: list[np.ndarray] = []
    control_step_freq = 50
    step = 0
    current_action = np.zeros(model.nu)

    while data.time < DURATION:
        if step % control_step_freq == 0:
            state = get_robot_state(data).astype(np.float32)
            current_action = network.forward(model, data, state)

        data.ctrl[:] = current_action
        ctrl_history.append(current_action.copy())
        mujoco.mj_step(model, data)

        if step % control_step_freq == 0:
            trajectory.append([float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])])

        step += 1

    x_final = float(data.qpos[0])
    fitness  = -(x_final - x0)  # lower = better, maximise forward X

    return {
        "fitness":      fitness,
        "trajectory":   trajectory,
        "control_cost": action_control_cost(ctrl_history),
    }


# ── Wrapper for ProcessPoolExecutor (locomotion) ──────────────────────────────


def _train_locomotion_worker(task: dict[str, Any]) -> dict[str, Any]:
    task["episode_fn"]   = run_episode_locomotion
    task["use_vision"]   = False
    task["waypoints"]    = None
    task["reach_radius"] = 0.0
    task["arena_radius"] = 1.0
    return train_body_serial(task)


# ── BodyBrainEvolution ────────────────────────────────────────────────────────


class BodyBrainEvolution:
    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,
            num_steps=BUDGET,
            target_population_size=MU,
            output_folder=DATA,
            db_file_name=f"database_{int(time.time())}.db",
            db_handling="delete",
        )
        self.executor: Optional[ProcessPoolExecutor] = None
        self.outer_gen: int = 0
        self.best_seen_fitness:  float = float("inf")
        self.best_seen_genotype: Optional[dict] = None
        self.best_seen_weights:  Optional[np.ndarray] = None

    # -- operations -----------------------------------------------------------

    def parent_selection(self, population: Population) -> Population:
        population = population.sort(sort="min", attribute="fitness_")
        for i, ind in enumerate(population):
            ind.tags = {**ind.tags, "ps": i < MU}
        return population

    def reproduction(self, population: Population) -> Population:
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        if not parents:
            parents = list(population)
        offspring = make_offspring(parents, LAM, RNG, NUM_MODULES, MAX_DEPTH)

        # For (mu,lambda): mark parents for re-evaluation each generation too
        if STRATEGY == "comma" or NUM_EVALS > 1:
            for ind in parents:
                ind.requires_eval = True

        population.extend(offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        console.rule(f"[bold magenta]Outer gen {self.outer_gen}")

        to_eval = [
            ind for ind in population
            if ind.alive and ind.tags.get("valid", True) and ind.requires_eval
        ]

        tasks = [
            {
                "body_hash":    genome_hash(ind.genotype["morph"]),
                "genome_dict":  ind.genotype["morph"],
                "rng_seed":     BASE_SEED + 1000 * self.outer_gen + idx,
                "brain_budget": BRAIN_BUDGET,
                "brain_pop":    BRAIN_POP,
                "duration":     DURATION,
                "num_evals":    NUM_EVALS,
            }
            for idx, ind in enumerate(to_eval)
        ]

        assert self.executor is not None
        t0 = time.time()
        results = list(self.executor.map(_train_locomotion_worker, tasks))
        elapsed = time.time() - t0

        for idx, (ind, res) in enumerate(zip(to_eval, results)):
            best_fit   = float(res["fitness"])
            best_w_lst = res["weights"]
            best_w     = np.array(best_w_lst) if best_w_lst is not None else None

            ind.fitness = best_fit if np.isfinite(best_fit) else float("inf")
            ind.tags = {
                **ind.tags,
                "best_brain":     best_w_lst or [],
                "learning_curve": res.get("learning_curve", []),
                "trajectory":     res.get("trajectory", []),
                "control_cost":   res.get("control_cost", 0.0),
                "yz_symmetry":    bilateral_symmetry_score(ind.genotype["morph"]),
            }
            ind.requires_eval = False

            self._write_jsonl(ind, idx)

            if best_fit < self.best_seen_fitness and best_w is not None:
                self.best_seen_fitness  = best_fit
                self.best_seen_genotype = dict(ind.genotype["morph"])
                self.best_seen_weights  = best_w.copy()
                self._save_checkpoint(tag=f"gen{self.outer_gen:03d}_body{idx:02d}")

        finite = [r["fitness"] for r in results if np.isfinite(r["fitness"])]
        stats = (f"min={np.min(finite):.3f}  avg={np.mean(finite):.3f}"
                 if finite else "all-infinite")
        console.log(f"  {len(to_eval)} bodies in {elapsed:.1f}s  {stats}")

        self.outer_gen += 1
        return population

    def survivor_selection(self, population: Population) -> Population:
        alive = [ind for ind in population if ind.alive]

        if STRATEGY == "plus":
            alive_sorted = sorted(alive, key=lambda i: i.fitness_ or float("inf"))
            survivors = set(id(i) for i in alive_sorted[:MU])
            for ind in population:
                if ind.alive and id(ind) not in survivors:
                    ind.alive = False

        elif STRATEGY == "comma":
            # Keep only offspring (non-parent), then best mu of those
            offspring_alive = [i for i in alive if not i.tags.get("ps", False)]
            offspring_sorted = sorted(offspring_alive, key=lambda i: i.fitness_ or float("inf"))
            survivors = set(id(i) for i in offspring_sorted[:MU])
            for ind in population:
                if ind.alive and id(ind) not in survivors:
                    ind.alive = False

        elif STRATEGY in ("nsga2-efficiency", "nsga2-symmetry"):
            valid = [i for i in alive if np.isfinite(i.fitness_ or float("inf"))]
            invalid = [i for i in alive if not np.isfinite(i.fitness_ or float("inf"))]
            # Kill invalid individuals first
            for ind in invalid:
                ind.alive = False

            if STRATEGY == "nsga2-efficiency":
                objectives = [
                    (i.fitness_ or float("inf"), i.tags.get("control_cost", float("inf")))
                    for i in valid
                ]
            else:
                objectives = [
                    (i.fitness_ or float("inf"), -i.tags.get("yz_symmetry", 0.0))
                    for i in valid
                ]

            chosen = nsga2_survivor_selection(valid, MU, objectives)
            survivors = set(id(i) for i in chosen)
            for ind in valid:
                if id(ind) not in survivors:
                    ind.alive = False

        survivors_alive = [i for i in population if i.alive]
        finite = [i.fitness_ for i in survivors_alive if i.fitness_ is not None and np.isfinite(i.fitness_)]
        if finite:
            console.log(f"[green]Survivors:[/green] avg={np.mean(finite):.3f}  min={np.min(finite):.3f}")
        return population

    # -- persistence ----------------------------------------------------------

    def _write_jsonl(self, ind: Individual, idx: int) -> None:
        record = {
            "gen":            self.outer_gen,
            "ind_id":         ind.id,
            "parent_ids":     ind.genotype.get("parent_ids", []),
            "fitness":        ind.fitness_ if ind.fitness_ is not None else None,
            "learning_curve": ind.tags.get("learning_curve", []),
            "trajectory":     ind.tags.get("trajectory", []),
            "food_positions": None,
            "control_cost":   ind.tags.get("control_cost", 0.0),
            "yz_symmetry":    ind.tags.get("yz_symmetry", 0.0),
        }
        with JSONL_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _save_checkpoint(self, tag: str) -> None:
        if self.best_seen_weights is None or self.best_seen_genotype is None:
            return
        sub = CHECKPOINTS / tag
        sub.mkdir(exist_ok=True, parents=True)
        np.save(sub / "best_weights.npy", self.best_seen_weights)
        with (sub / "best_genome.json").open("w") as fh:
            json.dump(self.best_seen_genotype, fh, indent=2)
        console.log(f"  [cyan]checkpoint → {sub}  fitness={self.best_seen_fitness:.3f}[/cyan]")

    # -- main -----------------------------------------------------------------

    def evolve(self) -> Optional[Individual]:
        console.log("[yellow]Initialising population...[/yellow]")
        population = Population([create_individual(NUM_MODULES, MAX_DEPTH) for _ in range(MU)])

        with ProcessPoolExecutor(
            max_workers=BRAIN_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=init_worker,
            initargs=(BASE_SEED,),
        ) as executor:
            self.executor = executor
            population = self.evaluate(population)

            ops = [
                EAOperation(self.parent_selection),
                EAOperation(self.reproduction),
                EAOperation(self.evaluate),
                EAOperation(self.survivor_selection),
            ]
            ea = EA(
                population,
                operations=ops,
                num_steps=BUDGET,
                db_file_path=self.config.db_file_path,
                db_handling=self.config.db_handling,
                quiet=self.config.quiet,
            )
            ea.run()
            self.executor = None
            return ea.get_solution("best", only_alive=False)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    console.rule("[bold magenta]Symmetry-Pressure — Forward Locomotion[/bold magenta]")
    console.log(
        f"strategy={STRATEGY}  (mu+lam)=({MU}+{LAM})  budget={BUDGET}  "
        f"brain_budget={BRAIN_BUDGET}  brain_pop={BRAIN_POP}  "
        f"num_evals={NUM_EVALS}  seed={BASE_SEED}"
    )

    start = time.time()
    evo = BodyBrainEvolution()
    best = evo.evolve()
    elapsed = time.time() - start

    console.rule("[bold green]Done[/bold green]")
    if best is not None and best.fitness_ is not None:
        console.log(f"Best fitness: {best.fitness:.3f}")
    console.log(f"Best seen:    {evo.best_seen_fitness:.3f}")
    console.log(f"Elapsed:      {elapsed / 60:.1f} min")
    console.log(f"Data →        {DATA}")


if __name__ == "__main__":
    main()
    gc.disable()
