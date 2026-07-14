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


import argparse
import gc
import json
import os
import random
import time
import warnings
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
    make_offspring,
    mutate_morph,
    nsga2_survivor_selection,
    register_episode_fn,
    train_body_parallel,
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
parser.add_argument("--dur",           type=float, default=30.0, help="Episode duration (s)")
parser.add_argument("--max-modules",   type=int,   default=25)
parser.add_argument("--max-depth",     type=int,   default=25)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--strategy-type",
                    choices=["plus", "comma", "nsga2-efficiency", "nsga2-symmetry"],
                    default="plus",
                    help="Survivor selection strategy")
parser.add_argument("--repeat-evals",  action="store_true",
                    help="Re-evaluate parents each generation (plus strategy only)")
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
REPEAT_EVALS  = args.repeat_evals

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)
CHECKPOINTS = DATA / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True, parents=True)
JSONL_PATH  = DATA / "run_data.jsonl"
TIMING_PATH = DATA / "timing.jsonl"

RNG = np.random.default_rng(BASE_SEED)

with (DATA / "run_config.json").open("w") as _fh:
    json.dump(vars(args), _fh, indent=2)

# ── Episode runner (locomotion) ───────────────────────────────────────────────

SETTLE_DURATION = 3.0   # seconds of zero-action settling before the episode starts
HINGE_CONTACT_LIMIT = 200  # unique rotor-ground contact events above which glitch penalty applies
HINGE_CONTACT_PENALTY = 0.005  # per unique rotor-ground contact event
HINGE_GLITCH_FITNESS = 1.0  # penalty fitness for glitched morphologies (must be > any real score; framework minimises)
CTRL_ALPHA = 0.5  # control blending factor (0=no change, 1=instant)
HEIGHT_PENALTY_THRESHOLD = 0.21  # m — only penalise spawn height above this


def _build_rotor_geom_ids(model: mujoco.MjModel) -> set[int]:
    """Return geom IDs belonging to hinge rotor bodies only."""
    ids: set[int] = set()
    for i in range(model.ngeom):
        if model.geom(i).name.endswith("-rotor"):
            ids.add(i)
    return ids


def _floor_geom_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


def run_episode_locomotion(
    ctx: dict[str, Any],
    waypoints: Any,  # unused for locomotion; kept for uniform interface
    weights: np.ndarray,
    record: bool = False,
) -> dict[str, Any]:
    model: mujoco.MjModel = ctx["model"]
    data: mujoco.MjData   = ctx["data"]
    network: Network      = ctx["network"]

    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    rotor_geom_ids = _build_rotor_geom_ids(model)
    floor_id       = _floor_geom_id(model)

    # ── Settling phase: no actions, let the robot drop and stabilise ──────────
    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)

    # Reference state is taken after settling
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    x0             = float(data.qpos[0])
    y0             = float(data.qpos[1])
    initial_height = float(data.xpos[core_id, 2])  # z of core after settling

    trajectory: list[list[float]] = []
    ctrl_history: list[np.ndarray] = []
    control_step_freq = 100
    step = 0
    current_action = np.zeros(model.nu)
    c_hinge = 0  # unique rotor-ground contact event counter
    prev_rotor_contacts: set[int] = set()

    episode_end = SETTLE_DURATION + DURATION
    while data.time < episode_end:
        if step % control_step_freq == 0:
            state = get_robot_state(data).astype(np.float32)
            raw_action = network.forward(model, data, state)
            # Alpha-blend towards new action and clip to servo range
            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -np.pi / 2, np.pi / 2,
            )

        data.ctrl[:] = current_action
        if record:
            ctrl_history.append(current_action.copy())
            if step % control_step_freq == 0:
                trajectory.append([float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])])
        mujoco.mj_step(model, data)

        # Count unique rotor-ground contact events (transition into contact only)
        curr_rotor_contacts: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor_id and g2 in rotor_geom_ids:
                curr_rotor_contacts.add(g2)
            elif g2 == floor_id and g1 in rotor_geom_ids:
                curr_rotor_contacts.add(g1)
        c_hinge += len(curr_rotor_contacts - prev_rotor_contacts)
        prev_rotor_contacts = curr_rotor_contacts

        step += 1

    x_final = float(data.qpos[0])
    y_final = float(data.qpos[1])
    x_displacement = x_final - x0

    if c_hinge > HINGE_CONTACT_LIMIT:
        fitness = HINGE_GLITCH_FITNESS
    else:
        # Maximise x_displacement, penalise tall spawn heights and rotor contacts.
        # Negated because the framework minimises.
        height_penalty = initial_height if initial_height > HEIGHT_PENALTY_THRESHOLD else 0.0
        fitness = -(x_displacement - height_penalty - HINGE_CONTACT_PENALTY * c_hinge)

    return {
        "fitness":      fitness,
        "trajectory":   trajectory,
        "control_cost": action_control_cost(ctrl_history),
        "c_hinge":      c_hinge,
        "initial_height": initial_height,
    }


_EPISODE_FN_NAME = "locomotion"
register_episode_fn(_EPISODE_FN_NAME, run_episode_locomotion)


def _train_locomotion_worker(task: dict[str, Any]) -> dict[str, Any]:
    task["episode_fn"]      = run_episode_locomotion
    task["episode_fn_name"] = _EPISODE_FN_NAME
    task["use_vision"]      = False
    task["waypoints"]       = None
    task["reach_radius"]    = 0.0
    task["arena_radius"]    = 1.0
    return train_body_parallel(task)


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

        if STRATEGY == "plus" and REPEAT_EVALS:
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
                "brain_workers": BRAIN_WORKERS,
                "duration":     DURATION,
            }
            for idx, ind in enumerate(to_eval)
        ]

        t0 = time.time()
        results = [_train_locomotion_worker(t) for t in tasks]
        wall_elapsed = time.time() - t0

        eval_times: list[float] = []
        for idx, (ind, res) in enumerate(zip(to_eval, results)):
            best_fit   = float(res["fitness"])
            best_w_lst = res["weights"]
            best_w     = np.array(best_w_lst) if best_w_lst is not None else None
            eval_time  = float(res.get("eval_time_s", 0.0))
            eval_times.append(eval_time)

            ind.fitness = best_fit if np.isfinite(best_fit) else float("inf")
            ind.tags = {
                **ind.tags,
                "best_brain":      best_w_lst or [],
                "learning_curve":  res.get("learning_curve", []),
                "trajectory":      res.get("trajectory", []),
                "control_cost":    res.get("control_cost", 0.0),
                "yz_symmetry":     bilateral_symmetry_score(ind.genotype["morph"]),
                "eval_time_s":     eval_time,
                "c_hinge":         res.get("c_hinge", 0),
                "initial_height":  res.get("initial_height", 0.0),
            }
            ind.requires_eval = False

            self._write_jsonl(ind, idx)
            self._save_checkpoint(
                tag=f"gen{self.outer_gen:03d}_body{idx:02d}",
                genotype=dict(ind.genotype["morph"]),
                weights=best_w,
                fitness=best_fit,
            )

            if best_fit < self.best_seen_fitness and best_w is not None:
                self.best_seen_fitness  = best_fit
                self.best_seen_genotype = dict(ind.genotype["morph"])
                self.best_seen_weights  = best_w.copy()

        finite = [r["fitness"] for r in results if np.isfinite(r["fitness"])]
        stats = (f"min={np.min(finite):.3f}  avg={np.mean(finite):.3f}"
                 if finite else "all-infinite")
        console.log(f"  {len(to_eval)} bodies in {wall_elapsed:.1f}s  {stats}")
        self._write_timing(wall_elapsed, eval_times)

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
            "genome_hash":    genome_hash(ind.genotype["morph"]),
            "eval_time_s":    ind.tags.get("eval_time_s", 0.0),
            "c_hinge":        ind.tags.get("c_hinge", 0),
            "initial_height": ind.tags.get("initial_height", 0.0),
        }
        with JSONL_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _write_timing(self, wall_elapsed: float, eval_times: list[float]) -> None:
        n = len(eval_times)
        record = {
            "gen":                self.outer_gen,
            "n_individuals":      n,
            "wall_time_s":        round(wall_elapsed, 3),
            "mean_eval_time_s":   round(float(np.mean(eval_times)) if eval_times else 0.0, 3),
            "min_eval_time_s":    round(float(np.min(eval_times)) if eval_times else 0.0, 3),
            "max_eval_time_s":    round(float(np.max(eval_times)) if eval_times else 0.0, 3),
            "total_eval_time_s":  round(float(np.sum(eval_times)) if eval_times else 0.0, 3),
        }
        with TIMING_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _save_checkpoint(self, tag: str, genotype: dict, weights: Optional[np.ndarray],
                         fitness: float) -> None:
        if weights is None or not np.isfinite(fitness):
            return
        sub = CHECKPOINTS / tag
        sub.mkdir(exist_ok=True, parents=True)
        np.save(sub / "best_weights.npy", weights)
        with (sub / "best_genome.json").open("w") as fh:
            json.dump(genotype, fh, indent=2)
        from shared import bilateral_symmetry_score as _bss
        from ariel.ec.genotypes.tree.tree_genome import TreeGenome as _TG
        try:
            num_mods = len(_TG.from_dict(genotype).nodes)
        except Exception:
            num_mods = 0
        with (sub / "meta.json").open("w") as fh:
            json.dump({
                "gen":         self.outer_gen,
                "fitness":     fitness,
                "yz_symmetry": _bss(genotype),
                "num_modules": num_mods,
            }, fh, indent=2)

    # -- main -----------------------------------------------------------------

    def evolve(self) -> Optional[Individual]:
        console.log("[yellow]Initialising population...[/yellow]")
        population = Population([create_individual(NUM_MODULES, MAX_DEPTH) for _ in range(MU)])

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
        f"repeat_evals={REPEAT_EVALS}  seed={BASE_SEED}"
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
