"""
Symmetry-pressure investigation — forward / multidirection / turn_avg tasks.

Outer loop:  (mu+lambda | mu,lambda | nsga2-efficiency | nsga2-symmetry) ES
             over bodies (tree / tree_symmetric / cppn genome, see --genome-type
             and genome_adapter.py) using the ariel.ec engine.
Inner loop:  One or more separate CMA-ES runs per body via
             shared.train_skill_for_body — the exact same skill-training
             kernel gecko_food_skills.py uses for its loco/left/right skills,
             so all task scripts in this sweep train skills identically and
             only differ in which skills are trained and how their fitnesses
             are combined into a body fitness.

--task selects the skill set:
  forward:        1 skill — forward translation (same reward as gecko_food_skills.py's
                   "loco" skill). Body fitness = that skill's fitness.
  multidirection:  5 skills — forward translation rotated by i*72 degrees for
                   i in 0..4 (the robot's spawn orientation is never changed,
                   only the reward-projection axis rotates). Body fitness =
                   mean of the 5 directional fitnesses.
  turn_avg:        3 skills — forward translation, turn-left, turn-right
                   (same reward primitives as gecko_food_skills.py's
                   loco/left/right). Body fitness = mean of the 3.

Unlike gecko_food_skills.py, there is no loco-fitness gate here: every
configured skill is always trained for every individual, since the point of
these tasks is a genuine average across skills rather than a gated pipeline.

Data saved per individual during evolution (run_data.jsonl):
  gen, ind_id, parent_ids, fitness, task, genome_type,
  per_skill: {name: {fitness, learning_curve, eval_time_s}},
  yz_symmetry, genome_hash, eval_time_s
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from rich.console import Console
from rich.traceback import install

from ariel.ec import EA, EAOperation, EASettings, Individual, Population

import genome_adapter
from shared import (
    LOCO_DURATION,
    TURN_DURATION,
    SkillReward,
    nsga2_survivor_selection,
    train_skill_for_body,
)

install()

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Symmetry-pressure: forward / multidirection / turn_avg locomotion tasks"
)
parser.add_argument("--task",
                    choices=["forward", "multidirection", "turn_avg"],
                    required=True)
parser.add_argument("--genome-type",
                    choices=["tree", "tree_symmetric", "cppn"],
                    default="tree",
                    help="Body genome representation: plain TreeGenome, TreeGenome with "
                         "enforced y_zero bilateral symmetry, or a CPPN-NEAT genome")
parser.add_argument("--budget",        type=int,   default=40)
parser.add_argument("--pop",           type=int,   default=20,  help="mu")
parser.add_argument("--lam",           type=int,   default=20,  help="lambda")
parser.add_argument("--skill-budget",  type=int,   default=100,
                    help="CMA-ES generations per trained skill (matches "
                         "gecko_food_skills.py's LOCO_BUDGET/TURN_BUDGET)")
parser.add_argument("--brain-workers", type=int,   default=16,
                    help="Parallel workers for each skill's CMA-ES population "
                         "(matches CMA-ES popsize=16 by default)")
parser.add_argument("--max-modules",   type=int,   default=25)
parser.add_argument("--max-depth",     type=int,   default=25)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--strategy-type",
                    choices=["plus", "comma", "nsga2-efficiency", "nsga2-symmetry"],
                    default="plus")
parser.add_argument("--repeat-evals",  action="store_true",
                    help="Re-evaluate parents each generation (plus strategy only)")
parser.add_argument("--time-limit",    type=float, default=None,
                    help="Wall-clock seconds; stop after current generation completes")
args = parser.parse_args()

TASK           = args.task
GENOME_TYPE    = args.genome_type
BUDGET         = args.budget
MU             = args.pop
LAM            = args.lam
SKILL_BUDGET   = args.skill_budget
BRAIN_WORKERS  = max(1, args.brain_workers)
NUM_MODULES    = args.max_modules
MAX_DEPTH      = args.max_depth
BASE_SEED      = args.seed
STRATEGY       = args.strategy_type
REPEAT_EVALS   = args.repeat_evals
TIME_LIMIT     = args.time_limit

TOURNAMENT_SIZE = 4
N_DIRECTIONS    = 5   # multidirection: 360 / 5 = 72 degrees apart

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME / TASK
DATA.mkdir(exist_ok=True, parents=True)
CHECKPOINTS = DATA / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True, parents=True)
JSONL_PATH  = DATA / "run_data.jsonl"
TIMING_PATH = DATA / "timing.jsonl"

RNG = np.random.default_rng(BASE_SEED)

with (DATA / "run_config.json").open("w") as _fh:
    json.dump(vars(args), _fh, indent=2)

ADAPTER = genome_adapter.genome_adapter_from_cli(GENOME_TYPE, NUM_MODULES)


# ── Skill sets per task ────────────────────────────────────────────────────────


def _skills_for_task(task: str) -> list[tuple[str, SkillReward]]:
    if task == "forward":
        return [("fwd", SkillReward(kind="translate", angle_deg=0.0, duration=LOCO_DURATION))]
    if task == "multidirection":
        step = 360.0 / N_DIRECTIONS
        return [
            (f"dir{i}", SkillReward(kind="translate", angle_deg=i * step, duration=LOCO_DURATION))
            for i in range(N_DIRECTIONS)
        ]
    if task == "turn_avg":
        return [
            ("fwd",   SkillReward(kind="translate", angle_deg=0.0, duration=LOCO_DURATION)),
            ("left",  SkillReward(kind="rotate", turn_sign=+1, duration=TURN_DURATION)),
            ("right", SkillReward(kind="rotate", turn_sign=-1, duration=TURN_DURATION)),
        ]
    raise ValueError(f"Unknown task: {task!r}")


SKILLS = _skills_for_task(TASK)


# ── Per-individual pipeline: train every configured skill, average fitness ───


def _train_body(genome_dict: dict, seed: int) -> dict[str, Any]:
    per_skill: dict[str, dict[str, Any]] = {}
    skill_fitnesses: list[float] = []
    weights: dict[str, Optional[np.ndarray]] = {}

    for i, (name, reward) in enumerate(SKILLS):
        w, learning_curve, eval_time = train_skill_for_body(
            reward, genome_dict, SKILL_BUDGET, BRAIN_WORKERS, seed + i,
            to_spec_fn=ADAPTER.to_spec,
        )
        fitness = float(min(learning_curve)) if learning_curve else float("inf")
        console.log(f"    {name}: best_fitness={fitness:.4f}")
        skill_fitnesses.append(fitness)
        weights[name] = w
        per_skill[name] = {
            "fitness":        fitness,
            "learning_curve": learning_curve,
            "eval_time_s":    eval_time,
        }

    return {
        "fitness":    float(np.mean(skill_fitnesses)) if skill_fitnesses else float("inf"),
        "per_skill":  per_skill,
        "weights":    weights,
        "eval_time_s": sum(s["eval_time_s"] for s in per_skill.values()),
    }


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
        self.selected_parents: list[Individual] = []
        self.best_seen_fitness:  float = float("inf")
        self.best_seen_genotype: Optional[dict] = None
        self.best_seen_weights:  Optional[dict[str, np.ndarray]] = None

    # -- operations -----------------------------------------------------------

    def parent_selection(self, population: Population) -> Population:
        contenders = list(population)
        k = min(TOURNAMENT_SIZE, len(contenders))
        winners: list[Individual] = []
        for _ in range(MU):
            bracket = RNG.choice(len(contenders), size=k, replace=False)
            winner = min(
                (contenders[i] for i in bracket),
                key=lambda ind: ind.fitness_ if ind.fitness_ is not None else float("inf"),
            )
            winners.append(winner)
        self.selected_parents = winners
        winner_ids = {id(w) for w in winners}
        for ind in population:
            ind.tags = {**ind.tags, "ps": id(ind) in winner_ids}
        return population

    def reproduction(self, population: Population) -> Population:
        parents = self.selected_parents or list(population)
        offspring = ADAPTER.make_offspring(parents, LAM, RNG, NUM_MODULES, MAX_DEPTH)

        if STRATEGY == "plus" and REPEAT_EVALS:
            for ind in parents:
                ind.requires_eval = True

        population.extend(offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        console.rule(f"[bold magenta]Outer gen {self.outer_gen} — task={TASK}[/bold magenta]")

        to_eval = [
            ind for ind in population
            if ind.alive and ind.tags.get("valid", True) and ind.requires_eval
        ]

        t0 = time.time()
        eval_times: list[float] = []
        results: list[dict[str, Any]] = []
        for idx, ind in enumerate(to_eval):
            genome_dict = ind.genotype[ADAPTER.genotype_key]
            console.log(f"  [{idx + 1}/{len(to_eval)}] training body {ADAPTER.hash(genome_dict)[:8]}...")
            res = _train_body(
                genome_dict=genome_dict,
                seed=BASE_SEED + 1000 * self.outer_gen + idx,
            )
            results.append(res)
            eval_times.append(res["eval_time_s"])
        wall_elapsed = time.time() - t0

        for idx, (ind, res) in enumerate(zip(to_eval, results)):
            best_fit = float(res["fitness"])

            ind.fitness = best_fit if np.isfinite(best_fit) else float("inf")
            ind.tags = {
                **ind.tags,
                "per_skill":   res["per_skill"],
                "yz_symmetry": ADAPTER.symmetry_score(ind.genotype[ADAPTER.genotype_key]),
                "eval_time_s": eval_times[idx],
                "control_cost": 0.0,  # no controller/episode-level control cost for these tasks
            }
            ind.requires_eval = False

            self._write_jsonl(ind)

            self._save_checkpoint(
                tag=f"gen{self.outer_gen:03d}_body{idx:02d}",
                genotype=dict(ind.genotype[ADAPTER.genotype_key]),
                weights=res["weights"],
                fitness=best_fit,
            )

            if best_fit < self.best_seen_fitness:
                self.best_seen_fitness  = best_fit
                self.best_seen_genotype = dict(ind.genotype[ADAPTER.genotype_key])
                self.best_seen_weights  = {k: (v.copy() if v is not None else None)
                                            for k, v in res["weights"].items()}

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
            offspring_alive = [i for i in alive if not i.tags.get("ps", False)]
            offspring_sorted = sorted(offspring_alive, key=lambda i: i.fitness_ or float("inf"))
            survivors = set(id(i) for i in offspring_sorted[:MU])
            for ind in population:
                if ind.alive and id(ind) not in survivors:
                    ind.alive = False

        elif STRATEGY in ("nsga2-efficiency", "nsga2-symmetry"):
            valid   = [i for i in alive if np.isfinite(i.fitness_ or float("inf"))]
            invalid = [i for i in alive if not np.isfinite(i.fitness_ or float("inf"))]
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

    def _write_jsonl(self, ind: Individual) -> None:
        record = {
            "gen":         self.outer_gen,
            "ind_id":      ind.id,
            "parent_ids":  ind.genotype.get("parent_ids", []),
            "fitness":     ind.fitness_ if ind.fitness_ is not None else None,
            "task":        TASK,
            "genome_type": GENOME_TYPE,
            "per_skill":   ind.tags.get("per_skill", {}),
            "yz_symmetry": ind.tags.get("yz_symmetry", 0.0),
            "genome_hash": ADAPTER.hash(ind.genotype[ADAPTER.genotype_key]),
            "eval_time_s": ind.tags.get("eval_time_s", 0.0),
        }
        with JSONL_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _write_timing(self, wall_elapsed: float, eval_times: list[float]) -> None:
        n = len(eval_times)
        record = {
            "gen":               self.outer_gen,
            "n_individuals":     n,
            "wall_time_s":       round(wall_elapsed, 3),
            "mean_eval_time_s":  round(float(np.mean(eval_times)) if eval_times else 0.0, 3),
            "min_eval_time_s":   round(float(np.min(eval_times)) if eval_times else 0.0, 3),
            "max_eval_time_s":   round(float(np.max(eval_times)) if eval_times else 0.0, 3),
            "total_eval_time_s": round(float(np.sum(eval_times)) if eval_times else 0.0, 3),
        }
        with TIMING_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _save_checkpoint(self, tag: str, genotype: dict, weights: dict[str, Optional[np.ndarray]],
                         fitness: float) -> None:
        if not weights or not np.isfinite(fitness):
            return
        sub = CHECKPOINTS / tag
        sub.mkdir(exist_ok=True, parents=True)
        for name, w in weights.items():
            if w is not None:
                np.save(sub / f"{name}_weights.npy", w)
        with (sub / "best_genome.json").open("w") as fh:
            json.dump(genotype, fh, indent=2)
        with (sub / "meta.json").open("w") as fh:
            json.dump({
                "gen":         self.outer_gen,
                "fitness":     fitness,
                "yz_symmetry": ADAPTER.symmetry_score(genotype),
                "num_modules": ADAPTER.module_count(genotype),
                "task":        TASK,
                "genome_type": GENOME_TYPE,
            }, fh, indent=2)

    # -- main -----------------------------------------------------------------

    def evolve(self) -> Optional[Individual]:
        console.log("[yellow]Initialising population...[/yellow]")
        population = Population([
            ADAPTER.create_individual(RNG, NUM_MODULES, MAX_DEPTH) for _ in range(MU)
        ])

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
        t_start = time.time()
        for _ in range(BUDGET):
            if TIME_LIMIT is not None and time.time() - t_start >= TIME_LIMIT:
                console.log(f"[yellow]Time limit {TIME_LIMIT:.0f}s reached — stopping early[/yellow]")
                break
            ea.step()
        return ea.get_solution("best", only_alive=False)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    console.rule(f"[bold magenta]Symmetry-Pressure — {TASK}[/bold magenta]")
    console.log(
        f"task={TASK}  genome_type={GENOME_TYPE}  skills={[name for name, _ in SKILLS]}  "
        f"strategy={STRATEGY}  (mu+lam)=({MU}+{LAM})  budget={BUDGET}  "
        f"skill_budget={SKILL_BUDGET}  brain_workers={BRAIN_WORKERS}  "
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
