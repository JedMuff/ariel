"""
Body-brain evolution on forward locomotion (E0, E1).

Outer loop:  (mu + lambda) evolution strategy over bodies (--genome-type
             tree / tree_symmetric / tree_tied, see genomes.py) using the
             ariel.ec engine, tournament parent selection.
Inner loop:  CMA-ES over each body's controller (--brain, see brains.py) for
             --skill-budget generations of popsize 20 (sim.train_skill_for_body).
             Body fitness = the forward skill's best fitness (minimised:
             -(normalised score - penalties), see sim.run_skill_episode).

With --repeat-evals, surviving parents are retrained every generation, so a
body's fitness is not a lucky draw it keeps forever.

Writes to ./__data__/evolve/forward/:
  run_config.json   the CLI arguments, eval profile and hinge actuator
  run_data.jsonl    one record per evaluated body: gen, ind_id, parent_ids,
                    fitness, per_skill {fwd: {fitness, learning_curve,
                    eval_time_s, raw_score, height_penalty, jerk_penalty,
                    score}}, yz_symmetry, tied_frac, genome_hash, eval_time_s
  timing.jsonl      wall time per generation
  checkpoints/genGGG_bodyII/  best_genome.json, fwd_weights.npy, meta.json

Usage:
    python evolve.py --genome-type tree_symmetric --brain bang_bang --seed 42 \\
        --budget 30 --pop 10 --lam 20 --repeat-evals --brain-workers 20
"""

import os

# One BLAS/OpenMP thread per process, set before numpy loads: otherwise the
# parent's CMA-ES linear algebra on the DANN's 705-dim covariance spins up an
# OpenBLAS thread per machine core, which starves the --brain-workers
# simulation processes (~3.5x slower).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import gc
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.traceback import install

from ariel.ec import EA, EAOperation, EASettings, Individual, Population
from ariel.ec.genotypes.tree.tying import tied_fraction

import genomes
from brains import BRAIN_KINDS
from sim import PROFILE, ZOO_HINGE, HingeActuator, genome_hash, train_skill_for_body, use_zoo_modules_globally, zoo_spec_fn

install()

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Body-brain evolution on forward locomotion")
parser.add_argument("--genome-type",   choices=genomes.GENOME_TYPES, default="tree")
parser.add_argument("--tie-prob",      type=float, default=genomes.DEFAULT_TIE_PROB,
                    help="tree_tied: chance a placed module gets a twin")
parser.add_argument("--brain",         choices=BRAIN_KINDS, default="bang_bang")
parser.add_argument("--budget",        type=int,   default=30, help="outer generations")
parser.add_argument("--pop",           type=int,   default=10, help="mu")
parser.add_argument("--lam",           type=int,   default=20, help="lambda")
parser.add_argument("--skill-budget",  type=int,   default=PROFILE.skill_budget,
                    help="inner CMA-ES generations per body (x popsize 20 evaluations)")
parser.add_argument("--brain-workers", type=int,   default=20, help="parallel episode workers")
parser.add_argument("--max-modules",   type=int,   default=25)
parser.add_argument("--max-depth",     type=int,   default=25)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--repeat-evals",  action="store_true", help="retrain surviving parents every generation")
parser.add_argument("--hinge-kp",      type=float, default=None, help="hinge gain (default: zoo hinge)")
parser.add_argument("--hinge-kv",      type=float, default=None, help="hinge damping (default: zoo hinge)")
parser.add_argument("--hinge-armature", type=float, default=None, help="hinge armature (default: zoo hinge)")
parser.add_argument("--hinge-limit",   action=argparse.BooleanOptionalAction, default=None,
                    help="hard +-90 deg hinge joint limit (default: on, as the zoo hinge)")
parser.add_argument("--time-limit",    type=float, default=None,
                    help="wall-clock seconds; stop after the current generation completes")
args = parser.parse_args()

TASK           = "forward"
GENOME_TYPE    = args.genome_type
BUDGET         = args.budget
MU             = args.pop
LAM            = args.lam
SKILL_BUDGET   = args.skill_budget
PROFILE_RUN    = replace(PROFILE, skill_budget=SKILL_BUDGET)
BRAIN_WORKERS  = max(1, args.brain_workers)
NUM_MODULES    = args.max_modules
MAX_DEPTH      = args.max_depth
BASE_SEED      = args.seed
REPEAT_EVALS   = args.repeat_evals
TIME_LIMIT     = args.time_limit
BRAIN          = args.brain
_HINGE_ARGS    = {"kp": args.hinge_kp, "kv": args.hinge_kv, "armature": args.hinge_armature,
                  "limited": args.hinge_limit}
HINGE_ACTUATOR = HingeActuator(**{**asdict(ZOO_HINGE), **{k: v for k, v in _HINGE_ARGS.items() if v is not None}})

TOURNAMENT_SIZE = 4

DATA = Path.cwd() / "__data__" / "evolve" / TASK
DATA.mkdir(exist_ok=True, parents=True)
CHECKPOINTS = DATA / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True, parents=True)
JSONL_PATH  = DATA / "run_data.jsonl"
TIMING_PATH = DATA / "timing.jsonl"

RNG = np.random.default_rng(BASE_SEED)

with (DATA / "run_config.json").open("w") as _fh:
    json.dump({**vars(args), "task": TASK, "eval_profile": "dsocial_zoo", "zoo_modules": True,
               "control_step_freq": PROFILE.control_step_freq, "eval_profile_settings": asdict(PROFILE_RUN),
               "hinge_actuator": asdict(HINGE_ACTUATOR)}, _fh, indent=2)

ADAPTER = genomes.genome_adapter(GENOME_TYPE, tie_prob=args.tie_prob)
TO_SPEC = zoo_spec_fn(HINGE_ACTUATOR)
use_zoo_modules_globally()  # the genome operators' collision checks use the zoo geometry too


def _train_body(genome_dict: dict, seed: int) -> dict[str, Any]:
    w, learning_curve, eval_time, best_info = train_skill_for_body(
        genome_dict, SKILL_BUDGET, BRAIN_WORKERS, seed, to_spec_fn=TO_SPEC, brain_kind=BRAIN, profile=PROFILE,
    )
    fitness = float(min(learning_curve)) if learning_curve else float("inf")
    console.log(f"    fwd: best_fitness={fitness:.4f}")
    return {
        "fitness":     fitness,
        "per_skill":   {"fwd": {"fitness": fitness, "learning_curve": learning_curve, "eval_time_s": eval_time,
                                **best_info}},
        "weights":     {"fwd": w},
        "eval_time_s": eval_time,
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
        self.best_seen_fitness: float = float("inf")

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

        if REPEAT_EVALS:
            for ind in parents:
                ind.requires_eval = True

        population.extend(offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        console.rule(f"[bold magenta]Outer gen {self.outer_gen}[/bold magenta]")

        to_eval = [
            ind for ind in population
            if ind.alive and ind.tags.get("valid", True) and ind.requires_eval
        ]

        t0 = time.time()
        results: list[dict[str, Any]] = []
        for idx, ind in enumerate(to_eval):
            genome_dict = ind.genotype["morph"]
            console.log(f"  [{idx + 1}/{len(to_eval)}] training body {genome_hash(genome_dict)[:8]}...")
            results.append(_train_body(genome_dict, seed=BASE_SEED + 1000 * self.outer_gen + idx))
        wall_elapsed = time.time() - t0
        eval_times = [r["eval_time_s"] for r in results]

        for idx, (ind, res) in enumerate(zip(to_eval, results)):
            best_fit = float(res["fitness"])
            ind.fitness = best_fit if np.isfinite(best_fit) else float("inf")
            ind.tags = {
                **ind.tags,
                "per_skill":   res["per_skill"],
                "yz_symmetry": ADAPTER.symmetry_score(ind.genotype["morph"]),
                "eval_time_s": res["eval_time_s"],
            }
            ind.requires_eval = False

            self._write_jsonl(ind)
            self._save_checkpoint(f"gen{self.outer_gen:03d}_body{idx:02d}", dict(ind.genotype["morph"]),
                                  res["weights"], best_fit)
            self.best_seen_fitness = min(self.best_seen_fitness, best_fit)

        finite = [r["fitness"] for r in results if np.isfinite(r["fitness"])]
        stats = f"min={np.min(finite):.3f}  avg={np.mean(finite):.3f}" if finite else "all-infinite"
        console.log(f"  {len(to_eval)} bodies in {wall_elapsed:.1f}s  {stats}")
        self._write_timing(wall_elapsed, eval_times)

        self.outer_gen += 1
        return population

    def survivor_selection(self, population: Population) -> Population:
        alive_sorted = sorted((ind for ind in population if ind.alive), key=lambda i: i.fitness_ or float("inf"))
        survivors = set(id(i) for i in alive_sorted[:MU])
        for ind in population:
            if ind.alive and id(ind) not in survivors:
                ind.alive = False

        finite = [i.fitness_ for i in population if i.alive and i.fitness_ is not None and np.isfinite(i.fitness_)]
        if finite:
            console.log(f"[green]Survivors:[/green] avg={np.mean(finite):.3f}  min={np.min(finite):.3f}")
        return population

    # -- persistence ----------------------------------------------------------

    def _write_jsonl(self, ind: Individual) -> None:
        morph = ind.genotype["morph"]
        record = {
            "gen":         self.outer_gen,
            "ind_id":      ind.id,
            "parent_ids":  ind.genotype.get("parent_ids", []),
            "fitness":     ind.fitness_ if ind.fitness_ is not None else None,
            "task":        TASK,
            "genome_type": GENOME_TYPE,
            "per_skill":   ind.tags.get("per_skill", {}),
            "yz_symmetry": ind.tags.get("yz_symmetry", 0.0),
            "tied_frac":   tied_fraction(morph) if GENOME_TYPE == "tree_tied" else None,
            "genome_hash": genome_hash(morph),
            "eval_time_s": ind.tags.get("eval_time_s", 0.0),
        }
        with JSONL_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _write_timing(self, wall_elapsed: float, eval_times: list[float]) -> None:
        t = np.array(eval_times) if eval_times else np.zeros(1)
        record = {
            "gen":               self.outer_gen,
            "n_individuals":     len(eval_times),
            "wall_time_s":       round(wall_elapsed, 3),
            "mean_eval_time_s":  round(float(t.mean()), 3),
            "min_eval_time_s":   round(float(t.min()), 3),
            "max_eval_time_s":   round(float(t.max()), 3),
            "total_eval_time_s": round(float(t.sum()), 3),
        }
        with TIMING_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _save_checkpoint(self, tag: str, genotype: dict, weights: dict[str, Optional[np.ndarray]],
                         fitness: float) -> None:
        if not np.isfinite(fitness):
            return
        sub = CHECKPOINTS / tag
        sub.mkdir(exist_ok=True, parents=True)
        for name, w in weights.items():
            np.save(sub / f"{name}_weights.npy", w)
        with (sub / "best_genome.json").open("w") as fh:
            json.dump(genotype, fh, indent=2)
        with (sub / "meta.json").open("w") as fh:
            json.dump({
                "gen":         self.outer_gen,
                "fitness":     fitness,
                "yz_symmetry": ADAPTER.symmetry_score(genotype),
                "num_modules": genomes.module_count(genotype),
                "task":        TASK,
                "genome_type": GENOME_TYPE,
            }, fh, indent=2)

    # -- main -----------------------------------------------------------------

    def evolve(self) -> Optional[Individual]:
        console.log("[yellow]Initialising population...[/yellow]")
        population = Population([ADAPTER.create_individual(RNG, NUM_MODULES, MAX_DEPTH) for _ in range(MU)])
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


def main() -> None:
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)

    console.rule("[bold magenta]Body-brain evolution — forward[/bold magenta]")
    console.log(
        f"genome_type={GENOME_TYPE}  brain={BRAIN}  (mu+lam)=({MU}+{LAM})  budget={BUDGET}  "
        f"skill_budget={SKILL_BUDGET}  hinge={HINGE_ACTUATOR.label}  brain_workers={BRAIN_WORKERS}  "
        f"repeat_evals={REPEAT_EVALS}  seed={BASE_SEED}"
    )

    start = time.time()
    evo = BodyBrainEvolution()
    best = evo.evolve()

    console.rule("[bold green]Done[/bold green]")
    if best is not None and best.fitness_ is not None:
        console.log(f"Best fitness: {best.fitness:.3f}")
    console.log(f"Best seen:    {evo.best_seen_fitness:.3f}")
    console.log(f"Elapsed:      {(time.time() - start) / 60:.1f} min")
    console.log(f"Data →        {DATA}")


if __name__ == "__main__":
    main()
    gc.disable()
