"""
Experiment 1: body evolution with controller learning on forward locomotion.

Outer loop  (mu + lambda) = (20 + 20) over bodies of one genome (genomes.py),
            config.PROTOCOL.generations generations after the initial one.
            Parents by tournament of 4; each child is crossover (60%) and/or
            mutation, redrawn until it is a valid body.
Inner loop  every body's controller (--brain) is trained from scratch with
            CMA-ES for --evals evaluations (sim.train_controller); a body's
            score is the best displacement found (m).
Re-evaluation  every survivor is retrained with a fresh seed each generation,
            together with the new offspring, and keeps only its latest score.

Generation g trains its bodies with seeds seed + 1000 g + i (i = position in
the generation's training order). The trainings of a generation run
concurrently on one pool of --workers processes.

Output (--out DIR):
  config.json          protocol, arguments, git commit
  evaluations.jsonl    one line per training: gen, id, parents, op, attempts,
                       score, train_seed, wall_s, stopped_early, curve
                       {best, median, sigma per CMA-ES generation}, body
                       measures (n_modules, n_hinges, symmetry, tied_frac),
                       genome_hash, and the genome itself the first time a body
                       is trained (born == gen)
  generations.jsonl    per generation: the population after selection (ids and
                       scores, best first), its rejection / fallback counts, wall time
  weights/genGGG.npz   best controller weights of every training in generation
                       GGG, keyed by body id
  state.pkl            the population and RNG states after the last finished
                       generation: a rerun with the same --out resumes from it

Usage:
    python evolve.py --genome tree_symmetric --brain dann --evals 200 --seed 42 \\
        --workers 32 --out runs/e1/tree_symmetric/dann/e200/s42
"""

import os

# One BLAS/OpenMP thread per process, set before numpy loads: the CMA-ES linear
# algebra of a large controller would otherwise start a thread per core and
# starve the simulation workers.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import pickle
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

import genomes
from config import BRAIN_KINDS, GENOME_TYPES, PROTOCOL, Protocol
from sim import genome_hash, make_pool, train_controllers

STATE_VERSION = 1


@dataclass
class Body:
    id: int
    genome: dict
    born: int                       # generation it was created in
    parents: list[int] = field(default_factory=list)
    op: str = "random"              # random / crossover / mutation / fallback
    attempts: int = 0               # draws until valid (random bodies: invalid draws before it)
    score: float = float("-inf")    # latest training's best displacement (m)


@dataclass
class State:
    gen: int                        # last finished generation
    population: list[Body]
    next_id: int
    rng: np.random.Generator
    py_random: tuple
    version: int = STATE_VERSION


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class Run:
    def __init__(self, genome: str, brain: str, evals: int, seed: int, workers: int, out: Path,
                 protocol: Protocol = PROTOCOL) -> None:
        self.genome, self.brain, self.evals, self.seed = genome, brain, evals, seed
        self.workers, self.out, self.protocol = workers, out, protocol
        self.ops = genomes.genome_ops(genome, protocol)
        (out / "weights").mkdir(parents=True, exist_ok=True)

    # ── persistence ───────────────────────────────────────────────────────────

    def write_config(self) -> None:
        cfg = {"experiment": "e1", "genome": self.genome, "brain": self.brain, "evals": self.evals,
               "seed": self.seed, "workers": self.workers, "protocol": asdict(self.protocol),
               "git_commit": git_commit()}
        path = self.out / "config.json"
        if path.exists():  # resuming (possibly with more generations): everything else must match
            old = json.loads(path.read_text())

            def key(c: dict) -> dict:
                return {k: c[k] for k in ("genome", "brain", "evals", "seed")} | \
                       {"protocol": {k: v for k, v in c["protocol"].items() if k != "generations"}}

            if key(old) != key(cfg):
                raise SystemExit(f"{path} belongs to a different run; use another --out")
            cfg = {**old, "protocol": cfg["protocol"],
                   "resumed_with_commits": [*old.get("resumed_with_commits", []), cfg["git_commit"]]}
        path.write_text(json.dumps(cfg, indent=2))

    def load_state(self) -> Optional[State]:
        path = self.out / "state.pkl"
        if not path.exists():
            return None
        state: State = pickle.loads(path.read_bytes())
        # Drop whatever a crash left of the unfinished generation.
        for name in ("evaluations.jsonl", "generations.jsonl"):
            p = self.out / name
            if p.exists():
                lines = [ln for ln in p.read_text().splitlines() if json.loads(ln)["gen"] <= state.gen]
                p.write_text("".join(ln + "\n" for ln in lines))
        for w in (self.out / "weights").glob("gen*.npz"):
            if int(w.stem[3:]) > state.gen:
                w.unlink()
        return state

    def save_state(self, state: State) -> None:
        tmp = self.out / "state.pkl.tmp"
        tmp.write_bytes(pickle.dumps(state))
        tmp.replace(self.out / "state.pkl")

    # ── one generation ────────────────────────────────────────────────────────

    def train(self, gen: int, bodies: list[Body], pool) -> list[dict]:
        """Train every body (in place: score), and log the trainings."""
        seeds = [self.seed + 1000 * gen + i for i in range(len(bodies))]
        jobs = [(b.genome, self.brain, self.evals, s) for b, s in zip(bodies, seeds)]
        results = train_controllers(jobs, pool, concurrent=len(jobs), protocol=self.protocol)
        records, weights = [], {}
        for b, s, r in zip(bodies, seeds, results):
            b.score = r.score
            weights[str(b.id)] = r.weights
            rec = {"gen": gen, "id": b.id, "born": b.born, "parents": b.parents, "op": b.op,
                   "attempts": b.attempts, "score": r.score, "train_seed": s, "evals": r.evals,
                   "stopped_early": r.stopped_early, "wall_s": round(r.wall_s, 2),
                   "curve": {k: [round(x, 5) for x in v] for k, v in r.curve.items()},
                   **genomes.describe(b.genome), "genome_hash": genome_hash(b.genome)}
            if b.born == gen:
                rec["genome"] = b.genome
            records.append(rec)
        np.savez_compressed(self.out / "weights" / f"gen{gen:03d}.npz", **weights)
        return records

    def log(self, gen: int, records: list[dict], population: list[Body], wall_s: float) -> None:
        with (self.out / "evaluations.jsonl").open("a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in records)
        new = [r for r in records if r["born"] == gen]
        line = {"gen": gen, "ids": [b.id for b in population], "scores": [b.score for b in population],
                "new": len(new), "fallbacks": sum(r["op"] == "fallback" for r in new),
                "attempts": sum(r["attempts"] for r in new), "wall_s": round(wall_s, 1)}
        with (self.out / "generations.jsonl").open("a") as fh:
            fh.write(json.dumps(line) + "\n")

    def select_parents(self, population: list[Body], rng: np.random.Generator) -> list[Body]:
        k = min(self.protocol.tournament, len(population))
        return [max((population[i] for i in rng.choice(len(population), size=k, replace=False)),
                    key=lambda b: b.score)
                for _ in range(self.protocol.mu)]

    def survivors(self, bodies: list[Body]) -> list[Body]:
        return sorted(bodies, key=lambda b: -b.score)[: self.protocol.mu]

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, time_limit: Optional[float] = None) -> None:
        self.write_config()
        state = self.load_state()
        t_start = time.time()
        with make_pool(self.workers) as pool:
            if state is None:
                random.seed(self.seed)
                rng = np.random.default_rng(self.seed)
                t0 = time.time()
                population = []
                for i in range(self.protocol.mu):
                    g, rejected = genomes.random_body(self.ops, self.protocol)
                    population.append(Body(id=i, genome=g, born=0, attempts=rejected))
                records = self.train(0, population, pool)
                population = self.survivors(population)
                self.log(0, records, population, time.time() - t0)
                state = State(0, population, self.protocol.mu, rng, random.getstate())
                self.save_state(state)
                print(f"gen 0: best {population[0].score:.3f} m", flush=True)
            else:
                random.setstate(state.py_random)
                print(f"resuming after generation {state.gen}", flush=True)

            while state.gen < self.protocol.generations:
                if time_limit is not None and time.time() - t_start >= time_limit:
                    print(f"time limit reached after generation {state.gen}", flush=True)
                    return
                gen, rng, population = state.gen + 1, state.rng, state.population
                t0 = time.time()
                parents = self.select_parents(population, rng)
                children = []
                for k in range(self.protocol.lam):
                    c = genomes.make_child(self.ops, [p.genome for p in parents], rng, self.protocol)
                    children.append(Body(id=state.next_id + k, genome=c.genome, born=gen,
                                         parents=[parents[i].id for i in c.parents], op=c.op, attempts=c.attempts))
                # Re-evaluation: survivors are retrained with the offspring.
                bodies = population + children
                records = self.train(gen, bodies, pool)
                population = self.survivors(bodies)
                self.log(gen, records, population, time.time() - t0)
                state = State(gen, population, state.next_id + self.protocol.lam, rng, random.getstate())
                self.save_state(state)
                print(f"gen {gen}: best {population[0].score:.3f} m  top-{len(population)} mean "
                      f"{np.mean([b.score for b in population]):.3f} m  ({time.time() - t0:.0f} s)", flush=True)
        (self.out / "DONE").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: body evolution with controller learning")
    parser.add_argument("--genome", choices=GENOME_TYPES, required=True)
    parser.add_argument("--brain", choices=BRAIN_KINDS, required=True)
    parser.add_argument("--evals", type=int, required=True, help="learning evaluations per body")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--generations", type=int, default=PROTOCOL.generations,
                        help="generations after the initial one (testing only; the paper uses the default)")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="wall-clock seconds; stop after the generation running at that time")
    args = parser.parse_args()
    protocol = replace(PROTOCOL, generations=args.generations)
    Run(args.genome, args.brain, args.evals, args.seed, args.workers, args.out, protocol).run(args.time_limit)


if __name__ == "__main__":
    main()
