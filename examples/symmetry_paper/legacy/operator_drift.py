"""
Do the genome operators change hinge count by themselves?

Evolved tied champions have about 50% more hinges than tree and symmetric ones.
This runs the outer loop of evolve.py without simulation or selection: (20 + 20),
parents and survivors chosen at random, the same operators (sim.make_offspring).
What the population's hinge count does here is what the operators do; the
difference from the real runs is what selection added.

Writes <out>/operator_drift.csv: genome, seed, gen, hinges, modules, tied_frac,
population means.

Usage:
    python operator_drift.py --seeds 20 --workers 60 -o OUT
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import genomes
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.tying import tied_fraction
from sim import joint_count, use_zoo_modules_globally

MU, LAM, GENS, MAX_MODULES, MAX_DEPTH = 20, 20, 30, 25, 25


def summary(pop: list, genome_type: str, seed: int, gen: int) -> dict:
    morphs = [i.genotype["morph"] for i in pop]
    trees = [TreeGenome.from_dict(m) for m in morphs]
    return {"genome": genome_type, "seed": seed, "gen": gen,
            "hinges": float(np.mean([joint_count(t) for t in trees])),
            "modules": float(np.mean([len(t.nodes) for t in trees])),
            "tied_frac": float(np.mean([tied_fraction(m) for m in morphs])) if genome_type == "tree_tied" else 0.0}


def drift(args: tuple[str, int]) -> list[dict]:
    genome_type, seed = args
    use_zoo_modules_globally()
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    adapter = genomes.genome_adapter(genome_type)
    pop = [adapter.create_individual(rng, MAX_MODULES, MAX_DEPTH) for _ in range(MU)]
    rows = [summary(pop, genome_type, seed, 0)]
    for gen in range(1, GENS + 1):
        children = [c for c in adapter.make_offspring(pop, LAM, rng, MAX_MODULES, MAX_DEPTH) if c.tags.get("valid", True)]
        pop = random.sample(pop + children, MU)
        rows.append(summary(pop, genome_type, seed, gen))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    args = parser.parse_args()
    jobs = [(g, 42 + s) for g in genomes.GENOME_TYPES for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        rows = [r for res in pool.imap_unordered(drift, jobs) for r in res]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["genome", "seed", "gen"])
    df.to_csv(args.out_dir / "operator_drift.csv", index=False)
    print(df[df["gen"].isin([0, 10, 20, 30])].groupby(["genome", "gen"])[["hinges", "modules", "tied_frac"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
