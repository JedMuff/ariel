"""
Bodies for the fixed-body learning curves (E2).

  random   --n bodies per genome drawn from the genome's initial distribution
           (sim.create_individual, as evolve.py's first generation, zoo
           geometry), matched on hinge count: a pool of --pool bodies per
           genome is binned by hinge count and the n are taken from the bins
           both genomes fill, in proportion to the smaller count, so the
           genomes' hinge-count distributions are identical.
  evolved  the best last-generation body of every tree and tree_symmetric run
           under --runs-root (E1), with that run's brain, seed and score.

Writes one JSON line per body: body_id, body_source, genome_type, genome,
n_modules, n_hinges, n_limbs (and run, run_brain, run_seed, run_score for
evolved bodies).

Usage:
    python sample_bodies.py random --n 40 -o bodies_random.jsonl
    python sample_bodies.py evolved --runs-root /scratch/$USER/ariel_paper_e1 -o bodies_evolved.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import runs
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from replay import body_descriptors
from sim import create_individual, use_zoo_modules_globally

GENOMES = ("tree", "tree_symmetric")
AXIS = {"tree": None, "tree_symmetric": MirrorAxis.Y_ZERO}


def allocate(counts: dict[int, int], n: int) -> dict[int, int]:
    """n split over bins in proportion to counts (largest remainder), at most counts[b] per bin."""
    total = sum(counts.values())
    if total < n:
        raise SystemExit(f"only {total} hinge-matched bodies in the pool for {n}: raise --pool")
    exact = {b: n * c / total for b, c in counts.items()}
    alloc = {b: int(np.floor(x)) for b, x in exact.items()}
    for b in sorted(exact, key=lambda b: (alloc[b] - exact[b], b))[:n - sum(alloc.values())]:
        alloc[b] += 1
    return alloc


def sample_random(n: int, pool: int, max_modules: int, max_depth: int, seed: int) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    use_zoo_modules_globally()
    bodies = {g: [] for g in GENOMES}
    for g in GENOMES:
        for _ in range(pool):
            genome = create_individual(max_modules, max_depth, symmetry_axis=AXIS[g]).genotype["morph"]
            bodies[g].append({"genome_type": g, "genome": genome, **body_descriptors(genome)})
    by_bin = {g: Counter(b["n_hinges"] for b in bodies[g]) for g in GENOMES}
    common = {h: min(by_bin["tree"][h], by_bin["tree_symmetric"][h]) for h in by_bin["tree"] if h in by_bin["tree_symmetric"]}
    alloc = allocate(common, n)
    rng = np.random.default_rng(seed)
    out = []
    for g in GENOMES:
        k = 0
        for h, m in sorted(alloc.items()):
            members = [b for b in bodies[g] if b["n_hinges"] == h]
            for i in rng.choice(len(members), size=m, replace=False):
                out.append({"body_id": f"rand_{g}_{k:03d}", "body_source": "random", **members[i]})
                k += 1
    return out


def sample_evolved(root: Path) -> list[dict]:
    df = runs.load_runs(root)
    out = []
    for r in df[df["genome"].isin(GENOMES)].sort_values("run").itertuples():
        ckpt = runs.best_checkpoint(Path(r.path))
        if ckpt is None:
            continue
        genome = json.loads((ckpt / "best_genome.json").read_text())
        out.append({"body_id": f"evo_{r.brain}_{r.genome}_s{r.seed}", "body_source": "evolved",
                    "genome_type": r.genome, "genome": genome, **body_descriptors(genome),
                    "run": r.run, "run_brain": r.brain, "run_seed": int(r.seed), "run_score": float(r.score)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["random", "evolved"])
    parser.add_argument("--n", type=int, default=40, help="random: bodies per genome")
    parser.add_argument("--pool", type=int, default=400, help="random: candidates per genome")
    parser.add_argument("--max-modules", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-root", type=Path, help="evolved: E1 output root")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    if args.source == "random":
        bodies = sample_random(args.n, args.pool, args.max_modules, args.max_depth, args.seed)
    else:
        if args.runs_root is None:
            parser.error("evolved needs --runs-root")
        bodies = sample_evolved(args.runs_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for b in bodies:
            fh.write(json.dumps(b) + "\n")
    for g in GENOMES:
        hs = [b["n_hinges"] for b in bodies if b["genome_type"] == g]
        ms = [b["n_modules"] for b in bodies if b["genome_type"] == g]
        if hs:
            print(f"{g:15s} n={len(hs):3d}  hinges mean {np.mean(hs):.2f}  modules mean {np.mean(ms):.2f}")
    print(f"{len(bodies)} bodies -> {args.out}")


if __name__ == "__main__":
    main()
