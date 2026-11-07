"""
Experiment 2: random bodies, drawn from each genome's initial distribution
(genomes.random_body, as evolve.py's first generation: redrawn until valid).

Writes one JSON line per body: body_id, genome_type, index, genome, rejected
(invalid draws before it), and the body measures of genomes.describe
(n_modules, n_hinges, symmetry, tied_frac).

Usage:
    python sample_bodies.py --n 80 -o bodies.jsonl                 # every genome
    python sample_bodies.py --n 80 --genomes tree grn -o bodies.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import genomes
from config import GENOME_TYPES


def sample(genome_type: str, n: int, seed: int) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    ops = genomes.genome_ops(genome_type)
    out = []
    for k in range(n):
        g, rejected = genomes.random_body(ops)
        out.append({"body_id": f"{genome_type}_{k:03d}", "genome_type": genome_type, "index": k,
                    "rejected": rejected, **genomes.describe(g), "genome": g})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=80, help="bodies per genome")
    ap.add_argument("--genomes", nargs="+", choices=GENOME_TYPES, default=list(GENOME_TYPES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for g in args.genomes:
        bodies = sample(g, args.n, args.seed + 1000 * GENOME_TYPES.index(g))
        rows += bodies
        acc = args.n / (args.n + sum(b["rejected"] for b in bodies))
        print(f"{g:15s} modules {np.mean([b['n_modules'] for b in bodies]):5.1f}  "
              f"hinges {np.mean([b['n_hinges'] for b in bodies]):5.1f}  valid draws {acc:.0%}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{len(rows)} bodies -> {args.out}")


if __name__ == "__main__":
    main()
