"""
The run lists the Slurm scripts work through.

  e1     one line per Experiment 1 run: genome brain evals seed out-subdir
         (5 genomes x 4 brains x 3 budgets x --seeds, out-subdir
         e1/<genome>/<brain>/e<evals>/s<seed>)
  pilot  the same with one seed per condition (pilot/<genome>/...)
  e2     one line per Experiment 2 job: body-line brain (every body of
         --bodies with every brain)

Lines are sorted so the expensive runs (800 evaluations, ANN) come first,
which keeps the tail of an array short.

Usage:
    python conditions.py e1 --seeds 20 > e1.tsv
    python conditions.py e1 --seeds 20 --evals 800 > e1_800.tsv
    python conditions.py pilot > pilot.tsv
    python conditions.py e2 --bodies bodies.jsonl > e2.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import BRAIN_KINDS, BUDGETS, GENOME_TYPES

FIRST_SEED = 42
COST = {"ann": 1.3}  # relative cost per evaluation, for ordering only


def e1_lines(seeds: int, budgets: list[int], prefix: str) -> list[str]:
    rows = [(g, b, e, FIRST_SEED + s) for e in budgets for b in BRAIN_KINDS for g in GENOME_TYPES for s in range(seeds)]
    rows.sort(key=lambda r: (-r[2] * COST.get(r[1], 1.0), r[3], r[1], r[0]))
    return [f"{g}\t{b}\t{e}\t{s}\t{prefix}/{g}/{b}/e{e}/s{s}" for g, b, e, s in rows]


def e2_lines(bodies: Path) -> list[str]:
    n = len(bodies.read_text().splitlines())
    rows = [(i, b) for b in BRAIN_KINDS for i in range(n)]
    rows.sort(key=lambda r: -COST.get(r[1], 1.0))
    return [f"{i}\t{b}" for i, b in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("e1", "pilot", "e2"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--evals", type=int, nargs="+", default=list(BUDGETS))
    ap.add_argument("--bodies", type=Path, help="e2: sample_bodies.py output")
    args = ap.parse_args()
    if args.what == "e2":
        if args.bodies is None:
            ap.error("e2 needs --bodies")
        lines = e2_lines(args.bodies)
    elif args.what == "pilot":
        lines = e1_lines(1, args.evals, "pilot")
    else:
        lines = e1_lines(args.seeds, args.evals, "e1")
    sys.stdout.write("".join(x + "\n" for x in lines))


if __name__ == "__main__":
    main()
