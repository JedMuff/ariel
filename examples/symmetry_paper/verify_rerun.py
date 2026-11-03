"""
Check that same-seed reruns retrace the original runs.

  fixed bodies   every training under --long must match the same-named training
                 under --short on best_so_far at every generation --short has
  evolution      optional, with --evolution-new/--evolution-old: runs of the same genome
                 and seed must have the same score

A mismatch means the longer curves are not continuations of the old ones (for
example a code change, or floating-point differences between nodes).

Usage:
    python verify_rerun.py --short $FINAL_ROOT/e2 --long $FINAL_ROOT/e2_x3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import runs

TOL = 1e-9


def best_curve(run_dir: Path) -> list[float]:
    with (run_dir / "curve.jsonl").open() as fh:
        return [json.loads(line)["best_so_far"] for line in fh]


def check_fixed(short: Path, long: Path) -> int:
    n = same = missing = 0
    mismatched = []
    for final in sorted(short.glob("*/*/final.json")):
        old = final.parent
        new = long / old.relative_to(short)
        if not (new / "final.json").exists():
            missing += 1
            continue
        a, b = best_curve(old), best_curve(new)
        n += 1
        if len(b) >= len(a) and all(abs(x - y) <= TOL for x, y in zip(a, b)):
            same += 1
        else:
            mismatched.append(str(old.relative_to(short)))
    print(f"fixed bodies: {same}/{n} trainings retrace the original; {missing} not finished yet")
    for m in mismatched[:20]:
        print(f"  differs: {m}")
    return len(mismatched)


def check_evolution(new: Path, old: Path) -> int:
    a = runs.load_runs(new).set_index(["genome", "seed"])["score"]
    b = runs.load_runs(old).set_index(["genome", "seed"])["score"]
    both = a.index.intersection(b.index)
    diff = (a[both] - b[both]).abs()
    print(f"evolution: {(diff <= TOL).sum()}/{len(both)} runs reproduce the original score")
    for key, d in diff[diff > TOL].items():
        print(f"  differs by {d:.4f}: {key}")
    return int((diff > TOL).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--short", type=Path, help="original fixed-body root (holds <set>/<training>/)")
    parser.add_argument("--long", type=Path, help="longer fixed-body root, same layout")
    parser.add_argument("--evolution-new", type=Path)
    parser.add_argument("--evolution-old", type=Path)
    args = parser.parse_args()
    bad = 0
    if args.short and args.long:
        bad += check_fixed(args.short, args.long)
    if args.evolution_new and args.evolution_old:
        bad += check_evolution(args.evolution_new, args.evolution_old)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
