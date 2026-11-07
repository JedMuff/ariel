"""
Experiment 2: train one brain on one fixed body, once per seed.

Takes line --line of a bodies file (sample_bodies.py) and trains --brain on it
for --evals evaluations with every seed, the seeds running concurrently on
one pool of --workers processes.

Output (--out-root/<body_id>__<brain>/):
  config.json     body, brain, evals, seeds, protocol, git commit
  results.jsonl   one line per seed: seed, score (best displacement, m),
                  evals, stopped_early, wall_s, curve {best, median, sigma
                  per CMA-ES generation}
  weights.npz     best weights per seed (key "s<seed>")
Seeds already in results.jsonl are skipped, so a rerun completes the set.

Usage:
    python train_fixed_body.py --bodies bodies.jsonl --line 0 --brain dann --evals 800 \\
        --seeds 42-61 --workers 32 --out-root runs/e2
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import BRAIN_KINDS, PROTOCOL
from evolve import git_commit
from sim import make_pool, train_controllers


def parse_seeds(text: list[str]) -> list[int]:
    """'42-61' or '42 43 44' -> seeds."""
    seeds = []
    for part in text:
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return seeds


def done_seeds(out: Path) -> set[int]:
    path = out / "results.jsonl"
    return {json.loads(x)["seed"] for x in path.read_text().splitlines()} if path.exists() else set()


def train(body: dict, brain: str, evals: int, seeds: list[int], workers: int, out: Path, concurrent: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"experiment": "e2", **{k: body[k] for k in body if k != "genome"}, "brain": brain, "evals": evals,
           "seeds": seeds, "protocol": asdict(PROTOCOL), "git_commit": git_commit(), "genome": body["genome"]}
    if not (out / "config.json").exists():
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
    todo = [s for s in seeds if s not in done_seeds(out)]
    if not todo:
        print(f"{out.name}: done", flush=True)
        return
    weights_path = out / "weights.npz"
    with make_pool(workers) as pool:
        for i in range(0, len(todo), concurrent):
            batch = todo[i:i + concurrent]
            results = train_controllers([(body["genome"], brain, evals, s) for s in batch], pool, len(batch))
            weights = dict(np.load(weights_path)) if weights_path.exists() else {}
            weights |= {f"s{s}": r.weights for s, r in zip(batch, results)}
            np.savez_compressed(out / "weights.tmp.npz", **weights)
            (out / "weights.tmp.npz").replace(weights_path)
            with (out / "results.jsonl").open("a") as fh:
                for s, r in zip(batch, results):
                    fh.write(json.dumps({"seed": s, "score": r.score, "evals": r.evals,
                                         "stopped_early": r.stopped_early, "wall_s": round(r.wall_s, 2),
                                         "curve": {k: [round(x, 5) for x in v] for k, v in r.curve.items()}}) + "\n")
            print(f"{out.name}: seeds {batch[0]}-{batch[-1]}: "
                  f"{', '.join(f'{r.score:.3f}' for r in results)} m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2: one brain on one fixed body, many seeds")
    ap.add_argument("--bodies", type=Path, required=True)
    ap.add_argument("--line", type=int, required=True, help="0-based line of --bodies")
    ap.add_argument("--brain", choices=BRAIN_KINDS, required=True)
    ap.add_argument("--evals", type=int, default=800)
    ap.add_argument("--seeds", nargs="+", default=["42-61"], help="e.g. 42-61 or 42 43 44")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--concurrent", type=int, default=20, help="seeds trained at once")
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()
    body = json.loads(args.bodies.read_text().splitlines()[args.line])
    train(body, args.brain, args.evals, parse_seeds(args.seeds), args.workers,
          args.out_root / f"{body['body_id']}__{args.brain}", args.concurrent)


if __name__ == "__main__":
    main()
