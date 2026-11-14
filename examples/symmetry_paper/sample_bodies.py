"""
Experiment 2: random bodies, drawn from each genome's initial distribution
(genomes.random_body, as evolve.py's first generation: redrawn until valid).

Writes one JSON line per body: body_id, genome_type, index, genome, rejected
(invalid draws before it), and the body measures of genomes.describe
(n_modules, n_hinges, symmetry, tied_frac).

Experiment 2b (--like): shape-matched random bodies. For each genome in
--genomes, a pool of --pool random bodies (the same distribution) is drawn in
parallel, in chunks with their own seeds, and its bodies with a hinge count
the targets need are measured at rest (bodies.body_measures) and for mirror
asymmetry (bodies.mirror_asymmetry). Within each hinge count, pool bodies are
assigned one-to-one to the target bodies of --like (the tree_symmetric bodies
of the E2 file; Hungarian assignment, no reuse) on the SHAPE measures,
standardised by the targets' SD. Each matched body gets body_id
<genome>_m<target index>, matched_to, match_dist, asymmetry and shape_* /
target_* measures. A sidecar <out>.shape.json holds the measures, the scale
and every target's measures, so the analysis uses the same space. A balance
table (standardised mean difference per measure and module count, matched
minus targets, and the matched bodies' mean asymmetry) is printed and stored in the sidecar.

Usage:
    python sample_bodies.py --n 80 -o bodies.jsonl                 # every genome
    python sample_bodies.py --n 80 --genomes tree grn -o bodies.jsonl
    python sample_bodies.py --like e2/bodies.jsonl --genomes tree tree_tied tree_tied_full grn \\
        --pool 20000 --workers 32 -o e2m/bodies.jsonl
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing as mp
import random
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import genomes
from config import GENOME_TYPES

TARGET = "tree_symmetric"
SHAPE = ("n_leg", "n_leg_yaw", "contact_len", "leg_reach", "height", "n_pitch")
CHUNK = 100  # pool bodies per worker task (one seed each)


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


# ── Experiment 2b: shape matching ─────────────────────────────────────────────


def measure(genome: dict) -> dict | None:
    """Rest-pose shape measures and mirror asymmetry; None if the body cannot be simulated."""
    from bodies import body_measures, mirror_asymmetry

    try:
        m = body_measures(genome)
        return {**{k: m[k] for k in SHAPE}, "asymmetry": mirror_asymmetry(genome)}
    except Exception:  # noqa: BLE001 -- an unbuildable body is just not a candidate
        return None


def _pool_chunk(task: tuple[str, int, int, tuple[int, ...]]) -> list[dict]:
    """CHUNK random bodies of one genome from one seed; those with a needed hinge count are measured."""
    genome_type, seed, n, hinges = task
    random.seed(seed)
    np.random.seed(seed)
    ops = genomes.genome_ops(genome_type)
    out = []
    for _ in range(n):
        g, _ = genomes.random_body(ops)
        d = genomes.describe(g)
        if d["n_hinges"] in hinges and (m := measure(g)) is not None:
            out.append({"genome_type": genome_type, **d, **m, "pool_seed": seed, "genome": g})
    return out


def scale(targets: np.ndarray) -> np.ndarray:
    sd = targets.std(axis=0)
    sd[sd == 0] = 1.0
    return sd


def match(targets: list[dict], pool: list[dict], sd: np.ndarray) -> list[dict]:
    """One pool body per target, with its hinge count, closest on SHAPE / sd (Hungarian, no reuse)."""
    out = []
    for h in sorted({t["n_hinges"] for t in targets}):
        ti = [i for i, t in enumerate(targets) if t["n_hinges"] == h]
        ci = [i for i, c in enumerate(pool) if c["n_hinges"] == h]
        if len(ci) < len(ti):
            raise SystemExit(f"{pool[0]['genome_type'] if pool else '?'}: only {len(ci)} pool bodies with {h} "
                             f"hinges for {len(ti)} targets: raise --pool")
        t = np.array([[targets[i][k] for k in SHAPE] for i in ti], float)
        c = np.array([[pool[i][k] for k in SHAPE] for i in ci], float)
        dist = np.linalg.norm((t[:, None] - c[None]) / sd, axis=-1)
        for a, b in zip(*linear_sum_assignment(dist)):
            tgt, body = targets[ti[a]], pool[ci[b]]
            out.append({**{k: v for k, v in body.items() if k not in SHAPE},
                        "matched_to": tgt["body_id"], "match_dist": float(dist[a, b]),
                        **{f"shape_{k}": body[k] for k in SHAPE}, **{f"target_{k}": tgt[k] for k in SHAPE},
                        "target_index": tgt["index"]})
    return out


def balance(matched: list[dict], targets: list[dict], sd: np.ndarray) -> dict[str, float]:
    """Standardised mean difference (matched - targets) per measure and for the module
    count; the matched bodies' mean asymmetry (the targets' is 0, so no SMD)."""
    out = {k: (np.mean([m[f"shape_{k}"] for m in matched]) - np.mean([t[k] for t in targets])) / s
           for k, s in zip(SHAPE, sd)}
    tv = np.array([t["n_modules"] for t in targets], float)
    out["n_modules"] = (np.mean([m["n_modules"] for m in matched]) - tv.mean()) / tv.std()
    out["asymmetry_mean"] = np.mean([m["asymmetry"] for m in matched])
    return {k: round(float(v), 3) for k, v in out.items()}


def sample_matched(like: Path, genome_types: list[str], pool: int, workers: int, seed: int) -> tuple[list[dict], dict]:
    targets = [b for b in map(json.loads, like.read_text().splitlines()) if b["genome_type"] == TARGET]
    ctx = mp.get_context("forkserver")
    with ctx.Pool(workers) as p:
        tm = p.map(measure, [t["genome"] for t in targets], chunksize=1)
        if any(m is None for m in tm):
            raise SystemExit("a target body cannot be measured")
        targets = [{**t, **m} for t, m in zip(targets, tm)]
        sd = scale(np.array([[t[k] for k in SHAPE] for t in targets], float))
        hinges = tuple(sorted({t["n_hinges"] for t in targets}))
        need = Counter(t["n_hinges"] for t in targets)
        rows, side = [], {"measures": SHAPE, "scale": dict(zip(SHAPE, sd.tolist())), "target": TARGET,
                          "pool": pool, "seed": seed, "balance": {}, "pool_hinges": {},
                          "targets": [{"body_id": t["body_id"], "n_hinges": t["n_hinges"], "n_modules": t["n_modules"],
                                       "asymmetry": t["asymmetry"], **{k: t[k] for k in SHAPE}} for t in targets]}
        for g in genome_types:
            gi = GENOME_TYPES.index(g)
            tasks = [(g, 1_000_000 * (seed + 1) + 10_000 * gi + c, min(CHUNK, pool - c * CHUNK), hinges)
                     for c in range(-(-pool // CHUNK))]
            cands = [b for chunk in p.imap(_pool_chunk, tasks) for b in chunk]
            have = Counter(c["n_hinges"] for c in cands)
            side["pool_hinges"][g] = {str(h): have[h] for h in hinges}
            short = {h: (have[h], n) for h, n in need.items() if have[h] < n}
            if short:
                raise SystemExit(f"{g}: pool too small for hinge counts {short} (have, need): raise --pool")
            matched = match(targets, cands, sd)
            for m in matched:
                m["body_id"] = f"{g}_m{m['target_index']:03d}"
                m["index"] = m.pop("target_index")
                m["rejected"] = None
            matched.sort(key=lambda m: m["index"])
            side["balance"][g] = balance(matched, targets, sd)
            rows += matched
            print(f"{g:15s} pool {len(cands):6d} usable  balance (SMD) "
                  + "  ".join(f"{k} {v:+.2f}" for k, v in side["balance"][g].items()), flush=True)
    return rows, side


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=80, help="bodies per genome")
    ap.add_argument("--genomes", nargs="+", choices=GENOME_TYPES, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--like", type=Path, help="E2 bodies file: match --genomes to its tree_symmetric bodies")
    ap.add_argument("--pool", type=int, default=20000, help="--like: random bodies drawn per genome")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="--like: processes")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.like is not None:
        gs = args.genomes or [g for g in GENOME_TYPES if g != TARGET]
        rows, side = sample_matched(args.like, gs, args.pool, args.workers, args.seed)
        args.out.with_suffix(".shape.json").write_text(json.dumps(side, indent=1))
    else:
        rows = []
        for g in args.genomes or list(GENOME_TYPES):
            bodies = sample(g, args.n, args.seed + 1000 * GENOME_TYPES.index(g))
            rows += bodies
            acc = args.n / (args.n + sum(b["rejected"] for b in bodies))
            print(f"{g:15s} modules {np.mean([b['n_modules'] for b in bodies]):5.1f}  "
                  f"hinges {np.mean([b['n_hinges'] for b in bodies]):5.1f}  valid draws {acc:.0%}", flush=True)
    with args.out.open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{len(rows)} bodies -> {args.out}")


if __name__ == "__main__":
    main()
