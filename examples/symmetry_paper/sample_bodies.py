"""
Bodies for the fixed-body learning curves (E2).

  random         --n bodies per genome drawn from the genome's initial distribution
                 (sim.create_individual, as evolve.py's first generation, zoo
                 geometry), matched on hinge count: a pool of --pool bodies per
                 genome is binned by hinge count and the n are taken from the bins
                 both genomes fill, in proportion to the smaller count, so the
                 genomes' hinge-count distributions are identical.
  evolved        the best last-generation body of every tree and tree_symmetric run
                 under --runs-root (E1), with that run's brain, seed and score.
  tied           tree_tied bodies with the hinge counts of the symmetric bodies in
                 --like (one per symmetric body, from a pool of --pool).
  shape_matched  tree bodies matched to the symmetric bodies in --like on rest
                 shape: a pool of --pool tree bodies is measured
                 (analyse_bodies.body_measures) and, within each hinge count, the
                 pool bodies are assigned to the symmetric ones (Hungarian, no
                 reuse) on the standardised SHAPE_MEASURES. Each gets matched_to
                 and match_dist. Tests whether body shape, not mirroring, carries
                 the symmetric advantage.

Writes one JSON line per body: body_id, body_source, genome_type, genome,
n_modules, n_hinges, n_limbs (and run, run_brain, run_seed, run_score for
evolved bodies).

Usage:
    python sample_bodies.py random --n 40 -o bodies_random.jsonl
    python sample_bodies.py evolved --runs-root /scratch/$USER/ariel_paper_e1 -o bodies_evolved.jsonl
    python sample_bodies.py tied --like bodies_random.jsonl --pool 400 -o bodies_tied.jsonl
    python sample_bodies.py shape_matched --like bodies_random.jsonl --pool 3000 --workers 20 -o bodies_shape_matched.jsonl
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import runs
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from replay import body_descriptors
from analyse_bodies import body_measures
from genomes import DEFAULT_TIE_PROB
from sim import create_individual, use_zoo_modules_globally

GENOMES = ("tree", "tree_symmetric", "tree_tied")
AXIS = {"tree": None, "tree_symmetric": MirrorAxis.Y_ZERO, "tree_tied": None}
TIE_PROB = {"tree_tied": DEFAULT_TIE_PROB}
# Rest-shape measures that account for most of the random-body gap (analyse_bodies.py).
SHAPE_MEASURES = ("n_leg", "n_leg_yaw", "contact_len", "leg_reach", "height", "n_pitch")


def random_genome(genome_type: str, max_modules: int, max_depth: int) -> dict:
    return create_individual(max_modules, max_depth, symmetry_axis=AXIS[genome_type],
                             tie_prob=TIE_PROB.get(genome_type)).genotype["morph"]


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
    genomes = ("tree", "tree_symmetric")
    bodies = {g: [] for g in genomes}
    for g in genomes:
        for _ in range(pool):
            genome = random_genome(g, max_modules, max_depth)
            bodies[g].append({"genome_type": g, "genome": genome, **body_descriptors(genome)})
    by_bin = {g: Counter(b["n_hinges"] for b in bodies[g]) for g in genomes}
    common = {h: min(by_bin["tree"][h], by_bin["tree_symmetric"][h]) for h in by_bin["tree"] if h in by_bin["tree_symmetric"]}
    alloc = allocate(common, n)
    rng = np.random.default_rng(seed)
    out = []
    for g in genomes:
        k = 0
        for h, m in sorted(alloc.items()):
            members = [b for b in bodies[g] if b["n_hinges"] == h]
            for i in rng.choice(len(members), size=m, replace=False):
                out.append({"body_id": f"rand_{g}_{k:03d}", "body_source": "random", **members[i]})
                k += 1
    return out


def _pool(genome_type: str, pool: int, max_modules: int, max_depth: int, seed: int) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    use_zoo_modules_globally()
    out = []
    for _ in range(pool):
        genome = random_genome(genome_type, max_modules, max_depth)
        out.append({"genome_type": genome_type, "genome": genome, **body_descriptors(genome)})
    return out


def _like_symmetric(like: Path, genome_type: str = "tree_symmetric") -> list[dict]:
    return [b for b in map(json.loads, like.read_text().splitlines()) if b["genome_type"] == genome_type]


def sample_tied(like: Path, pool: int, max_modules: int, max_depth: int, seed: int) -> list[dict]:
    """One tree_tied body per symmetric body in like, with its hinge count."""
    need = Counter(b["n_hinges"] for b in _like_symmetric(like))
    cands = _pool("tree_tied", pool, max_modules, max_depth, seed)
    rng = np.random.default_rng(seed)
    out = []
    for h, m in sorted(need.items()):
        members = [b for b in cands if b["n_hinges"] == h]
        if len(members) < m:
            raise SystemExit(f"only {len(members)} tied bodies with {h} hinges in the pool for {m}: raise --pool")
        for i in rng.choice(len(members), size=m, replace=False):
            out.append({"body_id": f"rand_tree_tied_{len(out):03d}", "body_source": "random", **members[i]})
    return out


def _measure(genome: dict) -> dict:
    try:
        return body_measures(genome)
    except Exception:  # noqa: BLE001 -- an unbuildable body is just not a candidate
        return {}


def sample_shape_matched(like: Path, pool: int, max_modules: int, max_depth: int, seed: int,
                         workers: int, measures: tuple[str, ...] = SHAPE_MEASURES,
                         target_genome: str = "tree_symmetric", body_source: str = "shape_matched",
                         id_prefix: str = "match_tree") -> list[dict]:
    """Tree bodies matched one-to-one to the target_genome bodies in like on
    hinge count (exactly) and measures (standardised Euclidean distance).
    Defaults: the E2 shape-matched control. E5: target_genome tree_tied,
    measures ("n_leg",), body_source leg_matched."""
    SHAPE = measures
    sym = _like_symmetric(like, target_genome)
    cands = _pool("tree", pool, max_modules, max_depth, seed)
    with Pool(workers) as p:
        sym_m = p.map(_measure, [b["genome"] for b in sym])
        cand_m = p.map(_measure, [b["genome"] for b in cands])
    keep = [i for i, m in enumerate(cand_m) if m]
    cands, cand_m = [cands[i] for i in keep], [cand_m[i] for i in keep]
    S = np.array([[m[k] for k in SHAPE] for m in sym_m], dtype=float)
    C = np.array([[m[k] for k in SHAPE] for m in cand_m], dtype=float)
    sd = np.vstack([S, C]).std(axis=0)
    sd[sd == 0] = 1.0
    out = []
    for h in sorted({b["n_hinges"] for b in sym}):
        si = [i for i, b in enumerate(sym) if b["n_hinges"] == h]
        ci = [i for i, b in enumerate(cands) if b["n_hinges"] == h]
        if len(ci) < len(si):
            raise SystemExit(f"only {len(ci)} tree bodies with {h} hinges in the pool for {len(si)}: raise --pool")
        dist = np.linalg.norm((S[si][:, None] - C[ci][None]) / sd, axis=-1)
        for a, c in zip(*linear_sum_assignment(dist)):
            b = cands[ci[c]]
            out.append({"body_id": f"{id_prefix}_{sym[si[a]]['body_id'].rsplit('_', 1)[1]}", "body_source": body_source,
                        **b, "matched_to": sym[si[a]]["body_id"], "match_dist": float(dist[a, c]),
                        **{f"shape_{k}": cand_m[ci[c]][k] for k in SHAPE},
                        **{f"target_{k}": sym_m[si[a]][k] for k in SHAPE}})
    return sorted(out, key=lambda b: b["body_id"])


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
    parser.add_argument("source", choices=["random", "evolved", "tied", "shape_matched"])
    parser.add_argument("--n", type=int, default=40, help="random: bodies per genome")
    parser.add_argument("--pool", type=int, default=400, help="random: candidates per genome")
    parser.add_argument("--max-modules", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-root", type=Path, help="evolved: E1 output root")
    parser.add_argument("--like", type=Path, help="tied, shape_matched: bodies file whose symmetric bodies to match")
    parser.add_argument("--measures", nargs="+", default=list(SHAPE_MEASURES),
                        help="shape_matched: measures to match on (E5: n_leg)")
    parser.add_argument("--target-genome", default="tree_symmetric",
                        help="shape_matched: genome of the --like bodies to match (E5: tree_tied)")
    parser.add_argument("--body-source", default="shape_matched", help="shape_matched: body_source label (E5: leg_matched)")
    parser.add_argument("--id-prefix", default="match_tree", help="shape_matched: body_id prefix (E5: legmatch_tree)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="shape_matched: measuring processes")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    if args.source in ("tied", "shape_matched") and args.like is None:
        parser.error(f"{args.source} needs --like")
    if args.source == "random":
        bodies = sample_random(args.n, args.pool, args.max_modules, args.max_depth, args.seed)
    elif args.source == "tied":
        bodies = sample_tied(args.like, args.pool, args.max_modules, args.max_depth, args.seed)
    elif args.source == "shape_matched":
        bodies = sample_shape_matched(args.like, args.pool, args.max_modules, args.max_depth, args.seed, args.workers,
                                      tuple(args.measures), args.target_genome, args.body_source, args.id_prefix)
        d = [b["match_dist"] for b in bodies]
        print(f"match distance (standardised): median {np.median(d):.2f}, max {max(d):.2f}")
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
