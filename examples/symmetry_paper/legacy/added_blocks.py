"""
Does adding bricks at two places cost less than adding one?

The theory: a tied genome places added blocks in two places rather than one, which
often costs less fitness, so good lineages survive better. Evolution cannot test
this (pure additions are about 0.5% of asexual children), so this makes the
mutations directly, on the same parents.

Parents: the best --per-run bodies of each E1 bang-bang run's final population.
Arms, all with bricks only, so hinge count and controller size do not change:
  identical  no change: retraining alone (training luck, regression to the mean)
  one        one brick at one random free site
  two_same   two bricks with the same rotation at two sites (the theory)
  two_diff   two bricks with different rotations at two sites (is identity needed?)
  chain      two bricks stacked at one site (same mass, one place)

Pre-stated primary test, before any result: two_same - one > 0.

--mode train     each mutant retrained from scratch for 200 evaluations, as in
                 evolution. This is the score selection would see.
--mode replay    the parent's own controller on each mutant, one episode. No
                 training noise, so it measures the immediate physical damage.
--mode analyse   tables, with parents as the pairing and runs (seeds) as the
                 bootstrap unit, since two parents of a run are related.

Writes <out>/train.csv, replay.csv, and the tables train_*.csv, replay_*.csv.

Usage:
    python added_blocks.py --mode replay --data __data__/ariel_paper_v2 -o OUT
    python added_blocks.py --mode train  --data __data__/ariel_paper_v2 -o OUT
    python added_blocks.py --mode analyse -o OUT
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import json
import random
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import runs
from ariel.body_phenotypes.robogen_lite.config import ALLOWED_ROTATIONS, ModuleType
from ariel.ec.genotypes.tree.collision import try_place
from ariel.ec.genotypes.tree.operators import add_node
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.tying import TiedGenome, _body_checker, _free_sites
from mechanics import knockout_episode
from sim import PROFILE, build_world, profile_for, train_skill_for_body, use_zoo_modules_globally, zoo_spec_fn

ARMS = ("identical", "one", "two_same", "two_diff", "chain")
TRAIN_REPS, REPLAY_REPS = 4, 20
GENS = 10   # CMA-ES generations of the E1 runs: 200 evaluations
BRICK_ROTATIONS = [r.name for r in ALLOWED_ROTATIONS[ModuleType.BRICK]]


# ── Mutants ───────────────────────────────────────────────────────────────────


def add_bricks(genome: dict, arm: str, rng: random.Random) -> Optional[tuple[dict, list[tuple[int, str]]]]:
    """(the genome with bricks added, their sites), or None if they do not fit.
    The result has no ties: it is only evaluated, never bred."""
    tree = TreeGenome.from_dict({"nodes": copy.deepcopy(genome["nodes"]), "edges": copy.deepcopy(genome["edges"])})
    if arm == "identical":
        return tree.to_dict(), []
    g = TiedGenome(tree, [])
    checker = _body_checker(tree)
    next_id = max(tree.nodes) + 1
    rot_a = rng.choice(BRICK_ROTATIONS)
    rot_b = rng.choice([r for r in BRICK_ROTATIONS if r != rot_a])
    sites = [s for s in _free_sites(g) if s[0] in checker.frames]
    rng.shuffle(sites)

    def place_first(rot: str, candidates: list[tuple[int, str]]) -> Optional[tuple[int, tuple[int, str]]]:
        nonlocal next_id
        for pid, face in candidates:
            if try_place(checker, next_id, pid, face, "BRICK", rot):
                add_node(tree, pid, face, next_id, "BRICK", rot)
                next_id += 1
                return next_id - 1, (pid, face)
        return None

    first = place_first(rot_a, sites)
    if first is None:
        return None
    new, site = first
    placed = [site]
    if arm == "one":
        return tree.to_dict(), placed
    if arm == "chain":
        on_it = _free_sites(g, {new})
        rng.shuffle(on_it)
        second = place_first(rot_b, on_it)
    else:
        rest = [s for s in sites if s != site and s[0] in checker.frames]
        second = place_first(rot_a if arm == "two_same" else rot_b, rest)
    if second is None:
        return None
    placed.append(second[1])
    return tree.to_dict(), placed


def parents_of(data: Path, per_run: int) -> list[dict]:
    out = []
    for run_dir in sorted((data / "e1").glob("e1_bang_bang_tree*")):
        cfg = runs.run_config(run_dir)
        final = runs.final_checkpoints(run_dir)
        rated = sorted((json.loads((c / "meta.json").read_text())["fitness"], str(c)) for c in final)
        threshold = -rated[-1][0]
        out += [{"run": run_dir.name, "genome": cfg["genome_type"], "seed": int(cfg["seed"]), "rank": r, "ckpt": c,
                 "parent_score": -f, "entry_threshold": threshold} for r, (f, c) in enumerate(rated[:per_run])]
    return out


def _seed(*parts: object) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode())


def _mutant(parent: dict, arm: str, rep: int, genome: dict) -> Optional[tuple[dict, list]]:
    rng = random.Random(_seed(parent["run"], parent["rank"], arm, rep))
    return add_bricks(genome, arm, rng)


def _job(args: tuple[str, dict, str, int]) -> Optional[dict]:
    mode, parent, arm, rep = args
    use_zoo_modules_globally()
    ckpt = Path(parent["ckpt"])
    cfg = runs.run_config(ckpt.parents[4])
    genome = json.loads((ckpt / "best_genome.json").read_text())
    mutant = _mutant(parent, arm, rep, genome)
    base = {k: parent[k] for k in ("run", "genome", "seed", "rank", "parent_score", "entry_threshold")} | {"arm": arm, "rep": rep}
    if mutant is None:
        return {**base, "skipped": True}
    child, sites = mutant
    to_spec = zoo_spec_fn(runs.hinge_actuator(cfg))
    model, _ = build_world(child, to_spec)
    parent_model, _ = build_world(genome, to_spec)
    if model.nu != parent_model.nu:
        return {**base, "skipped": True, "reason": "hinge count changed"}
    if mode == "replay":
        weights = np.load(ckpt / "fwd_weights.npy")
        score = knockout_episode(cfg, child, weights, "bang_bang", None)
        return {**base, "skipped": False, "score": score, "sites": len(sites), "modules": len(child["nodes"])}
    profile = replace(profile_for(cfg), skill_budget=GENS)
    _, curve, seconds, _ = train_skill_for_body(child, GENS, 1, _seed("train", parent["run"], parent["rank"], arm, rep) % 2**31,
                                                to_spec_fn=to_spec, brain_kind="bang_bang", profile=profile)
    return {**base, "skipped": False, "score": -min(curve), "sites": len(sites), "modules": len(child["nodes"]), "seconds": seconds}


def sweep(mode: str, data: Path, out: Path, per_run: int, workers: int, limit: int) -> None:
    path = out / f"{mode}.csv"
    done = set()
    if path.exists():
        prev = pd.read_csv(path)
        done = set(zip(prev["run"], prev["rank"], prev["arm"], prev["rep"]))
    reps = TRAIN_REPS if mode == "train" else REPLAY_REPS
    jobs = [(mode, p, a, r) for p in parents_of(data, per_run) for a in ARMS for r in range(reps)
            if (p["run"], p["rank"], a, r) not in done]
    if limit:
        jobs = jobs[:limit]
    print(f"{mode}: {len(jobs)} jobs, {len(done)} done", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            pd.DataFrame([fut.result()]).reindex(
                columns=["run", "genome", "seed", "rank", "parent_score", "entry_threshold", "arm", "rep", "skipped", "score", "sites", "modules", "seconds"]
            ).to_csv(path, mode="a", header=header, index=False)
            header = False
            if i % 200 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)


# ── Analysis ──────────────────────────────────────────────────────────────────


def contrast(wide: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> dict:
    """Mean a - b over parents, bootstrap over runs."""
    d = (wide[a] - wide[b]).dropna()
    by_run = d.groupby("run").mean().to_numpy()
    boot = rng.choice(by_run, (10_000, len(by_run))).mean(1)
    return {"contrast": f"{a} - {b}", "gap": d.mean(), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
            "runs": len(by_run), "runs_positive": int((by_run > 0).sum())}


def analyse_one(df: pd.DataFrame, name: str, out: Path, rng: np.random.Generator) -> None:
    print(f"\n===== {name} =====")
    ok = df[~df["skipped"].astype(bool)].copy()
    print("skipped mutants:", df.groupby("arm")["skipped"].apply(lambda s: s.astype(bool).mean()).round(3).to_dict())
    ok["near"] = ok["score"] >= ok["parent_score"] - 0.25
    ok["above"] = ok["score"] > ok["parent_score"]
    ok["would_enter"] = ok["score"] > ok["entry_threshold"]
    keys = ["genome", "run", "rank"]
    wide = ok.pivot_table(index=keys, columns="arm", values="score", aggfunc="mean").reset_index(level="genome")
    wide.index = wide.index.droplevel("rank").rename("run")
    means = ok.groupby(["genome", "arm"]).agg(score=("score", "mean"), parent=("parent_score", "mean"), near=("near", "mean"),
                                              above=("above", "mean"), would_enter=("would_enter", "mean"), n=("score", "size")).reset_index()
    means["loss_from_parent"] = means["parent"] - means["score"]
    means.to_csv(out / f"{name}_means.csv", index=False)
    print(means.round(3).to_string(index=False))

    rows = []
    pairs = [("two_same", "one"), ("two_same", "two_diff"), ("two_same", "chain"), ("two_diff", "chain"),
             ("one", "identical"), ("two_same", "identical"), ("two_diff", "identical"), ("chain", "identical")]
    for genome in (*runs.GENOME_LABEL, "all"):
        w = wide if genome == "all" else wide[wide["genome"] == genome]
        rows += [{"parents": genome, **contrast(w, a, b, rng)} for a, b in pairs]
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{name}_contrasts.csv", index=False)
    print(table.round(3).to_string(index=False))

    # Across genomes: is the same mutation less costly on tied than on tree parents?
    rows = []
    wide2 = wide.copy()
    for arm in ARMS[1:]:
        wide2["cost"] = wide2["identical"] - wide2[arm]
        by = wide2.groupby(["genome", "seed_key"] if "seed_key" in wide2 else ["genome", wide2.index])["cost"].mean().reset_index()
        by["seed"] = by.iloc[:, 1].str.extract(r"_s(\d+)_")[0].astype(int)
        p = by.pivot_table(index="seed", columns="genome", values="cost")
        d = (p["tree_tied"] - p["tree"]).dropna().to_numpy()
        boot = rng.choice(d, (10_000, len(d))).mean(1)
        rows.append({"arm": arm, "cost_tree": p["tree"].mean(), "cost_tied": p["tree_tied"].mean(), "cost_sym": p["tree_symmetric"].mean(),
                     "tied_minus_tree": d.mean(), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
    across = pd.DataFrame(rows)
    across.to_csv(out / f"{name}_across_genomes.csv", index=False)
    print("Cost relative to the identical arm, per parent genome:")
    print(across.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("replay", "train", "analyse"), required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--per-run", type=int, default=2)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--limit", type=int, default=0, help="test: only this many jobs")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "analyse":
        pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
        rng = np.random.default_rng(0)
        for name in ("replay", "train"):
            p = args.out_dir / f"{name}.csv"
            if p.exists():
                analyse_one(pd.read_csv(p), name, args.out_dir, rng)
        return
    sweep(args.mode, args.data, args.out_dir, args.per_run, args.workers, args.limit)


if __name__ == "__main__":
    main()
