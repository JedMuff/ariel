"""Body symmetry of every individual in the zoo fe10 runs, plus random genomes.

Scores each individual's body with sim_ariel/symmetry.py (geometric_symmetry:
sym, strict, paired, plane; tree_symmetry about both axes) and writes, under
--out-dir:
    individuals.csv  one row per individual and run: run keys (budget,
                     scheme, x, k, rep), id, parent_id, time_of_birth,
                     time_of_death, distance (mean of the final evaluations),
                     init_fitness, module counts and the symmetry scores
    survivors.csv    (budget, scheme, x, k, rep, generation, individual_id)
                     of every generation's survivors, from novelty_history
    random.csv       the same scores for random genomes, --random per added
                     module count 1..--random-max-modules, grown like each
                     run's initial population (random_tree, which
                     random_individual calls with 12 added modules only, so
                     every initial body has 13): a baseline per body size

Bodies are built with the zoo modules (sim preset "zoo"), as in the runs.

Usage:
    uv run examples/d_social_learning/analysis/symmetry_scores.py \\
        --root __data__/social/ariel_zoo_fe10 --out-dir __data__/social/symmetry
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

# Appended, not prepended: d_social_learning/ariel would shadow the ariel package.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from curve_utils import read_individuals  # noqa: E402
from core.novelty_history import read_run_history  # noqa: E402

RUN_RE = re.compile(r"(?P<budget>ig\d+)/(?P<scheme>[^/]+)/x(?P<x>\d+)/k(?P<k>\d+)/rep_(?P<rep>\d+)_")


def _init_worker() -> None:
    import simulator_dependent_functions as sdf
    sdf.simulator = sdf.SIMULATOR_ARIEL
    sdf.set_sim_preset("zoo")


def body_scores(morph: dict) -> dict:
    from sim_ariel.symmetry import geometric_symmetry, tree_symmetry
    types = [n["type"] for n in morph["nodes"].values()]
    return {
        "n_modules": len(types),
        "n_hinges": types.count("HINGE"),
        "n_bricks": types.count("BRICK"),
        **geometric_symmetry(morph),
        "tree_y": tree_symmetry(morph, "y_zero"),
        "tree_d": tree_symmetry(morph, "x_equals_y"),
    }


def _score_run(rep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = RUN_RE.search(rep_dir.as_posix()).groupdict()
    keys["x"] = int(keys["x"]) / 10
    df = read_individuals(rep_dir / "database.db", "time_of_birth, time_of_death, genotype_, tags_", "1")
    cache: dict[str, dict] = {}
    rows = []
    for r in df.itertuples():
        morph = json.loads(r.genotype_)["morph"]
        tags = json.loads(r.tags_) if r.tags_ else {}
        key = json.dumps(morph, sort_keys=True)
        if key not in cache:
            cache[key] = body_scores(morph)
        rows.append({**keys, "id": r.id, "parent_id": tags.get("parent_id"),
                     "time_of_birth": r.time_of_birth, "time_of_death": r.time_of_death,
                     "distance": tags.get("distance"), "init_fitness": tags.get("init_fitness"),
                     **cache[key]})
    hist = read_run_history(rep_dir)
    hist = hist[hist.survivor.astype(bool)].drop_duplicates(["generation", "individual_id"])
    surv = hist[["generation", "individual_id"]].assign(**keys)
    return pd.DataFrame(rows), surv


def _score_random(job: tuple[int, int]) -> dict:
    from ariel.ec.genotypes.tree.operators import random_tree
    added, seed = job
    random.seed(seed)
    return {"added": added, "seed": seed, **body_scores(random_tree(added).to_dict())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="__data__/social/ariel_zoo_fe10")
    parser.add_argument("--out-dir", default="__data__/social/symmetry")
    parser.add_argument("--random", type=int, default=1000, help="random genomes per size for the baseline")
    parser.add_argument("--random-max-modules", type=int, default=20)
    parser.add_argument("--random-only", action="store_true", help="only (re)write random.csv")
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N runs (testing)")
    args = parser.parse_args()

    runs = [] if args.random_only else sorted(
        p.parent for p in Path(args.root).glob("ig*/*/x*/k*/rep_*/database.db"))[: args.limit]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=_init_worker) as pool:
        if runs:
            results = pool.map(_score_run, runs, chunksize=1)
            pd.concat([r[0] for r in results]).to_csv(out / "individuals.csv", index=False)
            pd.concat([r[1] for r in results]).to_csv(out / "survivors.csv", index=False)
        if args.random:
            jobs = [(m, m * args.random + i) for m in range(1, args.random_max_modules + 1) for i in range(args.random)]
            pd.DataFrame(pool.map(_score_random, jobs, chunksize=64)).to_csv(out / "random.csv", index=False)
    print(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
