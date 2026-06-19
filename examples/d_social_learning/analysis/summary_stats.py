"""Generate summary.csv from ARIEL and EvoGym experiment databases.

Usage:
    uv run examples/d_social_learning/analysis/summary_stats.py \
        [--data-dir __data__/social] [--out __data__/social/summary.csv]

Reads both ARIEL and EvoGym databases via the same Archive.by_generation() API
(identical SQLite schema from ariel.ec and evogym/db.py respectively).
"""

import argparse
import csv
import sys
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).parent        # d_social_learning/analysis/
_SOCIAL_DIR = _ANALYSIS_DIR.parent           # d_social_learning/
_EVOGYM_DIR = _SOCIAL_DIR / "evogym"         # contains db.py

# Add evogym/ subdir only (not _SOCIAL_DIR, which contains ariel/ and evogym/
# subdirectories that shadow the installed packages).
for _p in [str(_EVOGYM_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMES = [
    "darwinian", "lamarckian", "random", "random_many",
    "best", "best_many", "similar", "similar_many",
]
X_VALUES = [0.0, 0.5, 1.0]
DOMAINS = ["ariel", "evogym"]


def _x_str(x: float) -> str:
    return str(x).replace(".", "")


def _load_archive(db_path: Path, domain: str):
    if domain == "ariel":
        from ariel.ec.archive import Archive
    else:
        from db import Archive
    return Archive(db_path)


def process_run(db_path: Path, domain: str, scheme: str, x: float, rep: int) -> list[dict]:
    try:
        archive = _load_archive(db_path, domain)
        gen_min, gen_max = archive.generation_range
    except Exception as exc:
        print(f"  [skip] {db_path}: {exc}")
        return []

    rows = []
    for gen in range(gen_min, gen_max + 1):
        pop = archive.by_generation(gen)
        if not pop:
            continue
        fitnesses = [ind.fitness_ for ind in pop if ind.fitness_ is not None]
        distances = [ind.tags_.get("distance", 0.0) for ind in pop if ind.tags_]
        novelties = [ind.tags_.get("novelty", 0.0) for ind in pop if ind.tags_]
        if not fitnesses:
            continue
        rows.append({
            "domain": domain,
            "scheme": scheme,
            "x": x,
            "rep": rep,
            "gen": gen,
            "best_fitness": max(fitnesses),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "best_distance": max(distances) if distances else 0.0,
            "mean_novelty": sum(novelties) / len(novelties) if novelties else 0.0,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="__data__/social")
    parser.add_argument("--out", default="__data__/social/summary.csv")
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["domain", "scheme", "x", "rep", "gen",
                  "best_fitness", "mean_fitness", "best_distance", "mean_novelty"]

    all_rows: list[dict] = []

    for domain in DOMAINS:
        for scheme in SCHEMES:
            for x in X_VALUES:
                for rep in range(args.reps):
                    db_path = data_dir / domain / scheme / f"x{_x_str(x)}" / f"rep_{rep}" / "database.db"
                    if not db_path.exists():
                        continue
                    print(f"  {domain}/{scheme}/x{x}/rep{rep}")
                    rows = process_run(db_path, domain, scheme, x, rep)
                    all_rows.extend(rows)

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
