"""Package the final ARIEL and EvoGym data (reps 0-9, the runs behind the
paper's figures) as three download tiers per domain:

    <out>/<domain>/<domain>_1_core.zip              per-individual CSVs, runs.csv,
                                                    derived/ figure inputs
    <out>/<domain>/<domain>_2_learning_curves.zip   per-individual CMA-ES curves
    <out>/<domain>/<domain>_3_raw_databases.zip     the original SQLite run databases
    <out>/<domain>/SHA256SUMS

One README covers everything: <out>/README.md (from
examples/d_social_learning/publication/README.md).
All tiers share the key run_id + individual_id.

Sources:
  ARIEL   tier 1: __data__/final_results/ariel (export_final_results.py output),
                  filtered to the chosen reps
          tiers 2-3: the run directories under __data__/social/ariel_zoo_fe10,
                  matched to runs.csv by their start time stamp
  EvoGym  all tiers: __data__/final_results/evogym/social/evogym/<scheme>/<x>/k8/
                  rep_N_novSTRUCT/database.db (+ the two summary CSVs next to it)
  derived: __data__/social/figures/final_results/{cache/*.csv,stats.csv}

Usage (from the repo root):
    uv run examples/d_social_learning/analysis/export_publication.py --domain ariel --tiers 1 2
    uv run examples/d_social_learning/analysis/export_publication.py --domain evogym --tiers 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_final_results import _db_files, parse_reps  # noqa: E402

FINAL = Path("__data__/final_results")
ARIEL_RAW = Path("__data__/social/ariel_zoo_fe10")
EVOGYM_BASE = FINAL / "evogym/social"
FIGS = Path("__data__/social/figures/final_results")
TEMPLATES = Path(__file__).resolve().parent.parent / "publication"
X_DIRS = {"x00": 0.0, "x05": 0.5, "x10": 1.0}
EVOGYM_SCHEMES = ["darwinian", "lamarckian", "random", "random_many", "best", "best_many",
                  "similar_STRUCT", "similar_many_STRUCT"]
EVOGYM_DESCRIPTOR = ["relative_activity", "size", "compactness", "elongation", "symmetry"]
TIER_NAMES = {1: "1_core", 2: "2_learning_curves", 3: "3_raw_databases"}
# The paper's scheme names, used for the scheme columns, run_ids and folders
PAPER_NAMES = {"darwinian": "Darwinian", "lamarckian": "Lamarckian", "random": "Random",
               "random_many": "Random-Many", "best": "Best", "best_many": "Best-Many",
               "similar_TOPDOWN": "Similar", "similar_many_TOPDOWN": "Similar-Many",
               "similar_STRUCT": "Similar", "similar_many_STRUCT": "Similar-Many"}


def xdir(x: float) -> str:
    return f"x{round(x * 10):02d}"


# ----------------------------------------------------------------------------- ARIEL

def ariel_runs(reps: list[int]) -> pd.DataFrame:
    """runs.csv rows for the chosen reps, with the matching raw run directory."""
    runs = pd.read_csv(FINAL / "ariel/runs.csv")
    runs = runs[runs.rep.isin(reps)].sort_values("run_id").reset_index(drop=True)
    dirs = []
    for r in runs.itertuples():
        stamp = r.started.replace("-", "").replace(":", "").replace("T", "_")
        root = ARIEL_RAW / f"ig{r.inner_gens}" / r.scheme / xdir(r.x) / f"k{r.k_inherit}"
        hits = list(root.glob(f"rep_{r.rep}_fe*_nov*_{stamp}"))
        if len(hits) != 1:
            sys.exit(f"{r.run_id}: expected one run dir {root}/rep_{r.rep}_*_{stamp}, found {hits}")
        dirs.append(hits[0])
    runs["run_dir"] = dirs
    runs["src_id"] = runs.run_id
    runs["scheme"] = runs.scheme.map(PAPER_NAMES)
    runs["run_id"] = [f"ig{r.inner_gens}_{r.scheme}_{xdir(r.x)}_rep{r.rep:02d}" for r in runs.itertuples()]
    return runs


def published(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.drop(columns=[c for c in ("run_dir", "src_id", "db") if c in runs])


def ariel_conditions(runs: pd.DataFrame):
    """(source folder, published folder, runs) per condition."""
    for (ig, src, x), g in runs.groupby(["inner_gens", runs.src_id.str.extract(r"^ig\d+_(.+)_x\d\d_rep")[0], "x"]):
        yield Path(f"ig{ig}") / src / xdir(x), Path(f"ig{ig}") / g.scheme.iat[0] / xdir(x), g


def ariel_tier1(stage: Path, runs: pd.DataFrame) -> None:
    for src, rel, g in ariel_conditions(runs):
        dest = stage / "data" / rel
        dest.mkdir(parents=True, exist_ok=True)
        new_id = dict(zip(g.src_id, g.run_id, strict=False))
        for name in ("individuals.csv.gz", "selection_history.csv.gz"):
            df = pd.read_csv(FINAL / "ariel/data" / src / name, dtype=str, keep_default_na=False)
            df = df[df.run_id.isin(new_id)]
            assert df.run_id.nunique() == len(g), (src, name)
            df["run_id"] = df.run_id.map(new_id)
            df.to_csv(dest / name, index=False)
        print(f"  core {rel}", flush=True)
    published(runs).to_csv(stage / "runs.csv", index=False)
    write_derived(stage, "ariel", ["ariel_ig10", "ariel_ig20"])


def _ariel_curves(job: tuple[str, str]) -> pd.DataFrame:
    run_id, run_dir = job
    rows = {}
    for db in _db_files(Path(run_dir)):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for iid, lc in con.execute("SELECT id, json_extract(tags_, '$.learning_curve') FROM individual "
                                   "WHERE fitness_ IS NOT NULL"):
            rows[iid] = lc  # later parts win, as in export_final_results.export_run
        con.close()
    return pd.DataFrame({"run_id": run_id, "individual_id": list(rows), "learning_curve": list(rows.values())}
                        ).sort_values("individual_id")


def ariel_tier2(stage: Path, runs: pd.DataFrame, workers: int) -> None:
    with Pool(workers) as pool:
        for _, rel, g in ariel_conditions(runs):
            dfs = pool.map(_ariel_curves, list(zip(g.run_id, map(str, g.run_dir), strict=False)))
            dest = stage / "data" / rel
            dest.mkdir(parents=True, exist_ok=True)
            pd.concat(dfs).to_csv(dest / "learning_curves.csv.gz", index=False)
            print(f"  curves {rel}", flush=True)
    published(runs).to_csv(stage / "runs.csv", index=False)


def ariel_tier3(stage: Path, runs: pd.DataFrame) -> None:
    for r in runs.itertuples():
        dest = stage / "data" / f"ig{r.inner_gens}" / r.scheme / xdir(r.x) / f"rep_{r.rep:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        for f in [*_db_files(r.run_dir), r.run_dir / "run_config.json"]:
            if f.exists():
                os.link(f, dest / f.name)  # hard link: same file system, no copy
    published(runs).to_csv(stage / "runs.csv", index=False)


# ----------------------------------------------------------------------------- EvoGym

def evogym_runs(reps: list[int]) -> pd.DataFrame:
    rows = []
    for scheme in EVOGYM_SCHEMES:
        for xd, xv in X_DIRS.items():
            for rep in reps:
                db = EVOGYM_BASE / "evogym" / scheme / xd / "k8" / f"rep_{rep}_novSTRUCT" / "database.db"
                if not db.exists():
                    sys.exit(f"missing {db}")
                name = PAPER_NAMES[scheme]
                rows.append({"run_id": f"{name}_{xd}_rep{rep:02d}", "scheme": name, "x": xv, "rep": rep, "db": db})
    return pd.DataFrame(rows)


def _evogym_individuals(job: tuple[str, str]) -> tuple[pd.DataFrame, dict]:
    run_id, db = job
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ind = pd.read_sql(
        "SELECT id AS individual_id, time_of_birth AS birth_generation, time_of_death, alive AS alive_at_end,"
        " json_extract(tags_, '$.parent_id') AS parent_id,"
        " json_extract(tags_, '$.donor_ids') AS donor_ids,"
        " json_extract(tags_, '$.distance') AS fitness,"
        " json_extract(tags_, '$.init_fitness') AS init_fitness,"
        " json_extract(tags_, '$.novelty') AS novelty_last,"
        " fitness_ AS selection_score_last,"
        " json_extract(tags_, '$.descriptor') AS descriptor,"
        " json_extract(genotype_, '$.morph') AS morphology"
        " FROM individual ORDER BY id", con)
    con.close()
    ind.insert(0, "run_id", run_id)
    ind["death_generation"] = ind.time_of_death.where(ind.alive_at_end == 0).astype("Int64")
    ind["parent_id"] = ind.parent_id.astype("Int64")
    ind["donor_ids"] = ind.donor_ids.map(lambda s: ";".join(str(int(d)) for d in json.loads(s)) or None
                                         if s else None)
    desc = pd.DataFrame(ind.descriptor.map(json.loads).tolist(), columns=EVOGYM_DESCRIPTOR, index=ind.index)
    # Voxel materials are stored as floats in some generations and ints in others
    ind["morphology"] = ind.morphology.map(
        lambda m: json.dumps([[int(v) for v in row] for row in json.loads(m)], separators=(",", ":")))
    ind = pd.concat([ind.drop(columns=["descriptor", "time_of_death"]), desc], axis=1)
    cols = ["run_id", "individual_id", "birth_generation", "death_generation", "alive_at_end", "parent_id",
            "donor_ids", "fitness", "init_fitness", "novelty_last", "selection_score_last",
            *EVOGYM_DESCRIPTOR, "morphology"]
    info = dict(generations=int(ind.birth_generation.max()), n_individuals=len(ind))
    return ind[cols], info


def _evogym_curves(job: tuple[str, str]) -> pd.DataFrame:
    run_id, db = job
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    df = pd.read_sql("SELECT id AS individual_id, json_extract(tags_, '$.learning_curve') AS learning_curve "
                     "FROM individual ORDER BY id", con)
    con.close()
    df.insert(0, "run_id", run_id)
    return df


def evogym_runs_csv(runs: pd.DataFrame, info: dict | None = None) -> pd.DataFrame:
    out = published(runs).assign(
        inner_gens=10, inner_pop=10, k_inherit=8, generations=60, mu=100, **{"lambda": 100},
        selection="elitist", cma_sigma=0.45, mlp_hidden=16, novelty_metric="STRUCT",
        env="Walker (walker_long world)", episode_steps=499)
    if info:
        out["n_individuals"] = out.run_id.map(lambda r: info[r]["n_individuals"])
        out["last_generation"] = out.run_id.map(lambda r: info[r]["generations"])
    return out


def evogym_summaries(stage: Path, reps: list[int]) -> None:
    for src, name in [("summary-correct-novelty.csv", "generation_summary.csv"),
                      ("evogym_body_stats.csv", "body_stats.csv")]:
        df = pd.read_csv(EVOGYM_BASE / src)
        df = df[df.rep.isin(reps) & df.x.isin(X_DIRS.values())]
        df["scheme"] = df.scheme.map(PAPER_NAMES)
        df.to_csv(stage / name, index=False)


def evogym_tier1(stage: Path, runs: pd.DataFrame, reps: list[int], workers: int) -> None:
    info = {}
    with Pool(workers) as pool:
        for (scheme, x), g in runs.groupby(["scheme", "x"], sort=False):
            res = pool.map(_evogym_individuals, list(zip(g.run_id, map(str, g.db))))
            dest = stage / "data" / scheme / xdir(x)
            dest.mkdir(parents=True, exist_ok=True)
            pd.concat([d for d, _ in res]).to_csv(dest / "individuals.csv.gz", index=False)
            info |= dict(zip(g.run_id, [i for _, i in res]))
            print(f"  core {scheme} {xdir(x)}", flush=True)
    evogym_runs_csv(runs, info).to_csv(stage / "runs.csv", index=False)
    evogym_summaries(stage, reps)
    write_derived(stage, "evogym", ["evogym"])


def evogym_tier2(stage: Path, runs: pd.DataFrame, workers: int) -> None:
    with Pool(workers) as pool:
        for (scheme, x), g in runs.groupby(["scheme", "x"], sort=False):
            dfs = pool.map(_evogym_curves, list(zip(g.run_id, map(str, g.db))))
            dest = stage / "data" / scheme / xdir(x)
            dest.mkdir(parents=True, exist_ok=True)
            pd.concat(dfs).to_csv(dest / "learning_curves.csv.gz", index=False)
            print(f"  curves {scheme} {xdir(x)}", flush=True)
    evogym_runs_csv(runs).to_csv(stage / "runs.csv", index=False)


def evogym_tier3(stage: Path, runs: pd.DataFrame, reps: list[int]) -> None:
    for r in runs.itertuples():
        dest = stage / "data" / r.scheme / xdir(r.x) / f"rep_{r.rep:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        os.link(r.db, dest / "database.db")
    evogym_runs_csv(runs).to_csv(stage / "runs.csv", index=False)
    evogym_summaries(stage, reps)


# ----------------------------------------------------------------------------- shared

def write_derived(stage: Path, domain: str, tags: list[str]) -> None:
    d = stage / "derived"
    d.mkdir(exist_ok=True)
    for t in tags:
        for kind in ("curves", "donor_dist"):
            df = pd.read_csv(FIGS / "cache" / f"{kind}_{t}.csv")
            df["scheme"] = df.scheme.map(PAPER_NAMES)
            df.to_csv(d / f"{kind}_{t}.csv", index=False)
    stats = pd.read_csv(FIGS / "stats.csv")
    stats = stats[stats.domain == domain].dropna(axis=1, how="all")
    stats["scheme"] = stats.scheme.map(lambda v: PAPER_NAMES.get(v, v))
    stats.to_csv(d / "stats.csv", index=False)


def zip_stage(stage: Path, zip_path: Path) -> None:
    zip_path.unlink(missing_ok=True)
    subprocess.run(["zip", "-qr", str(zip_path.resolve()), "."], cwd=stage, check=True)
    print(f"Saved {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)", flush=True)


def update_sums(folder: Path) -> None:
    lines = []
    for z in sorted(folder.glob("*.zip")):
        h = hashlib.sha256()
        with z.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {z.name}\n")
    (folder / "SHA256SUMS").write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=["ariel", "evogym"], required=True)
    ap.add_argument("--tiers", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    ap.add_argument("--reps", default="0-9")
    ap.add_argument("--out", type=Path, default=Path("__data__/publication"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep-stage", action="store_true", help="keep the unzipped staging folder")
    args = ap.parse_args()

    reps = parse_reps(args.reps)
    folder = args.out / args.domain
    folder.mkdir(parents=True, exist_ok=True)
    runs = ariel_runs(reps) if args.domain == "ariel" else evogym_runs(reps)
    print(f"{args.domain}: {len(runs)} runs, reps {reps[0]}-{reps[-1]}")

    for tier in args.tiers:
        name = f"{args.domain}_{TIER_NAMES[tier]}"
        stage = folder / "stage" / name
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        print(f"Building {name}", flush=True)
        if args.domain == "ariel":
            {1: lambda: ariel_tier1(stage, runs),
             2: lambda: ariel_tier2(stage, runs, args.workers),
             3: lambda: ariel_tier3(stage, runs)}[tier]()
        else:
            {1: lambda: evogym_tier1(stage, runs, reps, args.workers),
             2: lambda: evogym_tier2(stage, runs, args.workers),
             3: lambda: evogym_tier3(stage, runs, reps)}[tier]()
        zip_stage(stage, folder / f"{name}.zip")
        if not args.keep_stage:
            shutil.rmtree(stage)
    if not any((folder / "stage").iterdir()):
        (folder / "stage").rmdir()
    update_sums(folder)
    print(f"Updated {folder / 'SHA256SUMS'}")
    shutil.copy(TEMPLATES / "README.md", args.out / "README.md")


if __name__ == "__main__":
    main()
