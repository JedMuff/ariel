"""Export ARIEL zoo runs to the flat __data__/final_results/ariel layout.

Reads the run directories written by experiment.py
(<data-dir>/ig<N>/<scheme>/<xNN>/k<K>/rep_<R>_fe<FE>_nov<M>_<stamp>/, with
database.db and, for resumed runs, database_part<N>.db) and writes, per
(budget, scheme, x),

    <out>/data/ig<N>/<scheme>/<xNN>/individuals.csv.gz
    <out>/data/ig<N>/<scheme>/<xNN>/selection_history.csv.gz

plus one row per run in <out>/runs.csv, in the format documented in
<out>/README.md. Conditions that already have files are left alone unless
--overwrite is given, and runs.csv rows are replaced by run_id, so exporting a
new condition (e.g. x = 0) only adds to the dataset.

Usage:
    uv run python examples/d_social_learning/analysis/export_final_results.py \\
        --budgets 10 --x x00 --reps 0-9
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMES = ["darwinian", "lamarckian", "random", "random_many", "best", "best_many",
           "similar_TOPDOWN", "similar_many_TOPDOWN"]
DESCRIPTOR = ["B_branching", "L_limbs", "E_limb_length", "C_coverage", "J_joints",
              "S_symmetry", "D_module_diversity", "P_footprint"]
N_FINAL = 10
IND_COLS = (["run_id", "individual_id", "birth_generation", "death_generation", "alive_at_end",
             "parent_id", "mutation_op", "mutation_rotation", "mutation_n_pruned", "donor_ids",
             "fitness", "fitness_cma_best", "displacement", "init_fitness", "mean_jerk",
             "hinge_contacts"]
            + [f"final_eval_fitness_{i}" for i in range(1, N_FINAL + 1)]
            + [f"final_eval_displacement_{i}" for i in range(1, N_FINAL + 1)]
            + ["novelty_last", "selection_score_last"] + DESCRIPTOR + ["eval_seed", "morphology"])


def _obj(v):
    """A tag value that may be stored as a JSON/Python literal string."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return ast.literal_eval(v)
    return v


def _db_files(run: Path) -> list[Path]:
    parts = sorted(run.glob("database_part*.db"), key=lambda p: int(re.search(r"part(\d+)", p.name)[1]))
    return [run / "database.db", *parts]


def export_run(run: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, hist = {}, []
    for db in _db_files(run):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for iid, tob, fit, geno, tags in con.execute(
                "SELECT id, time_of_birth, fitness_, genotype_, tags_ FROM individual "
                "WHERE fitness_ IS NOT NULL"):
            rows[iid] = (tob, json.loads(geno), json.loads(tags))  # later parts win
        hist.append(pd.read_sql("SELECT generation, individual_id, novelty, fitness AS selection_score, "
                                "survivor AS survived FROM novelty_history", con))
        con.close()

    sel = (pd.concat(hist).drop_duplicates(["generation", "individual_id"], keep="last")
           .sort_values(["generation", "selection_score"], ascending=[True, False]))
    last_gen = int(sel.generation.max())
    last = sel.groupby("individual_id").tail(1).set_index("individual_id")
    removed = sel[sel.survived == 0].groupby("individual_id").generation.max()
    alive = set(sel[(sel.generation == last_gen) & (sel.survived == 1)].individual_id)

    out = []
    for iid, (tob, geno, t) in sorted(rows.items()):
        mut = _obj(t.get("mutation")) or {}
        parent = _obj(t.get("parent_id"))
        donors = _obj(t.get("donor_ids")) or []
        fe_fit = _obj(t.get("final_eval_fitness")) or [None] * N_FINAL
        fe_disp = _obj(t.get("final_eval_displacement")) or [None] * N_FINAL
        desc = _obj(t.get("descriptor")) or [None] * len(DESCRIPTOR)
        morph = _obj(geno["morph"])
        r = {
            "run_id": run_id, "individual_id": iid, "birth_generation": tob,
            "death_generation": None if iid in alive else removed.get(iid),
            "alive_at_end": int(iid in alive),
            "parent_id": parent,
            "mutation_op": mut.get("op"),
            "mutation_rotation": None if "rotation" not in mut else int(mut["rotation"]),
            "mutation_n_pruned": mut.get("n_pruned"),
            "donor_ids": ";".join(str(int(d)) for d in donors) or None,
            "fitness": _obj(t.get("distance")),
            "fitness_cma_best": _obj(t.get("distance_cma")),
            "displacement": _obj(t.get("displacement")),
            "init_fitness": _obj(t.get("init_fitness")),
            "mean_jerk": _obj(t.get("mean_jerk")),
            # Truncated to an int, as in the existing export
            "hinge_contacts": None if t.get("c_hinge") is None else int(float(_obj(t["c_hinge"]))),
            "novelty_last": last.novelty.get(iid),
            "selection_score_last": last.selection_score.get(iid),
            "eval_seed": _obj(t.get("eval_seed")),
            "morphology": json.dumps(morph, separators=(",", ":")),
        }
        r |= {f"final_eval_fitness_{i + 1}": v for i, v in enumerate(fe_fit)}
        r |= {f"final_eval_displacement_{i + 1}": v for i, v in enumerate(fe_disp)}
        r |= dict(zip(DESCRIPTOR, desc))
        out.append(r)
    ind = pd.DataFrame(out, columns=IND_COLS)
    for c in ["death_generation", "parent_id", "mutation_rotation", "mutation_n_pruned"]:
        ind[c] = ind[c].astype("Int64")
    sel.insert(0, "run_id", run_id)

    cfg = json.loads((run / "run_config.json").read_text())
    a = cfg["args"]
    info = {
        "run_id": run_id, "scheme": a["scheme"], "x": a["x"], "inner_gens": a["inner_gens"],
        "rep": a["rep"], "k_inherit": a["k_inherit"], "generations": last_gen, "mu": a["pop"],
        "lambda": a["lam"], "selection": a["selection"], "inner_pop": a["inner_pop"],
        "cma_sigma": a["sigma"], "mlp_hidden": a["hidden"], "final_evals": a["final_evals"],
        "novelty_metric": a["novelty_metric"], "sim_preset": a["sim_preset"], "seed": cfg["seed"],
        "seed_resume": None, "started": cfg["started"], "started_resume": None,
        "n_individuals": len(ind), "last_generation": last_gen,
    }
    return ind, sel.reset_index(drop=True)[["run_id", "generation", "individual_id", "novelty",
                                             "selection_score", "survived"]], info


def newest_run(root: Path, rep: int) -> Path | None:
    runs = sorted(d for d in root.glob(f"rep_{rep}_fe*_nov*_*") if (d / "database.db").exists())
    return runs[-1] if runs else None


def parse_reps(s: str) -> list[int]:
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("__data__/social/ariel_zoo_fe10"))
    ap.add_argument("--out", type=Path, default=Path("__data__/final_results/ariel"))
    ap.add_argument("--budgets", type=int, nargs="+", default=[10])
    ap.add_argument("--x", nargs="+", default=["x00"], help="x folders, e.g. x00 x05")
    ap.add_argument("--schemes", nargs="+", default=SCHEMES)
    ap.add_argument("--reps", default="0-9")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    runs_csv = args.out / "runs.csv"
    runs = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame()
    new_info = []
    for ig in args.budgets:
        for scheme in args.schemes:
            for xd in args.x:
                dest = args.out / "data" / f"ig{ig}" / scheme / xd
                if (dest / "individuals.csv.gz").exists() and not args.overwrite:
                    print(f"skip {dest} (exists; --overwrite to replace)")
                    continue
                inds, sels = [], []
                for rep in parse_reps(args.reps):
                    run = newest_run(args.data_dir / f"ig{ig}" / scheme / xd / f"k{args.k}", rep)
                    if run is None:
                        print(f"  missing: ig{ig} {scheme} {xd} rep {rep}")
                        continue
                    run_id = f"ig{ig}_{scheme}_{xd}_rep{rep:02d}"
                    ind, sel, info = export_run(run, run_id)
                    inds.append(ind), sels.append(sel), new_info.append(info)
                if not inds:
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                pd.concat(inds).to_csv(dest / "individuals.csv.gz", index=False)
                pd.concat(sels).to_csv(dest / "selection_history.csv.gz", index=False)
                print(f"wrote {dest} ({len(inds)} runs)")

    if new_info:
        new = pd.DataFrame(new_info)
        runs = pd.concat([runs[~runs.run_id.isin(new.run_id)] if len(runs) else runs, new], ignore_index=True)
        runs.to_csv(runs_csv, index=False)
        print(f"updated {runs_csv} (+{len(new)} rows, {len(runs)} total)")


if __name__ == "__main__":
    main()
