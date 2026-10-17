"""How much of each scheme's score survives re-scoring with perturbed starts,
and why: best-of-N vs mean-of-N scores, fragility vs reliance on lifetime
learning, donor vs parent quality, and controller weight norms.

Runs made with experiment.py --final-evals N store both scores per robot:
tags['distance_cma'] (CMA-ES's best single episode, unperturbed start) and
tags['distance'] (mean over N episodes with a jittered start; selection and
the donor schemes used this one). For runs without --final-evals,
distance_cma is taken to be distance, so the script also summarises older
data sets (e.g. the zoo pilot) on the same scale.

Reads <report-dir>/parta_included_runs.csv (parta_report.py) and the run
databases. Writes to <out-dir>:
    robots.csv.gz      one row per evaluated robot: scores, parent, donors,
                       ||theta*|| and ||theta_0||
    inflation.csv      per scheme/x: mean distance, distance_cma and their
                       ratio, for the generation-30 survivors and for the
                       offspring of generations 21-30 (per run, then mean/SD)
    fragility.csv      per scheme/x: fragility (1 - distance/distance_cma)
                       and learning share ((distance_cma - init) /
                       distance_cma) of the generation 21-30 offspring, plus
                       the Spearman correlation of the two over runs
    donor_quality.csv  per scheme/x: mean donor distance / parent distance
                       over all births (per run, then mean/SD)
    weight_norm.csv    per scheme/x: mean ||theta*|| of the survivors at
                       generations 0, 10, 20, 30

Usage:
    uv run examples/d_social_learning/analysis/final_eval_robustness.py \
        --report-dir __data__/social/figures/zoo_budget_fe10/report/ig10 \
        --out-dir __data__/social/figures/zoo_budget_fe10/robustness/ig10
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LATE = (21, 30)
NORM_GENS = (0, 10, 20, 30)


def load_robots(job: tuple) -> pd.DataFrame:
    scheme, x, rep, db = job
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, time_of_birth, time_of_death, alive, "
        "json_extract(tags_, '$.distance'), json_extract(tags_, '$.distance_cma'), "
        "json_extract(tags_, '$.init_fitness'), json_extract(tags_, '$.parent_id'), "
        "json_extract(tags_, '$.donor_ids'), json_extract(tags_, '$.theta'), "
        "json_extract(tags_, '$.init_theta') "
        "FROM individual WHERE fitness_ IS NOT NULL"
    ).fetchall()
    con.close()
    out = []
    for rid, gen, tod, alive, dist, dist_cma, init, parent, donors, theta, theta0 in rows:
        out.append(dict(
            scheme=scheme, x=x, rep=rep, id=rid, gen=gen, tod=tod, alive=alive,
            distance=dist, distance_cma=dist if dist_cma is None else dist_cma,
            init_fitness=init, parent_id=parent, donor_ids=donors or "[]",
            theta_norm=np.linalg.norm(json.loads(theta)) if theta else np.nan,
            init_theta_norm=np.linalg.norm(json.loads(theta0)) if theta0 else np.nan,
        ))
    return pd.DataFrame(out)


def spearman(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Spearman rho and p, NaN when either side is constant (runs without --final-evals)."""
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan, np.nan
    res = stats.spearmanr(a, b)
    return res.statistic, res.pvalue


def survivors(r: pd.DataFrame, g: int, gens: int) -> pd.DataFrame:
    """The mu individuals alive after selection at generation g (as parta_report.py)."""
    return r[(r.gen <= g) & ((r.tod > g) | ((r.alive == 1) & (g == gens)))]


def summarise(per_run: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = per_run.groupby(["scheme", "x"])[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    out["n"] = per_run.groupby(["scheme", "x"]).size()
    return out.reset_index()


def per_run_tables(robots: pd.DataFrame, gens: int) -> dict[str, pd.DataFrame]:
    infl, frag, donor, norm = [], [], [], []
    for (scheme, x, rep), r in robots.groupby(["scheme", "x", "rep"]):
        key = dict(scheme=scheme, x=x, rep=rep)
        surv = survivors(r, gens, gens)
        late = r[(r.gen >= LATE[0]) & (r.gen <= LATE[1])]
        infl.append(key | dict(
            surv_distance=surv.distance.mean(), surv_distance_cma=surv.distance_cma.mean(),
            surv_ratio=surv.distance.mean() / surv.distance_cma.mean(),
            late_distance=late.distance.mean(), late_distance_cma=late.distance_cma.mean(),
            late_ratio=late.distance.mean() / late.distance_cma.mean(),
        ))
        frag.append(key | dict(
            fragility=1 - late.distance.mean() / late.distance_cma.mean(),
            learning_share=(late.distance_cma - late.init_fitness).mean() / late.distance_cma.mean(),
        ))
        dist = r.set_index("id").distance
        births = r[r.gen > 0]
        ratios = []
        for parent, donors in zip(births.parent_id, births.donor_ids):
            donors = [d for d in json.loads(donors) if d in dist.index]
            if donors and parent in dist.index:
                ratios.append((dist[donors].mean(), dist[parent]))
        if ratios:
            d, p = np.array(ratios).T
            donor.append(key | dict(donor_distance=d.mean(), parent_distance=p.mean(),
                                    donor_parent_ratio=d.mean() / p.mean()))
        norm.append(key | {f"norm_g{g}": survivors(r, g, gens).theta_norm.mean() for g in NORM_GENS}
                    | dict(late_init_norm=late.init_theta_norm.mean()))
    return dict(inflation=pd.DataFrame(infl), fragility=pd.DataFrame(frag),
                donor_quality=pd.DataFrame(donor), weight_norm=pd.DataFrame(norm))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gens", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(Path(args.report_dir) / "parta_included_runs.csv")
    with Pool(args.workers) as pool:
        robots = pd.concat(pool.map(load_robots, [(r.scheme, r.x, r.rep, r.db) for r in runs.itertuples()],
                                    chunksize=1), ignore_index=True)
    robots.to_csv(out / "robots.csv.gz", index=False)

    tables = per_run_tables(robots, args.gens)
    for name, per_run in tables.items():
        per_run.to_csv(out / f"{name}_per_run.csv", index=False)
        cols = [c for c in per_run.columns if c not in ("scheme", "x", "rep")]
        summary = summarise(per_run, cols)
        if name == "fragility":
            rho = per_run.groupby(["scheme", "x"]).apply(
                lambda d: spearman(d.learning_share, d.fragility)[0], include_groups=False)
            summary["spearman_within_cell"] = summary.set_index(["scheme", "x"]).index.map(rho)
        summary.to_csv(out / f"{name}.csv", index=False)
    f = tables["fragility"]
    cell = summarise(f, ["fragility", "learning_share"])
    pd.DataFrame([dict(level="runs", n=len(f), **dict(zip(("rho", "p"), spearman(f.learning_share, f.fragility)))),
                  dict(level="cells", n=len(cell),
                       **dict(zip(("rho", "p"), spearman(cell.learning_share_mean, cell.fragility_mean))))]
                 ).to_csv(out / "fragility_correlation.csv", index=False)
    print(f"Saved robustness tables to {out} ({len(robots)} robots, {len(runs)} runs)")


if __name__ == "__main__":
    main()
