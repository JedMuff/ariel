"""Per-run curves and donor-offspring distances for the final ARIEL and EvoGym
data sets (__data__/final_results), in one schema, cached as CSVs for
final_results_figures.py.

The two domains come in different formats:
  - ARIEL:  data/<budget>/<scheme>/<x05|x10>/{individuals,selection_history}.csv.gz
            (see its README.md), TOPDOWN schemes, mu = lambda = 20, k = 5
  - EvoGym: evogym/<scheme>/<x05|x10>/k8/rep_N_novSTRUCT/database.db plus the
            per-generation pool summary summary-correct-novelty.csv (whose
            novelty was recomputed; the DB tags hold only each individual's last
            value), STRUCT schemes, mu = lambda = 100, k = 8

Both are mapped onto parta_report.per_run_curves's columns, so survivors,
best-so-far, head start (born_init_disp: offspring displacement with the
inherited controller, before learning) and learning gain are computed the
same way as in the earlier reports. Scheme names are normalised to the ARIEL
ones (similar_STRUCT -> similar_TOPDOWN) so both domains share colours.

Outputs, in <out-dir>:
    curves_<tag>.csv       scheme, x, rep, gen, surv_disp, best_so_far_disp,
                           born_init_disp, born_learn_gain, pool_novelty,
                           surv_diversity (mean pairwise morphological
                           distance among the generation's survivors)
    donor_dist_<tag>.csv   one row per offspring with donors: scheme, x, rep,
                           gen, donor_dist (mean over its donors), init_fitness,
                           disp
with <tag> = ariel_ig10, ariel_ig20 or evogym.

Usage (from the repo root):
    uv run examples/d_social_learning/analysis/final_results_data.py --domain ariel --budget ig10
    uv run examples/d_social_learning/analysis/final_results_data.py --domain evogym
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parta_report import per_run_curves  # noqa: E402
from sim_ariel.tree_edit_distance import top_down_tree_distance  # noqa: E402

ROOT = Path("__data__/final_results")
OUT = Path("__data__/social/figures/final_results/cache")
GENS = 60
X_DIRS = {"x00": 0.0, "x05": 0.5, "x10": 1.0}
ARIEL_SCHEMES = ["darwinian", "lamarckian", "random", "random_many", "best", "best_many",
                 "similar_TOPDOWN", "similar_many_TOPDOWN"]
EVOGYM_NAMES = {"similar_STRUCT": "similar_TOPDOWN", "similar_many_STRUCT": "similar_many_TOPDOWN"}


def aligned_hamming_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised sim_evogym.evogym_body_descriptors.aligned_hamming_distance
    for N pairs of equal square voxel grids, a and b of shape (N, n, n).

    That function shifts each body to every placement that keeps all its voxels
    in the grid and takes the minimum number of differing cells. Only the
    relative offset of the two bodies matters, so here both are moved to the
    top-left corner and every offset at which both still fit is tried: the
    differing cells are nnz(a) + nnz(b) minus, over the cells where both are
    filled, 1 for a different material and 2 for the same one."""
    n = a.shape[1]
    a, b = _to_corner(a), _to_corner(b)
    ha, wa = _extent(a)
    hb, wb = _extent(b)
    na = (a != 0).sum(axis=(1, 2))
    nb = (b != 0).sum(axis=(1, 2))
    best = np.zeros(len(a))
    for dr in range(-n + 1, n):
        for dc in range(-n + 1, n):
            # b is placed at (dr, dc) relative to a; both must fit in the grid.
            ok = (dr >= -(n - ha)) & (dr <= n - hb) & (dc >= -(n - wa)) & (dc <= n - wb)
            if not ok.any():
                continue
            ra, rb = (dr, 0) if dr >= 0 else (0, -dr)
            ca, cb = (dc, 0) if dc >= 0 else (0, -dc)
            # cell (i, j) of b sits on cell (i + dr, j + dc) of a
            sa = a[:, ra:n - rb, ca:n - cb]
            sb = b[:, rb:n - ra, cb:n - ca]
            both = (sa != 0) & (sb != 0)
            score = (both * (1 + (sa == sb))).sum(axis=(1, 2))
            best = np.where(ok, np.maximum(best, score), best)
    return na + nb - best


def _extent(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounding-box height and width of each body already in the top-left corner."""
    rows, cols = (v != 0).any(axis=2), (v != 0).any(axis=1)
    n = v.shape[1]
    return (n - np.argmax(rows[:, ::-1], axis=1), n - np.argmax(cols[:, ::-1], axis=1))


def _to_corner(v: np.ndarray) -> np.ndarray:
    out = np.zeros_like(v)
    for i, body in enumerate(v):
        nz = np.argwhere(body != 0)
        if len(nz):
            r0, c0 = nz.min(axis=0)
            out[i, :v.shape[1] - r0, :v.shape[2] - c0] = body[r0:, c0:]
    return out


def survivor_diversity(ind: pd.DataFrame, ids: str, pair_dist) -> pd.DataFrame:
    """Per generation of one run: mean pairwise morphological distance among
    its survivors (as parta_report.per_run_curves defines them). pair_dist
    maps two equal-length id arrays to their distances; survivors persist
    across generations, so each pair is computed once.
    """
    cache: dict[tuple[int, int], float] = {}
    rows = []
    for g in range(GENS + 1):
        surv = ind[(ind.gen <= g) & ((ind.tod > g) | ((ind.alive == 1) & (g == GENS)))][ids].to_numpy()
        i, j = np.triu_indices(len(surv), k=1)
        pairs = [(min(a, b), max(a, b)) for a, b in zip(surv[i], surv[j], strict=False)]
        new = [q for q in dict.fromkeys(pairs) if q not in cache]
        if new:
            a, b = map(np.array, zip(*new, strict=False))
            cache.update(zip(new, pair_dist(a, b), strict=False))
        rows.append({"gen": g, "surv_diversity": np.mean([cache[q] for q in pairs])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ARIEL
# ---------------------------------------------------------------------------

def _ariel_diversity(run: pd.DataFrame) -> pd.DataFrame:
    morph = {i: json.loads(m) for i, m in zip(run.individual_id, run.morphology, strict=False)}
    div = survivor_diversity(run, "individual_id", lambda a, b: [top_down_tree_distance(morph[i], morph[j])
                                                               for i, j in zip(a, b, strict=False)])
    return div.assign(scheme=run.scheme.iat[0], x=run.x.iat[0], rep=run.rep.iat[0])


def _ariel_donor_dist(run: pd.DataFrame) -> pd.DataFrame:
    morph = {i: json.loads(m) for i, m in zip(run.individual_id, run.morphology)}
    rows = []
    for r in run[run.donor_ids.notna() & (run.gen > 0)].itertuples():
        donors = [int(float(d)) for d in r.donor_ids.split(";")]
        dist = np.mean([top_down_tree_distance(morph[r.individual_id], morph[d]) for d in donors])
        rows.append(dict(scheme=r.scheme, x=r.x, rep=r.rep, gen=r.gen, donor_dist=dist,
                         init_fitness=r.init_fitness, disp=r.disp))
    return pd.DataFrame(rows)


def load_ariel(budget: str, reps: list[int], workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    inds, hists = [], []
    for scheme in ARIEL_SCHEMES:
        for xd, xv in X_DIRS.items():
            d = ROOT / "ariel/data" / budget / scheme / xd
            if not d.exists():  # x = 0 is not collected yet
                print(f"  skipping missing {d}")
                continue
            ind = pd.read_csv(d / "individuals.csv.gz", dtype={"donor_ids": str},
                              usecols=["run_id", "individual_id", "birth_generation", "death_generation",
                                       "alive_at_end", "displacement", "init_fitness", "donor_ids", "morphology"])
            hist = pd.read_csv(d / "selection_history.csv.gz", usecols=["run_id", "generation", "novelty"])
            for df in (ind, hist):
                df["scheme"], df["x"] = scheme, xv
                df["rep"] = df.run_id.str.extract(r"rep(\d+)$")[0].astype(int)
            inds.append(ind[ind.rep.isin(reps)])
            hists.append(hist[hist.rep.isin(reps)])
    ind = pd.concat(inds, ignore_index=True).rename(columns={
        "birth_generation": "gen", "death_generation": "tod", "alive_at_end": "alive", "displacement": "disp"})
    # Alive at the end: no death generation; survives every generation after birth.
    ind["tod"] = ind.tod.fillna(GENS)
    ind["distance"] = ind.disp
    hist = pd.concat(hists, ignore_index=True)

    curves = per_run_curves(ind.drop(columns="morphology"), pd.DataFrame(), GENS)
    nov = hist.groupby(["scheme", "x", "rep", "generation"]).novelty.mean().rename("pool_novelty").reset_index()
    curves = curves.merge(nov.rename(columns={"generation": "gen"}), on=["scheme", "x", "rep", "gen"], how="left")

    runs = [r for _, r in ind.groupby("run_id")]
    with Pool(workers) as pool:
        donor = pd.concat(pool.map(_ariel_donor_dist, [r for r in runs if r.scheme.iat[0] != "darwinian"]),
                          ignore_index=True)
        div = pd.concat(pool.map(_ariel_diversity, runs), ignore_index=True)
    curves = curves.merge(div, on=["scheme", "x", "rep", "gen"], how="left")
    return curves, donor


# ---------------------------------------------------------------------------
# EvoGym
# ---------------------------------------------------------------------------

def _evogym_run(job: tuple[str, float, int, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scheme, xv, rep, db = job
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ind = pd.read_sql(
        "SELECT id, alive, time_of_birth AS gen, time_of_death AS tod,"
        " json_extract(tags_, '$.distance') AS disp,"
        " json_extract(tags_, '$.init_fitness') AS init_fitness,"
        " json_extract(tags_, '$.donor_ids') AS donor_ids,"
        " json_extract(genotype_, '$.morph') AS morph"
        " FROM individual", con)
    con.close()
    ind["scheme"], ind["x"], ind["rep"] = scheme, xv, rep
    ind["distance"] = ind.disp
    curves = per_run_curves(ind.drop(columns=["morph", "donor_ids"]), pd.DataFrame(), GENS)
    morph = {i: np.array(json.loads(m)) for i, m in zip(ind.id, ind.morph, strict=False)}
    div = survivor_diversity(ind, "id", lambda a, b: aligned_hamming_batch(np.stack([morph[i] for i in a]),
                                                                         np.stack([morph[j] for j in b])))
    curves = curves.merge(div, on="gen", how="left")

    donor = pd.DataFrame()
    if scheme != "darwinian":
        kids = ind[(ind.gen > 0) & ind.donor_ids.notna()].copy()
        kids["donors"] = kids.donor_ids.map(json.loads)
        kids = kids[kids.donors.map(len) > 0]
        pairs = kids[["id", "donors"]].explode("donors")
        dist = aligned_hamming_batch(np.stack([morph[i] for i in pairs.id]),
                                     np.stack([morph[int(d)] for d in pairs.donors]))
        mean_dist = pd.Series(dist, index=pairs.index).groupby(level=0).mean()
        donor = kids.assign(donor_dist=mean_dist)[["scheme", "x", "rep", "gen", "donor_dist", "init_fitness", "disp"]]
    print(f"  {scheme} x={xv} rep {rep}", flush=True)
    return curves, donor


def load_evogym(reps: list[int], workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ROOT / "evogym/social"
    jobs = []
    for sdir in sorted((base / "evogym").iterdir()):
        for xd, xv in X_DIRS.items():
            for rep in reps:
                db = sdir / xd / "k8" / f"rep_{rep}_novSTRUCT" / "database.db"
                if not db.exists():
                    sys.exit(f"missing {db}")
                jobs.append((EVOGYM_NAMES.get(sdir.name, sdir.name), xv, rep, db))
    with Pool(workers) as pool:
        res = pool.map(_evogym_run, jobs)
    curves = pd.concat([c for c, _ in res], ignore_index=True)
    donor = pd.concat([d for _, d in res], ignore_index=True)

    summ = pd.read_csv(base / "summary-correct-novelty.csv")
    summ["scheme"] = summ.scheme.replace(EVOGYM_NAMES)
    summ = summ[summ.x.isin(X_DIRS.values()) & summ.rep.isin(reps)]
    nov = summ[["scheme", "x", "rep", "gen", "mean_novelty"]].rename(columns={"mean_novelty": "pool_novelty"})
    return curves.merge(nov, on=["scheme", "x", "rep", "gen"], how="left"), donor


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["ariel", "evogym"], required=True)
    parser.add_argument("--budget", default="ig10", help="ARIEL inner budget folder")
    parser.add_argument("--reps", type=int, default=10, help="use reps 0 .. N-1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--root", default=str(ROOT),
                        help="data root holding ariel/data/... (e.g. an unzipped ariel_1_core)")
    args = parser.parse_args()
    ROOT = Path(args.root)
    reps = list(range(args.reps))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.domain == "ariel":
        tag = f"ariel_{args.budget}"
        curves, donor = load_ariel(args.budget, reps, args.workers)
    else:
        tag = "evogym"
        curves, donor = load_evogym(reps, args.workers)
    curves.to_csv(out / f"curves_{tag}.csv", index=False)
    donor.to_csv(out / f"donor_dist_{tag}.csv", index=False)
    print(f"Saved {out}/curves_{tag}.csv ({curves.groupby(['scheme', 'x']).rep.nunique().min()}"
          f"-{curves.groupby(['scheme', 'x']).rep.nunique().max()} runs per condition) and donor_dist_{tag}.csv")


if __name__ == "__main__":
    main()
