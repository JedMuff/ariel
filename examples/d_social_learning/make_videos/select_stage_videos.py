"""Pick the individuals for the evolution-stage videos and write their
metadata to __data__/videos/manifest.csv.

For every (domain, scheme, x): the run (rep 0-9) whose final population
reached the highest displacement (surv_best_disp at generation 60 in the
final_results caches). In that run, at generations 0, 15, 30, 45 and 60, the
surviving population (born at or before g, not removed by g's selection)
is ranked by displacement and its best, median and worst member are picked.

Per pick, the manifest also holds what the overlay shows: displacement,
fitness, novelty and selection score from the selection at generation
max(g, 1) (generation 0 has no selection, and its members were first compared
at generation 1), and the head start (init_fitness). ARIEL stores these per
generation (selection_history). EvoGym stores only each individual's last
value, so the generation's pool novelty (k-NN aligned Hamming distance, k = 15,
as experiment.py) and selection score are recomputed here. The recomputed pool
means are checked against summary-correct-novelty.csv.

Usage (from the repo root):
    uv run examples/d_social_learning/make_videos/select_stage_videos.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
from export_publication import PAPER_NAMES, ariel_runs, xdir  # noqa: E402
from final_results_data import aligned_hamming_batch  # noqa: E402

FINAL = Path("__data__/final_results")
CACHE = Path("__data__/social/figures/final_results/cache")
OUT = Path("__data__/videos")
STAGES = [0, 15, 30, 45, 60]
GENS = 60
K_NOVELTY = 15
REPS = list(range(10))
ARIEL_SCHEMES = ["darwinian", "lamarckian", "random", "random_many", "best", "best_many",
                 "similar_TOPDOWN", "similar_many_TOPDOWN"]
EVOGYM_SRC = {"similar_TOPDOWN": "similar_STRUCT", "similar_many_TOPDOWN": "similar_many_STRUCT"}


def best_reps(cache_csv: Path) -> dict[tuple[str, float], int]:
    c = pd.read_csv(cache_csv)
    c = c[(c.gen == GENS) & c.rep.isin(REPS)]
    best = c.loc[c.groupby(["scheme", "x"]).surv_best_disp.idxmax()]
    return {(r.scheme, r.x): int(r.rep) for r in best.itertuples()}


def survivors(ind: pd.DataFrame, g: int) -> pd.DataFrame:
    """Members of the population after generation g's selection."""
    return ind[(ind.birth <= g) & (ind.death.isna() | (ind.death > g))]


def picks(ind: pd.DataFrame) -> list[dict]:
    rows = []
    for g in STAGES:
        pop = survivors(ind, g).sort_values("disp", ascending=False, kind="stable").reset_index(drop=True)
        for label, i in [("best", 0), ("median", len(pop) // 2), ("worst", len(pop) - 1)]:
            rows.append(dict(generation=g, rank=label, pop_rank=i + 1, pop_size=len(pop),
                             individual_id=int(pop.id[i])))
    return rows


# ----------------------------------------------------------------------------- ARIEL

def select_ariel() -> list[dict]:
    best = best_reps(CACHE / "curves_ariel_ig10.csv")
    runs = ariel_runs(REPS)
    runs = runs[runs.inner_gens == 10]
    out = []
    for scheme in ARIEL_SCHEMES:
        for xd, xv in [("x00", 0.0), ("x05", 0.5), ("x10", 1.0)]:
            rep = best[(scheme, xv)]
            src_id = f"ig10_{scheme}_{xd}_rep{rep:02d}"
            run = runs[runs.src_id == src_id].iloc[0]
            d = FINAL / "ariel/data/ig10" / scheme / xd
            ind = pd.read_csv(d / "individuals.csv.gz", usecols=[
                "run_id", "individual_id", "birth_generation", "death_generation", "displacement", "fitness",
                "init_fitness", "final_eval_fitness_1", "final_eval_displacement_1"])
            ind = ind[ind.run_id == src_id].rename(columns={
                "individual_id": "id", "birth_generation": "birth", "death_generation": "death", "displacement": "disp"})
            hist = pd.read_csv(d / "selection_history.csv.gz")
            hist = hist[hist.run_id == src_id].set_index(["generation", "individual_id"])
            ind = ind.set_index("id", drop=False)
            for p in picks(ind):
                r = ind.loc[p["individual_id"]]
                sel_gen = max(p["generation"], 1)
                h = hist.loc[(sel_gen, p["individual_id"])]
                out.append(dict(
                    domain="ariel", scheme=PAPER_NAMES[scheme], x=xv, rep=rep, **p,
                    displacement=r.disp, fitness=r.fitness, init_fitness=r.init_fitness,
                    novelty=h.novelty, selection_score=h.selection_score, selection_gen=sel_gen,
                    episode_fitness=r.final_eval_fitness_1, episode_displacement=r.final_eval_displacement_1,
                    source=str(run.run_dir)))
            print(f"  ariel {PAPER_NAMES[scheme]} {xd}: rep {rep}", flush=True)
    return out


# ----------------------------------------------------------------------------- EvoGym

def _pool_scores(ind: pd.DataFrame, morph: dict[int, np.ndarray], gen: int, x: float) -> pd.DataFrame:
    """Novelty and selection score of every member of generation gen's
    selection pool (the survivors of gen - 1 plus the offspring born at gen),
    as experiment.py computes them."""
    pool = pd.concat([survivors(ind, gen - 1), ind[ind.birth == gen]]).drop_duplicates("id")
    ids = pool.id.to_numpy()
    n = len(ids)
    i, j = np.triu_indices(n, k=1)
    stack = np.stack([morph[k] for k in ids])
    dist = np.zeros((n, n))
    dist[i, j] = dist[j, i] = aligned_hamming_batch(stack[i], stack[j])
    k = min(K_NOVELTY, n - 1)
    nov = np.sort(dist, axis=1)[:, 1:k + 1].mean(axis=1)
    d = pool.disp.to_numpy(dtype=float)
    fd, fn = d[np.isfinite(d)], nov[np.isfinite(nov)]
    nd = (d - fd.min()) / (fd.max() - fd.min()) if fd.max() > fd.min() else np.zeros(n)
    nn = (nov - fn.min()) / (fn.max() - fn.min()) if fn.max() > fn.min() else np.zeros(n)
    return pd.DataFrame({"id": ids, "novelty": nov, "selection_score": nd * x + nn * (1 - x)}).set_index("id")


def select_evogym() -> list[dict]:
    best = best_reps(CACHE / "curves_evogym.csv")
    summary = pd.read_csv(FINAL / "evogym/social/summary-correct-novelty.csv")
    out, worst_nov = [], 0.0
    for scheme in ARIEL_SCHEMES:
        src = EVOGYM_SRC.get(scheme, scheme)
        for xd, xv in [("x00", 0.0), ("x05", 0.5), ("x10", 1.0)]:
            rep = best[(scheme, xv)]
            db = FINAL / "evogym/social/evogym" / src / xd / "k8" / f"rep_{rep}_novSTRUCT" / "database.db"
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            ind = pd.read_sql(
                "SELECT id, alive, time_of_birth AS birth, time_of_death AS tod,"
                " json_extract(tags_, '$.distance') AS disp,"
                " json_extract(tags_, '$.init_fitness') AS init_fitness,"
                " json_extract(genotype_, '$.morph') AS morph FROM individual", con)
            con.close()
            ind["death"] = ind.tod.where(ind.alive == 0)
            morph = {i: np.array(json.loads(m)) for i, m in zip(ind.id, ind.morph)}
            ind = ind.drop(columns="morph").set_index("id", drop=False)
            scores = {}
            for p in picks(ind):
                sel_gen = max(p["generation"], 1)
                if sel_gen not in scores:
                    scores[sel_gen] = _pool_scores(ind, morph, sel_gen, xv)
                    ref = summary[(summary.scheme == src) & (summary.x == xv) & (summary.rep == rep)
                                  & (summary.gen == sel_gen)].mean_novelty
                    worst_nov = max(worst_nov, abs(scores[sel_gen].novelty.mean() - float(ref.iloc[0])))
                r, s = ind.loc[p["individual_id"]], scores[sel_gen].loc[p["individual_id"]]
                out.append(dict(
                    domain="evogym", scheme=PAPER_NAMES[scheme], x=xv, rep=rep, **p,
                    displacement=r.disp, fitness=r.disp, init_fitness=r.init_fitness,
                    novelty=s.novelty, selection_score=s.selection_score, selection_gen=sel_gen,
                    episode_fitness=r.disp, episode_displacement=r.disp, source=str(db)))
            print(f"  evogym {PAPER_NAMES[scheme]} {xd}: rep {rep}", flush=True)
    print(f"EvoGym recomputed pool novelty vs summary-correct-novelty.csv: max |diff of means| = {worst_nov:.2e}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(select_ariel() + select_evogym())
    m["path"] = [str(OUT / r.domain / f"{r.scheme}_{xdir(r.x)}" /
                     f"g{r.generation:02d}_{r.rank}_id{r.individual_id}.mp4") for r in m.itertuples()]
    m.to_csv(OUT / "manifest.csv", index=False)
    print(f"Saved {OUT / 'manifest.csv'}: {len(m)} videos "
          f"({m.groupby('domain').size().to_dict()})")


if __name__ == "__main__":
    main()
