"""Numbers behind two observations in the results section that the curve
caches do not hold:

  - controller drift: mean norm of the learned controller weights at
    generations 1 and 60, per scheme, x, rep. A single inherited controller
    accumulates every learning step and grows; an average of k does not.
  - ARIEL mobility: hinges per offspring and the share that cannot move
    (no hinge) or moves more than 0.5 m, generations 41-60, per scheme and x.
  - ARIEL lineages: distinct generation-0 ancestors among the survivors, and
    how often a parent and its child (one per survivor, in the same selection
    pool) share a fate, generations 1-20.
  - ARIEL body size: survivor modules and survivor diversity divided by the
    size of the two trees, at generation 60.
  - ARIEL x = 0 replay: the recorded selection pools of generations 1-20
    re-selected with other novelty rules, one step at a time.

The ARIEL export (__data__/final_results/ariel) has no controller weights, so
the ARIEL norms come from the raw run databases.

Writes __data__/social/figures/final_results/observations.csv.

Usage (from the repo root):
    uv run examples/d_social_learning/analysis/final_results_observations.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ARIEL_RAW = Path("__data__/social/ariel_zoo_fe10/ig10")
ARIEL = Path("__data__/final_results/ariel/data/ig10")
EVOGYM = Path("__data__/final_results/evogym/social/evogym")
OUT = Path("__data__/social/figures/final_results/observations.csv")
X_DIRS = {"x00": 0.0, "x05": 0.5, "x10": 1.0}
REPS = range(10)
GENS = (1, 60)
EVOGYM_NAMES = {"similar_STRUCT": "similar_TOPDOWN", "similar_many_STRUCT": "similar_many_TOPDOWN"}
K_NOVELTY = 15

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim_ariel.tree_edit_distance import top_down_tree_distance  # noqa: E402


def _norms(job: tuple[str, str, float, int, Path]) -> list[dict]:
    dom, scheme, xv, rep, db = job
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = pd.read_sql("SELECT time_of_birth AS gen, json_extract(tags_, '$.theta') AS theta FROM individual "
                    f"WHERE time_of_birth IN {GENS} AND json_extract(tags_, '$.theta') IS NOT NULL", con)
    con.close()
    q["norm"] = q.theta.map(lambda t: np.linalg.norm(json.loads(t)))
    return [dict(domain=dom, what="theta_norm", scheme=scheme, x=xv, rep=rep, gen=g, value=v)
            for g, v in q.groupby("gen").norm.mean().items()]


def _jobs() -> list[tuple]:
    jobs = []
    for xd, xv in X_DIRS.items():
        for sdir in sorted(EVOGYM.iterdir()):
            for rep in REPS:
                jobs.append(("evogym", EVOGYM_NAMES.get(sdir.name, sdir.name), xv, rep,
                             sdir / xd / "k8" / f"rep_{rep}_novSTRUCT" / "database.db"))
        for sdir in sorted(ARIEL_RAW.iterdir()):
            for rep in REPS:
                runs = sorted((sdir / xd / "k5").glob(f"rep_{rep}_fe10_novTOPDOWN_*"))
                if runs:
                    # Resumed runs hold generation 60 in database_part2.db
                    for db in sorted(runs[-1].glob("database*.db")):
                        jobs.append(("ariel", sdir.name, xv, rep, db))
    return jobs


def ariel_mobility() -> list[dict]:
    rows = []
    for sdir in sorted(ARIEL.iterdir()):
        for xd, xv in X_DIRS.items():
            f = sdir / xd / "individuals.csv.gz"
            if not f.exists():
                continue
            d = pd.read_csv(f, usecols=["run_id", "birth_generation", "displacement", "morphology"])
            d = d[d.run_id.str.extract(r"rep(\d+)$")[0].astype(int).isin(REPS) & (d.birth_generation > 40)]
            hinges = d.morphology.str.count('"HINGE"')
            for what, v in (("hinges", hinges.mean()), ("share_no_hinge", (hinges == 0).mean()),
                            ("share_moving_0.5m", (d.displacement > 0.5).mean())):
                rows.append(dict(domain="ariel", what=what, scheme=sdir.name, x=xv, value=v))
    return rows


def _rep(run_id: pd.Series) -> pd.Series:
    return run_id.str.extract(r"rep(\d+)$")[0].astype(int)


def _survivors(r: pd.DataFrame, g: int) -> pd.DataFrame:
    return r[(r.birth_generation <= g) & ((r.tod > g) | ((r.alive_at_end == 1) & (g == 60)))]


def _ariel_lineages(job: tuple[str, float, Path]) -> list[dict]:
    scheme, xv, d = job
    ind = pd.read_csv(d / "individuals.csv.gz", usecols=["run_id", "individual_id", "birth_generation",
                                                          "death_generation", "alive_at_end", "parent_id",
                                                          "morphology"])
    ind = ind[_rep(ind.run_id).isin(REPS)]
    ind["tod"] = ind.death_generation.fillna(60)
    ind["size"] = ind.morphology.str.count('"type"')
    sel = pd.read_csv(d / "selection_history.csv.gz")
    survived = sel.set_index(["run_id", "generation", "individual_id"]).survived
    rows, fates = [], []
    for rid, r in ind.groupby("run_id"):
        parent = dict(zip(r.individual_id, r.parent_id))
        root: dict[int, int] = {}
        for i in r.individual_id:  # ids increase with birth, so parents come first
            root[i] = i if pd.isna(parent[i]) else root[int(parent[i])]
        for g in (10, 60):
            rows.append(dict(what=f"founders_gen{g}", value=len({root[i] for i in _survivors(r, g).individual_id})))
        s = _survivors(r, 60)
        morph = {i: json.loads(m) for i, m in zip(s.individual_id, s.morphology)}
        size = dict(zip(s.individual_id, s["size"]))
        ids = s.individual_id.to_numpy()
        i, j = np.triu_indices(len(ids), k=1)
        dist = np.array([top_down_tree_distance(morph[a], morph[b]) for a, b in zip(ids[i], ids[j])])
        rows += [dict(what="survivor_modules_gen60", value=s["size"].mean()),
                 dict(what="diversity_per_size_gen60",
                      value=(dist / np.array([size[a] + size[b] for a, b in zip(ids[i], ids[j])])).mean())]
        for k in r[r.birth_generation.between(1, 20)].itertuples():
            c = survived.get((rid, k.birth_generation, k.individual_id))
            p = survived.get((rid, k.birth_generation, int(k.parent_id)))
            if c is not None and p is not None:
                fates.append(c == p)
    rows.append(dict(what="parent_child_same_fate", value=np.mean(fates)))
    return [dict(domain="ariel", scheme=scheme, x=xv, **r) for r in rows]


def _replay(job: tuple[str, int]) -> list[dict]:
    """Re-select the recorded x = 0 pools of generations 1-20 with other rules
    and record how often a parent and its child share a fate."""
    scheme, rep = job
    d = ARIEL / scheme / "x00"
    rid = f"ig10_{scheme}_x00_rep{rep:02d}"
    ind = pd.read_csv(d / "individuals.csv.gz", usecols=["run_id", "individual_id", "birth_generation",
                                                          "parent_id", "morphology"])
    ind = ind[ind.run_id == rid]
    morph = {i: json.loads(m) for i, m in zip(ind.individual_id, ind.morphology)}
    parent, born = dict(zip(ind.individual_id, ind.parent_id)), dict(zip(ind.individual_id, ind.birth_generation))
    sel = pd.read_csv(d / "selection_history.csv.gz")
    fates: dict[str, list[bool]] = {}
    for g, pool in sel[(sel.run_id == rid) & (sel.generation <= 20)].groupby("generation"):
        ids = pool.individual_id.to_numpy()
        n, pos = len(ids), {i: j for j, i in enumerate(ids)}
        dist = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                dist[a, b] = dist[b, a] = top_down_tree_distance(morph[ids[a]], morph[ids[b]])
        pairs = [(j, pos[int(parent[i])]) for j, i in enumerate(ids) if born[i] == g and int(parent[i]) in pos]
        rules = {}
        for k in (1, 5, K_NOVELTY):
            dd = dist + np.diag(np.full(n, np.inf))
            rules[f"k{k}"] = set(np.argsort(-np.sort(dd, axis=1)[:, :k].mean(axis=1))[:n // 2].tolist())
        # Keep one at a time, each the farthest from those already kept
        kept = [int(np.argmax(dist.sum(axis=1)))]
        while len(kept) < n // 2:
            kept.append(max((j for j in range(n) if j not in kept), key=lambda j: dist[j, kept].min()))
        rules["greedy"] = set(kept)
        for name, s in rules.items():
            fates.setdefault(name, []).extend((a in s) == (b in s) for a, b in pairs)
    return [dict(domain="ariel", what=f"replay_same_fate_{name}", scheme=scheme, x=0.0, value=np.mean(v))
            for name, v in fates.items()]


def main() -> None:
    with Pool(16) as pool:
        norms = pd.DataFrame([r for rs in pool.map(_norms, _jobs()) for r in rs])
        lin =pd.DataFrame([r for rs in pool.map(_ariel_lineages, [(s.name, xv, s / xd) for s in sorted(ARIEL.iterdir())
                                                                     for xd, xv in X_DIRS.items() if (s / xd).exists()])
                            for r in rs])
        replay = pd.DataFrame([r for rs in pool.map(_replay, [(s.name, rep) for s in sorted(ARIEL.iterdir())
                                                              for rep in REPS]) for r in rs])
    # A resumed ARIEL run can list a generation in both parts; keep one value per rep
    norms = norms.groupby(["domain", "what", "scheme", "x", "rep", "gen"]).value.mean().reset_index()
    norms = norms.groupby(["domain", "what", "scheme", "x", "gen"]).value.mean().reset_index()
    lin =lin.groupby(["domain", "what", "scheme", "x"]).value.mean().reset_index()
    replay = replay.groupby(["domain", "what", "scheme", "x"]).value.mean().reset_index()
    out = pd.concat([norms, pd.DataFrame(ariel_mobility()), lin, replay], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
