"""
Are mutations less harmful in tied genomes, so that good lineages last longer?

From the E1 evolution runs, each offspring against its parent. Controllers are
trained from scratch for every body, so a child's score is its body's score under
a new training, and parents are selected, so part of their score is luck. A child
identical to its parent therefore also tends to score lower (regression to the
mean); identical children are that baseline.

Bodies are compared by module: a module is identified by its path of faces from
the core, which works for every genome. Between parent and child:
  added      paths only in the child; sites: added paths whose parent path is
             in the parent body, i.e. the places modules were attached
  removed    paths only in the parent
  changed    paths in both, with another type or rotation
Classes of asexual children (one parent, no crossover):
  identical, add (only added), remove (only removed), change (only changed), mixed.

Reported per brain and genome, with seeds as the unit (paired tied - tree by seed,
bootstrap CI over seeds, Wilcoxon p):
  delta         child - parent score (m), and the share of children near or above parent
  adjusted      genome effect on delta given parent score and parent size, OLS,
                cluster bootstrap over seeds
  sites         delta of add children by number of sites and modules added
  survival      share of children that enter the population; of those from the
                parent's best quartile; age and founder diversity of the final population

Writes to --out: offspring.csv, delta_by_seed.csv, delta_by_class.csv, adjusted.csv,
sites.csv, survival.csv

Usage:
    python analyse_lineages.py --data __data__/ariel_paper_v2 --out __data__/ariel_paper_v2/paper/lineages
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import runs

BRAINS = ("bang_bang", "bang_bang_drift", "bang_bang_free")
GENOMES = ("tree", "tree_tied", "tree_symmetric")
MU = 20
NEAR = 0.25     # m; "near the parent": about one training's seed-to-seed spread on fixed bodies


# ── Body differences ──────────────────────────────────────────────────────────


def module_paths(genome: dict) -> dict[tuple[str, ...], tuple[str, str]]:
    """Path of faces from the core -> (type, rotation), for every reachable module."""
    children: dict[int, list[tuple[int, str]]] = {}
    for e in genome["edges"]:
        children.setdefault(int(e["parent"]), []).append((int(e["child"]), str(e["face"])))
    nodes = {int(k): v for k, v in genome["nodes"].items()}
    out: dict[tuple[str, ...], tuple[str, str]] = {}
    stack: list[tuple[int, tuple[str, ...]]] = [(0, ())]
    while stack:
        n, path = stack.pop()
        out[path] = (nodes[n]["type"], nodes[n]["rotation"])
        stack += [(c, (*path, f)) for c, f in children.get(n, [])]
    return out


def body_diff(parent: dict, child: dict) -> dict:
    p, c = module_paths(parent), module_paths(child)
    added = set(c) - set(p)
    removed = set(p) - set(c)
    changed = sum(p[k] != c[k] for k in set(p) & set(c))
    sites = sum(k[:-1] in p for k in added)
    hinges = lambda m: sum(v[0] == "HINGE" for v in m.values())  # noqa: E731
    return {"added": len(added), "removed": len(removed), "changed": changed, "sites": sites,
            "parent_modules": len(p), "child_modules": len(c), "parent_hinges": hinges(p), "child_hinges": hinges(c),
            "added_hinges": sum(c[k][0] == "HINGE" for k in added)}


def change_class(d: dict) -> str:
    a, r, c = d["added"] > 0, d["removed"] > 0, d["changed"] > 0
    if not (a or r or c):
        return "identical"
    if a and not (r or c):
        return "add"
    if r and not (a or c):
        return "remove"
    if c and not (a or r):
        return "change"
    return "mixed"


# ── One run ───────────────────────────────────────────────────────────────────


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """(one row per valid offspring, run-level lineage measures)."""
    cfg = runs.run_config(run_dir)
    d = runs.data_dir(run_dir)
    recs = [json.loads(line) for line in (d / "run_data.jsonl").open()]
    con = sqlite3.connect(next(d.glob("*.db")))
    genomes = {i: json.loads(g) if isinstance(g, str) else g for i, g in con.execute("select id, genotype_ from individual")}
    con.close()
    invalid = sum(1 for g in genomes.values() if g is None)
    fit = {r["ind_id"]: -r["fitness"] for r in recs if r["fitness"] is not None and np.isfinite(r["fitness"])}
    gen = {r["ind_id"]: r["gen"] for r in recs}
    parents = {r["ind_id"]: r["parent_ids"] for r in recs}

    # Plus selection, replayed: who is alive after each generation; rank within the parents' population.
    last = max(gen.values())
    pop: list[int] = sorted((i for i in fit if gen[i] == 0), key=lambda i: -fit[i])[:MU]
    rank_before: dict[int, dict[int, float]] = {}
    entered: dict[int, bool] = {i: True for i in pop}
    for g in range(1, last + 1):
        rank_before[g] = {i: k / max(len(pop) - 1, 1) for k, i in enumerate(pop)}
        new = [i for i in fit if gen[i] == g]
        both = sorted(pop + new, key=lambda i: -fit[i])[:MU]
        alive = set(both)
        for i in new:
            entered[i] = i in alive
        pop = both

    rows = []
    for i, g in gen.items():
        if g == 0 or i not in fit:
            continue
        ps = [p for p in parents[i] if p in fit]
        if not ps or len(ps) != len(parents[i]):
            continue
        row = {"brain": cfg["brain"], "genome": cfg["genome_type"], "seed": int(cfg["seed"]), "gen": g, "ind": i,
               "n_parents": len(ps), "score": fit[i], "parent_score": float(np.mean([fit[p] for p in ps])),
               "best_parent": max(fit[p] for p in ps), "entered": entered[i],
               "parent_rank": float(np.mean([rank_before[g].get(p, np.nan) for p in ps]))}
        if len(ps) == 1:
            m_p, m_c = genomes[ps[0]].get("morph"), genomes[i].get("morph")
            row |= body_diff(m_p, m_c)
            row["change"] = change_class(row)
        rows.append(row)

    # Founders behind the final population, and how old it is.
    def founders(i: int, seen: Optional[set[int]] = None) -> set[int]:
        out: set[int] = set()
        stack, seen = [i], seen or set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if not parents.get(x):
                out.add(x)
            else:
                stack += [p for p in parents[x] if p in gen]
        return out

    roots = [founders(i) for i in pop]
    champ = max(pop, key=lambda i: fit[i])
    # Lineage line of the champion: its parent chain through the better parent.
    chain, x = [], champ
    while parents.get(x):
        ps = [p for p in parents[x] if p in fit]
        if not ps:
            break
        chain.append(fit[x] - max(fit[p] for p in ps))
        x = max(ps, key=lambda p: fit[p])
    run = {"brain": cfg["brain"], "genome": cfg["genome_type"], "seed": int(cfg["seed"]),
           "invalid_children": invalid, "final_mean_age": float(np.mean([last - gen[i] for i in pop])),
           "final_founders_all": len(set().union(*roots)),
           "final_top_founder_share": Counter(f for r in roots for f in r).most_common(1)[0][1] / MU,
           "champ_gen": gen[champ], "champ_depth": len(chain), "champ_step_gain": float(np.mean(chain)) if chain else np.nan,
           "champ_steps_down": float(np.mean(np.array(chain) < 0)) if chain else np.nan,
           "gen0_best": max(fit[i] for i in fit if gen[i] == 0), "final_best": fit[champ]}
    return pd.DataFrame(rows), run


def _job(p: str) -> tuple[pd.DataFrame, dict]:
    return load_run(Path(p))


# ── Statistics ────────────────────────────────────────────────────────────────


def paired(wide: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> dict:
    d = (wide[a] - wide[b]).dropna().to_numpy()
    boot = rng.choice(d, (10_000, len(d))).mean(1)
    return {"gap": d.mean(), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
            "p": wilcoxon(d).pvalue if np.any(d != 0) else np.nan, "wins": int((d > 0).sum()), "n": len(d)}


def by_seed(df: pd.DataFrame, value: str, rng: np.random.Generator) -> list[dict]:
    wide = df.groupby(["seed", "genome"])[value].mean().unstack()
    means = wide.mean().to_dict()
    out = []
    for a, b in (("tree_tied", "tree"), ("tree_symmetric", "tree"), ("tree_tied", "tree_symmetric")):
        out.append({"value": value, "comparison": f"{runs.GENOME_LABEL[a]}-{runs.GENOME_LABEL[b]}",
                    "mean_a": means.get(a), "mean_b": means.get(b), **paired(wide, a, b, rng)})
    return out


def adjusted(df: pd.DataFrame, rng: np.random.Generator, covariates: list[str]) -> dict:
    """delta ~ genome + covariates, OLS, cluster bootstrap over seeds."""
    def fit(d: pd.DataFrame) -> np.ndarray:
        x = np.c_[np.ones(len(d)), d["genome"] == "tree_tied", d["genome"] == "tree_symmetric", d[covariates]].astype(float)
        return np.linalg.lstsq(x, d["delta"].to_numpy(), rcond=None)[0]
    groups = {s: g for s, g in df.groupby("seed")}
    keys = list(groups)
    boot = np.array([fit(pd.concat([groups[s] for s in rng.choice(keys, len(keys))], ignore_index=True)) for _ in range(1000)])
    est = fit(df)
    return {"tied": est[1], "tied_lo": np.percentile(boot[:, 1], 2.5), "tied_hi": np.percentile(boot[:, 1], 97.5),
            "sym": est[2], "sym_lo": np.percentile(boot[:, 2], 2.5), "sym_hi": np.percentile(boot[:, 2], 97.5),
            **{f"per_{c}": est[3 + i] for i, c in enumerate(covariates)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)

    dirs = [str(d) for d in sorted((args.data / "e1").iterdir()) if d.is_dir() and d.name.startswith("e1_bang_bang")]
    with Pool(args.workers) as pool:
        results = pool.map(_job, dirs)
    off = pd.concat([r[0] for r in results], ignore_index=True)
    run = pd.DataFrame([r[1] for r in results])
    off["delta"] = off["score"] - off["parent_score"]
    off["delta_best"] = off["score"] - off["best_parent"]
    off["near"] = off["delta"] >= -NEAR
    off["above"] = off["delta"] > 0
    off.to_csv(args.out / "offspring.csv", index=False)
    rng = np.random.default_rng(0)
    print(f"{len(off)} offspring of {len(run)} runs")

    seed_rows, class_rows, adj_rows, site_rows, surv_rows = [], [], [], [], []
    for brain in BRAINS:
        o, asex = off[off["brain"] == brain], off[(off["brain"] == brain) & (off["n_parents"] == 1)]
        sex = o[o["n_parents"] == 2]
        changed = asex[asex["change"] != "identical"]
        for name, sub in (("asexual, body changed", changed), ("crossover", sex)):
            for value in ("delta", "near", "above"):
                seed_rows += [{"brain": brain, "children": name, **r} for r in by_seed(sub, value, rng)]

        tab = asex.groupby(["genome", "change"]).agg(children=("delta", "size"), delta=("delta", "mean"), median=("delta", "median"),
                                                     near=("near", "mean"), above=("above", "mean"), entered=("entered", "mean"),
                                                     parent=("parent_score", "mean"), added=("added", "mean"), sites=("sites", "mean"))
        class_rows.append(tab.reset_index().assign(brain=brain))

        adds = asex[asex["change"] == "add"].copy()
        adds["sites_bin"] = adds["sites"].clip(upper=3)
        adds["added_bin"] = pd.cut(adds["added"], [0, 1, 2, 4, 99], labels=["1", "2", "3-4", "5+"])
        s = adds.groupby(["genome", "sites_bin", "added_bin"], observed=True).agg(
            children=("delta", "size"), delta=("delta", "mean"), near=("near", "mean"), entered=("entered", "mean"), parent=("parent_score", "mean"))
        site_rows.append(s[s["children"] >= 10].reset_index().assign(brain=brain))

        for name, sub, cov in (("asexual changed: parent score", changed, ["parent_score"]),
                               ("asexual changed: + parent size", changed, ["parent_score", "parent_modules", "parent_hinges"]),
                               ("identical children: parent score", asex[asex["change"] == "identical"], ["parent_score"])):
            if sub.groupby("genome").size().min() >= 20:
                adj_rows.append({"brain": brain, "model": name, "children": len(sub), **adjusted(sub, rng, cov)})

        top = o[o["parent_rank"] <= 0.25]
        for name, sub, value in (("all children entered", o, "entered"), ("top-quartile parents' children entered", top, "entered")):
            surv_rows += [{"brain": brain, "measure": name, **r} for r in by_seed(sub, value, rng)]
        rb = run[run["brain"] == brain]
        for value in ("final_mean_age", "final_founders_all", "final_top_founder_share", "champ_gen", "champ_depth",
                      "champ_step_gain", "champ_steps_down", "gen0_best", "final_best", "invalid_children"):
            surv_rows += [{"brain": brain, "measure": "run", **r} for r in by_seed(rb, value, rng)]

    seed_t, class_t, adj_t = pd.DataFrame(seed_rows), pd.concat(class_rows), pd.DataFrame(adj_rows)
    site_t, surv_t = pd.concat(site_rows), pd.DataFrame(surv_rows)
    seed_t.to_csv(args.out / "delta_by_seed.csv", index=False)
    class_t.to_csv(args.out / "delta_by_class.csv", index=False)
    adj_t.to_csv(args.out / "adjusted.csv", index=False)
    site_t.to_csv(args.out / "sites.csv", index=False)
    surv_t.to_csv(args.out / "survival.csv", index=False)

    print("\n== Child vs parent, paired over seeds ==")
    print(seed_t.round(3).to_string(index=False))
    print("\n== By kind of change (asexual) ==")
    print(class_t.round(3).to_string(index=False))
    print("\n== Genome effect on delta (m), adjusted ==")
    print(adj_t.round(3).to_string(index=False))
    print("\n== Added modules, by sites and modules ==")
    print(site_t.round(3).to_string(index=False))
    print("\n== Survival and lineages ==")
    print(surv_t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
