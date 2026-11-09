"""
Reading results: Experiment 1 runs (evolve.py) and Experiment 2 trainings
(train_fixed_body.py), paired statistics and the figure style.

An E1 run is a directory with config.json ("experiment": "e1"),
evaluations.jsonl, generations.jsonl and weights/. Its main score, fixed
before the runs, is the mean score of its final population (the mu bodies
after selection in the last generation, all retrained that generation). The
champion is the best of them.

An E2 directory (<body_id>__<brain>) has config.json ("experiment": "e2"),
results.jsonl (one line per seed) and weights.npz.

Episode (replay.py, mechanics.py): one trained controller on one body, with
the score its training stored, so a replay can be checked against it.

Paired statistics pair E1 runs by seed: every condition uses the same seeds.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import BRAIN_LABEL, GENOME_LABEL, GENOME_TYPES

N_BOOT = 10_000

# Figures: categorical slots validated with the dataviz palette checker (light
# surface); identity is always also given by a legend or label.
GENOME_COLOR = dict(zip(GENOME_TYPES, ("#eb6834", "#2a78d6", "#1baf7a", "#9a958a", "#8e5bc9")))
INK, INK_2, GRID = "#1f1f1d", "#6b6a63", "#e4e3de"


def style(ax) -> None:  # noqa: ANN001
    """Recessive axes: no top/right spines, light y grid, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def config(run_dir: Path) -> dict:
    p = Path(run_dir) / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def run_name(run_dir: Path) -> str:
    """A name unique across the experiment, for output folders and tables: E1
    <genome>__<brain>__e<evals>__s<seed> (run directories are all called
    s<seed>); E2 the directory name (<body_id>__<brain>)."""
    cfg = config(run_dir)
    if cfg.get("experiment") == "e1":
        return f"{cfg['genome']}__{cfg['brain']}__e{cfg['evals']}__s{cfg['seed']}"
    return Path(run_dir).name


def find(roots: list[Path], experiment: str) -> list[Path]:
    """Run directories of an experiment ("e1" or "e2") anywhere under the roots, sorted."""
    out = [p.parent for root in roots for p in Path(root).rglob("config.json")
           if config(p.parent).get("experiment") == experiment]
    return sorted(set(out))


# ── Experiment 1 ──────────────────────────────────────────────────────────────


def generations(run_dir: Path) -> list[dict]:
    return _jsonl(Path(run_dir) / "generations.jsonl")


def evaluations(run_dir: Path) -> list[dict]:
    return _jsonl(Path(run_dir) / "evaluations.jsonl")


def run_row(run_dir: Path) -> Optional[dict]:
    """One row per E1 run: condition, progress, main score and champion."""
    cfg, gens = config(run_dir), generations(run_dir)
    if not cfg or not gens:
        return None
    last = gens[-1]
    return {
        "run": str(run_dir), "genome": cfg["genome"], "brain": cfg["brain"], "evals": int(cfg["evals"]),
        "seed": int(cfg["seed"]), "last_gen": last["gen"], "done": (Path(run_dir) / "DONE").exists(),
        "score": float(np.mean(last["scores"])), "champion": float(max(last["scores"])),
        "fallbacks": sum(g["fallbacks"] for g in gens), "wall_h": sum(g["wall_s"] for g in gens) / 3600,
    }


def load_e1(*roots: Path) -> pd.DataFrame:
    rows = [r for d in find(list(roots), "e1") if (r := run_row(d))]
    return pd.DataFrame(rows)


def curve_by_gen(run_dir: Path) -> pd.DataFrame:
    """Per generation: best and mean score of the population after selection."""
    return pd.DataFrame([{"gen": g["gen"], "best": max(g["scores"]), "mean": float(np.mean(g["scores"]))}
                         for g in generations(run_dir)])


def genomes_by_id(run_dir: Path) -> dict[int, dict]:
    """Every body's genome (logged with its first training)."""
    return {e["id"]: e["genome"] for e in evaluations(run_dir) if "genome" in e}


# ── Episodes ──────────────────────────────────────────────────────────────────


@dataclass
class Episode:
    run: str                 # run directory name
    key: str                 # within the run: g<gen>_b<id> or s<seed>
    genome: dict
    brain: str
    weights: np.ndarray
    score: float             # the score the training stored (m)
    meta: dict = field(default_factory=dict)


def e1_episodes(run_dir: Path, which: str = "final") -> Iterator[Episode]:
    """final: the final population; champion: its best; generations 'g:0,10,49':
    every body trained in those generations (survivors and offspring); or
    's:0,10,49': only the survivors of those generations, best first."""
    run_dir = Path(run_dir)
    cfg, gens, evals = config(run_dir), generations(run_dir), evaluations(run_dir)
    genome_of = genomes_by_id(run_dir)
    if which in ("final", "champion"):
        last = gens[-1]
        order = sorted(zip(last["scores"], last["ids"]), reverse=True)
        picks = [(last["gen"], i) for _, i in (order[:1] if which == "champion" else order)]
    elif which.startswith("g:"):
        wanted = {int(g) for g in which[2:].split(",")}
        picks = [(e["gen"], e["id"]) for e in evals if e["gen"] in wanted]
    elif which.startswith("s:"):
        wanted = {int(g) for g in which[2:].split(",")}
        picks = [(g["gen"], i) for g in gens if g["gen"] in wanted
                 for _, i in sorted(zip(g["scores"], g["ids"]), reverse=True)]
    else:
        raise ValueError(f"unknown selection {which!r}")
    record = {(e["gen"], e["id"]): e for e in evals}
    weights: dict[int, np.lib.npyio.NpzFile] = {}
    for gen, i in picks:
        if gen not in weights:
            weights[gen] = np.load(run_dir / "weights" / f"gen{gen:03d}.npz")
        rec = record[(gen, i)]
        yield Episode(run_name(run_dir), f"g{gen:03d}_b{i}", genome_of[i], cfg["brain"], weights[gen][str(i)],
                      rec["score"], {"gen": gen, "id": i, "born": rec["born"], "train_seed": rec["train_seed"]})


def e2_episodes(run_dir: Path, which: str = "all") -> Iterator[Episode]:
    """all: every seed; champion: the best seed."""
    run_dir = Path(run_dir)
    cfg, results = config(run_dir), _jsonl(Path(run_dir) / "results.jsonl")
    if which == "champion":
        results = sorted(results, key=lambda r: -r["score"])[:1]
    w = np.load(run_dir / "weights.npz")
    for r in results:
        yield Episode(run_name(run_dir), f"s{r['seed']}", cfg["genome"], cfg["brain"], w[f"s{r['seed']}"], r["score"],
                      {"seed": r["seed"], "body_id": cfg["body_id"]})


def episodes(run_dir: Path, which: str = "final") -> Iterator[Episode]:
    """Episodes of an E1 run or an E2 directory ('final' means every seed for E2)."""
    if config(run_dir).get("experiment") == "e2":
        return e2_episodes(run_dir, "champion" if which == "champion" else "all")
    return e1_episodes(run_dir, which)


def identity(run_dir: Path) -> dict:
    """Condition columns of a run (E1 or E2)."""
    cfg = config(run_dir)
    out = {"run": run_name(run_dir), "path": str(run_dir), "experiment": cfg.get("experiment"), "brain": cfg.get("brain"),
           "evals": cfg.get("evals")}
    if cfg.get("experiment") == "e2":
        return {**out, "genome": cfg["genome_type"], "body_id": cfg["body_id"]}
    return {**out, "genome": cfg.get("genome"), "seed": cfg.get("seed")}


# ── Experiment 2 ──────────────────────────────────────────────────────────────


def load_e2(*roots: Path) -> pd.DataFrame:
    """One row per training (body, brain, seed)."""
    rows = []
    for d in find(list(roots), "e2"):
        cfg = config(d)
        for r in _jsonl(d / "results.jsonl"):
            rows.append({"body_id": cfg["body_id"], "genome": cfg["genome_type"], "brain": cfg["brain"],
                         "evals": cfg["evals"], "seed": r["seed"], "score": r["score"],
                         "stopped_early": r["stopped_early"], "n_modules": cfg.get("n_modules"),
                         "n_hinges": cfg.get("n_hinges")})
    return pd.DataFrame(rows)


# ── Labels ────────────────────────────────────────────────────────────────────


def genome_label(g: str) -> str:
    return GENOME_LABEL.get(g, g)


def brain_label(b: str) -> str:
    return BRAIN_LABEL.get(b, b)


# ── Paired statistics ─────────────────────────────────────────────────────────


def paired_gap(df: pd.DataFrame, a: str, b: str, value: str = "score", by: str = "genome",
               pair_on: str = "seed", rng: Optional[np.random.Generator] = None) -> dict:
    """Paired a - b difference of value between the rows with by == a and by == b,
    matched on pair_on: mean, percentile bootstrap 95% CI over pairs, Wilcoxon
    signed-rank p, wins and n."""
    rng = rng or np.random.default_rng(0)
    nan = {"gap": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "p": math.nan, "wins": 0, "n": 0}
    wide = df[df[by].isin([a, b])].pivot_table(index=pair_on, columns=by, values=value, aggfunc="mean")
    if a not in wide or b not in wide:
        return nan
    d = (wide[a] - wide[b]).dropna().to_numpy()
    if len(d) == 0:
        return nan
    boot = rng.choice(d, (N_BOOT, len(d))).mean(axis=1)
    p = wilcoxon(d).pvalue if len(d) >= 2 and np.any(d != 0) else math.nan
    return {"gap": float(d.mean()), "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
            "p": float(p), "wins": int((d > 0).sum()), "n": len(d)}


def gap_table(df: pd.DataFrame, groups: list[str], a: str = "tree_symmetric", b: str = "tree", **kw) -> pd.DataFrame:
    """paired_gap of a vs b per combination of the group columns."""
    by, value = kw.get("by", "genome"), kw.get("value", "score")
    rows = []
    for key, g in df.groupby(groups):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(groups, key)), **paired_gap(g, a, b, **kw),
                     f"mean_{a}": g.loc[g[by] == a, value].mean(), f"mean_{b}": g.loc[g[by] == b, value].mean()})
    return pd.DataFrame(rows)
