"""
Run discovery and statistics shared by the analysis scripts.

An evolution run (evolve.py, or the investigation's gecko_skill_tasks.py, so
the E0 data collected before this branch loads too) is a directory holding
__data__/{evolve,gecko_skill_tasks}/forward/ with run_config.json,
run_data.jsonl and checkpoints/genGGG_bodyII/. Its score is the best forward
score of its last generation (-fitness: the normalised score in [0, 2] minus
penalties), steadier than the best ever under --repeat-evals.

Paired statistics pair runs by seed (every cell uses seeds 42 + rep).
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sim import ZOO_HINGE, HingeActuator

DATA_SUBDIRS = ("evolve", "gecko_skill_tasks")
GENOME_LABEL = {"tree": "tree", "tree_symmetric": "symmetric", "tree_tied": "tied"}
BRAIN_LABEL = {"bang_bang": "bang_bang", "bang_bang_free": "bang_bang (free frequency)",
               "bang_bang_drift": "bang_bang (shared + drift)", "dann": "DANN"}
N_BOOT = 10_000
_CKPT_RE = re.compile(r"^gen(\d+)_body(\d+)$")

# Figures: categorical slots validated with the dataviz palette checker
# (light surface); identity is always also given by a legend or label.
GENOME_COLOR = {"tree": "#eb6834", "tree_symmetric": "#2a78d6", "tree_tied": "#1baf7a"}
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


# ── One run ───────────────────────────────────────────────────────────────────


def data_dir(run_dir: Path) -> Optional[Path]:
    """The run's __data__/<script>/forward directory, or None."""
    for sub in DATA_SUBDIRS:
        d = run_dir / "__data__" / sub / "forward"
        if (d / "run_config.json").exists():
            return d
    return None


def run_config(run_dir: Path) -> dict:
    d = data_dir(run_dir)
    return json.loads((d / "run_config.json").read_text()) if d else {}


def hinge_actuator(cfg: dict) -> HingeActuator:
    """The run's hinge (runs predating --hinge-kp etc. used the zoo hinge)."""
    return HingeActuator(**cfg["hinge_actuator"]) if "hinge_actuator" in cfg else ZOO_HINGE


def fitness_by_gen(run_dir: Path) -> dict[int, list[float]]:
    d = data_dir(run_dir)
    by_gen: dict[int, list[float]] = defaultdict(list)
    if d is None or not (d / "run_data.jsonl").exists():
        return by_gen
    for line in (d / "run_data.jsonl").open():
        rec = json.loads(line)
        fit = rec.get("fitness")
        if fit is not None and math.isfinite(fit):
            by_gen[rec["gen"]].append(float(fit))
    return by_gen


def checkpoints_by_gen(run_dir: Path) -> dict[int, list[Path]]:
    """gen -> checkpoint dirs of that generation."""
    d = data_dir(run_dir)
    by_gen: dict[int, list[Path]] = defaultdict(list)
    if d is None or not (d / "checkpoints").exists():
        return by_gen
    for ckpt in sorted((d / "checkpoints").iterdir()):
        m = _CKPT_RE.match(ckpt.name)
        if m and (ckpt / "meta.json").exists():
            by_gen[int(m.group(1))].append(ckpt)
    return by_gen


def best_checkpoint(run_dir: Path, gen: Optional[int] = None) -> Optional[Path]:
    """The lowest-fitness checkpoint of generation gen (default: the last)."""
    by_gen = checkpoints_by_gen(run_dir)
    if not by_gen:
        return None
    ckpts = by_gen.get(max(by_gen) if gen is None else gen, [])
    rated = [(json.loads((c / "meta.json").read_text()).get("fitness", math.inf), c) for c in ckpts]
    return min(rated, key=lambda x: x[0])[1] if rated else None


def run_row(run_dir: Path) -> Optional[dict]:
    """One row per run: brain, genome, seed, skill budget, hinge, last gen, score."""
    cfg = run_config(run_dir)
    by_gen = fitness_by_gen(run_dir)
    if not cfg or not by_gen:
        return None
    last = max(by_gen)
    act = hinge_actuator(cfg)
    return {
        "run": run_dir.name, "brain": cfg.get("brain", "ann"), "genome": cfg["genome_type"],
        "seed": int(cfg["seed"]), "skill_budget": int(cfg["skill_budget"]),
        "evals_per_body": int(cfg["skill_budget"]) * 20,
        "hinge": act.label, "kp": act.kp, "kv": act.kv, "armature": act.armature, "limited": act.limited,
        "last_gen": last, "score": -min(by_gen[last]), "path": str(run_dir),
    }


def load_runs(*roots: Path, pattern: str = "*") -> pd.DataFrame:
    """run_row for every finished run directly under the roots."""
    rows = [r for root in roots for d in sorted(Path(root).glob(pattern)) if d.is_dir() and (r := run_row(d))]
    return pd.DataFrame(rows)


# ── Paired statistics ─────────────────────────────────────────────────────────


def paired_gap(df: pd.DataFrame, a: str, b: str, value: str = "score", by: str = "genome",
               pair_on: str = "seed", rng: Optional[np.random.Generator] = None) -> dict:
    """Paired a - b difference of value between the rows with by == a and by == b,
    matched on pair_on: mean, percentile bootstrap 95% CI over pairs, Wilcoxon
    signed-rank p, wins and n."""
    rng = rng or np.random.default_rng(0)
    wide = df[df[by].isin([a, b])].pivot_table(index=pair_on, columns=by, values=value, aggfunc="mean")
    if a not in wide or b not in wide:
        return {"gap": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "p": math.nan, "wins": 0, "n": 0}
    d = (wide[a] - wide[b]).dropna().to_numpy()
    if len(d) == 0:
        return {"gap": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "p": math.nan, "wins": 0, "n": 0}
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
