"""
Shared helpers for analyzing __data__/sympress_* symmetry-pressure sweep
run directories (see slurm/run_symmetry_pressure_sweep.sh).
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "__data__" / "ariel_symmetry_pressure_sweep"

TASKS = ["forward", "multidirection", "turn_avg", "food"]
GENOME_TYPES = ["tree", "tree_symmetric", "cppn"]
N_REPS = 5

_RUN_TAG_RE = re.compile(
    r"^sympress_(?P<task>forward|multidirection|turn_avg|food)_"
    r"(?P<genome>tree_symmetric|tree|cppn)_rep(?P<rep>\d+)_"
    r"(?P<jobid>\d+)_(?P<taskidx>\d+)$"
)


def parse_run_tag(name: str) -> Optional[dict]:
    m = _RUN_TAG_RE.match(name)
    if not m:
        return None
    info = m.groupdict()
    info["rep"] = int(info["rep"])
    return info


def discover_run_dirs() -> list[Path]:
    return sorted(
        p for p in DATA_ROOT.glob("sympress_*")
        if p.is_dir() and parse_run_tag(p.name) is not None
    )


def analysis_dir(run_dir: Path, task: str, genome: str) -> Path:
    """Collated analysis output dir for one run: <sweep_root>/analysis/<task>/<genome>/,
    where <sweep_root> is run_dir's parent (e.g. __data__/ariel_symmetry_pressure_sweep).
    Every plot/video for this run lands here, filename-prefixed with run_dir.name.
    """
    return run_dir.parent / "analysis" / task / genome


def analysis_overview_path(run_dir: Path) -> Path:
    return run_dir.parent / "analysis" / "overview.png"


def data_subdir(run_dir: Path, task: str) -> Path:
    """Canonical top-level __data__/<script>/[<task>] dir for a run."""
    if task == "food":
        return run_dir / "__data__" / "gecko_food_skills"
    return run_dir / "__data__" / "gecko_skill_tasks" / task


def checkpoints_dir(run_dir: Path, task: str) -> Path:
    return data_subdir(run_dir, task) / "checkpoints"


def run_data_path(run_dir: Path, task: str) -> Path:
    return data_subdir(run_dir, task) / "run_data.jsonl"


def run_config(run_dir: Path, task: str) -> dict:
    p = data_subdir(run_dir, task) / "run_config.json"
    return json.loads(p.read_text()) if p.exists() else {}


_CKPT_RE = re.compile(r"^gen(\d+)_body\d+$")


def checkpoint_generations(run_dir: Path, task: str) -> dict[int, list[Path]]:
    """gen -> checkpoint dirs for that generation."""
    base = checkpoints_dir(run_dir, task)
    by_gen: dict[int, list[Path]] = {}
    if not base.exists():
        return by_gen
    for ckpt in base.iterdir():
        m = _CKPT_RE.match(ckpt.name)
        if not m:
            continue
        by_gen.setdefault(int(m.group(1)), []).append(ckpt)
    return by_gen


def last_generation(run_dir: Path, task: str) -> Optional[int]:
    gens = checkpoint_generations(run_dir, task)
    return max(gens) if gens else None


def checkpoints_ranked_by_fitness(run_dir: Path, task: str) -> list[Path]:
    """Every checkpoint dir across all generations, best (lowest) fitness
    first — not just the last generation's, since gecko_food_skills.py
    gates expensive left/right-skill training on a loco-fitness threshold:
    a body that fails the gate gets a checkpoint with only loco_weights.npy
    and a raw loco fitness (not a comparable food fitness) in meta.json. In
    a generation where every candidate happens to be gated, the "best"
    checkpoint there has no renderable skills at all, so callers should
    fall through to earlier generations rather than give up on the run.
    """
    ckpts = [c for gen_ckpts in checkpoint_generations(run_dir, task).values() for c in gen_ckpts]
    rated = []
    for c in ckpts:
        meta_path = c / "meta.json"
        if not meta_path.exists():
            continue
        fit = json.loads(meta_path.read_text()).get("fitness", float("inf"))
        rated.append((fit, c))
    rated.sort(key=lambda x: x[0])
    return [c for _, c in rated]


def load_fitness_by_gen(run_dir: Path, task: str) -> dict[int, list[float]]:
    path = run_data_path(run_dir, task)
    by_gen: dict[int, list[float]] = defaultdict(list)
    if not path.exists():
        return by_gen
    for line in path.open():
        rec = json.loads(line)
        fit = rec.get("fitness")
        if fit is None or not math.isfinite(fit):
            continue
        by_gen[rec["gen"]].append(float(fit))
    return by_gen


def best_checkpoint_in_generation(run_dir: Path, task: str, gen: int) -> Optional[Path]:
    ckpts = checkpoint_generations(run_dir, task).get(gen, [])
    best, best_fit = None, float("inf")
    for c in ckpts:
        meta_path = c / "meta.json"
        if not meta_path.exists():
            continue
        fit = json.loads(meta_path.read_text()).get("fitness", float("inf"))
        if fit < best_fit:
            best, best_fit = c, fit
    return best
