"""Data loading and ancestry-graph construction for the interactive ancestry
tree viewer (`ancestry_tree_prepare.py` / `ancestry_tree_view.py`).

Reads one sweep run directory (see `sweep_common.py` for the general
`sympress_<task>_<genome>_rep<N>_<jobid>_<idx>` layout this also understands)
and builds a graph where every individual gets one node per generation of its
observed lifespan (origin generation through its last logged evaluation),
assigned a stable "lane" (row index) that's reused across all of that
individual's generations -- so a persisting individual draws as a straight
horizontal line rather than a diagonal one, and every edge spans exactly one
generation-column, with no exceptions.

A tournament-selected parent is re-evaluated in the *same* generation as the
offspring it produces (verified against real data: an elite selected as a
parent gets `requires_eval=True` set in the same `reproduction()` call that
creates its offspring, so both are logged in the same `evaluate()` call --
`gecko_skill_tasks.py:223-232`). So the genealogically correct anchor for a
"reproduction" edge is the parent's most recent *completed* occurrence
strictly before the child's generation -- i.e. gen-1, once gaps are bridged.

Gaps arise because a "plus"-strategy elite can stay alive for several
generations without being re-evaluated every single round (it's only
re-evaluated when tournament-selected as a parent that round). Since "plus"
elitism guarantees an individual can never drop out and later reappear
(`survivor_selection` permanently sets `alive=False`), it's safe to infer it
was continuously alive between any two of its confirmed occurrences -- so
those gaps are bridged with synthetic "still alive, not re-evaluated this
round" placeholder nodes at the same lane (`is_passthrough=True`, no fitness
or images), guaranteeing every edge -- reproduction and survival alike -- is
exactly one generation wide.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sweep_common

logger = logging.getLogger(__name__)

_CKPT_RE = re.compile(r"^gen(\d+)_body(\d+)$")


# ── Run discovery ────────────────────────────────────────────────────────────


@dataclass
class RunLayout:
    run_dir: Path
    task: str
    genome_type: str
    data_dir: Path
    run_config_path: Path
    run_data_path: Path
    checkpoints_dir: Path
    run_config: dict


def discover_run_layout(run_dir: Path, task_override: Optional[str] = None) -> RunLayout:
    run_dir = run_dir.resolve()

    tag = sweep_common.parse_run_tag(run_dir.name)
    task = task_override or (tag["task"] if tag else None)
    genome_type = tag["genome"] if tag else None

    if task is None or genome_type is None:
        # Fall back: glob for run_config.json under __data__/{gecko_skill_tasks/*|gecko_food_skills}
        candidates = sorted((run_dir / "__data__").glob("*/run_config.json")) + sorted(
            (run_dir / "__data__").glob("*/*/run_config.json")
        )
        if task_override is not None:
            candidates = [
                c for c in candidates
                if c.parent.name == task_override or c.parent.parent.name == "gecko_skill_tasks"
                and c.parent.name == task_override
            ]
        if not candidates:
            raise FileNotFoundError(f"No run_config.json found under {run_dir}/__data__/")
        if len(candidates) > 1 and task_override is None:
            names = [str(c.relative_to(run_dir)) for c in candidates]
            raise ValueError(
                f"Multiple run_config.json candidates under {run_dir}, pass --task to disambiguate: {names}"
            )
        run_config_path = candidates[0]
        data_dir = run_config_path.parent
        run_config = json.loads(run_config_path.read_text())
        task = run_config.get("task")
        genome_type = run_config.get("genome_type")
        if task is None:
            # food-task run_config.json has no "task" key (verified) -- infer
            # from directory name, else from the first run_data.jsonl row.
            if data_dir.name == "gecko_food_skills" or data_dir.parent.name == "gecko_food_skills":
                task = "food"
            else:
                first_row = _first_jsonl_row(data_dir / "run_data.jsonl")
                task = first_row.get("task") if first_row else data_dir.name
        if genome_type is None:
            first_row = _first_jsonl_row(data_dir / "run_data.jsonl")
            genome_type = first_row.get("genome_type") if first_row else "tree"
    else:
        data_dir = sweep_common.data_subdir(run_dir, task)
        run_config_path = data_dir / "run_config.json"
        run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}

    run_data_path = data_dir / "run_data.jsonl"
    checkpoints_dir = data_dir / "checkpoints"
    if not run_data_path.exists():
        raise FileNotFoundError(f"run_data.jsonl not found at {run_data_path}")

    return RunLayout(
        run_dir=run_dir,
        task=task,
        genome_type=genome_type,
        data_dir=data_dir,
        run_config_path=run_config_path,
        run_data_path=run_data_path,
        checkpoints_dir=checkpoints_dir,
        run_config=run_config,
    )


def _first_jsonl_row(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def load_run_data(run_data_path: Path) -> list[dict]:
    """Load run_data.jsonl, preserving file order (required for checkpoint
    correspondence -- see `index_checkpoints`/`build_ancestry_graph`)."""
    rows: list[dict] = []
    with run_data_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_checkpoints(checkpoints_dir: Path) -> dict[tuple[int, int], Path]:
    """{(gen, body_idx): checkpoint_dir}, parsed from whatever zero-pad width
    exists on disk -- does not assume a fixed `gen{:03d}_body{:02d}` format."""
    index: dict[tuple[int, int], Path] = {}
    if not checkpoints_dir.exists():
        return index
    for child in checkpoints_dir.iterdir():
        if not child.is_dir():
            continue
        m = _CKPT_RE.match(child.name)
        if not m:
            continue
        index[(int(m.group(1)), int(m.group(2)))] = child
    return index


# ── Ancestry graph ───────────────────────────────────────────────────────────


@dataclass
class TreeNode:
    node_id: str
    gen: int
    ind_id: int
    lane: int
    fitness: Optional[float]
    yz_symmetry: Optional[float]
    genome_hash: Optional[str]
    checkpoint_dir: Optional[Path] = None
    has_images: bool = False
    genome_image: Optional[str] = None
    phenotype_image: Optional[str] = None
    parents: list[dict] = field(default_factory=list)
    parent_ids_raw: list[int] = field(default_factory=list)
    parent_data_missing: bool = False
    is_invalid: bool = False
    is_passthrough: bool = False

    def to_json_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "gen": self.gen,
            "ind_id": self.ind_id,
            "lane": self.lane,
            "fitness": self.fitness,
            "fitness_display": f"{self.fitness:.3f}" if self.fitness is not None else None,
            "yz_symmetry": self.yz_symmetry,
            "genome_hash": self.genome_hash,
            "has_images": self.has_images,
            "genome_image": self.genome_image,
            "phenotype_image": self.phenotype_image,
            "parents": self.parents,
            "parent_ids_raw": self.parent_ids_raw,
            "parent_data_missing": self.parent_data_missing,
            "is_invalid": self.is_invalid,
            "is_passthrough": self.is_passthrough,
        }


@dataclass
class TreeEdge:
    from_node_id: str
    to_node_id: str
    kind: str

    def to_json_dict(self) -> dict:
        return {"from_node_id": self.from_node_id, "to_node_id": self.to_node_id, "kind": self.kind}


@dataclass
class AncestryGraph:
    nodes: dict[str, TreeNode]
    edges: list[TreeEdge]
    num_generations: int
    num_lanes: int
    checkpoint_matches: int
    checkpoint_fitness_mismatches: int
    orphan_individuals: int
    invalid_reconstructed: int = 0
    passthrough_created: int = 0


def _reconstruct_invalid_ind_ids(rows: list[dict], pop: int, lam: int) -> dict[int, int]:
    """Return {ind_id: generation} for individuals that were created but
    failed genome validity (see genome_adapter.py's `_MAX_OFFSPRING_ATTEMPTS`
    retry loop / shared.py's `make_offspring`) and so were never evaluated or
    written to run_data.jsonl at all.

    `ind_id` is a simple monotonic counter (`shared._next_ind_id`) incremented
    once per individual *created*, valid or not -- gen 0's `pop` individuals
    (via `create_individual`, which retries internally until valid, so this
    block is always gap-free) get ids [1, pop]; each subsequent generation's
    `reproduction()` call creates exactly `lam` new ids (valid or not) in one
    contiguous block right after the previous one, via `make_offspring`'s
    "keep appending until len(offspring) == lam" loop. So any id within a
    generation's predicted contiguous block that never appears as any row's
    `ind_id` anywhere in run_data.jsonl was created, failed validity, and was
    silently dropped. Verified exactly (0 mismatches, 505/505 ids) against a
    real completed run.

    This is inherently approximate for a run's *final* generation if the run
    was stopped mid-generation (time limit) rather than by exhausting the
    budget -- a "gap" there could be an individual that was simply never
    reached, not one that failed validity. There is no way to distinguish
    the two from run_data.jsonl alone.
    """
    if not rows:
        return {}

    first_seen_gen: dict[int, int] = {}
    for row in rows:
        iid = row["ind_id"]
        if iid not in first_seen_gen:
            first_seen_gen[iid] = row["gen"]
    seen_ids = set(first_seen_gen)
    max_gen = max(row["gen"] for row in rows)

    invalid: dict[int, int] = {}
    for gen in range(0, max_gen + 1):
        if gen == 0:
            lo, hi = 1, pop
        else:
            lo = pop + lam * (gen - 1) + 1
            hi = pop + lam * gen
        for iid in range(lo, hi + 1):
            if iid not in seen_ids:
                invalid[iid] = gen
    return invalid


@dataclass
class _Entity:
    """One individual's observed lifespan, prior to lane assignment."""
    ind_id: int
    origin_gen: int
    last_gen: int
    real_gens: dict[int, dict]      # gen -> augmented row (empty for invalid entities)
    parent_ids: list[int]
    is_invalid: bool


def build_ancestry_graph(
    rows: list[dict],
    ckpt_index: dict[tuple[int, int], Path],
    pop: Optional[int] = None,
    lam: Optional[int] = None,
) -> AncestryGraph:
    checkpoint_matches = 0
    checkpoint_fitness_mismatches = 0

    # Pass 1: attach a checkpoint_dir to each row (positional correspondence
    # -- Nth row, 0-indexed, within a generation's block == checkpoints/
    # gen{G}_body{N}), and group rows by ind_id. Row order within a
    # generation must be preserved for this, hence no re-sorting here.
    gen_body_counters: dict[int, int] = {}
    rows_by_ind: dict[int, list[dict]] = defaultdict(list)
    max_gen = 0
    for row in rows:
        gen = row["gen"]
        max_gen = max(max_gen, gen)
        body_idx = gen_body_counters.get(gen, 0)
        gen_body_counters[gen] = body_idx + 1
        ckpt = ckpt_index.get((gen, body_idx))

        augmented = dict(row)
        augmented["_checkpoint_dir"] = ckpt
        if ckpt is not None:
            checkpoint_matches += 1
            meta_path = ckpt / "meta.json"
            if meta_path.exists() and row.get("fitness") is not None:
                try:
                    meta_fitness = json.loads(meta_path.read_text()).get("fitness")
                    if meta_fitness is not None and abs(meta_fitness - row["fitness"]) > 1e-6:
                        checkpoint_fitness_mismatches += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        rows_by_ind[row["ind_id"]].append(augmented)

    # Pass 2: build one _Entity per individual -- real ones (origin/last gen
    # from their logged rows) and invalid ones (single-generation span, no
    # real_gens, no known parent_ids -- see module docstring / user decision:
    # never-evaluated individuals have zero logged data to connect them to a
    # parent).
    entities: dict[int, _Entity] = {}
    for ind_id, ind_rows in rows_by_ind.items():
        ind_rows.sort(key=lambda r: r["gen"])
        real_gens = {r["gen"]: r for r in ind_rows}
        entities[ind_id] = _Entity(
            ind_id=ind_id,
            origin_gen=ind_rows[0]["gen"],
            last_gen=ind_rows[-1]["gen"],
            real_gens=real_gens,
            parent_ids=list(ind_rows[0].get("parent_ids") or []),
            is_invalid=False,
        )

    invalid_reconstructed = 0
    if pop is not None and lam is not None:
        for iid, gen in _reconstruct_invalid_ind_ids(rows, pop, lam).items():
            entities[iid] = _Entity(
                ind_id=iid, origin_gen=gen, last_gen=gen,
                real_gens={}, parent_ids=[], is_invalid=True,
            )
            invalid_reconstructed += 1

    # Pass 3: lane assignment. Process generations in order; free lanes
    # whose occupant's span ended last generation before assigning lanes to
    # entities newly starting their span this generation (reuse-first, per
    # user decision), so the tree stays close to the actual concurrent
    # population width instead of growing unboundedly.
    entities_by_origin: dict[int, list[int]] = defaultdict(list)
    for e in entities.values():
        entities_by_origin[e.origin_gen].append(e.ind_id)

    active_lanes: dict[int, int] = {}   # ind_id -> lane, currently in-span
    free_lanes: list[int] = []
    next_lane = 0
    lane_of: dict[int, int] = {}

    for gen in range(0, max_gen + 1):
        for ind_id in [i for i, lane in active_lanes.items() if entities[i].last_gen == gen - 1]:
            free_lanes.append(active_lanes.pop(ind_id))
        free_lanes.sort()
        for ind_id in sorted(entities_by_origin.get(gen, [])):
            if free_lanes:
                lane = free_lanes.pop(0)
            else:
                lane = next_lane
                next_lane += 1
            active_lanes[ind_id] = lane
            lane_of[ind_id] = lane

    num_lanes = next_lane

    # Pass 4: create one node per entity per generation in its span (real if
    # logged that generation, else a synthetic passthrough placeholder).
    nodes: dict[str, TreeNode] = {}
    node_id_by_ind_gen: dict[tuple[int, int], str] = {}
    passthrough_created = 0
    for e in entities.values():
        for gen in range(e.origin_gen, e.last_gen + 1):
            node_id = f"g{gen}_i{e.ind_id}"
            row = e.real_gens.get(gen)
            is_passthrough = not e.is_invalid and row is None
            if is_passthrough:
                passthrough_created += 1
            node = TreeNode(
                node_id=node_id, gen=gen, ind_id=e.ind_id, lane=lane_of[e.ind_id],
                fitness=row.get("fitness") if row else None,
                yz_symmetry=row.get("yz_symmetry") if row else None,
                genome_hash=row.get("genome_hash") if row else None,
                parent_ids_raw=list(e.parent_ids) if gen == e.origin_gen else [],
                is_invalid=e.is_invalid,
                is_passthrough=is_passthrough,
            )
            if row is not None:
                node.checkpoint_dir = row.get("_checkpoint_dir")
            nodes[node_id] = node
            node_id_by_ind_gen[(e.ind_id, gen)] = node_id

    # Pass 5: edges. Reproduction (solid) at an entity's origin, from each
    # parent's node at gen-1 -- guaranteed present, since a parent's span
    # must cover gen-1 (it was alive and tournament-selected as a parent
    # this round). Survival (dotted) for every continuation within an
    # entity's own span, gen-1 -> gen, always a same-lane single-column hop
    # by construction.
    edges: list[TreeEdge] = []
    orphan_individuals = 0
    for e in entities.values():
        origin_node_id = node_id_by_ind_gen[(e.ind_id, e.origin_gen)]
        origin_node = nodes[origin_node_id]
        if e.parent_ids:
            for pid in e.parent_ids:
                parent_node_id = node_id_by_ind_gen.get((pid, e.origin_gen - 1))
                if parent_node_id is not None:
                    origin_node.parents.append({"parent_node_id": parent_node_id, "kind": "reproduction"})
                    edges.append(TreeEdge(parent_node_id, origin_node_id, "reproduction"))
                else:
                    origin_node.parents.append({"parent_node_id": None, "kind": "missing"})
                    origin_node.parent_data_missing = True
        elif e.origin_gen != 0 and not e.is_invalid:
            orphan_individuals += 1
            logger.warning(
                "Individual (origin gen=%d, ind_id=%d) has empty parent_ids and gen>0 "
                "-- treating as an unexpected root.", e.origin_gen, e.ind_id,
            )

        for gen in range(e.origin_gen + 1, e.last_gen + 1):
            prev_node_id = node_id_by_ind_gen[(e.ind_id, gen - 1)]
            this_node_id = node_id_by_ind_gen[(e.ind_id, gen)]
            nodes[this_node_id].parents.append({"parent_node_id": prev_node_id, "kind": "survival"})
            edges.append(TreeEdge(prev_node_id, this_node_id, "survival"))

    # has_images / genome_image / phenotype_image are filled in later by
    # ancestry_tree_prepare.py once images are actually rendered (or found
    # cached) for each node's checkpoint.

    return AncestryGraph(
        nodes=nodes,
        edges=edges,
        num_generations=max_gen + 1,
        num_lanes=num_lanes,
        checkpoint_matches=checkpoint_matches,
        checkpoint_fitness_mismatches=checkpoint_fitness_mismatches,
        orphan_individuals=orphan_individuals,
        invalid_reconstructed=invalid_reconstructed,
        passthrough_created=passthrough_created,
    )


# ── Manifest I/O ─────────────────────────────────────────────────────────────


def write_manifest(path: Path, run_meta: dict, nodes: list[TreeNode], edges: list[TreeEdge]) -> None:
    manifest = {
        "schema_version": 1,
        "run": run_meta,
        "nodes": [n.to_json_dict() for n in nodes],
        "edges": [e.to_json_dict() for e in edges],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
