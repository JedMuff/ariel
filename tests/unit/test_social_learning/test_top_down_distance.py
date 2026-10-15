"""Tests for the TOPDOWN novelty metric and the novelty_history table
(examples/d_social_learning)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "d_social_learning"))

from core.novelty_history import (  # noqa: E402
    COLUMNS,
    history_entry,
    read_history,
    read_run_history,
    write_rows,
)
from sim_ariel.tree_edit_distance import top_down_tree_distance  # noqa: E402


def genome(nodes: dict[int, tuple[str, str]], edges: list[tuple[int, int, str]]) -> dict:
    return {
        "nodes": {str(k): {"type": t, "rotation": r} for k, (t, r) in nodes.items()},
        "edges": [{"parent": p, "child": c, "face": f} for p, c, f in edges],
    }


CORE = ("CORE", "DEG_0")
BRICK = ("BRICK", "DEG_0")
HINGE = ("HINGE", "DEG_0")

# core -FRONT-> brick -FRONT-> hinge
CHAIN = genome({0: CORE, 1: BRICK, 2: HINGE}, [(0, 1, "FRONT"), (1, 2, "FRONT")])


def test_identical_trees_are_zero() -> None:
    assert top_down_tree_distance(CHAIN, CHAIN) == 0.0


def test_node_ids_do_not_matter() -> None:
    renumbered = genome({0: CORE, 7: BRICK, 3: HINGE}, [(0, 7, "FRONT"), (7, 3, "FRONT")])
    assert top_down_tree_distance(CHAIN, renumbered) == 0.0


def test_leaf_insertion_costs_node_plus_edge() -> None:
    longer = genome(
        {0: CORE, 1: BRICK, 2: HINGE, 3: BRICK},
        [(0, 1, "FRONT"), (1, 2, "FRONT"), (2, 3, "FRONT")],
    )
    assert top_down_tree_distance(CHAIN, longer) == 2.0
    assert top_down_tree_distance(longer, CHAIN) == 2.0


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        # type change on one node
        (genome({0: CORE, 1: HINGE, 2: HINGE}, [(0, 1, "FRONT"), (1, 2, "FRONT")]), 1.0),
        # rotation change on one node
        (genome({0: CORE, 1: ("BRICK", "DEG_90"), 2: HINGE}, [(0, 1, "FRONT"), (1, 2, "FRONT")]), 1.0),
        # child attached to a different face
        (genome({0: CORE, 1: BRICK, 2: HINGE}, [(0, 1, "LEFT"), (1, 2, "FRONT")]), 1.0),
    ],
)
def test_single_substitutions_cost_one(other: dict, expected: float) -> None:
    assert top_down_tree_distance(CHAIN, other) == expected


def test_whole_subtree_deletion() -> None:
    core_only = genome({0: CORE}, [])
    # Two non-root nodes, each node + its parent edge.
    assert top_down_tree_distance(CHAIN, core_only) == 4.0


def test_children_are_matched_optimally_across_faces() -> None:
    a = genome({0: CORE, 1: BRICK, 2: HINGE}, [(0, 1, "LEFT"), (0, 2, "RIGHT")])
    b = genome({0: CORE, 1: HINGE, 2: BRICK}, [(0, 1, "LEFT"), (0, 2, "RIGHT")])
    # Matching by face relabels both nodes (2); matching by type crosses both
    # faces (2). Either way 2 -- the assignment must not pay for both.
    assert top_down_tree_distance(a, b) == 2.0


def test_novelty_history_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "database.db"
    entry = history_entry(3, "TOPDOWN", 4.5, float("-inf"), True)
    assert entry["fitness"] is None  # -inf isn't valid JSON
    rows = pd.DataFrame(
        [(3, 11, "TOPDOWN", 4.5, None, 1, "live"), (3, 12, "TOPDOWN", 2.0, 0.5, 0, "live")],
        columns=COLUMNS,
    )
    write_rows(db, rows)
    write_rows(tmp_path / "database_part2.db", rows.iloc[:1])  # duplicated restored row
    assert len(read_history(db, "TOPDOWN")) == 2
    assert read_history(db, "STRUCT").empty
    assert len(read_run_history(tmp_path, "TOPDOWN")) == 2
