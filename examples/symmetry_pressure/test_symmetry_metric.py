"""
Visual + numeric test of bilateral_symmetry_score.

For each test morphology:
  - Print old MorphologicalMeasures.yz_symmetry  (expected: always 0 due to bug)
  - Print new bilateral_symmetry_score            (expected: correct value)
  - Save a birds-eye PNG to /tmp/sym_test/

Usage:
    cd /path/to/ariel
    python examples/symmetry_pressure/test_symmetry_metric.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mujoco
import numpy as np

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.utils.morphological_descriptor import MorphologicalMeasures
from shared import bilateral_symmetry_score, build_world_for_body, random_tree

OUT_DIR = Path("/tmp/sym_test")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ── Hand-crafted test cases ───────────────────────────────────────────────────
#
# Coordinate convention (from bilateral_symmetry_score BFS):
#   FRONT/BACK  → ±x   (forward/backward)
#   RIGHT/LEFT  → ±y   (left-right)
#   TOP/BOTTOM  → ±z   (up-down)
#
# Bilateral symmetry = left-right mirror about the XZ plane (y=0).

CASES: list[tuple[str, dict, float]] = [

    # ── Perfectly symmetric ──────────────────────────────────────────────────

    ("perfect_2arm", {
        "nodes": {
            "0": {"type": "CORE",  "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},   # right arm
            "2": {"type": "HINGE", "rotation": "DEG_0"},   # left arm (mirror)
        },
        "edges": [
            {"parent": 0, "child": 1, "face": "RIGHT"},
            {"parent": 0, "child": 2, "face": "LEFT"},
        ],
    }, 1.0),

    ("perfect_4arm", {
        "nodes": {
            "0": {"type": "CORE",  "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},
            "2": {"type": "HINGE", "rotation": "DEG_0"},
            "3": {"type": "BRICK", "rotation": "DEG_0"},
            "4": {"type": "BRICK", "rotation": "DEG_0"},
            "5": {"type": "HINGE", "rotation": "DEG_0"},
            "6": {"type": "HINGE", "rotation": "DEG_0"},
        },
        "edges": [
            {"parent": 0, "child": 1, "face": "RIGHT"},
            {"parent": 0, "child": 2, "face": "LEFT"},
            {"parent": 0, "child": 3, "face": "FRONT"},
            {"parent": 0, "child": 4, "face": "BACK"},
            {"parent": 1, "child": 5, "face": "FRONT"},
            {"parent": 2, "child": 6, "face": "FRONT"},
        ],
    }, 1.0),

    ("perfect_deep", {
        # CORE - BRICK(RIGHT) - HINGE(RIGHT)
        #      - BRICK(LEFT)  - HINGE(RIGHT)
        "nodes": {
            "0": {"type": "CORE",  "rotation": "DEG_0"},
            "1": {"type": "BRICK", "rotation": "DEG_0"},
            "2": {"type": "BRICK", "rotation": "DEG_0"},
            "3": {"type": "HINGE", "rotation": "DEG_0"},
            "4": {"type": "HINGE", "rotation": "DEG_0"},
        },
        "edges": [
            {"parent": 0, "child": 1, "face": "RIGHT"},
            {"parent": 0, "child": 2, "face": "LEFT"},
            {"parent": 1, "child": 3, "face": "FRONT"},
            {"parent": 2, "child": 4, "face": "FRONT"},
        ],
    }, 1.0),

    # ── Partially symmetric ──────────────────────────────────────────────────

    ("partial_one_extra_arm", {
        # Right arm has an extra child, left arm does not
        "nodes": {
            "0": {"type": "CORE",  "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},   # right
            "2": {"type": "HINGE", "rotation": "DEG_0"},   # left (mirror of 1)
            "3": {"type": "BRICK", "rotation": "DEG_0"},   # extra on right — breaks sym
        },
        "edges": [
            {"parent": 0, "child": 1, "face": "RIGHT"},
            {"parent": 0, "child": 2, "face": "LEFT"},
            {"parent": 1, "child": 3, "face": "FRONT"},
        ],
    }, 0.5),   # 1 matched pair, 2 off-midline total → 0.5

    ("partial_type_mismatch", {
        # Arms present on both sides but different types
        "nodes": {
            "0": {"type": "CORE",  "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},   # right
            "2": {"type": "BRICK", "rotation": "DEG_0"},   # left — wrong type
        },
        "edges": [
            {"parent": 0, "child": 1, "face": "RIGHT"},
            {"parent": 0, "child": 2, "face": "LEFT"},
        ],
    }, 0.0),

    # ── Completely asymmetric ────────────────────────────────────────────────

    ("asymmetric_spine_only", {
        # All modules on the midline (y=0) — no off-midline modules → score 0.0
        "nodes": {
            "0": {"type": "CORE",  "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},
            "2": {"type": "HINGE", "rotation": "DEG_0"},
            "3": {"type": "HINGE", "rotation": "DEG_0"},
            "4": {"type": "HINGE", "rotation": "DEG_0"},
        },
        "edges": [
            {"parent": 0, "child": 1, "face": "FRONT"},
            {"parent": 1, "child": 2, "face": "FRONT"},
            {"parent": 2, "child": 3, "face": "FRONT"},
            {"parent": 3, "child": 4, "face": "FRONT"},
        ],
    }, 0.0),

    ("asymmetric_one_arm", {
        # Arm only on the right, nothing on the left
        "nodes": {
            "0": {"type": "CORE",  "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},
            "2": {"type": "HINGE", "rotation": "DEG_0"},
            "3": {"type": "HINGE", "rotation": "DEG_0"},
            "4": {"type": "HINGE", "rotation": "DEG_0"},
        },
        "edges": [
            {"parent": 0, "child": 1, "face": "RIGHT"},
            {"parent": 1, "child": 2, "face": "FRONT"},
            {"parent": 2, "child": 3, "face": "FRONT"},
            {"parent": 0, "child": 4, "face": "FRONT"},
        ],
    }, 0.0),

    # ── Random morphologies ──────────────────────────────────────────────────

    ("random_small",  random_tree(6).to_dict(),  None),
    ("random_medium", random_tree(12).to_dict(), None),
    ("random_large",  random_tree(20).to_dict(), None),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def old_symmetry(genome_dict: dict) -> float:
    """MorphologicalMeasures.yz_symmetry — the broken baseline."""
    try:
        graph = TreeGenome.from_dict(genome_dict).to_networkx()
        m = MorphologicalMeasures(graph)
        return float(m.yz_symmetry)
    except Exception as e:
        return float("nan")


def render_birdseye(genome_dict: dict, out_path: Path, label: str) -> bool:
    try:
        model, data, _, _ = build_world_for_body(
            genome_dict, reach_radius=0.2, arena_radius=1.5, add_food_target=False
        )
        renderer = mujoco.Renderer(model, height=512, width=512)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="overview_cam")
        frame = renderer.render()
        renderer.close()

        import cv2
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out_path), bgr)
        return True
    except Exception as e:
        print(f"    render failed: {e}")
        return False


# ── Run ───────────────────────────────────────────────────────────────────────

print(f"{'Case':<30}  {'Old (yz)':>10}  {'New':>10}  {'Expected':>10}  {'Pass?'}")
print("-" * 75)

all_pass = True
for name, genome_dict, expected in CASES:
    old = old_symmetry(genome_dict)
    new = bilateral_symmetry_score(genome_dict)

    if expected is not None:
        ok = abs(new - expected) < 1e-9
        all_pass = all_pass and ok
        pass_str = "PASS" if ok else "FAIL"
        exp_str  = f"{expected:.3f}"
    else:
        pass_str = "—"
        exp_str  = "—"

    print(f"{name:<30}  {old:>10.3f}  {new:>10.3f}  {exp_str:>10}  {pass_str}")

    label = f"{name}  new={new:.2f}"
    img_path = OUT_DIR / f"{name}.png"
    render_birdseye(genome_dict, img_path, label)

print()
print(f"Images written to {OUT_DIR}/")
print("Overall:", "ALL PASS" if all_pass else "FAILURES detected")
