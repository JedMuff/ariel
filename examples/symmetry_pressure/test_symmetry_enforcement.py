"""
Verify that symmetry-enforced genome generation, mutation, and crossover
always produce bilaterally-symmetric bodies.

For each mirror axis (y_zero, x_equals_y) and each of the three code paths
(random generation, mutation, crossover), generates a batch of genomes and
checks `bilateral_symmetry_score` == 1.0 for every one. A genome with zero
off-midline modules is treated as trivially symmetric even though the score
function itself returns 0.0 for that case (its documented convention, shared
with the pre-existing hand-crafted "asymmetric_spine_only" case in
test_symmetry_metric.py) — this script accounts for that instead of treating
it as a failure.

Usage:
    cd /path/to/ariel
    python examples/symmetry_pressure/test_symmetry_enforcement.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from ariel.ec.genotypes.tree.operators import (
    crossover_subtree_symmetric,
    random_tree_symmetric,
)
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from shared import bilateral_symmetry_score, mutate_morph_symmetric

N_TRIALS = 30
N_MUTATION_STEPS = 10
MAX_MODULES = 25
MAX_DEPTH = 25

AXES: list[tuple[str, MirrorAxis]] = [
    ("y_zero", MirrorAxis.Y_ZERO),
    ("x_equals_y", MirrorAxis.X_EQUALS_Y),
]


def has_off_midline_modules(genome_dict: dict, axis_name: str) -> bool:
    """Whether the score has any off-midline modules to compare (see module docstring)."""
    from ariel.ec.genotypes.tree.tree_genome import TreeGenome

    mirror_axis = MirrorAxis.Y_ZERO if axis_name == "y_zero" else MirrorAxis.X_EQUALS_Y
    graph = TreeGenome.from_dict(genome_dict).to_networkx()
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if not roots:
        return False
    root = roots[0]

    from ariel.ec.genotypes.tree.symmetry import mirror_face

    paths: dict = {root: ()}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in graph.successors(node):
            if child in paths:
                continue
            face = (graph.get_edge_data(node, child) or {}).get("face", "FRONT")
            paths[child] = (*paths[node], face)
            queue.append(child)

    def mirror_path(path: tuple) -> tuple:
        return tuple(mirror_face(f, mirror_axis, is_outer=(i == 0)) for i, f in enumerate(path))

    return any(path != mirror_path(path) for path in paths.values())


def is_symmetric(genome_dict: dict, axis_name: str) -> bool:
    score = bilateral_symmetry_score(genome_dict, axis=axis_name)
    if score == 1.0:
        return True
    return score == 0.0 and not has_off_midline_modules(genome_dict, axis_name)


def check_batch(label: str, axis_name: str, genome_dicts: list[dict]) -> tuple[int, int]:
    n_pass = 0
    for gd in genome_dicts:
        validate_genome_dict(gd)
        if is_symmetric(gd, axis_name):
            n_pass += 1
    n_total = len(genome_dicts)
    status = "PASS" if n_pass == n_total else "FAIL"
    print(f"  {label:<28} {n_pass:>4}/{n_total:<4}  {status}")
    return n_pass, n_total


def main() -> None:
    random.seed(0)
    np_rng = np.random.default_rng(0)

    print(f"{'Check':<30} {'Result':>10}  Status")
    print("-" * 55)

    grand_pass = 0
    grand_total = 0

    for axis_name, axis in AXES:
        print(f"\n[{axis_name}]")

        # ── Random generation ────────────────────────────────────────────
        randoms = [random_tree_symmetric(MAX_MODULES, axis).to_dict() for _ in range(N_TRIALS)]
        p, t = check_batch("random generation", axis_name, randoms)
        grand_pass += p
        grand_total += t

        # ── Mutation (repeated) ──────────────────────────────────────────
        from ariel.ec.genotypes.tree.tree_genome import TreeGenome

        mutated: list[dict] = []
        for _ in range(N_TRIALS):
            genome = random_tree_symmetric(MAX_MODULES, axis)
            for _ in range(N_MUTATION_STEPS):
                genome = mutate_morph_symmetric(genome, np_rng, MAX_MODULES, axis)
            mutated.append(genome.to_dict())
        p, t = check_batch(f"mutation ({N_MUTATION_STEPS} steps)", axis_name, mutated)
        grand_pass += p
        grand_total += t

        # ── Crossover ─────────────────────────────────────────────────────
        crossed: list[dict] = []
        for _ in range(N_TRIALS):
            g1 = random_tree_symmetric(MAX_MODULES, axis)
            g2 = random_tree_symmetric(MAX_MODULES, axis)
            c1, c2 = crossover_subtree_symmetric(g1, g2, axis)
            crossed.append(c1.to_dict())
            crossed.append(c2.to_dict())
        p, t = check_batch("crossover", axis_name, crossed)
        grand_pass += p
        grand_total += t

    print()
    print(f"Overall: {grand_pass}/{grand_total}", "ALL PASS" if grand_pass == grand_total else "FAILURES detected")
    sys.exit(0 if grand_pass == grand_total else 1)


if __name__ == "__main__":
    main()
