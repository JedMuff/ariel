"""
Interactively view symmetry-enforced bodies in a live MuJoCo window.

For visual (human) verification that random/mutated/crossed-over bodies
produced by the symmetry-enforcement machinery
(ariel.ec.genotypes.tree.operators.random_tree_symmetric /
shared.mutate_morph_symmetric / operators.crossover_subtree_symmetric) are
actually bilaterally symmetric — rotate the camera around the body in the
window and check it mirrors cleanly, including that 45/90-degree-rotated
hinges on one side look like true mirror images of the other side.

Requires a display (mujoco.viewer.launch opens an interactive window) — run
locally, not over the headless SLURM/MUJOCO_GL=egl path.

Usage:
    cd /path/to/ariel
    python examples/symmetry_pressure/view_symmetric_morphs.py \\
        --axis y_zero --source random

    python examples/symmetry_pressure/view_symmetric_morphs.py \\
        --axis x_equals_y --source mutated --num-mutations 5

    python examples/symmetry_pressure/view_symmetric_morphs.py \\
        --axis y_zero --source crossover --child 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mujoco
import numpy as np
from mujoco import viewer

from ariel.ec.genotypes.tree.operators import (
    crossover_subtree_symmetric,
    random_tree_symmetric,
)
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from shared import bilateral_symmetry_score, build_world_for_body, mutate_morph_symmetric

parser = argparse.ArgumentParser(description="Live-view a symmetry-enforced body")
parser.add_argument("--axis", choices=["y_zero", "x_equals_y"], required=True)
parser.add_argument("--source", choices=["random", "mutated", "crossover"], required=True)
parser.add_argument("--max-modules", type=int, default=25)
parser.add_argument("--seed", type=int, default=None, help="Omit for a new random body each run")
parser.add_argument("--num-mutations", type=int, default=1, help="--source mutated only")
parser.add_argument("--child", type=int, choices=[0, 1], default=0, help="--source crossover only")
args = parser.parse_args()

if args.seed is not None:
    import random

    random.seed(args.seed)

axis = MirrorAxis.Y_ZERO if args.axis == "y_zero" else MirrorAxis.X_EQUALS_Y


def build_genome():
    if args.source == "random":
        return random_tree_symmetric(args.max_modules, axis)

    if args.source == "mutated":
        genome = random_tree_symmetric(args.max_modules, axis)
        rng = np.random.default_rng(args.seed)
        for _ in range(args.num_mutations):
            genome = mutate_morph_symmetric(genome, rng, args.max_modules, axis)
        return genome

    # crossover
    g1 = random_tree_symmetric(args.max_modules, axis)
    g2 = random_tree_symmetric(args.max_modules, axis)
    c1, c2 = crossover_subtree_symmetric(g1, g2, axis)
    return c1 if args.child == 0 else c2


def main() -> None:
    genome = build_genome()
    genome_dict = genome.to_dict()

    score = bilateral_symmetry_score(genome_dict, axis=args.axis)
    print(f"axis={args.axis} source={args.source} n_modules={len(genome.nodes)} "
          f"bilateral_symmetry_score={score:.3f}")
    print("Rotate the camera in the viewer window to visually confirm mirror symmetry.")

    model, data, _, _ = build_world_for_body(
        genome_dict, reach_radius=0.2, arena_radius=1.5, add_food_target=False
    )
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    viewer.launch(model, data)


if __name__ == "__main__":
    main()
