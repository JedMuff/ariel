"""
The five body genomes, behind one interface, and the shared rules for valid
bodies and offspring.

  tree            Standard: unconstrained tree genome
  tree_symmetric  Symmetric: mirror-symmetric about the y = 0 plane
  tree_tied       Tied-matched: each placed module gets a randomly placed twin
                  with probability 0.47 (~64% of modules tied, as mirrored in
                  tree_symmetric), never at its mirror position
                  (ariel.ec.genotypes.tree.tying)
  tree_tied_full  Tied-full: tree_tied with every module tied (probability 1)
  grn             Indirect: the R2 genetic regulatory network of Miras and
                  Feenstra (2026), grown in 2D (ariel.ec.genotypes.grn)

A genome is a dict: the tree genome of the body it builds (nodes, edges), plus
"ties" (tied genomes) or "grn" (the regulatory genome the body was grown from).
The simulator reads only nodes and edges, so every genome builds the same way.

Valid body (is_valid): at most PROTOCOL.max_modules modules besides the core,
at least min_hinges hinges, depth at most max_depth, and it builds. The
genomes' own operators do not bound body size (tied twins, mirroring and
subtree crossover can all grow a body), so the cap is enforced here, the same
way for every genome: a random body is redrawn until valid, and a child is
redrawn from its parents (fresh crossover and mutation) until valid. A child
that fails offspring_attempts times becomes a mutated clone of one parent, or
failing that an exact copy ("fallback").
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ariel.body_phenotypes.robogen_lite.config import ALLOWED_ROTATIONS, IDX_OF_CORE, ModuleType
from ariel.ec.genotypes.tree.collision import prune_colliding_subtrees, prune_colliding_subtrees_symmetric
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    crossover_subtree,
    crossover_subtree_symmetric,
    get_tree_depth,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    random_tree_symmetric,
)
from ariel.ec.genotypes.tree.symmetry import MirrorAxis, mirror_face, symmetrize_genome
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.tying import crossover_tied, mutate_morph_tied, random_tree_tied, tied_fraction
from ariel.ec.genotypes.tree.validation import validate_genome_dict

import sim  # noqa: F401  (sets the zoo module geometry the collision checks use)
from config import GENOME_TYPES, PROTOCOL, TIE_PROB_FULL, TIE_PROB_MATCHED, Protocol

AXIS = MirrorAxis.Y_ZERO


@dataclass(frozen=True)
class GenomeOps:
    """random() -> genome; crossover(a, b) -> one child; mutate(genome, rng) -> genome.
    None of them changes its arguments."""

    name: str
    random: Callable[[], dict]
    crossover: Callable[[dict, dict], dict]
    mutate: Callable[[dict, np.random.Generator], dict]


# ── Tree mutation (Standard and Symmetric) ────────────────────────────────────


def _mutate_tree(genome: TreeGenome, rng: np.random.Generator, max_modules: int) -> TreeGenome:
    new = copy.deepcopy(genome)
    op = rng.choice(["point", "subtree", "shrink", "hoist"], p=[0.45, 0.35, 0.1, 0.1])
    if op == "point":
        mutate_replace_node(new)
    elif op == "subtree":
        mutate_subtree_replacement(new, max_modules=max_modules)
    elif op == "shrink":
        mutate_shrink(new)
    else:
        mutate_hoist(new)

    if rng.random() < 0.25:
        noncore = [nid for nid in new.nodes if nid != IDX_OF_CORE]
        if noncore:
            nid = random.choice(noncore)
            rots = [r.name for r in ALLOWED_ROTATIONS[ModuleType[new.nodes[nid]["type"]]]]
            if rots:
                new.nodes[nid]["rotation"] = random.choice(rots)

    _prune_invalid_edges(new)
    # Point/rotation changes and grafted subtrees can make parts overlap; drop
    # any subtree that now collides (FCL, real module geometry).
    prune_colliding_subtrees(new)
    return new


def _tree_ops(max_modules: int) -> GenomeOps:
    def crossover(a: dict, b: dict) -> dict:
        c1, c2 = crossover_subtree(TreeGenome.from_dict(a), TreeGenome.from_dict(b))
        return random.choice((c1, c2)).to_dict()

    return GenomeOps(
        "tree",
        random=lambda: random_tree(max_modules).to_dict(),
        crossover=crossover,
        mutate=lambda g, rng: _mutate_tree(TreeGenome.from_dict(g), rng, max_modules).to_dict(),
    )


def _symmetric_ops(max_modules: int) -> GenomeOps:
    def crossover(a: dict, b: dict) -> dict:
        c1, c2 = crossover_subtree_symmetric(TreeGenome.from_dict(a), TreeGenome.from_dict(b), AXIS)
        return random.choice((c1, c2)).to_dict()

    def mutate(g: dict, rng: np.random.Generator) -> dict:
        # Mirroring can create new overlaps (an arm hitting its own mirror
        # image), so colliding subtrees are pruned again, pairwise.
        sym = symmetrize_genome(_mutate_tree(TreeGenome.from_dict(g), rng, max_modules), AXIS)
        prune_colliding_subtrees_symmetric(sym, AXIS)
        return sym.to_dict()

    return GenomeOps(
        "tree_symmetric",
        random=lambda: random_tree_symmetric(max_modules, AXIS).to_dict(),
        crossover=crossover,
        mutate=mutate,
    )


def _tied_ops(name: str, max_modules: int, tie_prob: float) -> GenomeOps:
    return GenomeOps(
        name,
        random=lambda: random_tree_tied(max_modules, tie_prob).to_dict(),
        crossover=lambda a, b: random.choice(crossover_tied(a, b)),
        mutate=lambda g, rng: mutate_morph_tied(g, rng, max_modules, tie_prob),
    )


def _grn_ops(max_modules: int) -> GenomeOps:
    """R2 GRN. The genome is the regulatory values ("grn"); the body is grown
    from them (the original's cap counts the core, hence + 1) and, as for the
    tree genomes, subtrees that collide in ARIEL's geometry are dropped. The
    operators draw from Python's random, as the originals do."""
    from ariel.ec.genotypes.grn import GRN_R2, crossover_grn, develop_body, mutate_grn, random_grn  # noqa: PLC0415

    def body(values: list[float]) -> dict:
        tree = TreeGenome.from_dict(develop_body(values, GRN_R2, max_modules=max_modules + 1))
        prune_colliding_subtrees(tree)
        return {**tree.to_dict(), "grn": values}

    return GenomeOps(
        "grn",
        random=lambda: body(random_grn(GRN_R2, random)),
        crossover=lambda a, b: body(crossover_grn(a["grn"], b["grn"], GRN_R2, random)),
        mutate=lambda g, rng: body(mutate_grn(g["grn"], GRN_R2, random)),  # noqa: ARG005
    )


def genome_ops(name: str, protocol: Protocol = PROTOCOL) -> GenomeOps:
    m = protocol.max_modules
    if name == "tree":
        return _tree_ops(m)
    if name == "tree_symmetric":
        return _symmetric_ops(m)
    if name == "tree_tied":
        return _tied_ops(name, m, TIE_PROB_MATCHED)
    if name == "tree_tied_full":
        return _tied_ops(name, m, TIE_PROB_FULL)
    if name == "grn":
        return _grn_ops(m)
    raise ValueError(f"Unknown genome type: {name!r} (expected one of {GENOME_TYPES})")


# ── Body measures used by the validity check and the logs ─────────────────────


def module_count(genome: dict) -> int:
    """Modules besides the core."""
    return max(len(genome.get("nodes", {})) - 1, 0)


def hinge_count(genome: dict) -> int:
    return sum(1 for d in genome.get("nodes", {}).values() if d.get("type") == "HINGE")


def is_valid(genome: dict, protocol: Protocol = PROTOCOL) -> bool:
    if not protocol.min_hinges <= hinge_count(genome) or module_count(genome) > protocol.max_modules:
        return False
    try:
        validate_genome_dict(genome)
        tree = TreeGenome.from_dict(genome)
    except Exception:
        return False
    return get_tree_depth(tree) <= protocol.max_depth and sim.body_spec(genome) is not None


def bilateral_symmetry(genome: dict) -> float:
    """Share of off-midline modules with a same-type mirror partner about y = 0
    (two modules are partners if one's path of faces from the core is the
    other's with faces mirrored, the rule symmetrize_genome enforces); 0.0 if
    there are none."""
    graph = TreeGenome.from_dict(genome).to_networkx()
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if not roots:
        return 0.0
    paths: dict = {roots[0]: ()}
    queue = [roots[0]]
    while queue:
        node = queue.pop(0)
        for child in graph.successors(node):
            if child not in paths:
                paths[child] = (*paths[node], (graph.get_edge_data(node, child) or {}).get("face", "FRONT"))
                queue.append(child)
    by_path = {p: graph.nodes[n].get("type") for n, p in paths.items()}
    sym = off = 0
    seen: set = set()
    for path, t in by_path.items():
        mpath = tuple(mirror_face(f, AXIS, is_outer=(i == 0)) for i, f in enumerate(path))
        if path == mpath or path in seen:
            continue
        seen |= {path, mpath}
        off += 1
        sym += by_path.get(mpath) == t
    return sym / off if off else 0.0


def describe(genome: dict) -> dict:
    """The per-body numbers logged with every evaluation."""
    return {
        "n_modules": module_count(genome),
        "n_hinges": hinge_count(genome),
        "symmetry": round(bilateral_symmetry(genome), 4),
        "tied_frac": round(tied_fraction(genome), 4) if "ties" in genome else None,
    }


# ── Random bodies and offspring ───────────────────────────────────────────────


def random_body(ops: GenomeOps, protocol: Protocol = PROTOCOL) -> tuple[dict, int]:
    """(a valid random genome, the number of invalid draws before it)."""
    rejected = 0
    while True:
        genome = ops.random()
        if is_valid(genome, protocol):
            return genome, rejected
        rejected += 1


@dataclass
class Child:
    genome: dict
    parents: list[int]     # indices into the parent list
    op: str                # "crossover", "mutation" or "fallback"
    attempts: int          # draws until valid (offspring_attempts + ... for fallbacks)


def make_child(ops: GenomeOps, parents: list[dict], rng: np.random.Generator,
               protocol: Protocol = PROTOCOL) -> Child:
    """One valid child of the parent genomes: crossover of two parents with
    probability p_crossover (else a copy of one), then one mutation. Every
    attempt starts again from the parents."""
    for attempt in range(1, protocol.offspring_attempts + 1):
        if len(parents) >= 2 and rng.random() < protocol.p_crossover:
            i, j = (int(k) for k in rng.choice(len(parents), size=2, replace=False))
            genome, idx, op = ops.crossover(parents[i], parents[j]), [i, j], "crossover"
        else:
            i = int(rng.integers(len(parents)))
            genome, idx, op = parents[i], [i], "mutation"
        genome = ops.mutate(genome, rng)
        if is_valid(genome, protocol):
            return Child(genome, idx, op, attempt)
    # Fallback: a mutated clone of one parent, else that parent unchanged.
    i = int(rng.integers(len(parents)))
    for extra in range(1, protocol.offspring_attempts + 1):
        genome = ops.mutate(parents[i], rng)
        if is_valid(genome, protocol):
            return Child(genome, [i], "fallback", protocol.offspring_attempts + extra)
    return Child(copy.deepcopy(parents[i]), [i], "fallback", 2 * protocol.offspring_attempts)
