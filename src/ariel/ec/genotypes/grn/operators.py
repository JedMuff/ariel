"""Random GRN genomes, mutation and unequal crossover.

Ports of ``random_v1`` (geno_body_GRN_v3.py), ``mutate_body`` (_mutate_v1.py)
and ``crossover_v1`` (_crossover_v1.py) of karinemiras/revolve2, branch
plasticoding_v3 (Miras and Feenstra, 2026). They make the same calls on a
``random.Random``-like rng as the originals, so the same seed gives the same
result; unlike the originals they never change their arguments.
"""

from __future__ import annotations

import random as _random

from .develop import GRN_R2, GrnDesign

MAX_GENOME_LENGTH = 1000


def random_grn(design: GrnDesign = GRN_R2, rng=_random) -> list[float]:
    return [round(rng.uniform(0, 1), 2) for _ in range(design.genome_size)]


def mutate_grn(values: list[float], design: GrnDesign = GRN_R2, rng=_random) -> list[float]:  # noqa: ARG001
    """One mutation at a random position: perturbation (N(0, 0.1), clipped to
    [0, 1]), deletion, insertion of a random value, or swap with another position."""
    g = list(values)
    position = rng.sample(range(0, len(g)), 1)[0]
    kind = rng.sample(["perturbation", "deletion", "addition", "swap"], 1)[0]
    if kind == "perturbation":
        g[position] = min(1, max(0, round(g[position] + rng.normalvariate(0, 0.1), 2)))
    elif kind == "deletion":
        if len(g) > 1:
            g.pop(position)
    elif kind == "addition":
        g.insert(position, round(rng.uniform(0, 1), 2))
    elif len(g) > 1:  # swap
        position2 = rng.sample(range(0, len(g)), 1)[0]
        while position == position2:
            position2 = rng.sample(range(0, len(g)), 1)[0]
        g[position], g[position2] = g[position2], g[position]
    return g


def _promoters(values: list[float], design: GrnDesign) -> list[int]:
    sites, i = [], 0
    while i < len(values):
        if values[i] < design.promoter_threshold and len(values) - 1 - i >= design.gene_length:
            sites.append(i)
            i += design.gene_length
        i += 1
    return sites


def crossover_grn(a: list[float], b: list[float], design: GrnDesign = GRN_R2, rng=_random) -> list[float]:
    """Unequal crossover: the child's threshold is the parents' mean, followed by
    each parent's genes up to and including a random one of them, truncated to
    MAX_GENOME_LENGTH. (A parent without genes contributes nothing; the
    original fails on one.)"""
    child = [(a[0] + b[0]) / 2]
    for parent in (a[1:], b[1:]):
        sites = _promoters(parent, design)
        if sites:
            cut = rng.sample(sites, 1)[0]
            child += parent[0:cut + design.gene_length + 1]
    return child[:MAX_GENOME_LENGTH]
