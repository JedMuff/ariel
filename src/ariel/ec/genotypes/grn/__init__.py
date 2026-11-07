"""Indirect body encoding: the genetic regulatory network (GRN) of Miras and
Feenstra (2026), ported from karinemiras/revolve2 (plasticoding_v3).

    values = random_grn()                       # 151 values in [0, 1]
    body = develop_body(values, max_modules=26) # ARIEL tree genome dict

Bodies grow in the plane (no top or bottom attachments). See develop.py for
the development, convert.py for the mapping to ARIEL modules and
operators.py for mutation and crossover.
"""

from __future__ import annotations

from .convert import to_tree
from .develop import GRN_R2, GrnDesign, Module, develop, modules_of
from .operators import MAX_GENOME_LENGTH, crossover_grn, mutate_grn, random_grn


def develop_body(values: list[float], design: GrnDesign = GRN_R2, max_modules: int = 26) -> dict:
    """The ARIEL tree genome of the body grown from values. max_modules counts
    the core, as in the original."""
    return to_tree(develop(values, design, max_modules))


__all__ = [
    "GRN_R2", "MAX_GENOME_LENGTH", "GrnDesign", "Module", "crossover_grn", "develop", "develop_body",
    "modules_of", "mutate_grn", "random_grn", "to_tree",
]
