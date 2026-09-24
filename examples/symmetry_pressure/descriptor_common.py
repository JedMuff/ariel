"""
Genome-type-aware morphological descriptor computation for the
symmetry-pressure sweep, shared by plot_descriptors_over_generations.py and
plot_descriptors_vs_fitness_sweep.py.

Mirrors how genome_adapter.py's GenomeAdapter.symmetry_score/module_count
score each genome type in production, so descriptor values here match what
was actually logged (as "yz_symmetry") during the runs:
  - tree / tree_symmetric: genome dict decodes directly (it *is* the
    phenotype tree) via TreeGenome.from_dict(...).to_networkx(); symmetry
    uses shared.bilateral_symmetry_score (tree-schema-specific).
  - cppn: genome dict is a CPPN network, decoded to its phenotype graph via
    genome_adapter._cppn_decode; symmetry uses the graph-native
    MorphologicalMeasures.symmetry instead (bilateral_symmetry_score does
    not understand the CPPN genotype schema), reusing the same decode
    already needed for the other 7 descriptors rather than decoding twice.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.utils.morphological_descriptor import MorphologicalMeasures

from genome_adapter import _cppn_decode
from shared import bilateral_symmetry_score
from sweep_common import checkpoints_dir

DESCRIPTORS = [
    ("num_modules", "Num modules"),
    ("branching", "Branching (B)"),
    ("limbs", "Limbs (L)"),
    ("length_of_limbs", "Length of limbs (E)"),
    ("coverage", "Coverage (C)"),
    ("joints", "Joints (J)"),
    ("symmetry", "Symmetry (S)"),
    ("module_diversity", "Module diversity (D)"),
]


def genotype_to_graph(genotype_dict: dict, genome_type: str, max_modules: int) -> nx.DiGraph:
    if genome_type == "cppn":
        return _cppn_decode(genotype_dict, max_modules)
    return TreeGenome.from_dict(genotype_dict).to_networkx()


def compute_descriptors(genotype_dict: dict, genome_type: str, max_modules: int) -> dict[str, float] | None:
    try:
        graph = genotype_to_graph(genotype_dict, genome_type, max_modules)
        if graph.number_of_nodes() == 0:
            return None
        m = MorphologicalMeasures(graph)
    except Exception:
        return None

    if genome_type == "cppn":
        symmetry = float(m.symmetry)
    else:
        symmetry = bilateral_symmetry_score(genotype_dict)

    return {
        "num_modules": float(graph.number_of_nodes()),
        "branching": m.branching,
        "limbs": m.limbs,
        "length_of_limbs": m.length_of_limbs,
        "coverage": m.coverage,
        "joints": m.joints,
        "symmetry": symmetry,
        "module_diversity": m.module_diversity,
    }


def load_checkpoint_records(run_dir: Path, task: str, genome_type: str, max_modules: int) -> list[dict]:
    """One row per checkpoint: {"gen": int, "fitness": float, **descriptors}.

    Skips checkpoints with missing meta/genome files, non-finite fitness, or
    a genome that fails to decode/measure.
    """
    base = checkpoints_dir(run_dir, task)
    if not base.exists():
        return []
    records: list[dict] = []
    for ckpt in sorted(base.iterdir()):
        meta_path, genome_path = ckpt / "meta.json", ckpt / "best_genome.json"
        if not meta_path.exists() or not genome_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        fitness = meta.get("fitness")
        if fitness is None or not math.isfinite(fitness):
            continue
        genotype = json.loads(genome_path.read_text())
        desc = compute_descriptors(genotype, genome_type, max_modules)
        if desc is None:
            continue
        records.append({"gen": meta["gen"], "fitness": float(fitness), **desc})
    return records
