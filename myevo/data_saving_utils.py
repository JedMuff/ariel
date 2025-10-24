"""Data saving utilities for evolutionary robotics experiments.

This module provides comprehensive data saving functionality for evolutionary
algorithms, including per-generation folders, per-individual data, and final
databases with complete lineage information.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from networkx import DiGraph
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.decoders import save_graph_as_json
from ariel.ec import TreeGenotype

console = Console()


def save_individual_data(
    individual: Any,
    individual_dir: Path,
    learning_curve: list[float] | None = None,
    weights_dict: dict[str, np.ndarray] | None = None,
) -> dict[str, str]:
    """Save complete data for a single individual.

    Creates a directory for the individual and saves:
    - Body: Tree genotype as JSON
    - Initial brain: Initial controller weights as NPY file
    - Optimized brain: Optimized controller weights as NPY file (after CMA-ES)
    - Learning curve: CMA-ES fitness progression as CSV

    Parameters
    ----------
    individual : Any
        Individual object with .genotype, .fitness, .id attributes.
    individual_dir : Path
        Directory to save individual data in.
    learning_curve : list[float] | None, optional
        List of fitness values from CMA-ES optimization, by default None.
    weights_dict : dict[str, np.ndarray] | None, optional
        Dictionary with 'initial' and 'optimized' weights, by default None.

    Returns
    -------
    dict[str, str]
        Dictionary with file paths: 'body', 'initial_brain', 'optimized_brain', 'learning_curve'.
    """
    # Create individual directory
    individual_dir.mkdir(exist_ok=True, parents=True)

    file_paths = {}

    # Save body (tree genotype as JSON)
    tree = individual.genotype.tree if isinstance(individual.genotype, TreeGenotype) else individual.genotype
    body_path = individual_dir / "body.json"
    save_graph_as_json(tree, body_path)
    file_paths['body'] = str(body_path)

    # Save initial and optimized brain weights
    if weights_dict is not None:
        if 'initial' in weights_dict:
            initial_brain_path = individual_dir / "initial_brain.npy"
            np.save(initial_brain_path, weights_dict['initial'])
            file_paths['initial_brain'] = str(initial_brain_path)
        else:
            file_paths['initial_brain'] = ""

        if 'optimized' in weights_dict:
            optimized_brain_path = individual_dir / "optimized_brain.npy"
            np.save(optimized_brain_path, weights_dict['optimized'])
            file_paths['optimized_brain'] = str(optimized_brain_path)
        else:
            file_paths['optimized_brain'] = ""
    else:
        file_paths['initial_brain'] = ""
        file_paths['optimized_brain'] = ""

    # Save learning curve (CMA-ES fitness progression as CSV)
    if learning_curve is not None and len(learning_curve) > 0:
        learning_curve_path = individual_dir / "learning_curve.csv"
        with open(learning_curve_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'fitness'])
            for i, fitness in enumerate(learning_curve):
                writer.writerow([i, fitness])
        file_paths['learning_curve'] = str(learning_curve_path)
    else:
        file_paths['learning_curve'] = ""

    return file_paths


def save_generation_folder(
    population: list[Any],
    generation: int,
    save_dir: Path,
    learning_curves: dict[int, list[float]] | None = None,
    individual_weights: dict[int, dict[str, np.ndarray]] | None = None,
) -> None:
    """Save all individuals in a generation to a dedicated folder.

    Creates a generation folder with subfolders for each individual.

    Parameters
    ----------
    population : list[Any]
        List of Individual objects for this generation.
    generation : int
        Generation number.
    save_dir : Path
        Base save directory (e.g., __data__/script_name).
    learning_curves : dict[int, list[float]] | None, optional
        Dictionary mapping individual ID to learning curve data, by default None.
    individual_weights : dict[int, dict[str, np.ndarray]] | None, optional
        Dictionary mapping individual ID to weight dict with 'initial' and 'optimized' keys, by default None.
    """
    gen_dir = save_dir / f"generation_{generation}"
    gen_dir.mkdir(exist_ok=True, parents=True)

    # DEBUG
    weights_found = 0
    curves_found = 0

    for idx, individual in enumerate(population):
        individual_dir = gen_dir / f"individual_{idx}"

        # Get learning curve if available
        learning_curve = None
        if learning_curves is not None and individual.id is not None:
            learning_curve = learning_curves.get(individual.id)
            if learning_curve:
                curves_found += 1

        # Get weights if available
        weights_dict = None
        if individual_weights is not None and individual.id is not None:
            weights_dict = individual_weights.get(individual.id)
            if weights_dict:
                weights_found += 1

        # Save individual data
        save_individual_data(
            individual=individual,
            individual_dir=individual_dir,
            learning_curve=learning_curve,
            weights_dict=weights_dict,
        )

    console.print(f"[green]Saved generation {generation} data to:[/green] {gen_dir}")
    console.print(f"[yellow]DEBUG: Found weights for {weights_found}/{len(population)}, curves for {curves_found}/{len(population)}[/yellow]")


def save_final_database(
    all_individuals: list[Any],
    save_dir: Path,
    learning_curves: dict[int, list[float]] | None = None,
) -> None:
    """Save comprehensive database with all individuals and metadata.

    Creates both CSV and JSON files with complete information about all
    individuals, including fitness, parent IDs, and file paths.

    Parameters
    ----------
    all_individuals : list[Any]
        All individuals from all generations.
    save_dir : Path
        Base save directory.
    learning_curves : dict[int, list[float]] | None, optional
        Dictionary mapping individual ID to learning curve data, by default None.
    """
    # Prepare data records
    records = []

    # Group individuals by generation for indexing
    gen_individuals: dict[int, list[Any]] = {}
    for ind in all_individuals:
        gen = ind.time_of_birth
        if gen not in gen_individuals:
            gen_individuals[gen] = []
        gen_individuals[gen].append(ind)

    # Create index mapping for each generation
    gen_indices: dict[int, dict[int, int]] = {}
    for gen, inds in gen_individuals.items():
        gen_indices[gen] = {ind.id: idx for idx, ind in enumerate(inds)}

    # Build records
    for ind in all_individuals:
        gen = ind.time_of_birth
        ind_idx = gen_indices[gen][ind.id]

        # Get tree for counting parts/actuators
        tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype

        # Get parent IDs from tags
        parent1_id = ind.tags.get("parent1_id", None)
        parent2_id = ind.tags.get("parent2_id", None)

        # Get individual_uuid from tags (persists across multiprocessing)
        # Fall back to tree metadata if not in tags
        individual_id_str = ind.tags.get("individual_uuid")
        if individual_id_str is None:
            individual_id_str = tree.graph.get("_individual_id")

        # Determine directory path - use flat individuals/ folder
        if individual_id_str:
            directory = str(save_dir / "individuals" / f"individual_{individual_id_str}")
        else:
            # Fallback if no individual_id found
            directory = ""

        # Count parts and actuators
        # Note: In robogen_lite, actuators are modules with type="HINGE"
        num_parts = len(tree.nodes)
        num_actuators = sum(1 for _, data in tree.nodes(data=True) if data.get("type") == "HINGE")

        record = {
            "individual_id": ind.id,
            "individual_uuid": individual_id_str,
            "generation": gen,
            "fitness": ind.fitness if ind.fitness_ is not None else None,
            "parent1_id": parent1_id,
            "parent2_id": parent2_id,
            "directory": directory,
            "num_parts": num_parts,
            "num_actuators": num_actuators,
        }
        records.append(record)

    # Save as CSV
    csv_path = save_dir / "database.csv"
    if records:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        console.print(f"[bold green]Database (CSV) saved to:[/bold green] {csv_path}")

    # Save as JSON
    json_path = save_dir / "database.json"
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
    console.print(f"[bold green]Database (JSON) saved to:[/bold green] {json_path}")

    # Print summary statistics
    console.print(f"\n[cyan]Database Summary:[/cyan]")
    console.print(f"  Total individuals: {len(records)}")
    console.print(f"  Generations: {max(r['generation'] for r in records) + 1}")
    console.print(f"  Individuals with directories: {sum(1 for r in records if r['directory'])}")
    console.print(f"  Average actuators per individual: {np.mean([r['num_actuators'] for r in records]):.2f}")
