"""Pure novelty fitness experiment with parent-child distance tracking.

This script measures how novelty-only selection affects parent-child morphological
similarity over generations. Unlike locomotion-based evolution, fitness here is
PURELY based on morphological novelty (distance to k-nearest neighbors in archive).

Key features:
- Fitness = novelty score only (no locomotion component)
- No CMA-ES optimization (not needed without locomotion)
- Tracks tree edit distance between parent and offspring each generation
- Outputs distance statistics to CSV for analysis
"""

from __future__ import annotations

# Thread limits BEFORE importing numpy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import csv
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

# Apply custom config BEFORE importing ARIEL modules
CWD = Path.cwd()
sys.path.insert(0, str(CWD))
from myevo.config.custom_config import ALLOWED_FACES, ALLOWED_ROTATIONS
import ariel.body_phenotypes.robogen_lite.config as ariel_config
ariel_config.ALLOWED_FACES = ALLOWED_FACES
ariel_config.ALLOWED_ROTATIONS = ALLOWED_ROTATIONS

import matplotlib.pyplot as plt
import numpy as np
from networkx import DiGraph
from rich.console import Console
from rich.table import Table
from rich.traceback import install

from ariel.ec import Individual
from myevo.core import TreeGenotype
from myevo.core.weight_inheritance import tree_distance
from myevo.measures.novelty_fitness import KDTreeArchive, extract_morphological_vector
from myevo.measures.morphological_measures import Body, MorphologicalMeasures

# Global constants
SCRIPT_NAME = Path(__file__).stem
DATA = None
SEED = 42

install(show_locals=False)
console = Console()
RNG = np.random.default_rng(SEED)
random.seed(SEED)


def _compute_single_distance(args: tuple) -> tuple[int, float | None, bool]:
    """Compute tree distance for a single parent-child pair.

    Standalone function for multiprocessing (must be picklable).

    Parameters
    ----------
    args : tuple
        (index, child_tree, parent_tree, is_crossover, timeout)

    Returns
    -------
    tuple[int, float | None, bool]
        (index, distance, is_crossover)
    """
    idx, child_tree, parent_tree, is_crossover, timeout = args
    try:
        dist = tree_distance(child_tree, parent_tree, timeout=timeout)
        return (idx, dist, is_crossover)
    except Exception:
        return (idx, None, is_crossover)


class TreeWrapper:
    """Wrapper for tree genotypes to work with novelty archive."""
    def __init__(self, tree):
        self.tree = tree


class PureNoveltyEvolution:
    """Evolution system using pure novelty as fitness.

    This class evolves robot morphologies where fitness is determined solely
    by morphological novelty (distance to k-nearest neighbors in archive).
    It tracks parent-child distances to measure how novelty selection
    affects morphological exploration.
    """

    def __init__(
        self,
        mu: int = 30,
        lambda_: int = 30,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.2,
        mutate_after_crossover: bool = True,
        strategy_type: str = "plus",
        max_part_limit: int = 25,
        max_actuators: int = 12,
        mutation_strength: int = 1,
        mutation_reps: int = 1,
        novelty_k_neighbors: int = 1,
        seed: int = 42,
        num_workers: int = 1,
        verbose: bool = True,
    ):
        """Initialize pure novelty evolution system.

        Parameters
        ----------
        mu : int
            Population size (number of parents).
        lambda_ : int
            Number of offspring per generation.
        mutation_rate : float
            Proportion of offspring via mutation (0-1).
        crossover_rate : float
            Proportion of offspring via crossover (0-1).
        mutate_after_crossover : bool
            Whether to mutate after crossover.
        strategy_type : str
            'plus' (mu+lambda) or 'comma' (mu,lambda).
        max_part_limit : int
            Maximum robot parts.
        max_actuators : int
            Maximum actuators.
        mutation_strength : int
            Tree depth for mutations.
        mutation_reps : int
            Number of mutation repetitions.
        novelty_k_neighbors : int
            K for k-nearest neighbor novelty calculation.
        seed : int
            Random seed.
        verbose : bool
            Print progress info.
        """
        self.mu = mu
        self.lambda_ = lambda_
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutate_after_crossover = mutate_after_crossover
        self.strategy_type = strategy_type
        self.max_part_limit = max_part_limit
        self.max_actuators = max_actuators
        self.novelty_k_neighbors = novelty_k_neighbors
        self.num_workers = num_workers
        self.verbose = verbose

        # Calculate offspring counts
        self.num_mutate = int(lambda_ * mutation_rate)
        self.num_crossover = lambda_ - self.num_mutate

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Initialize tree genotype
        self.tree_genotype = TreeGenotype(
            max_part_limit=max_part_limit,
            max_actuators=max_actuators,
            mutation_strength=mutation_strength,
            mutation_reps=mutation_reps,
        )

        # Initialize novelty archive
        self.novelty_archive = KDTreeArchive(
            min_distance=None,
            feature_extractor=self._extract_morphological_features,
        )

        # Distance tracking
        self.generation_distances: list[dict[str, Any]] = []

        # Individual ID counter
        self._next_id = 0

        if verbose:
            console.print(
                f"\n[bold cyan]Pure Novelty Evolution Configuration[/bold cyan]\n"
                f"Strategy: ({strategy_type}) mu={mu}, lambda={lambda_}\n"
                f"Morphology: max_parts={max_part_limit}, max_actuators={max_actuators}\n"
                f"Novelty: k={novelty_k_neighbors} (pure novelty fitness)\n"
                f"Mutation rate: {mutation_rate:.0%}, Crossover rate: {crossover_rate:.0%}\n"
                f"Workers: {num_workers} (for distance calculations)\n"
            )

    def _get_next_id(self) -> int:
        """Get next unique individual ID."""
        id_ = self._next_id
        self._next_id += 1
        return id_

    def _extract_morphological_features(self, ind) -> np.ndarray:
        """Extract morphological features for novelty calculation."""
        tree = ind.tree if hasattr(ind, 'tree') else ind
        return extract_morphological_vector(
            MorphologicalMeasures(Body(tree, max_part_limit=self.max_part_limit))
        )

    def _calculate_novelty(self, tree: DiGraph) -> float:
        """Calculate novelty score for a tree genotype.

        Parameters
        ----------
        tree : DiGraph
            Tree genotype to evaluate.

        Returns
        -------
        float
            Novelty score (mean distance to k-nearest neighbors).
        """
        wrapper = TreeWrapper(tree)

        if len(self.novelty_archive) == 0:
            return 1.0  # First individual gets neutral novelty

        feature_vec = self.novelty_archive.feature_extractor(wrapper)

        # Query more neighbors to filter out self-matches
        k_query = min(self.novelty_k_neighbors + 5, len(self.novelty_archive))
        distances, _ = self.novelty_archive.kdtree.query(feature_vec, k=k_query)

        if np.isscalar(distances):
            distances = np.array([distances])
        else:
            distances = np.array(distances)

        # Filter out exact matches (distance ~0)
        eps = 1e-10
        filtered_distances = distances[distances > eps]

        if len(filtered_distances) == 0:
            return 1.0

        k_actual = min(self.novelty_k_neighbors, len(filtered_distances))
        return float(np.mean(filtered_distances[:k_actual]))

    def _add_to_archive(self, tree: DiGraph) -> None:
        """Add a tree to the novelty archive if unique."""
        wrapper = TreeWrapper(tree)

        if len(self.novelty_archive) > 0:
            feature_vec = self.novelty_archive.feature_extractor(wrapper)
            distances, _ = self.novelty_archive.kdtree.query(feature_vec, k=1)
            nearest_distance = distances if np.isscalar(distances) else distances[0]

            if nearest_distance < 1e-10:
                return  # Already in archive

        self.novelty_archive.add(wrapper)

    def _create_individual(self, tree: DiGraph, parent1_id: int | None = None,
                          parent2_id: int | None = None) -> Individual:
        """Create an Individual from a tree genotype."""
        ind = Individual(
            genotype=self.tree_genotype,
            id=self._get_next_id(),
        )
        # Store tree in tags dict (Individual is a Pydantic model, can't add arbitrary attrs)
        ind.tags = {
            "tree": tree,
            "parent1_id": parent1_id,
            "parent2_id": parent2_id,
        }
        return ind

    def _calculate_parent_child_distance(
        self,
        offspring: Individual,
        parent: Individual
    ) -> float | None:
        """Calculate tree edit distance between offspring and parent.

        Parameters
        ----------
        offspring : Individual
            Offspring individual.
        parent : Individual
            Parent individual.

        Returns
        -------
        float | None
            Tree edit distance, or None if calculation fails/times out.
        """
        try:
            offspring_tree = offspring.tags["tree"]
            parent_tree = parent.tags["tree"]
            return tree_distance(offspring_tree, parent_tree, timeout=2.0)
        except (TimeoutError, Exception) as e:
            if self.verbose:
                console.print(f"[yellow]Warning: tree_distance failed: {e}[/yellow]")
            return None

    def _generate_offspring(
        self,
        population: list[Individual]
    ) -> tuple[list[Individual], list[tuple[Individual, Individual | None]]]:
        """Generate offspring via mutation and crossover.

        Returns
        -------
        tuple[list[Individual], list[tuple[Individual, Individual | None]]]
            (offspring_list, parent_pairs) where parent_pairs[i] = (parent1, parent2 or None)
        """
        offspring = []
        parent_pairs = []

        # Build ID -> Individual mapping
        id_to_ind = {ind.id: ind for ind in population}

        # Mutation offspring
        for _ in range(self.num_mutate):
            parent = random.choice(population)
            child_tree = self.tree_genotype.mutate(parent.tags["tree"])
            child = self._create_individual(
                tree=child_tree,
                parent1_id=parent.id,
                parent2_id=None,
            )
            offspring.append(child)
            parent_pairs.append((parent, None))

        # Crossover offspring
        for _ in range(self.num_crossover):
            parent1, parent2 = random.sample(population, 2)
            # crossover returns tuple of two offspring, take the first one
            child_tree, _ = self.tree_genotype.crossover(
                parent1.tags["tree"],
                parent2.tags["tree"]
            )

            if self.mutate_after_crossover:
                child_tree = self.tree_genotype.mutate(child_tree)

            child = self._create_individual(
                tree=child_tree,
                parent1_id=parent1.id,
                parent2_id=parent2.id,
            )
            offspring.append(child)
            parent_pairs.append((parent1, parent2))

        return offspring, parent_pairs

    def _evaluate_population(self, population: list[Individual]) -> None:
        """Evaluate novelty fitness for all individuals."""
        for ind in population:
            novelty = self._calculate_novelty(ind.tags["tree"])
            ind.fitness = novelty
            ind.tags["novelty_score"] = novelty

        # Add all to archive after evaluation
        for ind in population:
            self._add_to_archive(ind.tags["tree"])

    def _select_survivors(
        self,
        population: list[Individual],
        offspring: list[Individual]
    ) -> list[Individual]:
        """Select survivors for next generation."""
        if self.strategy_type == "plus":
            # (mu+lambda): select from parents + offspring
            combined = population + offspring
        else:
            # (mu,lambda): select from offspring only
            combined = offspring

        # Sort by fitness (descending - higher novelty = better)
        combined.sort(key=lambda x: x.fitness or 0.0, reverse=True)

        return combined[:self.mu]

    def _track_generation_distances(
        self,
        generation: int,
        offspring: list[Individual],
        parent_pairs: list[tuple[Individual, Individual | None]],
    ) -> dict[str, Any]:
        """Track parent-child distances for a generation.

        Parameters
        ----------
        generation : int
            Current generation number.
        offspring : list[Individual]
            Offspring individuals.
        parent_pairs : list[tuple[Individual, Individual | None]]
            Parent pairs for each offspring.

        Returns
        -------
        dict
            Statistics for this generation.
        """
        distances = []
        mutation_distances = []
        crossover_distances = []

        # Build list of distance computation tasks
        tasks = []
        for idx, (child, (parent1, parent2)) in enumerate(zip(offspring, parent_pairs)):
            child_tree = child.tags["tree"]
            parent1_tree = parent1.tags["tree"]
            is_crossover = parent2 is not None

            # Task for parent1 distance
            tasks.append((len(tasks), child_tree, parent1_tree, is_crossover, 2.0))

            # Task for parent2 distance (if crossover)
            if parent2 is not None:
                parent2_tree = parent2.tags["tree"]
                tasks.append((len(tasks), child_tree, parent2_tree, True, 2.0))

        # Compute distances (parallel or sequential)
        if self.num_workers > 1 and len(tasks) > 1:
            # Parallel computation
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(_compute_single_distance, tasks))
        else:
            # Sequential computation
            results = [_compute_single_distance(t) for t in tasks]

        # Collect results
        for idx, dist, is_crossover in results:
            if dist is not None:
                distances.append(dist)
                if is_crossover:
                    crossover_distances.append(dist)
                else:
                    mutation_distances.append(dist)

        # Calculate statistics
        stats = {
            "generation": generation,
            "num_offspring": len(offspring),
            "num_distances": len(distances),
            "archive_size": len(self.novelty_archive),
        }

        if distances:
            stats.update({
                "mean_distance": np.mean(distances),
                "std_distance": np.std(distances),
                "min_distance": np.min(distances),
                "max_distance": np.max(distances),
                "median_distance": np.median(distances),
            })
        else:
            stats.update({
                "mean_distance": None,
                "std_distance": None,
                "min_distance": None,
                "max_distance": None,
                "median_distance": None,
            })

        if mutation_distances:
            stats.update({
                "mutation_mean_distance": np.mean(mutation_distances),
                "mutation_std_distance": np.std(mutation_distances),
                "mutation_count": len(mutation_distances),
            })
        else:
            stats.update({
                "mutation_mean_distance": None,
                "mutation_std_distance": None,
                "mutation_count": 0,
            })

        if crossover_distances:
            stats.update({
                "crossover_mean_distance": np.mean(crossover_distances),
                "crossover_std_distance": np.std(crossover_distances),
                "crossover_count": len(crossover_distances),
            })
        else:
            stats.update({
                "crossover_mean_distance": None,
                "crossover_std_distance": None,
                "crossover_count": 0,
            })

        # Population fitness stats
        fitnesses = [ind.fitness for ind in offspring if ind.fitness is not None]
        if fitnesses:
            stats.update({
                "mean_novelty": np.mean(fitnesses),
                "std_novelty": np.std(fitnesses),
                "max_novelty": np.max(fitnesses),
                "min_novelty": np.min(fitnesses),
            })

        return stats

    def initialize_population(self) -> list[Individual]:
        """Initialize population with diverse tree depths."""
        population = []

        # 10 small, 10 medium, 10 large trees
        for depth in [1, 2, 3]:
            for _ in range(10):
                tree = self.tree_genotype.random_tree(depth=depth)
                ind = self._create_individual(tree)
                population.append(ind)

        return population

    def run(self, num_generations: int) -> tuple[list[Individual], list[dict]]:
        """Run evolution with distance tracking.

        Parameters
        ----------
        num_generations : int
            Number of generations to run.

        Returns
        -------
        tuple[list[Individual], list[dict]]
            (final_population, generation_statistics)
        """
        if self.verbose:
            console.print(
                f"\n[bold cyan]Starting Pure Novelty Evolution[/bold cyan]\n"
                f"Generations: {num_generations}\n"
                f"Fitness: Pure Novelty (k={self.novelty_k_neighbors})\n"
            )

        # Initialize population
        population = self.initialize_population()

        # Evaluate initial population
        self._evaluate_population(population)

        # Track statistics for initial population
        initial_stats = {
            "generation": 0,
            "num_offspring": len(population),
            "num_distances": 0,  # No parents yet
            "archive_size": len(self.novelty_archive),
            "mean_distance": None,
            "std_distance": None,
            "min_distance": None,
            "max_distance": None,
            "median_distance": None,
            "mutation_mean_distance": None,
            "mutation_std_distance": None,
            "mutation_count": 0,
            "crossover_mean_distance": None,
            "crossover_std_distance": None,
            "crossover_count": 0,
        }
        fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
        if fitnesses:
            initial_stats.update({
                "mean_novelty": np.mean(fitnesses),
                "std_novelty": np.std(fitnesses),
                "max_novelty": np.max(fitnesses),
                "min_novelty": np.min(fitnesses),
            })
        self.generation_distances.append(initial_stats)

        # Evolution loop
        for gen in range(1, num_generations + 1):
            # Generate offspring
            offspring, parent_pairs = self._generate_offspring(population)

            # Evaluate offspring
            self._evaluate_population(offspring)

            # Track distances
            gen_stats = self._track_generation_distances(gen, offspring, parent_pairs)
            self.generation_distances.append(gen_stats)

            # Select survivors
            population = self._select_survivors(population, offspring)

            # Print progress
            if self.verbose:
                mean_dist = gen_stats.get("mean_distance")
                mean_nov = gen_stats.get("mean_novelty")
                dist_str = f"{mean_dist:.3f}" if mean_dist is not None else "N/A"
                nov_str = f"{mean_nov:.3f}" if mean_nov is not None else "N/A"
                console.print(
                    f"Gen {gen:3d}: "
                    f"mean_dist={dist_str:>7} | "
                    f"mean_novelty={nov_str:>7} | "
                    f"archive={gen_stats['archive_size']:4d}"
                )

        return population, self.generation_distances

    def save_statistics(self, filepath: Path) -> None:
        """Save generation statistics to CSV."""
        if not self.generation_distances:
            return

        fieldnames = list(self.generation_distances[0].keys())

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.generation_distances)

        console.print(f"[green]Saved statistics to {filepath}[/green]")

    def print_summary(self) -> None:
        """Print summary table of distance statistics."""
        table = Table(title="Parent-Child Distance Over Generations")
        table.add_column("Generation", justify="right")
        table.add_column("Mean Dist", justify="right")
        table.add_column("Std Dist", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Mean Novelty", justify="right")
        table.add_column("Archive", justify="right")

        for stats in self.generation_distances:
            gen = stats["generation"]
            mean = stats.get("mean_distance")
            std = stats.get("std_distance")
            min_d = stats.get("min_distance")
            max_d = stats.get("max_distance")
            nov = stats.get("mean_novelty")
            arch = stats.get("archive_size")

            table.add_row(
                str(gen),
                f"{mean:.3f}" if mean is not None else "N/A",
                f"{std:.3f}" if std is not None else "N/A",
                f"{min_d:.1f}" if min_d is not None else "N/A",
                f"{max_d:.1f}" if max_d is not None else "N/A",
                f"{nov:.3f}" if nov is not None else "N/A",
                str(arch) if arch is not None else "N/A",
            )

        console.print(table)

    def plot_statistics(self, save_dir: Path | None = None) -> None:
        """Plot generation statistics as separate figures.

        Parameters
        ----------
        save_dir : Path | None
            If provided, save figures to this directory. Otherwise display interactively.
        """
        if not self.generation_distances:
            console.print("[yellow]No statistics to plot[/yellow]")
            return

        generations = [s["generation"] for s in self.generation_distances]
        mean_distances = [s.get("mean_distance") for s in self.generation_distances]
        std_distances = [s.get("std_distance") for s in self.generation_distances]
        num_distances = [s.get("num_distances") for s in self.generation_distances]
        mean_novelties = [s.get("mean_novelty") for s in self.generation_distances]
        archive_sizes = [s.get("archive_size") for s in self.generation_distances]

        # Plot 1: Mean parent-child distance over generations with 95% CI
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        valid_data = [
            (g, m, s, n) for g, m, s, n in zip(generations, mean_distances, std_distances, num_distances)
            if m is not None and s is not None and n is not None and n > 0
        ]

        if valid_data:
            valid_gens = [d[0] for d in valid_data]
            valid_means = [d[1] for d in valid_data]
            valid_stds = [d[2] for d in valid_data]
            valid_counts = [d[3] for d in valid_data]

            ax1.plot(valid_gens, valid_means, "b-", linewidth=2, label="Mean Distance")
            # 95% confidence interval: mean ± 1.96 * (std / sqrt(n))
            ci_95 = [1.96 * s / np.sqrt(n) for s, n in zip(valid_stds, valid_counts)]
            lower = [m - ci for m, ci in zip(valid_means, ci_95)]
            upper = [m + ci for m, ci in zip(valid_means, ci_95)]
            ax1.fill_between(valid_gens, lower, upper, alpha=0.3, color="blue")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Tree Edit Distance")
        ax1.set_title("Parent-Child Distance (95% CI)")
        ax1.set_yticks(range(3, 13))
        ax1.set_ylim(3, 12)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            fig1.savefig(save_dir / "parent_child_distance.png", dpi=150, bbox_inches="tight")
            console.print(f"[green]Saved: {save_dir / 'parent_child_distance.png'}[/green]")
            plt.close(fig1)

        # Plot 2: Mean novelty over generations
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        valid_nov_gens = [g for g, n in zip(generations, mean_novelties) if n is not None]
        valid_novelties = [n for n in mean_novelties if n is not None]

        if valid_novelties:
            ax2.plot(valid_nov_gens, valid_novelties, "g-", linewidth=2)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Novelty Score")
        ax2.set_title("Mean Novelty")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            fig2.savefig(save_dir / "mean_novelty.png", dpi=150, bbox_inches="tight")
            console.print(f"[green]Saved: {save_dir / 'mean_novelty.png'}[/green]")
            plt.close(fig2)

        # Plot 3: Archive size over generations
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        valid_arch_gens = [g for g, a in zip(generations, archive_sizes) if a is not None]
        valid_archives = [a for a in archive_sizes if a is not None]

        if valid_archives:
            ax3.plot(valid_arch_gens, valid_archives, "r-", linewidth=2)
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Archive Size")
        ax3.set_title("Novelty Archive Growth")
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            fig3.savefig(save_dir / "archive_growth.png", dpi=150, bbox_inches="tight")
            console.print(f"[green]Saved: {save_dir / 'archive_growth.png'}[/green]")
            plt.close(fig3)

        # Plot 4: Mutation vs Crossover distances with 95% CI
        fig4, ax4 = plt.subplots(figsize=(10, 6))

        # Mutation data
        mut_data = [
            (g, s.get("mutation_mean_distance"), s.get("mutation_std_distance"), s.get("mutation_count"))
            for g, s in zip(generations, self.generation_distances)
        ]
        mut_valid = [(g, m, s, n) for g, m, s, n in mut_data if m is not None and s is not None and n and n > 0]

        if mut_valid:
            mut_gens = [d[0] for d in mut_valid]
            mut_means = [d[1] for d in mut_valid]
            mut_ci = [1.96 * d[2] / np.sqrt(d[3]) for d in mut_valid]
            ax4.plot(mut_gens, mut_means, "b-", linewidth=2, label="Mutation")
            ax4.fill_between(mut_gens, [m - c for m, c in zip(mut_means, mut_ci)],
                           [m + c for m, c in zip(mut_means, mut_ci)], alpha=0.3, color="blue")

        # Crossover data
        cross_data = [
            (g, s.get("crossover_mean_distance"), s.get("crossover_std_distance"), s.get("crossover_count"))
            for g, s in zip(generations, self.generation_distances)
        ]
        cross_valid = [(g, m, s, n) for g, m, s, n in cross_data if m is not None and s is not None and n and n > 0]

        if cross_valid:
            cross_gens = [d[0] for d in cross_valid]
            cross_means = [d[1] for d in cross_valid]
            cross_ci = [1.96 * d[2] / np.sqrt(d[3]) for d in cross_valid]
            ax4.plot(cross_gens, cross_means, "orange", linewidth=2, label="Crossover")
            ax4.fill_between(cross_gens, [m - c for m, c in zip(cross_means, cross_ci)],
                           [m + c for m, c in zip(cross_means, cross_ci)], alpha=0.3, color="orange")

        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Tree Edit Distance")
        ax4.set_title("Distance by Operator Type (95% CI)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            fig4.savefig(save_dir / "distance_by_operator.png", dpi=150, bbox_inches="tight")
            console.print(f"[green]Saved: {save_dir / 'distance_by_operator.png'}[/green]")
            plt.close(fig4)

        if not save_dir:
            plt.show()


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pure novelty fitness experiment with parent-child distance tracking"
    )

    parser.add_argument(
        "--num-generations", type=int, default=50,
        help="Number of generations (default: 50)"
    )
    parser.add_argument(
        "--mu", type=int, default=30,
        help="Population size (default: 30)"
    )
    parser.add_argument(
        "--lambda", type=int, default=30, dest="lambda_",
        help="Number of offspring per generation (default: 30)"
    )
    parser.add_argument(
        "--novelty-k", type=int, default=1,
        help="K for k-nearest neighbor novelty (default: 1)"
    )
    parser.add_argument(
        "--mutation-rate", type=float, default=0.8,
        help="Mutation rate (default: 0.8)"
    )
    parser.add_argument(
        "--strategy-type", type=str, default="plus", choices=["plus", "comma"],
        help="Selection strategy: plus (mu+lambda) or comma (mu,lambda)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of parallel workers for distance calculations (default: 1)"
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Name prefix for experiment directory"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Base directory for saving data (default: __data__)"
    )
    parser.add_argument(
        "--plot-only", type=str, default=None,
        help="Path to existing experiment directory to regenerate plots from CSV"
    )
    parser.add_argument(
        "--repetitions", type=int, default=1,
        help="Number of independent repetitions to run (default: 1)"
    )

    return parser.parse_args()


def plot_combined_statistics(experiment_dir: Path, num_reps: int) -> None:
    """Plot combined statistics across all repetitions.

    Parameters
    ----------
    experiment_dir : Path
        Base experiment directory containing rep_* subdirectories.
    num_reps : int
        Number of repetitions.
    """
    # Load all repetition data
    all_stats = []
    for rep in range(num_reps):
        csv_path = experiment_dir / f"rep_{rep}" / "distance_statistics.csv"
        if csv_path.exists():
            stats = load_statistics_from_csv(csv_path)
            for s in stats:
                s["rep"] = rep
            all_stats.extend(stats)

    if not all_stats:
        console.print("[yellow]No data to plot[/yellow]")
        return

    # Group by generation
    from collections import defaultdict
    by_generation: dict[int, list[dict]] = defaultdict(list)
    for s in all_stats:
        by_generation[s["generation"]].append(s)

    generations = sorted(by_generation.keys())

    # Calculate mean and 95% CI across repetitions for each metric
    def aggregate_metric(metric_name: str) -> tuple[list, list, list]:
        means, ci_lowers, ci_uppers = [], [], []
        for gen in generations:
            values = [s.get(metric_name) for s in by_generation[gen] if s.get(metric_name) is not None]
            if values:
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0
                ci = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0
                means.append(mean)
                ci_lowers.append(mean - ci)
                ci_uppers.append(mean + ci)
            else:
                means.append(None)
                ci_lowers.append(None)
                ci_uppers.append(None)
        return means, ci_lowers, ci_uppers

    # Plot 1: Combined parent-child distance
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    means, ci_lowers, ci_uppers = aggregate_metric("mean_distance")
    valid_idx = [i for i, m in enumerate(means) if m is not None]
    if valid_idx:
        valid_gens = [generations[i] for i in valid_idx]
        valid_means = [means[i] for i in valid_idx]
        valid_lower = [ci_lowers[i] for i in valid_idx]
        valid_upper = [ci_uppers[i] for i in valid_idx]
        ax1.plot(valid_gens, valid_means, "b-", linewidth=2)
        ax1.fill_between(valid_gens, valid_lower, valid_upper, alpha=0.3, color="blue")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Tree Edit Distance")
    ax1.set_title(f"Parent-Child Distance (95% CI, n={num_reps} reps)")
    ax1.set_yticks(range(3, 13))
    ax1.set_ylim(3, 12)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(experiment_dir / "combined_parent_child_distance.png", dpi=150, bbox_inches="tight")
    console.print(f"[green]Saved: {experiment_dir / 'combined_parent_child_distance.png'}[/green]")
    plt.close(fig1)

    # Plot 2: Combined mean novelty
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    means, ci_lowers, ci_uppers = aggregate_metric("mean_novelty")
    valid_idx = [i for i, m in enumerate(means) if m is not None]
    if valid_idx:
        valid_gens = [generations[i] for i in valid_idx]
        valid_means = [means[i] for i in valid_idx]
        valid_lower = [ci_lowers[i] for i in valid_idx]
        valid_upper = [ci_uppers[i] for i in valid_idx]
        ax2.plot(valid_gens, valid_means, "g-", linewidth=2)
        ax2.fill_between(valid_gens, valid_lower, valid_upper, alpha=0.3, color="green")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Novelty Score")
    ax2.set_title(f"Mean Novelty (95% CI, n={num_reps} reps)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(experiment_dir / "combined_mean_novelty.png", dpi=150, bbox_inches="tight")
    console.print(f"[green]Saved: {experiment_dir / 'combined_mean_novelty.png'}[/green]")
    plt.close(fig2)

    # Plot 3: Combined archive size
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    means, ci_lowers, ci_uppers = aggregate_metric("archive_size")
    valid_idx = [i for i, m in enumerate(means) if m is not None]
    if valid_idx:
        valid_gens = [generations[i] for i in valid_idx]
        valid_means = [means[i] for i in valid_idx]
        valid_lower = [ci_lowers[i] for i in valid_idx]
        valid_upper = [ci_uppers[i] for i in valid_idx]
        ax3.plot(valid_gens, valid_means, "r-", linewidth=2)
        ax3.fill_between(valid_gens, valid_lower, valid_upper, alpha=0.3, color="red")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Archive Size")
    ax3.set_title(f"Novelty Archive Growth (95% CI, n={num_reps} reps)")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(experiment_dir / "combined_archive_growth.png", dpi=150, bbox_inches="tight")
    console.print(f"[green]Saved: {experiment_dir / 'combined_archive_growth.png'}[/green]")
    plt.close(fig3)

    # Plot 4: Combined mutation vs crossover distances
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    # Mutation
    means, ci_lowers, ci_uppers = aggregate_metric("mutation_mean_distance")
    valid_idx = [i for i, m in enumerate(means) if m is not None]
    if valid_idx:
        valid_gens = [generations[i] for i in valid_idx]
        valid_means = [means[i] for i in valid_idx]
        valid_lower = [ci_lowers[i] for i in valid_idx]
        valid_upper = [ci_uppers[i] for i in valid_idx]
        ax4.plot(valid_gens, valid_means, "b-", linewidth=2, label="Mutation")
        ax4.fill_between(valid_gens, valid_lower, valid_upper, alpha=0.3, color="blue")

    # Crossover
    means, ci_lowers, ci_uppers = aggregate_metric("crossover_mean_distance")
    valid_idx = [i for i, m in enumerate(means) if m is not None]
    if valid_idx:
        valid_gens = [generations[i] for i in valid_idx]
        valid_means = [means[i] for i in valid_idx]
        valid_lower = [ci_lowers[i] for i in valid_idx]
        valid_upper = [ci_uppers[i] for i in valid_idx]
        ax4.plot(valid_gens, valid_means, "orange", linewidth=2, label="Crossover")
        ax4.fill_between(valid_gens, valid_lower, valid_upper, alpha=0.3, color="orange")

    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Tree Edit Distance")
    ax4.set_title(f"Distance by Operator Type (95% CI, n={num_reps} reps)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4.savefig(experiment_dir / "combined_distance_by_operator.png", dpi=150, bbox_inches="tight")
    console.print(f"[green]Saved: {experiment_dir / 'combined_distance_by_operator.png'}[/green]")
    plt.close(fig4)


def load_statistics_from_csv(filepath: Path) -> list[dict[str, Any]]:
    """Load generation statistics from CSV file.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.

    Returns
    -------
    list[dict]
        List of generation statistics dictionaries.
    """
    stats = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            parsed_row = {}
            for key, value in row.items():
                if value == '' or value == 'None':
                    parsed_row[key] = None
                else:
                    try:
                        # Try int first, then float
                        if '.' in value:
                            parsed_row[key] = float(value)
                        else:
                            parsed_row[key] = int(value)
                    except ValueError:
                        parsed_row[key] = value
            stats.append(parsed_row)
    return stats


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Handle plot-only mode
    if args.plot_only:
        plot_dir = Path(args.plot_only)
        csv_path = plot_dir / "distance_statistics.csv"

        if not csv_path.exists():
            console.print(f"[red]Error: CSV file not found at {csv_path}[/red]")
            return

        console.print(f"[bold]Loading statistics from:[/bold] {csv_path}")

        # Create a minimal evolution object just for plotting
        evolution = PureNoveltyEvolution(verbose=False)
        evolution.generation_distances = load_statistics_from_csv(csv_path)

        # Generate and save plots
        evolution.plot_statistics(save_dir=plot_dir)
        console.print(f"[bold green]Plots saved to:[/bold green] {plot_dir}")
        return

    global DATA, SEED, RNG

    # Create base data directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        dir_name = f"{args.experiment_name}_{timestamp}"
    else:
        dir_name = f"{SCRIPT_NAME}_{timestamp}"

    if args.data_dir:
        base_dir = Path(args.data_dir)
    else:
        base_dir = CWD / "__data__"

    experiment_dir = base_dir / dir_name
    experiment_dir.mkdir(exist_ok=True, parents=True)

    console.print(f"[bold]Experiment directory:[/bold] {experiment_dir}")
    console.print(f"[bold]Repetitions:[/bold] {args.repetitions}")

    # Run repetitions
    for rep in range(args.repetitions):
        # Calculate seed for this repetition
        rep_seed = args.seed + rep
        SEED = rep_seed
        random.seed(rep_seed)
        np.random.seed(rep_seed)
        RNG = np.random.default_rng(rep_seed)

        # Create repetition directory
        if args.repetitions > 1:
            DATA = experiment_dir / f"rep_{rep}"
            DATA.mkdir(exist_ok=True, parents=True)
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]Repetition {rep + 1}/{args.repetitions} (seed={rep_seed})[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        else:
            DATA = experiment_dir

        # Create evolution system
        evolution = PureNoveltyEvolution(
            mu=args.mu,
            lambda_=args.lambda_,
            mutation_rate=args.mutation_rate,
            crossover_rate=1.0 - args.mutation_rate,
            mutate_after_crossover=True,
            strategy_type=args.strategy_type,
            max_part_limit=25,
            max_actuators=12,
            novelty_k_neighbors=args.novelty_k,
            seed=rep_seed,
            num_workers=args.num_workers,
            verbose=True,
        )

        # Run evolution
        final_pop, stats = evolution.run(num_generations=args.num_generations)

        # Save and display results
        evolution.save_statistics(DATA / "distance_statistics.csv")
        evolution.print_summary()
        evolution.plot_statistics(save_dir=DATA)

        console.print(f"[green]Repetition {rep + 1} results saved to:[/green] {DATA}")

    # Generate combined plots if multiple repetitions
    if args.repetitions > 1:
        console.print(f"\n[bold cyan]Generating combined plots across {args.repetitions} repetitions...[/bold cyan]")
        plot_combined_statistics(experiment_dir, args.repetitions)

    console.print(f"\n[bold green]All results saved to:[/bold green] {experiment_dir.absolute()}")


if __name__ == "__main__":
    main()
