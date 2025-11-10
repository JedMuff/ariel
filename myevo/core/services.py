"""Service classes for evolution system.

This module provides focused service classes for common operations,
following the single responsibility principle.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from ariel.ec import Individual

console = Console()


class CacheManager:
    """Manages temporary cache files for CMA-ES states and other data.

    This prevents serialization of large objects through multiprocessing pipes
    by using disk-based caching instead.
    """

    def __init__(self, cache_dir: Path | str | None = None, verbose: bool = True):
        """Initialize cache manager.

        Parameters
        ----------
        cache_dir : Path | str | None, optional
            Directory for cache files. If None, creates temp directory.
        verbose : bool, optional
            Whether to print cleanup messages.
        """
        if cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "ariel_cmaes_cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.verbose = verbose
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def cleanup(self) -> int:
        """Remove all cached files.

        Returns
        -------
        int
            Number of files removed.
        """
        if not self.cache_dir.exists():
            return 0

        files = list(self.cache_dir.glob("**/*"))
        num_files = len([f for f in files if f.is_file()])

        if num_files > 0:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                console.print(f"[green]Cleaned {num_files} cache files[/green]")

        return num_files

    def get_status(self) -> dict[str, Any]:
        """Get cache status information.

        Returns
        -------
        dict
            Status info with 'exists', 'num_files', 'total_mb' keys.
        """
        if not self.cache_dir.exists():
            return {"exists": False, "num_files": 0, "total_mb": 0.0}

        files = list(self.cache_dir.glob("**/*"))
        file_list = [f for f in files if f.is_file()]
        total_bytes = sum(f.stat().st_size for f in file_list)

        return {
            "exists": True,
            "num_files": len(file_list),
            "total_mb": total_bytes / (1024 * 1024),
        }

    def get_path(self, filename: str) -> Path:
        """Get path for a cache file.

        Parameters
        ----------
        filename : str
            Name of the cache file.

        Returns
        -------
        Path
            Full path to the cache file.
        """
        return self.cache_dir / filename


class ResultsPersistence:
    """Handles saving evaluation results to disk.

    Consolidates all file I/O operations for brain weights, learning curves,
    metadata, and fitness components.
    """

    @staticmethod
    def save_brain_weights(
        save_dir: Path,
        initial_weights: np.ndarray,
        optimized_weights: np.ndarray,
    ) -> None:
        """Save initial and optimized brain weights.

        Parameters
        ----------
        save_dir : Path
            Directory to save weights to.
        initial_weights : np.ndarray
            Pre-optimization weights.
        optimized_weights : np.ndarray
            Post-optimization weights.
        """
        np.savez_compressed(save_dir / "initial_brain.npz", weights=initial_weights)
        np.savez_compressed(save_dir / "optimized_brain.npz", weights=optimized_weights)

    @staticmethod
    def save_metadata(save_dir: Path, metadata: dict[str, Any]) -> None:
        """Save metadata about controller and optimization.

        Parameters
        ----------
        save_dir : Path
            Directory to save metadata to.
        metadata : dict
            Metadata dictionary.
        """
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def save_learning_curve(
        save_dir: Path,
        learning_curve: list[float],
    ) -> None:
        """Save CMA-ES learning curve to CSV.

        Parameters
        ----------
        save_dir : Path
            Directory to save learning curve to.
        learning_curve : list[float]
            List of fitness values over iterations.
        """
        with open(save_dir / "learning_curve.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "fitness"])
            for i, fitness in enumerate(learning_curve):
                writer.writerow([i, fitness])

    @staticmethod
    def save_fitness_components(
        save_dir: Path,
        locomotion_fitness: float,
        novelty_score: float | None = None,
    ) -> None:
        """Save fitness components breakdown.

        Parameters
        ----------
        save_dir : Path
            Directory to save components to.
        locomotion_fitness : float
            Base locomotion fitness.
        novelty_score : float | None, optional
            Novelty score if enabled.
        """
        components = {
            "locomotion_fitness": float(locomotion_fitness),
            "novelty_score": novelty_score,
        }
        with open(save_dir / "fitness_components.json", "w") as f:
            json.dump(components, f, indent=2)

    @staticmethod
    def update_fitness_components(
        save_dir: Path,
        novelty_score: float,
    ) -> None:
        """Update existing fitness components file with novelty score.

        Parameters
        ----------
        save_dir : Path
            Directory containing fitness_components.json.
        novelty_score : float
            Novelty score to add.
        """
        components_path = save_dir / "fitness_components.json"
        if components_path.exists():
            with open(components_path, "r") as f:
                data = json.load(f)
            data["novelty_score"] = novelty_score
            with open(components_path, "w") as f:
                json.dump(data, f, indent=2)


class DatabasePersistence:
    """Handles database persistence to CSV and JSON.

    Extracts database operations from strategy class for better separation of concerns.
    """

    @staticmethod
    def save_snapshot(
        log_dir_base: Path | str,
        all_individuals: list[Individual],
        selected_individuals: list[Individual],
        current_generation: int,
    ) -> None:
        """Save database snapshot to CSV/JSON.

        Parameters
        ----------
        log_dir_base : Path | str
            Base directory for logging.
        all_individuals : list[Individual]
            All evaluated individuals in this generation.
        selected_individuals : list[Individual]
            Individuals selected to survive.
        current_generation : int
            Current generation number.
        """
        save_dir = Path(log_dir_base)
        csv_path = save_dir / "database.csv"
        json_path = save_dir / "database.json"

        # Create set of selected IDs
        selected_ids = {ind.id for ind in selected_individuals}

        # Prepare records
        records = []
        for ind in all_individuals:
            record = DatabasePersistence._create_record(
                ind, current_generation, selected_ids
            )
            records.append(record)

        # Save to CSV (append mode)
        if records:
            file_exists = csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerows(records)

        # Save to JSON (read, append, write)
        all_records = []
        if json_path.exists():
            with open(json_path, "r") as f:
                all_records = json.load(f)
        all_records.extend(records)
        with open(json_path, "w") as f:
            json.dump(all_records, f, indent=2)

    @staticmethod
    def _create_record(
        ind: Individual,
        current_generation: int,
        selected_ids: set[int],
    ) -> dict[str, Any]:
        """Create database record for an individual.

        Parameters
        ----------
        ind : Individual
            Individual to create record for.
        current_generation : int
            Current generation number.
        selected_ids : set[int]
            Set of selected individual IDs.

        Returns
        -------
        dict
            Record dictionary.
        """
        from myevo.core import TreeGenotype

        # Get tree for counting
        tree = (
            ind.genotype.tree
            if isinstance(ind.genotype, TreeGenotype)
            else ind.genotype
        )

        # Count parts and actuators
        num_parts = len(tree.nodes)
        num_actuators = sum(
            1 for _, data in tree.nodes(data=True) if data.get("type") == "HINGE"
        )

        # Extract from tags
        parent1_id = ind.tags.get("parent1_id", None) if ind.tags else None
        parent2_id = ind.tags.get("parent2_id", None) if ind.tags else None
        directory = ind.tags.get("log_dir", "") if ind.tags else ""
        locomotion_fitness = ind.tags.get("locomotion_fitness") if ind.tags else None
        novelty_score = ind.tags.get("novelty_score") if ind.tags else None
        cmaes_sigma = ind.tags.get("cmaes_sigma") if ind.tags else None
        cmaes_condition_number = (
            ind.tags.get("cmaes_condition_number") if ind.tags else None
        )
        cmaes_mean_fitness = ind.tags.get("cmaes_mean_fitness") if ind.tags else None
        cmaes_num_evaluations = (
            ind.tags.get("cmaes_num_evaluations") if ind.tags else None
        )

        return {
            "individual_id": ind.id,
            "birth_generation": ind.time_of_birth,
            "current_generation": current_generation,
            "selected": ind.id in selected_ids,
            "fitness": ind.fitness if ind.fitness is not None else None,
            "locomotion_fitness": locomotion_fitness,
            "novelty_score": novelty_score,
            "parent1_id": parent1_id,
            "parent2_id": parent2_id,
            "directory": directory,
            "num_parts": num_parts,
            "num_actuators": num_actuators,
            "cmaes_sigma": cmaes_sigma,
            "cmaes_condition_number": cmaes_condition_number,
            "cmaes_mean_fitness": cmaes_mean_fitness,
            "cmaes_num_evaluations": cmaes_num_evaluations,
        }
