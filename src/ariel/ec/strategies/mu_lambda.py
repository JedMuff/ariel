"""Mu+Lambda and Mu,Lambda evolutionary strategies.

This module implements the classic (μ+λ) and (μ,λ) evolution strategies,
adapted to work with ARIEL's Individual model and database persistence.
"""

from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sqlalchemy import Engine
from sqlmodel import Session
from tqdm import tqdm

from ariel.ec.a001 import Individual
from ariel.ec.evaluation import evaluate_population
from ariel.ec.genotypes.base import Genotype
from ariel.ec.selection import select_parents
from ariel.ec.strategies.weight_inheritance import ParentWeightManager


class MuLambdaStrategy:
    """(μ+λ) and (μ,λ) evolution strategies.

    Attributes
    ----------
    genotype : Genotype
        The genotype system defining mutation and crossover operations.
    population_size : int
        Size of the population (μ).
    num_offspring : int
        Number of offspring to generate per generation (λ).
    num_mutate : int
        Number of offspring created via mutation.
    num_crossover : int
        Number of offspring created via crossover.
    mutate_after_crossover : bool
        Whether to apply mutation to crossover offspring.
    strategy_type : str
        Either 'plus' (μ+λ) or 'comma' (μ,λ).
    selection_method : str
        Parent selection method.
    maximize : bool
        Whether to maximize or minimize fitness.
    verbose : bool
        Whether to print progress information.
    """

    def __init__(
        self,
        genotype: Genotype,
        population_size: int,
        num_offspring: int,
        num_mutate: int,
        num_crossover: int,
        mutate_after_crossover: bool = False,
        strategy_type: str = "comma",
        selection_method: str = "tournament",
        maximize: bool = True,
        verbose: bool = True,
        lamarckian_mode: bool = False,
        weight_crossover_mode: str = "closest_parent",
        weight_sigma: float = 1.0,
        covariance_inheritance_mode: str = "adaptive",
        sigma_inheritance_mode: str = "blend",
        post_evaluation_callback: Callable[[list[Individual]], None] | None = None,
        save_database_per_generation: bool = False,
        cache_dir: Path | str | None = None,
    ):
        """Initialize the Mu+Lambda or Mu,Lambda evolution strategy.

        Parameters
        ----------
        genotype : Genotype
            The genotype system defining genetic operations.
        population_size : int
            Size of the population (μ).
        num_offspring : int
            Number of offspring to generate per generation (λ).
        num_mutate : int
            Number of offspring created via mutation only.
        num_crossover : int
            Number of offspring created via crossover.
        mutate_after_crossover : bool, optional
            Whether to mutate offspring after crossover, by default False.
        strategy_type : str, optional
            Either 'plus' (μ+λ) or 'comma' (μ,λ), by default 'comma'.
        selection_method : str, optional
            Parent selection method, by default 'tournament'.
        maximize : bool, optional
            Whether to maximize fitness (True) or minimize (False), by default True.
        verbose : bool, optional
            Whether to print progress information, by default True.
        lamarckian_mode : bool, optional
            If True, offspring inherit optimized weights from parents (Lamarckian evolution).
            If False, offspring inherit initial weights from parents (non-Lamarckian evolution).
            By default False.
        weight_crossover_mode : str, optional
            How to combine parent weights for crossover offspring, by default "closest_parent".
            Options: "average", "parent1", "random", "closest_parent".
        weight_sigma : float, optional
            Range for random weight initialization when adapting to morphology changes,
            by default 1.0.
        covariance_inheritance_mode : str, optional
            How to adapt CMA-ES covariance matrices when morphology changes,
            by default "adaptive".
            Options: "adaptive" (preserve learned correlations), "reset" (identity),
            "preserve" (scale existing).
        sigma_inheritance_mode : str, optional
            How to adapt CMA-ES step size when morphology changes,
            by default "blend".
            Options: "blend" (formula-based), "reset" (sigma_init), "keep" (parent),
            "adaptive" (custom logic).
        post_evaluation_callback : Callable[[list[Individual]], None] | None, optional
            Optional callback function to be called after population evaluation.
            Receives the evaluated population as argument. Useful for custom
            post-processing like novelty recalculation. By default None.
        save_database_per_generation : bool, optional
            If True, saves database.csv and database.json after each generation,
            including all evaluated individuals (both selected and non-selected).
            Only works when log_dir_base is provided in evolve(). By default False.

            Note on folder saves: By default, only offspring (λ individuals) are
            evaluated per generation and get log folders created. For (μ+λ) strategies,
            parents don't get new folders unless reevaluate_parents=True is set.
            This means the database will contain all μ+λ records per generation, but
            only λ will have folders from the current generation (parents have folders
            from when they were originally evaluated in previous generations).

        Raises
        ------
        ValueError
            If strategy_type is not 'plus' or 'comma'.
        ValueError
            If num_mutate + num_crossover != num_offspring.
        """
        if strategy_type not in ["plus", "comma"]:
            msg = f"Strategy type must be 'plus' or 'comma', got '{strategy_type}'"
            raise ValueError(msg)

        if num_mutate + num_crossover != num_offspring:
            msg = (
                f"num_mutate ({num_mutate}) + num_crossover ({num_crossover}) "
                f"must equal num_offspring ({num_offspring})"
            )
            raise ValueError(msg)

        self.genotype = genotype
        self.population_size = population_size
        self.num_offspring = num_offspring
        self.num_mutate = num_mutate
        self.num_crossover = num_crossover
        self.mutate_after_crossover = mutate_after_crossover
        self.strategy_type = strategy_type
        self.selection_method = selection_method
        self.maximize = maximize
        self.verbose = verbose

        # Weight inheritance parameters
        self.lamarckian_mode = lamarckian_mode
        self.weight_crossover_mode = weight_crossover_mode
        self.weight_sigma = weight_sigma

        # CMA-ES inheritance parameters
        self.covariance_inheritance_mode = covariance_inheritance_mode
        self.sigma_inheritance_mode = sigma_inheritance_mode

        # Post-evaluation callback
        self.post_evaluation_callback = post_evaluation_callback

        # Database saving per generation
        self.save_database_per_generation = save_database_per_generation

        # Cache directory for CMA-ES states (experiment-specific to avoid conflicts)
        if cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "ariel_cmaes_cache"
        else:
            self.cache_dir = Path(cache_dir)

        # Weight managers for inheritance
        # Initial weights manager: stores pre-optimization weights (for non-Lamarckian mode)
        self._initial_weight_manager = ParentWeightManager(
            crossover_mode=weight_crossover_mode,
            sigma=weight_sigma,
        )
        # Optimized weights manager: stores post-optimization weights (for Lamarckian mode)
        self._optimized_weight_manager = ParentWeightManager(
            crossover_mode=weight_crossover_mode,
            sigma=weight_sigma,
        )

        # CMA-ES state managers for inheritance (Lamarckian CMA-ES)
        from ariel.ec.strategies.cmaes_inheritance import CMAESStateManager
        # Initial CMA-ES states: stores pre-optimization CMA-ES states
        self._initial_cmaes_manager = CMAESStateManager(
            sigma_init=weight_sigma,
            covariance_mode=covariance_inheritance_mode,
            sigma_mode=sigma_inheritance_mode,
        )
        # Optimized CMA-ES states: stores post-optimization CMA-ES states
        self._optimized_cmaes_manager = CMAESStateManager(
            sigma_init=weight_sigma,
            covariance_mode=covariance_inheritance_mode,
            sigma_mode=sigma_inheritance_mode,
        )

        # Evolution state
        self.current_generation = 0
        self.all_individuals: list[Individual] = []

    def initialize_population(
        self,
        engine: Engine | None = None,
        initial_population: list[Any] | None = None,
    ) -> list[Individual]:
        """Initialize the population.

        Parameters
        ----------
        engine : Engine | None, optional
            Database engine for persistence, by default None.
        initial_population : list[Any] | None, optional
            Initial genomes to use instead of random generation, by default None.

        Returns
        -------
        list[Individual]
            The initial population.
        """
        population = []

        if initial_population is not None:
            # Use provided genomes
            for i, genome in enumerate(initial_population):
                ind = Individual()
                ind.genotype = genome
                ind.time_of_birth = 0
                population.append(ind)
        else:
            # Generate random population
            genomes = self.genotype.random_population(self.population_size)
            for i, genome in enumerate(genomes):
                ind = Individual()
                ind.genotype = genome
                ind.time_of_birth = 0
                population.append(ind)

        # Save to database if engine provided
        if engine is not None:
            with Session(engine) as session:
                session.add_all(population)
                session.commit()
                for ind in population:
                    session.refresh(ind)

        return population

    def step(
        self,
        population: list[Individual],
        fitness_function: Callable[[Individual, str | None], float],
        generation: int,
        log_dir_base: str | None = None,
        num_workers: int = 1,
        reevaluate_parents: bool = False,
    ) -> list[Individual]:
        """Execute one generation of evolution.

        Parameters
        ----------
        population : list[Individual]
            Current population (μ individuals).
        fitness_function : Callable[[Individual, str | None], float]
            Fitness evaluation function taking (individual, log_dir) -> fitness.
        generation : int
            Current generation number.
        log_dir_base : str | None, optional
            Base directory for logging, by default None.
        num_workers : int, optional
            Number of parallel workers for evaluation, by default 1.
        reevaluate_parents : bool, optional
            Whether to reevaluate parents each generation (only for 'plus'), by default False.
            Set to True if you want all μ+λ individuals to have log folders created in each
            generation. When False, only offspring get new folders (parents retain folders
            from when they were originally evaluated).

        Returns
        -------
        list[Individual]
            The next generation population (μ individuals).
        """
        # Select parents for crossover and mutation
        num_parents_needed = self.num_crossover * 2 + self.num_mutate
        parents = select_parents(
            population,
            k=num_parents_needed,
            method=self.selection_method,
            maximize=self.maximize,
        )

        # Split parents into groups
        crossover_parents = parents[: self.num_crossover * 2]
        mutation_parents = parents[self.num_crossover * 2 :]

        # Generate offspring via crossover
        offspring = []
        for i in range(0, len(crossover_parents), 2):
            parent1 = crossover_parents[i]
            parent2 = crossover_parents[i + 1]

            # Perform crossover
            child_genome1, child_genome2 = self.genotype.crossover(
                parent1.genotype,
                parent2.genotype,
            )

            # Optionally mutate after crossover
            if self.mutate_after_crossover:
                child_genome1 = self.genotype.mutate(child_genome1)

            # Create Individual for first child
            child = Individual()
            child.genotype = child_genome1
            child.time_of_birth = generation
            child.tags = {
                "parent1_id": parent1.id,
                "parent2_id": parent2.id,
                "operator": "crossover" + (" + mutation" if self.mutate_after_crossover else ""),
            }
            offspring.append(child)

        # Generate offspring via mutation
        for parent in mutation_parents:
            # Perform mutation
            child_genome = self.genotype.mutate(parent.genotype)

            # Create Individual
            child = Individual()
            child.genotype = child_genome
            child.time_of_birth = generation
            child.tags = {
                "parent1_id": parent.id,
                "operator": "mutation",
            }
            offspring.append(child)

        # Assign IDs to offspring before evaluation (needed for log_dir paths)
        # Find the maximum ID in the current population
        max_id = max([ind.id for ind in population if ind.id is not None], default=0)
        next_id = max_id + 1
        for child in offspring:
            if child.id is None:
                child.id = next_id
                next_id += 1

        # Populate inherited weights for offspring before evaluation
        # This enables weight inheritance from parents
        self._populate_inherited_weights(offspring, population)

        # Evaluate offspring
        offspring = evaluate_population(
            offspring,
            fitness_function,
            generation,
            log_dir_base,
            num_workers,
        )

        # Call post-evaluation callback if provided (e.g., for novelty recalculation)
        if self.post_evaluation_callback is not None:
            self.post_evaluation_callback(offspring)

        # Extract and store weights from evaluated offspring
        # This captures both initial and optimized weights for future inheritance
        self._extract_and_store_weights(offspring)

        # Clean up temp cache files after extraction is complete
        # At this point, all offspring have been evaluated and their states extracted
        # The temporary cache files from parent inheritance are no longer needed
        import shutil
        if self.cache_dir.exists():
            try:
                files_before = len(list(self.cache_dir.glob("**/*")))
                if files_before > 0:
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    if self.verbose:
                        print(f"  Cleaned {files_before} temp cache files after offspring evaluation", flush=True)
            except Exception:
                pass  # Silently ignore cleanup errors during evaluation

        # Handle reevaluation of parents (only for plus strategy)
        if reevaluate_parents and self.strategy_type == "plus":
            population = evaluate_population(
                population,
                fitness_function,
                generation,
                log_dir_base,
                num_workers,
            )

        # Survivor selection
        if self.strategy_type == "plus":
            # (μ+λ): Select best μ from μ parents + λ offspring
            combined = population + offspring
        else:
            # (μ,λ): Select best μ from λ offspring only
            combined = offspring

        # Sort by fitness and select top μ
        combined.sort(
            key=lambda ind: ind.fitness or (float('-inf') if self.maximize else float('inf')),
            reverse=self.maximize,
        )
        next_population = combined[: self.population_size]

        # Store combined population for database saving (includes both selected and non-selected)
        self._last_combined_population = combined
        self._last_selected_population = next_population

        # Clean up old states to prevent memory leaks
        # Only keep states for the surviving population (μ individuals)
        self._cleanup_old_states(next_population)

        # Force garbage collection after each generation to prevent memory accumulation
        # This is especially important with high CMA-ES budgets
        gc.collect()

        return next_population

    def _populate_inherited_weights(
        self,
        offspring: list[Individual],
        population: list[Individual],
    ) -> None:
        """Populate inherited weights and CMA-ES states in offspring tags before evaluation.

        This method looks up parent weights and CMA-ES states and stores them in
        offspring.tags so that the fitness function can access them during evaluation.

        Parameters
        ----------
        offspring : list[Individual]
            Offspring to populate with inherited weights and CMA-ES states.
        population : list[Individual]
            Parent population (for tracking).
        """
        from ariel.ec import TreeGenotype

        # Track all evaluated individuals in weight and CMA-ES managers
        for ind in population:
            self._initial_weight_manager.add_evaluated_individual(ind)
            self._optimized_weight_manager.add_evaluated_individual(ind)

        # Choose weight manager based on Lamarckian mode
        weight_manager = self._optimized_weight_manager if self.lamarckian_mode else self._initial_weight_manager
        # Choose CMA-ES manager based on Lamarckian mode (same selection logic)
        cmaes_manager = self._optimized_cmaes_manager if self.lamarckian_mode else self._initial_cmaes_manager

        # Populate inherited weights for each offspring
        for child in offspring:
            # Get parent IDs from tags
            parent1_id = child.tags.get("parent1_id")
            parent2_id = child.tags.get("parent2_id")

            parent_weights_and_sizes = None
            chosen_parent_id = None  # Track which parent we chose

            if parent2_id is not None:
                # Crossover - use configured crossover mode
                parent1_data = weight_manager.learned_weights.get(parent1_id)
                parent2_data = weight_manager.learned_weights.get(parent2_id)

                if self.weight_crossover_mode == "closest_parent":
                    # Use tree distance to find closest parent
                    if parent1_data and parent2_data:
                        from ariel.ec.strategies.weight_inheritance import tree_distance
                        offspring_tree = child.genotype.tree if isinstance(child.genotype, TreeGenotype) else child.genotype
                        # Find parent individuals
                        parent1_ind = next((ind for ind in population if ind.id == parent1_id), None)
                        parent2_ind = next((ind for ind in population if ind.id == parent2_id), None)
                        if parent1_ind and parent2_ind:
                            parent1_tree = parent1_ind.genotype.tree if isinstance(parent1_ind.genotype, TreeGenotype) else parent1_ind.genotype
                            parent2_tree = parent2_ind.genotype.tree if isinstance(parent2_ind.genotype, TreeGenotype) else parent2_ind.genotype
                            dist1 = tree_distance(offspring_tree, parent1_tree)
                            dist2 = tree_distance(offspring_tree, parent2_tree)
                            if dist1 <= dist2:
                                parent_weights_and_sizes = parent1_data
                                chosen_parent_id = parent1_id
                            else:
                                parent_weights_and_sizes = parent2_data
                                chosen_parent_id = parent2_id
                        else:
                            parent_weights_and_sizes = parent1_data or parent2_data
                            chosen_parent_id = parent1_id if parent1_data else parent2_id
                    else:
                        parent_weights_and_sizes = parent1_data or parent2_data
                        chosen_parent_id = parent1_id if parent1_data else parent2_id
                elif self.weight_crossover_mode == "parent1":
                    parent_weights_and_sizes = parent1_data
                    chosen_parent_id = parent1_id
                elif self.weight_crossover_mode == "random":
                    if parent1_data and parent2_data:
                        import random
                        if random.random() < 0.5:
                            parent_weights_and_sizes = parent1_data
                            chosen_parent_id = parent1_id
                        else:
                            parent_weights_and_sizes = parent2_data
                            chosen_parent_id = parent2_id
                    else:
                        parent_weights_and_sizes = parent1_data or parent2_data
                        chosen_parent_id = parent1_id if parent1_data else parent2_id
                else:  # "average" or default
                    # For average mode, just use parent1 (averaging happens in weight_inheritance module)
                    parent_weights_and_sizes = parent1_data or parent2_data
                    chosen_parent_id = parent1_id if parent1_data else parent2_id
            else:
                # Mutation - inherit from single parent
                if parent1_id is not None:
                    parent_weights_and_sizes = weight_manager.learned_weights.get(parent1_id)
                    chosen_parent_id = parent1_id

            # Store inherited weights in tags for fitness function to access and adapt
            if parent_weights_and_sizes is not None:
                parent_weights, parent_layer_sizes = parent_weights_and_sizes
                child.tags["inherited_weights"] = parent_weights
                child.tags["inherited_layer_sizes"] = parent_layer_sizes

            # Save parent's CMA-ES state to a temporary cache file for multiprocessing
            # This avoids serializing large covariance matrices through pipes (which causes BrokenPipeError)
            if chosen_parent_id is not None:
                parent_cmaes_state = cmaes_manager.get_state(chosen_parent_id)
                if parent_cmaes_state is not None:
                    # Save to temporary file that worker can load
                    from ariel.ec.strategies.cmaes_inheritance import save_cmaes_state_to_disk

                    # Create temp directory if it doesn't exist
                    self.cache_dir.mkdir(exist_ok=True, parents=True)

                    # Save state with parent ID as filename
                    cache_file = self.cache_dir / f"parent_{chosen_parent_id}_state"
                    save_cmaes_state_to_disk(parent_cmaes_state, cache_file)

                    # Store file path in tags (much smaller than the full state object)
                    child.tags["inherited_cmaes_cache_path"] = str(cache_file)
                    child.tags["inherited_layer_sizes_cmaes"] = parent_cmaes_state.layer_sizes

    def _extract_and_store_weights(self, offspring: list[Individual]) -> None:
        """Extract weights and CMA-ES states from evaluated offspring and store in managers.

        After evaluation, the fitness function should have populated:
        - tags["initial_weights"]: Pre-optimization weights
        - tags["optimized_weights"]: Post-optimization weights
        - tags["layer_sizes"]: Neural network architecture
        - tags["initial_cmaes_cache_path"]: Path to initial CMA-ES state cache file
        - tags["optimized_cmaes_cache_path"]: Path to optimized CMA-ES state cache file
        - tags["optimized_cmaes_layer_sizes"]: Layer sizes for optimized state

        This method extracts these and stores them for future inheritance.

        Parameters
        ----------
        offspring : list[Individual]
            Evaluated offspring with weight and CMA-ES state information in tags.
        """
        from pathlib import Path
        from ariel.ec.strategies.cmaes_inheritance import load_cmaes_state_from_disk

        for child in offspring:
            # Extract weights and layer sizes from tags
            initial_weights = child.tags.get("initial_weights")
            optimized_weights = child.tags.get("optimized_weights")
            layer_sizes = child.tags.get("layer_sizes")

            # Load CMA-ES states from cache files (avoid pipe serialization)
            initial_cmaes_cache_path = child.tags.get("initial_cmaes_cache_path")
            initial_cmaes_layer_sizes = child.tags.get("initial_cmaes_layer_sizes")
            optimized_cmaes_cache_path = child.tags.get("optimized_cmaes_cache_path")
            optimized_cmaes_layer_sizes = child.tags.get("optimized_cmaes_layer_sizes")

            initial_cmaes_state = None
            optimized_cmaes_state = None

            if initial_cmaes_cache_path is not None and initial_cmaes_layer_sizes is not None:
                cache_path = Path(initial_cmaes_cache_path)
                if cache_path.exists():
                    initial_cmaes_state = load_cmaes_state_from_disk(cache_path, initial_cmaes_layer_sizes)

            if optimized_cmaes_cache_path is not None and optimized_cmaes_layer_sizes is not None:
                cache_path = Path(optimized_cmaes_cache_path)
                if cache_path.exists():
                    optimized_cmaes_state = load_cmaes_state_from_disk(cache_path, optimized_cmaes_layer_sizes)

            # Store in weight managers if available
            if initial_weights is not None and layer_sizes is not None:
                self._initial_weight_manager.store_weights(
                    child.id,
                    initial_weights,
                    layer_sizes,
                )

            if optimized_weights is not None and layer_sizes is not None:
                self._optimized_weight_manager.store_weights(
                    child.id,
                    optimized_weights,
                    layer_sizes,
                )

            # Store CMA-ES states in respective managers (CRITICAL FIX)
            # Initial manager gets initial state, optimized manager gets optimized state
            if initial_cmaes_state is not None:
                self._initial_cmaes_manager.store_state(child.id, initial_cmaes_state)

            if optimized_cmaes_state is not None:
                self._optimized_cmaes_manager.store_state(child.id, optimized_cmaes_state)

            # Add to evaluated individuals list
            self._initial_weight_manager.add_evaluated_individual(child)
            self._optimized_weight_manager.add_evaluated_individual(child)

    def _validate_state_separation(self, individual: Individual, verbose: bool = False) -> bool:
        """Validate that initial and optimized states are properly separated.

        This is a debugging/validation helper to ensure the inheritance system
        is working correctly. Checks that initial and optimized states differ
        when CMA-ES optimization is used.

        Parameters
        ----------
        individual : Individual
            Individual to validate states for.
        verbose : bool, optional
            Whether to print validation messages, by default False.

        Returns
        -------
        bool
            True if validation passes, False otherwise.
        """
        # Get states from tags
        initial_weights = individual.tags.get("initial_weights")
        optimized_weights = individual.tags.get("optimized_weights")
        initial_cmaes = individual.tags.get("initial_cmaes_state")
        optimized_cmaes = individual.tags.get("optimized_cmaes_state")
        num_evaluations = individual.tags.get("cmaes_num_evaluations", 0)

        # If CMA-ES was used (num_evaluations > 0), states should differ
        if num_evaluations > 0:
            # Check weights
            if initial_weights is not None and optimized_weights is not None:
                weights_differ = not np.allclose(initial_weights, optimized_weights)
                if verbose and not weights_differ:
                    print(f"WARNING: Individual {individual.id} has identical initial/optimized weights despite CMA-ES optimization!")

            # Check CMA-ES sigma
            if initial_cmaes is not None and optimized_cmaes is not None:
                sigma_differs = abs(initial_cmaes.sigma - optimized_cmaes.sigma) > 1e-6
                if verbose and not sigma_differs:
                    print(f"WARNING: Individual {individual.id} has identical initial/optimized sigma despite CMA-ES optimization!")

                # Check covariance condition number (should change during optimization)
                if initial_cmaes.condition_number is not None and optimized_cmaes.condition_number is not None:
                    condition_differs = abs(initial_cmaes.condition_number - optimized_cmaes.condition_number) > 1e-3
                    if verbose and not condition_differs:
                        print(f"WARNING: Individual {individual.id} has identical covariance condition numbers!")

                return True

        # If CMA-ES was not used, states should be identical (both default)
        return True

    def _cleanup_old_states(self, current_population: list[Individual]) -> None:
        """Clean up weights and CMA-ES states for individuals no longer in population.

        This prevents memory leaks by removing cached states for old individuals.
        Retains states for current population only.

        Parameters
        ----------
        current_population : list[Individual]
            The current population to retain states for.
        """
        import shutil
        import tempfile
        from pathlib import Path

        # Get IDs of individuals to keep (current population)
        keep_ids = {ind.id for ind in current_population if ind.id is not None}

        # Clean up weight managers
        self._initial_weight_manager.clear_old_weights(keep_ids)
        self._optimized_weight_manager.clear_old_weights(keep_ids)

        # Clean up CMA-ES state managers (keep only current population)
        # Note: CMAESStateManager doesn't have clear_old_states, so we need to add it
        # For now, we'll manually filter the internal dicts
        for manager in [self._initial_cmaes_manager, self._optimized_cmaes_manager]:
            old_ids = set(manager._states.keys()) - keep_ids
            for old_id in old_ids:
                manager._states.pop(old_id, None)
                manager._layer_sizes.pop(old_id, None)

        # AGGRESSIVE CLEANUP: Remove ALL temp cache files after every generation
        # This is critical because cache files accumulate faster than cleanup can track them
        # (e.g., 30 offspring × 2 states = 60 files per generation)
        if self.cache_dir.exists():
            try:
                # Get count before cleanup for logging
                files_before = len(list(self.cache_dir.glob("**/*")))
                if files_before > 0:
                    # Remove entire cache directory and recreate
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    if self.verbose:
                        print(f"Cleaned {files_before} temp cache files", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to clean temp cache: {e}", flush=True)

        # Clean up evaluated individuals list (keep only current population)
        # This prevents unbounded growth of the tracking list
        self._initial_weight_manager.all_evaluated_individuals = list(current_population)
        self._optimized_weight_manager.all_evaluated_individuals = list(current_population)

    def evolve(
        self,
        fitness_function: Callable[[Individual, str | None], float],
        num_generations: int,
        engine: Engine | None = None,
        initial_population: list[Any] | None = None,
        log_dir_base: str | None = None,
        num_workers: int = 1,
        reevaluate_parents: bool = False,
    ) -> list[Individual]:
        """Run the complete evolution process.

        Parameters
        ----------
        fitness_function : Callable[[Individual, str | None], float]
            Fitness evaluation function taking (individual, log_dir) -> fitness.
        num_generations : int
            Number of generations to evolve.
        engine : Engine | None, optional
            Database engine for persistence, by default None.
        initial_population : list[Any] | None, optional
            Initial genomes to use, by default None (random initialization).
        log_dir_base : str | None, optional
            Base directory for logging, by default None.
        num_workers : int, optional
            Number of parallel workers for evaluation, by default 1.
        reevaluate_parents : bool, optional
            Whether to reevaluate parents each generation, by default False.
            Set to True if you want all μ+λ individuals to have log folders created in each
            generation. When False, only offspring get new folders (parents retain folders
            from when they were originally evaluated).

        Returns
        -------
        list[Individual]
            All individuals from all generations.

        Examples
        --------
        >>> from ariel.ec.genotypes import TreeGenotype
        >>> genotype = TreeGenotype(max_part_limit=10)
        >>> strategy = MuLambdaStrategy(
        ...     genotype=genotype,
        ...     population_size=10,
        ...     num_offspring=20,
        ...     num_mutate=10,
        ...     num_crossover=10,
        ...     strategy_type='comma',
        ... )
        >>> def fitness_fn(individual, log_dir):
        ...     return len(individual.genotype.nodes())
        >>> all_inds = strategy.evolve(fitness_fn, num_generations=50)
        """
        evo_start = time.time()

        # Initialize population
        population = self.initialize_population(engine, initial_population)

        # Assign IDs to initial population before evaluation
        for i, ind in enumerate(population):
            if ind.id is None:
                ind.id = i

        # Evaluate initial population
        population = evaluate_population(
            population,
            fitness_function,
            generation=0,
            log_dir_base=log_dir_base,
            num_workers=num_workers,
        )

        # Call post-evaluation callback if provided (e.g., for novelty recalculation)
        if self.post_evaluation_callback is not None:
            self.post_evaluation_callback(population)

        # Extract and store weights from initial population
        self._extract_and_store_weights(population)

        # Save initial population to database
        if engine is not None:
            with Session(engine) as session:
                session.add_all(population)
                session.commit()

        # Track all individuals
        self.all_individuals = population.copy()

        # Save initial population to database snapshot if enabled
        if self.save_database_per_generation and log_dir_base is not None:
            # For initial generation, all individuals are "selected"
            self._save_database_snapshot(log_dir_base, population, population, 0)

        # Print initial stats
        if self.verbose:
            fitnesses = [ind.fitness or 0.0 for ind in population]
            best_fitness = max(fitnesses) if self.maximize else min(fitnesses)
            avg_fitness = np.mean(fitnesses)
            print(
                f"G:0 Time:{np.round(time.time() - evo_start, 2)}, "
                f"BestF={best_fitness:.4f}, AvgF={avg_fitness:.4f}",
                flush=True,
            )

        # Evolution loop with progress bar
        pbar = tqdm(range(1, num_generations + 1), disable=not self.verbose, desc="Evolution")
        for generation in pbar:
            gen_start = time.time()
            self.current_generation = generation

            # Perform one generation
            population = self.step(
                population,
                fitness_function,
                generation,
                log_dir_base,
                num_workers,
                reevaluate_parents,
            )

            # Save to database
            if engine is not None:
                with Session(engine) as session:
                    session.add_all(population)
                    session.commit()

            # Track all individuals
            self.all_individuals.extend(population)

            # Save database to CSV/JSON after each generation if enabled
            if self.save_database_per_generation and log_dir_base is not None:
                # Save all evaluated individuals (combined) with selection status
                self._save_database_snapshot(
                    log_dir_base,
                    self._last_combined_population,
                    self._last_selected_population,
                    generation
                )

            # Log memory usage every 5 generations
            if generation % 5 == 0:
                try:
                    import psutil
                    process = psutil.Process()
                    mem_info = process.memory_info()
                    mem_mb = mem_info.rss / (1024 * 1024)
                    if self.verbose:
                        print(f"\nG:{generation} Memory: {mem_mb:.1f} MB RSS", flush=True)
                except ImportError:
                    pass  # psutil not available

            # Update progress bar with stats
            if self.verbose:
                fitnesses = [ind.fitness or 0.0 for ind in population]
                best_fitness = max(fitnesses) if self.maximize else min(fitnesses)
                avg_fitness = np.mean(fitnesses)
                gen_time = time.time() - gen_start
                pbar.set_postfix({
                    'BestF': f'{best_fitness:.4f}',
                    'AvgF': f'{avg_fitness:.4f}',
                    'GenTime': f'{gen_time:.2f}s'
                })

        if self.verbose:
            total_time = time.time() - evo_start
            print(f"\nTime taken to evolve: {total_time:.2f} seconds")

        return self.all_individuals

    def _save_database_snapshot(
        self,
        log_dir_base: str,
        all_individuals: list[Individual],
        selected_individuals: list[Individual],
        current_generation: int,
    ) -> None:
        """Save database snapshot to CSV/JSON, appending current generation records.

        This method appends all evaluated individuals to database.csv, including
        both selected and non-selected individuals. For μ+λ strategies, this includes
        both parents and offspring evaluated in the generation.

        Parameters
        ----------
        log_dir_base : str
            Base directory for logging (where generation folders are stored).
        all_individuals : list[Individual]
            All individuals evaluated in this generation (parents + offspring for plus,
            offspring only for comma).
        selected_individuals : list[Individual]
            The individuals that were selected to survive to the next generation.
        current_generation : int
            The current generation number.
        """
        import csv
        import json
        from pathlib import Path

        save_dir = Path(log_dir_base)
        csv_path = save_dir / "database.csv"
        json_path = save_dir / "database.json"

        # Create set of selected individual IDs for quick lookup
        selected_ids = {ind.id for ind in selected_individuals}

        # Prepare records for all evaluated individuals in this generation
        records = []
        for ind in all_individuals:
            # Get tree for counting parts/actuators
            from ariel.ec import TreeGenotype
            tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype

            # Get parent IDs from tags
            parent1_id = ind.tags.get("parent1_id", None) if ind.tags else None
            parent2_id = ind.tags.get("parent2_id", None) if ind.tags else None

            # Get directory path from tags (set during evaluation)
            directory = ind.tags.get("log_dir", "") if ind.tags else ""

            # Count parts and actuators
            num_parts = len(tree.nodes)
            num_actuators = sum(1 for _, data in tree.nodes(data=True) if data.get("type") == "HINGE")

            # Extract fitness components from tags if available
            locomotion_fitness = ind.tags.get("locomotion_fitness") if ind.tags else None
            novelty_score = ind.tags.get("novelty_score") if ind.tags else None

            # Extract CMA-ES metrics from tags if available
            cmaes_sigma = ind.tags.get("cmaes_sigma") if ind.tags else None
            cmaes_condition_number = ind.tags.get("cmaes_condition_number") if ind.tags else None
            cmaes_mean_fitness = ind.tags.get("cmaes_mean_fitness") if ind.tags else None
            cmaes_num_evaluations = ind.tags.get("cmaes_num_evaluations") if ind.tags else None

            record = {
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
            records.append(record)

        # Append to CSV (create with header if doesn't exist)
        if records:
            file_exists = csv_path.exists()
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerows(records)

        # For JSON, read existing data, append, and save
        # (JSON is less efficient for incremental updates but maintained for compatibility)
        all_records = []
        if json_path.exists():
            with open(json_path, 'r') as f:
                all_records = json.load(f)
        all_records.extend(records)
        with open(json_path, 'w') as f:
            json.dump(all_records, f, indent=2)

    def get_best_individual(self, population: list[Individual]) -> Individual:
        """Get the best individual from a population.

        Parameters
        ----------
        population : list[Individual]
            The population to search.

        Returns
        -------
        Individual
            The individual with the best fitness.
        """
        if self.maximize:
            return max(population, key=lambda ind: ind.fitness or float('-inf'))
        else:
            return min(population, key=lambda ind: ind.fitness or float('inf'))
