"""Mu+Lambda and Mu,Lambda evolutionary strategies.

This module implements the classic (μ+λ) and (μ,λ) evolution strategies,
adapted to work with ARIEL's Individual model and database persistence.
"""

from __future__ import annotations

import time
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
        post_evaluation_callback: Callable[[list[Individual]], None] | None = None,
        save_database_per_generation: bool = False,
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
        post_evaluation_callback : Callable[[list[Individual]], None] | None, optional
            Optional callback function to be called after population evaluation.
            Receives the evaluated population as argument. Useful for custom
            post-processing like novelty recalculation. By default None.
        save_database_per_generation : bool, optional
            If True, saves database.csv and database.json after each generation.
            Only works when log_dir_base is provided in evolve(). By default False.

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

        # Post-evaluation callback
        self.post_evaluation_callback = post_evaluation_callback

        # Database saving per generation
        self.save_database_per_generation = save_database_per_generation

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

        return next_population

    def _populate_inherited_weights(
        self,
        offspring: list[Individual],
        population: list[Individual],
    ) -> None:
        """Populate inherited weights in offspring tags before evaluation.

        This method looks up parent weights and stores them in offspring.tags
        so that the fitness function can access them during evaluation.

        Parameters
        ----------
        offspring : list[Individual]
            Offspring to populate with inherited weights.
        population : list[Individual]
            Parent population (for tracking).
        """
        from ariel.ec import TreeGenotype

        # Track all evaluated individuals in weight managers
        for ind in population:
            self._initial_weight_manager.add_evaluated_individual(ind)
            self._optimized_weight_manager.add_evaluated_individual(ind)

        # Choose weight manager based on Lamarckian mode
        weight_manager = self._optimized_weight_manager if self.lamarckian_mode else self._initial_weight_manager

        # Populate inherited weights for each offspring
        for child in offspring:
            # Extract offspring tree for distance calculation (needed for closest_parent mode)
            offspring_tree = child.genotype.tree if isinstance(child.genotype, TreeGenotype) else child.genotype

            # Determine which parent to inherit from
            parent1_id = child.tags.get("parent1_id")
            parent2_id = child.tags.get("parent2_id")

            # Find parent individuals
            parent1_ind = None
            parent2_ind = None
            for ind in population:
                if ind.id == parent1_id:
                    parent1_ind = ind
                if parent2_id is not None and ind.id == parent2_id:
                    parent2_ind = ind

            # Verify parents were found (they should always be in the population)
            if parent1_id is not None and parent1_ind is None:
                raise RuntimeError(
                    f"Parent with ID {parent1_id} not found in population for offspring {child.id}. "
                    "This indicates a bug in the evolution strategy."
                )
            if parent2_id is not None and parent2_ind is None:
                raise RuntimeError(
                    f"Parent with ID {parent2_id} not found in population for offspring {child.id}. "
                    "This indicates a bug in the evolution strategy."
                )

            # Get parent weights and layer sizes
            parent_weights_and_sizes = None

            if parent2_ind is not None:
                # Crossover - combine parents based on crossover mode
                parent1_data = weight_manager.learned_weights.get(parent1_ind.id) if parent1_ind else None
                parent2_data = weight_manager.learned_weights.get(parent2_ind.id) if parent2_ind else None

                if self.weight_crossover_mode == "closest_parent" and offspring_tree is not None:
                    # Use closest parent
                    if parent1_data and parent2_data:
                        from ariel.ec.strategies.weight_inheritance import tree_distance
                        parent1_tree = parent1_ind.genotype.tree if isinstance(parent1_ind.genotype, TreeGenotype) else parent1_ind.genotype
                        parent2_tree = parent2_ind.genotype.tree if isinstance(parent2_ind.genotype, TreeGenotype) else parent2_ind.genotype
                        dist1 = tree_distance(offspring_tree, parent1_tree)
                        dist2 = tree_distance(offspring_tree, parent2_tree)
                        parent_weights_and_sizes = parent1_data if dist1 <= dist2 else parent2_data
                    elif parent1_data:
                        parent_weights_and_sizes = parent1_data
                    elif parent2_data:
                        parent_weights_and_sizes = parent2_data
                elif self.weight_crossover_mode == "parent1" and parent1_data:
                    parent_weights_and_sizes = parent1_data
                elif self.weight_crossover_mode == "random":
                    if parent1_data and parent2_data:
                        import random
                        parent_weights_and_sizes = parent1_data if random.random() < 0.5 else parent2_data
                    elif parent1_data:
                        parent_weights_and_sizes = parent1_data
                    elif parent2_data:
                        parent_weights_and_sizes = parent2_data
                else:  # average or default
                    # For average, just pick one parent for now (fitness function will handle averaging)
                    # This is a simplification - true averaging would need both parents' weights
                    parent_weights_and_sizes = parent1_data or parent2_data
            else:
                # Mutation - inherit from single parent
                if parent1_ind:
                    parent_weights_and_sizes = weight_manager.learned_weights.get(parent1_ind.id)

            # Store in tags for fitness function to access and adapt
            if parent_weights_and_sizes is not None:
                parent_weights, parent_layer_sizes = parent_weights_and_sizes
                child.tags["inherited_weights"] = parent_weights
                child.tags["inherited_layer_sizes"] = parent_layer_sizes

    def _extract_and_store_weights(self, offspring: list[Individual]) -> None:
        """Extract weights from evaluated offspring and store in weight managers.

        After evaluation, the fitness function should have populated:
        - tags["initial_weights"]: Pre-optimization weights
        - tags["optimized_weights"]: Post-optimization weights
        - tags["layer_sizes"]: Neural network architecture

        This method extracts these and stores them for future inheritance.

        Parameters
        ----------
        offspring : list[Individual]
            Evaluated offspring with weight information in tags.
        """
        for child in offspring:
            # Extract weights and layer sizes from tags
            initial_weights = child.tags.get("initial_weights")
            optimized_weights = child.tags.get("optimized_weights")
            layer_sizes = child.tags.get("layer_sizes")

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

            # Add to evaluated individuals list
            self._initial_weight_manager.add_evaluated_individual(child)
            self._optimized_weight_manager.add_evaluated_individual(child)

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
                self._save_database_snapshot(log_dir_base)

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

    def _save_database_snapshot(self, log_dir_base: str) -> None:
        """Save database snapshot to CSV/JSON.

        This method saves the database of all individuals seen so far
        to database.csv and database.json in the log_dir_base directory.

        Parameters
        ----------
        log_dir_base : str
            Base directory for logging (where generation folders are stored).
        """
        import csv
        import json
        from pathlib import Path

        save_dir = Path(log_dir_base)

        # Prepare data records
        records = []

        # Group individuals by generation for indexing
        gen_individuals: dict[int, list[Individual]] = {}
        for ind in self.all_individuals:
            gen = ind.time_of_birth
            if gen not in gen_individuals:
                gen_individuals[gen] = []
            gen_individuals[gen].append(ind)

        # Create index mapping for each generation
        gen_indices: dict[int, dict[int, int]] = {}
        for gen, inds in gen_individuals.items():
            gen_indices[gen] = {ind.id: idx for idx, ind in enumerate(inds)}

        # Build records
        for ind in self.all_individuals:
            gen = ind.time_of_birth
            ind_idx = gen_indices[gen][ind.id]

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

            record = {
                "individual_id": ind.id,
                "generation": gen,
                "fitness": ind.fitness if ind.fitness is not None else None,
                "locomotion_fitness": locomotion_fitness,
                "novelty_score": novelty_score,
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

        # Save as JSON
        json_path = save_dir / "database.json"
        with open(json_path, 'w') as f:
            json.dump(records, f, indent=2)

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
