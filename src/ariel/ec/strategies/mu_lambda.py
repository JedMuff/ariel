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

from ariel.ec.a001 import Individual
from ariel.ec.evaluation import evaluate_population
from ariel.ec.genotypes.base import Genotype
from ariel.ec.selection import select_parents


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
        fitness_function: Callable[[Any, str | None], float],
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
        fitness_function : Callable[[Any, str | None], float]
            Fitness evaluation function.
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

        # Evaluate offspring
        offspring = evaluate_population(
            offspring,
            fitness_function,
            generation,
            log_dir_base,
            num_workers,
        )

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

    def evolve(
        self,
        fitness_function: Callable[[Any, str | None], float],
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
        fitness_function : Callable[[Any, str | None], float]
            Fitness evaluation function taking (genome, log_dir) -> fitness.
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
        >>> def fitness_fn(genome, log_dir):
        ...     return len(genome.nodes())
        >>> all_inds = strategy.evolve(fitness_fn, num_generations=50)
        """
        evo_start = time.time()

        # Initialize population
        population = self.initialize_population(engine, initial_population)

        # Evaluate initial population
        population = evaluate_population(
            population,
            fitness_function,
            generation=0,
            log_dir_base=log_dir_base,
            num_workers=num_workers,
        )

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

        # Evolution loop
        for generation in range(1, num_generations + 1):
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

            # Print progress
            if self.verbose:
                fitnesses = [ind.fitness or 0.0 for ind in population]
                best_fitness = max(fitnesses) if self.maximize else min(fitnesses)
                avg_fitness = np.mean(fitnesses)
                print(
                    f"G:{generation} Time:{np.round(time.time() - gen_start, 2)}, "
                    f"BestF={best_fitness:.4f}, AvgF={avg_fitness:.4f}",
                    flush=True,
                )

        if self.verbose:
            total_time = time.time() - evo_start
            print(f"Time taken to evolve: {total_time:.2f} seconds")

        return self.all_individuals

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
