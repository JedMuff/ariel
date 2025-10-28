"""Evaluation module for ARIEL evolutionary algorithms.

This module provides fitness evaluation functions that work with the
Individual model and database persistence. It supports both sequential
and parallel evaluation of populations.
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Any, Callable

from sqlalchemy import Engine
from sqlmodel import Session

from ariel.ec.a001 import Individual


def evaluate_individual(
    individual: Individual,
    fitness_function: Callable[[Individual, str | None], float],
    generation: int,
    log_dir_base: str | None = None,
) -> Individual:
    """Evaluate fitness for a single individual.

    Parameters
    ----------
    individual : Individual
        The individual to evaluate.
    fitness_function : Callable[[Individual, str | None], float]
        Function that takes (individual, log_dir) and returns fitness value.
        The Individual parameter provides access to genotype and tags (e.g., for weight inheritance).
    generation : int
        Current generation number.
    log_dir_base : str | None, optional
        Base directory for logging individual data, by default None.
        If None, no logging directory is created.

    Returns
    -------
    Individual
        The individual with updated fitness and tags.

    Examples
    --------
    >>> def my_fitness(individual, log_dir):
    ...     return sum(individual.genotype)
    >>> ind = Individual(genotype=[1, 2, 3])
    >>> ind = evaluate_individual(ind, my_fitness, generation=0)
    >>> print(ind.fitness)
    """
    # Create logging directory if requested
    indiv_log_dir = None
    if log_dir_base is not None:
        gen_dir = os.path.join(log_dir_base, f"generation_{generation:02d}")
        indiv_log_dir = os.path.join(gen_dir, f"individual_{individual.id}")
        os.makedirs(indiv_log_dir, exist_ok=True)

    # Evaluate fitness - pass Individual object for access to genotype and tags
    fitness = fitness_function(individual, indiv_log_dir)
    individual.fitness = float(fitness)

    # Update metadata in tags (preserve existing tags like parent IDs)
    if individual.tags is None:
        individual.tags = {}
    individual.tags.update({
        "generation": generation,
        "log_dir": indiv_log_dir,
    })

    return individual


def evaluate_population(
    population: list[Individual],
    fitness_function: Callable[[Individual, str | None], float],
    generation: int,
    log_dir_base: str | None = None,
    num_workers: int = 1,
) -> list[Individual]:
    """Evaluate fitness for an entire population.

    Parameters
    ----------
    population : list[Individual]
        List of individuals to evaluate.
    fitness_function : Callable[[Individual, str | None], float]
        Function that takes (individual, log_dir) and returns fitness value.
        The Individual parameter provides access to genotype and tags.
    generation : int
        Current generation number.
    log_dir_base : str | None, optional
        Base directory for logging individual data, by default None.
    num_workers : int, optional
        Number of parallel workers for evaluation, by default 1.
        Set to 1 for sequential evaluation.

    Returns
    -------
    list[Individual]
        List of individuals with updated fitness values.

    Examples
    --------
    >>> def my_fitness(individual, log_dir):
    ...     return sum(individual.genotype)
    >>> population = [Individual(genotype=[i, i+1]) for i in range(10)]
    >>> population = evaluate_population(population, my_fitness, generation=0)
    """
    # Filter individuals that need evaluation
    to_evaluate = [ind for ind in population if ind.requires_eval]

    if not to_evaluate:
        return population  # Nothing to evaluate

    if num_workers == 1:
        # Sequential evaluation
        for individual in to_evaluate:
            evaluate_individual(individual, fitness_function, generation, log_dir_base)
    else:
        # Parallel evaluation
        # Prepare arguments for worker function
        args_list = [
            (ind, generation, log_dir_base, fitness_function)
            for ind in to_evaluate
        ]

        # Evaluate in parallel using multiprocessing pool
        with Pool(processes=num_workers) as pool:
            results = pool.map(_evaluate_worker, args_list)

        # Assign results back to individuals
        for individual, (fitness, log_dir, tags) in zip(to_evaluate, results):
            individual.fitness = fitness
            # Apply tags returned from worker (contains weights for inheritance)
            # This replaces all tags to ensure worker-populated data is preserved
            individual.tags = tags
            # Update with generation and log_dir
            individual.tags.update({
                "generation": generation,
                "log_dir": log_dir,
            })

    return population


def _evaluate_worker(
    args: tuple[Individual, int, str | None, Callable[[Individual, str | None], float]]
) -> tuple[float, str | None, dict]:
    """Worker function for parallel fitness evaluation.

    This is a module-level function required for multiprocessing.Pool.

    Parameters
    ----------
    args : tuple
        Tuple containing (individual, generation, log_dir_base, fitness_function).

    Returns
    -------
    tuple[float, str | None, dict]
        Tuple containing (fitness_value, log_directory, tags).
    """
    individual, generation, log_dir_base, fitness_function = args

    # Create logging directory if requested
    indiv_log_dir = None
    if log_dir_base is not None:
        gen_dir = os.path.join(log_dir_base, f"generation_{generation:02d}")
        indiv_log_dir = os.path.join(gen_dir, f"individual_{individual.id}")
        os.makedirs(indiv_log_dir, exist_ok=True)

    # Evaluate fitness - pass Individual object
    fitness = fitness_function(individual, indiv_log_dir)

    # Return tags populated during evaluation (contains weights for inheritance)
    return float(fitness), indiv_log_dir, individual.tags


def evaluate_and_commit(
    engine: Engine,
    population: list[Individual],
    fitness_function: Callable[[Individual, str | None], float],
    generation: int,
    log_dir_base: str | None = None,
    num_workers: int = 1,
) -> list[Individual]:
    """Evaluate population and commit results to database.

    Convenience function that combines evaluation and database persistence.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy database engine.
    population : list[Individual]
        List of individuals to evaluate.
    fitness_function : Callable[[Individual, str | None], float]
        Function that takes (individual, log_dir) and returns fitness value.
        The Individual parameter provides access to genotype and tags.
    generation : int
        Current generation number.
    log_dir_base : str | None, optional
        Base directory for logging individual data, by default None.
    num_workers : int, optional
        Number of parallel workers for evaluation, by default 1.

    Returns
    -------
    list[Individual]
        List of evaluated individuals.

    Examples
    --------
    >>> from ariel.ec.a001 import init_database
    >>> engine = init_database()
    >>> def my_fitness(individual, log_dir):
    ...     return sum(individual.genotype)
    >>> population = [Individual(genotype=[i]) for i in range(10)]
    >>> population = evaluate_and_commit(
    ...     engine, population, my_fitness, generation=0
    ... )
    """
    # Evaluate population
    population = evaluate_population(
        population,
        fitness_function,
        generation,
        log_dir_base,
        num_workers,
    )

    # Commit to database
    with Session(engine) as session:
        session.add_all(population)
        session.commit()
        # Refresh to get updated state
        for ind in population:
            session.refresh(ind)

    return population
