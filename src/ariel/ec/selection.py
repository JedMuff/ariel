"""Selection module for ARIEL evolutionary algorithms.

This module provides various parent selection methods for evolutionary
algorithms, including tournament selection, rank-based selection, and
proportional selection.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from ariel.ec.a001 import Individual


def tournament_selection(
    population: list[Individual],
    k: int,
    tournament_size: int = 3,
    maximize: bool = True,
) -> list[Individual]:
    """Select k individuals using tournament selection.

    Parameters
    ----------
    population : list[Individual]
        The population to select from.
    k : int
        Number of individuals to select.
    tournament_size : int, optional
        Number of individuals in each tournament, by default 3.
    maximize : bool, optional
        Whether to maximize fitness (True) or minimize (False), by default True.

    Returns
    -------
    list[Individual]
        List of k selected individuals.

    Examples
    --------
    >>> population = [Individual(genotype=[i], fitness_=float(i)) for i in range(10)]
    >>> selected = tournament_selection(population, k=5, tournament_size=3)
    >>> len(selected)
    5
    """
    selected = []

    for _ in range(k):
        # Randomly select tournament_size individuals
        tournament = random.sample(population, tournament_size)

        # Select the best from the tournament
        if maximize:
            winner = max(tournament, key=lambda ind: ind.fitness or float('-inf'))
        else:
            winner = min(tournament, key=lambda ind: ind.fitness or float('inf'))

        selected.append(winner)

    return selected


def rank_selection(
    population: list[Individual],
    k: int,
    maximize: bool = True,
) -> list[Individual]:
    """Select k individuals using rank-based selection.

    Individuals are ranked by fitness and selection probability is
    proportional to rank rather than raw fitness values.

    Parameters
    ----------
    population : list[Individual]
        The population to select from.
    k : int
        Number of individuals to select.
    maximize : bool, optional
        Whether to maximize fitness (True) or minimize (False), by default True.

    Returns
    -------
    list[Individual]
        List of k selected individuals.

    Examples
    --------
    >>> population = [Individual(genotype=[i], fitness_=float(i)) for i in range(10)]
    >>> selected = rank_selection(population, k=5)
    >>> len(selected)
    5
    """
    # Sort population by fitness
    sorted_pop = sorted(
        population,
        key=lambda ind: ind.fitness or (float('-inf') if maximize else float('inf')),
        reverse=maximize,
    )

    # Assign ranks (best = n, worst = 1)
    n = len(sorted_pop)
    ranks = np.arange(n, 0, -1)

    # Calculate selection probabilities from ranks
    probabilities = ranks / ranks.sum()

    # Select k individuals based on rank probabilities
    selected_indices = np.random.choice(n, size=k, replace=True, p=probabilities)
    selected = [sorted_pop[i] for i in selected_indices]

    return selected


def proportional_selection(
    population: list[Individual],
    k: int,
    maximize: bool = True,
) -> list[Individual]:
    """Select k individuals using fitness-proportional selection (roulette wheel).

    Selection probability is proportional to fitness value. For minimization
    problems, fitness values are inverted.

    Parameters
    ----------
    population : list[Individual]
        The population to select from.
    k : int
        Number of individuals to select.
    maximize : bool, optional
        Whether to maximize fitness (True) or minimize (False), by default True.

    Returns
    -------
    list[Individual]
        List of k selected individuals.

    Examples
    --------
    >>> population = [Individual(genotype=[i], fitness_=float(i+1)) for i in range(10)]
    >>> selected = proportional_selection(population, k=5)
    >>> len(selected)
    5
    """
    # Get fitness values
    fitnesses = np.array([ind.fitness or 0.0 for ind in population])

    # Handle minimization by inverting fitness
    if not maximize:
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        max_fitness = np.max(fitnesses)
        fitnesses = (max_fitness + epsilon) - fitnesses + epsilon

    # Ensure all fitnesses are positive
    min_fitness = np.min(fitnesses)
    if min_fitness < 0:
        fitnesses = fitnesses - min_fitness + 1e-10

    # Calculate selection probabilities
    total_fitness = fitnesses.sum()
    if total_fitness == 0:
        # If all fitnesses are zero, use uniform selection
        probabilities = np.ones(len(population)) / len(population)
    else:
        probabilities = fitnesses / total_fitness

    # Select k individuals based on fitness probabilities
    selected_indices = np.random.choice(
        len(population),
        size=k,
        replace=True,
        p=probabilities,
    )
    selected = [population[i] for i in selected_indices]

    return selected


def truncation_selection(
    population: list[Individual],
    k: int,
    maximize: bool = True,
) -> list[Individual]:
    """Select the k best individuals (truncation selection).

    Parameters
    ----------
    population : list[Individual]
        The population to select from.
    k : int
        Number of individuals to select.
    maximize : bool, optional
        Whether to maximize fitness (True) or minimize (False), by default True.

    Returns
    -------
    list[Individual]
        List of k best individuals.

    Examples
    --------
    >>> population = [Individual(genotype=[i], fitness_=float(i)) for i in range(10)]
    >>> selected = truncation_selection(population, k=5, maximize=True)
    >>> len(selected)
    5
    """
    # Sort population by fitness
    sorted_pop = sorted(
        population,
        key=lambda ind: ind.fitness or (float('-inf') if maximize else float('inf')),
        reverse=maximize,
    )

    # Return top k individuals
    return sorted_pop[:k]


def random_selection(
    population: list[Individual],
    k: int,
) -> list[Individual]:
    """Select k individuals randomly (uniform selection).

    Parameters
    ----------
    population : list[Individual]
        The population to select from.
    k : int
        Number of individuals to select.

    Returns
    -------
    list[Individual]
        List of k randomly selected individuals.

    Examples
    --------
    >>> population = [Individual(genotype=[i]) for i in range(10)]
    >>> selected = random_selection(population, k=5)
    >>> len(selected)
    5
    """
    return random.choices(population, k=k)


def select_parents(
    population: list[Individual],
    k: int,
    method: str = "tournament",
    maximize: bool = True,
    **kwargs,
) -> list[Individual]:
    """Select k parent individuals using specified selection method.

    This is a convenience function that dispatches to specific selection
    methods based on the 'method' parameter.

    Parameters
    ----------
    population : list[Individual]
        The population to select from.
    k : int
        Number of individuals to select.
    method : str, optional
        Selection method: 'tournament', 'rank', 'proportional', 'truncation',
        or 'random', by default 'tournament'.
    maximize : bool, optional
        Whether to maximize fitness (True) or minimize (False), by default True.
    **kwargs
        Additional arguments for specific selection methods:
        - tournament_size (int): For tournament selection, default 3.

    Returns
    -------
    list[Individual]
        List of k selected individuals.

    Raises
    ------
    ValueError
        If an unknown selection method is specified.

    Examples
    --------
    >>> population = [Individual(genotype=[i], fitness_=float(i)) for i in range(10)]
    >>> selected = select_parents(population, k=5, method='tournament')
    >>> len(selected)
    5
    """
    if method == "tournament":
        tournament_size = kwargs.get("tournament_size", 3)
        return tournament_selection(population, k, tournament_size, maximize)
    elif method == "rank":
        return rank_selection(population, k, maximize)
    elif method == "proportional":
        return proportional_selection(population, k, maximize)
    elif method == "truncation":
        return truncation_selection(population, k, maximize)
    elif method == "random":
        return random_selection(population, k)
    else:
        msg = f"Unknown selection method: {method}"
        raise ValueError(msg)


# Type alias for selection functions
SelectionFunction = Callable[[list[Individual], int], list[Individual]]
