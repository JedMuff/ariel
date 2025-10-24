"""Base genotype interface for ARIEL evolutionary algorithms.

This module provides an abstract base class that defines the interface
for all genotype representations in the ARIEL framework. Concrete genotype
implementations (e.g., TreeGenotype, NeuralDevelopmentalEncoding) should
inherit from this base class and implement the required methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

# Type variable for genotype representations
# This allows concrete classes to specify their own genotype type
GenotypeType = TypeVar("GenotypeType")


class Genotype(ABC):
    """Abstract base class for genotype representations.

    This class defines the interface that all genotype classes must implement
    to be compatible with ARIEL's evolutionary strategies. Concrete genotype
    classes should inherit from this and implement all abstract methods.

    The genotype can be any data structure (NetworkX graph, numpy array,
    nested list, etc.) as long as the required operations are implemented.

    Type Parameters
    ---------------
    GenotypeType
        The underlying data structure used to represent the genotype
        (e.g., nx.DiGraph for trees, np.ndarray for arrays)
    """

    @abstractmethod
    def random_genome(self) -> Any:
        """Generate a single random genome.

        Returns
        -------
        Any
            A randomly generated genome in the appropriate representation.

        Examples
        --------
        >>> genotype = TreeGenotype(max_part_limit=10)
        >>> genome = genotype.random_genome()
        """
        pass

    @abstractmethod
    def random_population(self, pop_size: int, **kwargs: Any) -> list[Any]:
        """Generate a population of random genomes.

        Parameters
        ----------
        pop_size : int
            Number of genomes to generate.
        **kwargs : Any
            Additional parameters specific to the genotype implementation.

        Returns
        -------
        list[Any]
            List of randomly generated genomes.

        Examples
        --------
        >>> genotype = TreeGenotype(max_part_limit=10)
        >>> population = genotype.random_population(pop_size=100, depth=4)
        """
        pass

    @abstractmethod
    def mutate(self, genome: Any) -> Any:
        """Mutate a single genome.

        Parameters
        ----------
        genome : Any
            The genome to mutate.

        Returns
        -------
        Any
            The mutated genome. May be a modified copy or a new genome.

        Examples
        --------
        >>> genotype = TreeGenotype(max_part_limit=10)
        >>> mutated = genotype.mutate(genome)
        """
        pass

    @abstractmethod
    def crossover(self, parent1: Any, parent2: Any) -> tuple[Any, Any]:
        """Perform crossover between two parent genomes.

        Parameters
        ----------
        parent1 : Any
            First parent genome.
        parent2 : Any
            Second parent genome.

        Returns
        -------
        tuple[Any, Any]
            Two offspring genomes resulting from crossover.

        Examples
        --------
        >>> genotype = TreeGenotype(max_part_limit=10)
        >>> child1, child2 = genotype.crossover(parent1, parent2)
        """
        pass

    def mutate_population(self, population: list[Any]) -> list[Any]:
        """Mutate an entire population.

        Default implementation applies mutation to each genome individually.
        Subclasses can override this for vectorized or batch mutations.

        Parameters
        ----------
        population : list[Any]
            List of genomes to mutate.

        Returns
        -------
        list[Any]
            List of mutated genomes.

        Examples
        --------
        >>> genotype = TreeGenotype(max_part_limit=10)
        >>> mutated_pop = genotype.mutate_population(population)
        """
        return [self.mutate(genome) for genome in population]

    def crossover_population(
        self,
        parents1: list[Any],
        parents2: list[Any],
    ) -> list[Any]:
        """Perform crossover on two lists of parent genomes.

        Default implementation pairs parents and produces one offspring per pair.
        Subclasses can override this for vectorized or batch crossover.

        Parameters
        ----------
        parents1 : list[Any]
            First list of parent genomes.
        parents2 : list[Any]
            Second list of parent genomes.

        Returns
        -------
        list[Any]
            List of offspring genomes (one per parent pair).

        Raises
        ------
        ValueError
            If parent lists have different lengths.

        Examples
        --------
        >>> genotype = TreeGenotype(max_part_limit=10)
        >>> offspring = genotype.crossover_population(parents1, parents2)
        """
        if len(parents1) != len(parents2):
            msg = f"Parent lists must have same length: {len(parents1)} != {len(parents2)}"
            raise ValueError(msg)

        offspring = []
        for p1, p2 in zip(parents1, parents2):
            child1, _child2 = self.crossover(p1, p2)
            # Only keep first child by default
            offspring.append(child1)

        return offspring
