"""Run morphological novelty search evolution.

This script runs mu+lambda evolution with novelty-based selection using
morphological feature vectors (branching, limbs, etc.) and visualizes results.
"""

import sys
import random
from pathlib import Path

# Setup paths
CWD = Path.cwd()
sys.path.insert(0, str(CWD / "myevo"))

# Monkey-patch config for no rotations
from config_no_rotation import ALLOWED_ROTATIONS
import ariel.body_phenotypes.robogen_lite.config as ariel_config
ariel_config.ALLOWED_ROTATIONS = ALLOWED_ROTATIONS

import numpy as np
from ariel.ec import MuLambdaStrategy, TreeGenotype
from ariel.ec.evaluation import evaluate_population
from morphological_measures import Body, MorphologicalMeasures
from novelty import KDTreeArchive, CompleteArchive, extract_morphological_vector


def run_novelty_evolution(
    # Evolution parameters
    mu=100,
    lambda_=100,
    num_generations=20,
    max_depth=3,
    max_parts=25,

    # Novelty parameters
    use_kdtree=True,
    min_distance=None,
    k_neighbors=5,

    # System parameters
    seed=42,
    output_dir=None,
):
    """Run novelty-based evolution.

    Parameters
    ----------
    mu : int
        Population size
    lambda_ : int
        Offspring per generation
    num_generations : int
        Number of generations to evolve
    max_depth : int
        Maximum tree depth
    max_parts : int
        Maximum robot parts
    use_kdtree : bool
        Use KDTree (fast) or CompleteArchive (stores all)
    min_distance : float or None
        Minimum distance for Poisson sampling (None = store all)
    k_neighbors : int
        Number of neighbors for novelty calculation
    seed : int
        Random seed
    output_dir : Path or None
        Output directory for results

    Returns
    -------
    dict
        Results dictionary containing:
        - all_individuals: list of all individuals from all generations
        - novelty_archive: the final novelty archive
        - measures_dict: dict mapping tree id to morphological measures
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Create output directory
    if output_dir is None:
        output_dir = CWD / "__data__" / "novelty_evolution"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Morphological Novelty Search")
    print(f"=" * 60)
    print(f"Population: μ={mu}, λ={lambda_}")
    print(f"Generations: {num_generations}")
    print(f"Archive: {'KDTree' if use_kdtree else 'Complete'} (min_dist={min_distance}, k={k_neighbors})")
    print(f"Output: {output_dir}")
    print()

    # Initialize genotype with collision checking enabled
    tree_genotype = TreeGenotype(
        max_part_limit=max_parts,
        max_actuators=12,
        default_depth=max_depth,
        mutation_strength=1,
        mutation_reps=1,
        enable_collision_repair=True,  # Enable genotype-level collision checking
        max_repair_iterations=100,      # Max iterations for collision repair
    )

    # Create novelty archive
    if use_kdtree:
        novelty_archive = KDTreeArchive(
            min_distance=min_distance,
            feature_extractor=lambda ind: extract_morphological_vector(
                MorphologicalMeasures(Body(ind.genotype.tree, max_part_limit=max_parts))
            ),
            adaptive=False,
        )
    else:
        novelty_archive = CompleteArchive(
            distance_fn=lambda ind1, ind2: euclidean_distance(
                MorphologicalMeasures(Body(ind1.genotype.tree, max_part_limit=max_parts)),
                MorphologicalMeasures(Body(ind2.genotype.tree, max_part_limit=max_parts))
            ),
            use_cache=True,
        )

    # Dictionary to store measures for later visualization (keyed by tree id)
    measures_dict = {}

    # Track current generation
    current_generation = {'gen': 0}

    # Fitness function: novelty score
    def novelty_fitness(genome, log_dir=None):
        """Fitness = novelty score based on morphological features."""
        try:
            from ariel.ec import TreeGenotype
            tree = genome.tree if isinstance(genome, TreeGenotype) else genome

            # Calculate morphological measures (with max_part_limit for normalized size)
            body = Body(tree, max_part_limit=max_parts)
            measures = MorphologicalMeasures(body)

            # Store measures for later visualization
            measures_dict[id(tree)] = measures

            # Create wrapper for archive
            class MorphWrapper:
                def __init__(self, tree, measures):
                    self.genotype = type('obj', (object,), {'tree': tree})()
                    self._measures = measures

            wrapper = MorphWrapper(tree, measures)

            # Generation 0: seed the archive with initial population, return 0 fitness
            if current_generation['gen'] == 0:
                novelty_archive.add(wrapper)
                return 0.0

            # Subsequent generations: calculate novelty and add to archive
            novelty_score = novelty_archive.novelty(wrapper, k=k_neighbors)
            novelty_archive.add(wrapper)

            return novelty_score if novelty_score != float('inf') else 100.0

        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    # Initialize evolution strategy
    strategy = MuLambdaStrategy(
        genotype=tree_genotype,
        population_size=mu,
        num_offspring=lambda_,
        num_mutate=int(lambda_ * 0.8),
        num_crossover=lambda_ - int(lambda_ * 0.8),
        mutate_after_crossover=True,
        strategy_type="comma",
        selection_method="tournament",
        maximize=True,
        verbose=False,
    )

    # Initialize population
    print("Initializing population...")
    population = strategy.initialize_population()
    population = evaluate_population(population, novelty_fitness, generation=0, num_workers=1)

    all_individuals = population.copy()

    # Print generation 0 stats
    print(f"Gen   0 | Archive: {len(novelty_archive):4d} | "
          f"BestNovelty:  0.000 | AvgNovelty:  0.000 (initial population)")

    # Evolution loop
    print("\nRunning evolution...")
    for gen in range(1, num_generations + 1):
        # Update generation counter for fitness function
        current_generation['gen'] = gen

        population = strategy.step(population, novelty_fitness, gen, num_workers=1)
        all_individuals.extend(population)

        # Print progress
        fitnesses = [ind.fitness or 0.0 for ind in population]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        print(f"Gen {gen:3d} | Archive: {len(novelty_archive):4d} | "
              f"BestNovelty: {best_fitness:6.3f} | AvgNovelty: {avg_fitness:6.3f}")

    print(f"\nEvolution complete! Final archive size: {len(novelty_archive)}")

    return {
        'all_individuals': all_individuals,
        'novelty_archive': novelty_archive,
        'measures_dict': measures_dict,
        'output_dir': output_dir,
    }


if __name__ == "__main__":
    # Run evolution
    results = run_novelty_evolution(
        mu=100,
        lambda_=100,
        num_generations=20,
        max_depth=3,
        max_parts=25,
        use_kdtree=True,
        min_distance=None,
        k_neighbors=5,
        seed=42,
    )

    print(f"\n{'='*60}")
    print(f"Results saved to: {results['output_dir'].absolute()}")
