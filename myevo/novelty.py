import heapq
import numpy as np
from scipy.spatial import KDTree
from typing import Callable, Optional, Union

class PoissonArchive:
    """
    Maintains an archive of individuals that are at least a minimum distance apart
    according to a provided distance function (e.g., tree edit distance).
    Also provides a novelty score function relative to the archive.
    """

    def __init__(self, min_distance, distance_fn, adaptive=False, use_cache=True):
        """
        Args:
            min_distance (float): Minimum allowed distance between any two archived individuals.
            distance_fn (callable): Function that takes two individuals and returns their distance.
            adaptive (bool): Whether to adaptively adjust the min_distance based on archive growth.
            use_cache (bool): Whether to cache distance computations for performance.
        """
        self.archive = []
        self.min_distance = min_distance
        self.distance_fn = distance_fn
        self.adaptive = adaptive
        self.use_cache = use_cache
        self._distance_cache = {} if use_cache else None
        self._insert_attempts = 0
        self._insert_successes = 0

    def _get_distance(self, ind1, ind2):
        """Get distance between two individuals, using cache if available."""
        if not self.use_cache:
            return self.distance_fn(ind1, ind2)

        # Create cache key (use id() for object identity)
        key = (id(ind1), id(ind2))
        reverse_key = (id(ind2), id(ind1))

        # Check cache
        if key in self._distance_cache:
            return self._distance_cache[key]
        if reverse_key in self._distance_cache:
            return self._distance_cache[reverse_key]

        # Compute and cache
        dist = self.distance_fn(ind1, ind2)
        self._distance_cache[key] = dist
        return dist

    def add(self, individual):
        """
        Attempts to add an individual to the archive.
        Returns True if added, False otherwise.
        """
        if not self.archive:
            self.archive.append(individual)
            self._insert_successes += 1
            return True

        for other in self.archive:
            dist = self._get_distance(individual, other)
            if dist < self.min_distance:
                # Too close to an existing member — reject
                self._insert_attempts += 1
                return False

        # Far enough from all others — accept
        self.archive.append(individual)
        self._insert_successes += 1
        self._insert_attempts += 1

        if self.adaptive:
            self._adapt_threshold()

        return True

    def novelty(self, individual, k=5):
        """
        Computes the novelty of an individual with respect to the archive.

        Novelty is defined as the average distance to the k nearest neighbors in the archive.

        Args:
            individual: The individual to evaluate.
            k (int): Number of nearest neighbors to consider.

        Returns:
            float: Novelty score (average of k nearest distances).
        """
        if not self.archive:
            # If archive is empty, return a large novelty (encourage adding)
            return float("inf")

        distances = [self._get_distance(individual, other) for other in self.archive]

        # Get the k smallest distances efficiently
        # k = min(k, len(distances))
        nearest = heapq.nsmallest(1, distances)
        # Return the average distance to the k nearest
        return nearest[0]#sum(nearest) / k

    def _adapt_threshold(self):
        """Simple adaptive strategy: adjust threshold based on insertion ratio."""
        if self._insert_attempts < 10:
            return  # avoid early noise

        ratio = self._insert_successes / self._insert_attempts
        if ratio > 0.5:
            self.min_distance *= 1.05
        elif ratio < 0.1:
            self.min_distance *= 0.95

        self._insert_attempts = 0
        self._insert_successes = 0

    def get_archive(self):
        """Returns the current list of archived individuals."""
        return self.archive

    def __len__(self):
        return len(self.archive)


def extract_morphological_vector(measures):
    """
    Extract a feature vector from morphological measures.

    Args:
        measures: MorphologicalMeasures object from myevo/morphological_measures.py

    Returns:
        numpy array: [branching, limbs, length_of_limbs, coverage, joints, proportion, symmetry, size]
    """
    # Choose proportion based on whether the robot is 2D or 3D
    proportion = measures.proportion_2d if measures.is_2d else measures.proportion_3d

    return np.array([
        measures.branching,
        measures.limbs,
        measures.length_of_limbs,
        measures.coverage,
        measures.joints,
        proportion,
        measures.symmetry,
        measures.size
    ], dtype=np.float64)


def euclidean_distance(vec1, vec2):
    """
    Compute Euclidean distance between two feature vectors.

    Args:
        vec1: numpy array or morphological measures object
        vec2: numpy array or morphological measures object

    Returns:
        float: Euclidean distance
    """
    # Convert to vectors if they are MorphologicalMeasures objects
    if not isinstance(vec1, np.ndarray):
        vec1 = extract_morphological_vector(vec1)
    if not isinstance(vec2, np.ndarray):
        vec2 = extract_morphological_vector(vec2)

    return np.linalg.norm(vec1 - vec2)


class CompleteArchive:
    """
    Archive that stores ALL individuals without any filtering.
    Useful for exhaustive novelty search without Poisson disk sampling.
    """

    def __init__(self, distance_fn, use_cache=True):
        """
        Args:
            distance_fn (callable): Function that takes two individuals and returns their distance.
            use_cache (bool): Whether to cache distance computations for performance.
        """
        self.archive = []
        self.distance_fn = distance_fn
        self.use_cache = use_cache
        self._distance_cache = {} if use_cache else None

    def _get_distance(self, ind1, ind2):
        """Get distance between two individuals, using cache if available."""
        if not self.use_cache:
            return self.distance_fn(ind1, ind2)

        # Create cache key (use id() for object identity)
        key = (id(ind1), id(ind2))
        reverse_key = (id(ind2), id(ind1))

        # Check cache
        if key in self._distance_cache:
            return self._distance_cache[key]
        if reverse_key in self._distance_cache:
            return self._distance_cache[reverse_key]

        # Compute and cache
        dist = self.distance_fn(ind1, ind2)
        self._distance_cache[key] = dist
        return dist

    def add(self, individual):
        """
        Add an individual to the archive (always succeeds).

        Returns True to indicate successful addition.
        """
        self.archive.append(individual)
        return True

    def novelty(self, individual, k=5):
        """
        Computes the novelty of an individual with respect to the archive.

        Novelty is defined as the average distance to the k nearest neighbors in the archive.

        Args:
            individual: The individual to evaluate.
            k (int): Number of nearest neighbors to consider.

        Returns:
            float: Novelty score (average of k nearest distances).
        """
        if not self.archive:
            return float("inf")

        distances = [self._get_distance(individual, other) for other in self.archive]

        # Get the k smallest distances efficiently
        nearest = heapq.nsmallest(min(k, len(distances)), distances)

        # Return the average distance to the k nearest
        return sum(nearest) / len(nearest)

    def get_archive(self):
        """Returns the current list of archived individuals."""
        return self.archive

    def __len__(self):
        return len(self.archive)


class KDTreeArchive:
    """
    Archive optimized for Euclidean distance using KD-tree for fast nearest neighbor search.
    Works with morphological feature vectors.

    Combines Poisson disk sampling (optional min_distance) with KD-tree spatial indexing
    for efficient nearest neighbor queries. This is much faster than linear search for
    large archives when using Euclidean distance.
    """

    def __init__(self, min_distance=None, feature_extractor=None, adaptive=False):
        """
        Args:
            min_distance (float, optional): Minimum allowed distance between archived individuals.
                                           If None, all individuals are added (like CompleteArchive).
            feature_extractor (callable, optional): Function to extract feature vector from individual.
                                                   Defaults to extract_morphological_vector.
            adaptive (bool): Whether to adaptively adjust the min_distance based on archive growth.
        """
        self.archive = []
        self.feature_vectors = []
        self.min_distance = min_distance
        self.feature_extractor = feature_extractor or extract_morphological_vector
        self.adaptive = adaptive
        self.kdtree = None
        self._insert_attempts = 0
        self._insert_successes = 0

    def _rebuild_kdtree(self):
        """Rebuild the KD-tree with current feature vectors."""
        if self.feature_vectors:
            self.kdtree = KDTree(np.array(self.feature_vectors))
        else:
            self.kdtree = None

    def add(self, individual):
        """
        Attempts to add an individual to the archive.

        If min_distance is None, always adds the individual.
        Otherwise, only adds if it's far enough from all existing members.

        Returns True if added, False otherwise.
        """
        # Extract feature vector
        feature_vec = self.feature_extractor(individual)

        if not self.archive:
            self.archive.append(individual)
            self.feature_vectors.append(feature_vec)
            self._rebuild_kdtree()
            self._insert_successes += 1
            return True

        # If no minimum distance constraint, always add
        if self.min_distance is None:
            self.archive.append(individual)
            self.feature_vectors.append(feature_vec)
            self._rebuild_kdtree()
            self._insert_successes += 1
            return True

        # Check if too close to any existing member using KD-tree
        distances, _ = self.kdtree.query(feature_vec, k=1)
        min_dist = distances if np.isscalar(distances) else distances[0]

        if min_dist < self.min_distance:
            # Too close to an existing member — reject
            self._insert_attempts += 1
            return False

        # Far enough from all others — accept
        self.archive.append(individual)
        self.feature_vectors.append(feature_vec)
        self._rebuild_kdtree()
        self._insert_successes += 1
        self._insert_attempts += 1

        if self.adaptive:
            self._adapt_threshold()

        return True

    def novelty(self, individual, k=5):
        """
        Computes the novelty of an individual with respect to the archive using KD-tree.

        Novelty is defined as the average distance to the k nearest neighbors in the archive.

        Args:
            individual: The individual to evaluate.
            k (int): Number of nearest neighbors to consider.

        Returns:
            float: Novelty score (average of k nearest distances).
        """
        if not self.archive:
            return float("inf")

        # Extract feature vector
        feature_vec = self.feature_extractor(individual)

        # Query KD-tree for k nearest neighbors
        k_actual = min(k, len(self.archive))
        distances, _ = self.kdtree.query(feature_vec, k=k_actual)

        # Handle both single and multiple neighbors
        if np.isscalar(distances):
            return float(distances)

        return np.mean(distances)

    def _adapt_threshold(self):
        """Simple adaptive strategy: adjust threshold based on insertion ratio."""
        if self._insert_attempts < 10:
            return  # avoid early noise

        ratio = self._insert_successes / self._insert_attempts
        if ratio > 0.5:
            self.min_distance *= 1.05
        elif ratio < 0.1:
            self.min_distance *= 0.95

        self._insert_attempts = 0
        self._insert_successes = 0

    def get_archive(self):
        """Returns the current list of archived individuals."""
        return self.archive

    def __len__(self):
        return len(self.archive)


# Example usage: Morphological Novelty Search with mu+lambda Evolution
if __name__ == "__main__":
    """
    Simple novelty search example using morphological measures and mu+lambda evolution.

    This demonstrates:
    1. Using morphological feature vectors (branching, limbs, etc.) for novelty
    2. KDTreeArchive for fast nearest neighbor search
    3. Mu+lambda evolution with novelty-based selection
    4. Plotting fitness and novelty over generations
    5. Displaying final morphologies in a grid
    """
    import sys
    import random
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Setup paths
    CWD = Path.cwd()
    sys.path.insert(0, str(CWD / "myevo"))

    # Monkey-patch config for no rotations
    from config_no_rotation import ALLOWED_ROTATIONS
    import ariel.body_phenotypes.robogen_lite.config as ariel_config
    ariel_config.ALLOWED_ROTATIONS = ALLOWED_ROTATIONS

    # ARIEL imports
    from ariel.ec import Individual, MuLambdaStrategy, TreeGenotype
    from ariel.ec.evaluation import evaluate_population
    from ariel.body_phenotypes.robogen_lite.decoders import draw_graph
    from morphological_measures import Body, MorphologicalMeasures

    # Configuration
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # Evolution parameters
    MU = 100              # Population size
    LAMBDA = 100          # Offspring per generation
    NUM_GENERATIONS = 20
    MAX_DEPTH = 3
    MAX_PARTS = 25

    # Novelty parameters
    USE_KDTREE = True           # Use KDTree (fast) or CompleteArchive (stores all)
    MIN_DISTANCE = None          # Minimum distance for Poisson sampling (None = store all)
    K_NEIGHBORS = 5             # Number of neighbors for novelty calculation

    # Create output directory
    DATA_DIR = CWD / "__data__" / "novelty_example"
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Novelty Search Example")
    print(f"=" * 60)
    print(f"Population: μ={MU}, λ={LAMBDA}")
    print(f"Generations: {NUM_GENERATIONS}")
    print(f"Archive: {'KDTree' if USE_KDTREE else 'Complete'} (min_dist={MIN_DISTANCE}, k={K_NEIGHBORS})")
    print(f"Output: {DATA_DIR}")
    print()

    # Initialize genotype
    tree_genotype = TreeGenotype(
        max_part_limit=MAX_PARTS,
        max_actuators=12,
        default_depth=MAX_DEPTH,
        mutation_strength=1,
        mutation_reps=1,
    )

    # Create novelty archive
    if USE_KDTREE:
        # KDTreeArchive with morphological feature extractor
        novelty_archive = KDTreeArchive(
            min_distance=MIN_DISTANCE,
            feature_extractor=lambda ind: extract_morphological_vector(
                MorphologicalMeasures(Body(ind.genotype.tree))
            ),
            adaptive=False,
        )
    else:
        # CompleteArchive stores everything
        novelty_archive = CompleteArchive(
            distance_fn=lambda ind1, ind2: euclidean_distance(
                MorphologicalMeasures(Body(ind1.genotype.tree)),
                MorphologicalMeasures(Body(ind2.genotype.tree))
            ),
            use_cache=True,
        )

    # Dictionary to store measures for later visualization (keyed by tree id)
    measures_dict = {}

    # Track current generation
    current_generation = {'gen': 0}

    # Fitness function: novelty score
    def novelty_fitness(genome, log_dir=None):
        """Fitness = novelty score based on morphological features.

        Generation 0 gets fitness=0 and seeds the archive.
        Subsequent generations get novelty-based fitness.
        """
        try:
            # genome is a DiGraph (tree structure)
            from ariel.ec import TreeGenotype

            # Extract tree from TreeGenotype if needed
            tree = genome.tree if isinstance(genome, TreeGenotype) else genome

            # Calculate morphological measures
            body = Body(tree)
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
            novelty_score = novelty_archive.novelty(wrapper, k=K_NEIGHBORS)
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
        population_size=MU,
        num_offspring=LAMBDA,
        num_mutate=int(LAMBDA * 0.8),
        num_crossover=LAMBDA - int(LAMBDA * 0.8),
        mutate_after_crossover=True,
        strategy_type="comma",  # μ+λ
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
    for gen in range(1, NUM_GENERATIONS + 1):
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

    # === Plot fitness (novelty) over generations ===
    max_gen = max(ind.time_of_birth for ind in all_individuals)
    best_novelty = []
    avg_novelty = []

    for gen in range(max_gen + 1):
        gen_inds = [ind for ind in all_individuals if ind.time_of_birth == gen]
        if gen_inds:
            best_novelty.append(max(ind.fitness or 0.0 for ind in gen_inds))
            avg_novelty.append(np.mean([ind.fitness or 0.0 for ind in gen_inds]))

    plt.figure(figsize=(10, 6))
    plt.plot(best_novelty, linewidth=2, color="#2E86AB", label="Best Novelty")
    plt.plot(avg_novelty, linewidth=2, color="#F6C85F", label="Avg Novelty")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Novelty Score", fontsize=12)
    plt.title("Morphological Novelty Search: Novelty over Generations", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = DATA_DIR / "novelty_history.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()  # Close the figure
    print(f"\nNovelty plot saved to: {plot_path}")

    # === Display final morphologies in a grid ===
    # Get the final generation individuals
    final_gen = max(ind.time_of_birth for ind in all_individuals)
    final_individuals = [ind for ind in all_individuals if ind.time_of_birth == final_gen]

    # Sort by novelty (descending)
    final_individuals.sort(key=lambda ind: ind.fitness or 0.0, reverse=True)

    # Create grid of morphologies
    n_display = min(12, len(final_individuals))  # Display up to 12
    rows = 3
    cols = 4

    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    fig.suptitle("Final Generation Morphologies (Sorted by Novelty)", fontsize=16)

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        if idx < n_display:
            ind = final_individuals[idx]

            # Draw the graph morphology
            from ariel.ec import TreeGenotype
            tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
            save_path = DATA_DIR / f"final_morph_{idx}.png"

            # Draw and save the graph
            graph_drawn = False
            try:
                draw_graph(tree, title=f"Novelty: {ind.fitness:.2f}",
                          save_file=save_path)
                plt.close('all')  # Close all figures to free memory
                graph_drawn = True
            except ValueError as e:
                # NetworkX has a bug with edge label positioning on simple graphs
                # Create a simple text representation instead
                num_nodes = len(tree.nodes())
                num_edges = len(tree.edges())
                ax.text(0.5, 0.5,
                       f"Robot {idx}\n{num_nodes} nodes\n{num_edges} edges\nNovelty: {ind.fitness:.2f}",
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)

                # Add morphological features below
                tree_id = id(tree)
                if tree_id in measures_dict:
                    m = measures_dict[tree_id]
                    text = (f"B:{m.branching:.2f} L:{m.limbs:.2f}\n"
                           f"C:{m.coverage:.2f} S:{m.symmetry:.2f}")
                    ax.text(0.5, 0.3, text, transform=ax.transAxes,
                           ha='center', fontsize=8, family='monospace')
                ax.axis('off')
                continue  # Skip to next individual
            except Exception as e:
                print(f"Error drawing graph for individual {idx}: {e}")
                ax.text(0.5, 0.5, f"Error\n{str(e)[:30]}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            # Load and display the image
            if graph_drawn and save_path.exists():
                from PIL import Image
                img = Image.open(save_path)
                ax.imshow(img)
                ax.axis('off')

                # Add text with morphological features
                tree_id = id(tree)
                if tree_id in measures_dict:
                    m = measures_dict[tree_id]
                    text = (f"B:{m.branching:.2f} L:{m.limbs:.2f}\n"
                           f"C:{m.coverage:.2f} S:{m.symmetry:.2f}")
                    ax.text(0.5, -0.05, text, transform=ax.transAxes,
                           ha='center', fontsize=8, family='monospace')
        else:
            ax.axis('off')

    plt.tight_layout()
    grid_path = DATA_DIR / "final_morphologies_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')  # Save the grid figure explicitly
    print(f"Morphology grid saved to: {grid_path}")
    plt.close(fig)  # Close the grid figure
    # plt.show()  # Comment out to avoid blocking

    print(f"\n{'='*60}")
    print(f"Results saved to: {DATA_DIR.absolute()}")
