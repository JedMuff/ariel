"""Evolutionary morphology optimization using mu-lambda strategy with locomotion fitness.

This script demonstrates:
1. Evolving robot morphologies using TreeGenotype and MuLambdaStrategy
2. Evaluating morphologies based on locomotion performance in MuJoCo simulation
3. Each morphology gets a neural network controller for actuation
4. Fitness = forward displacement during simulation
5. Optional novelty search for diversity maintenance

Key features:
- Tree genotype → Robot morphology → MuJoCo simulation → Fitness
- Mu-lambda evolutionary strategy (configurable mu+lambda or mu,lambda)
- All hyperparameters exposed in main script
- Support for parallel evaluation
- Visualization and progress tracking
"""

# Enable modern type annotations
from __future__ import annotations

# IMPORTANT: Set thread limits BEFORE importing numpy/mujoco to prevent nested parallelism
# When using multiprocessing, each worker should use only 1 thread
import os
os.environ["OMP_NUM_THREADS"] = "1"          # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"          # Intel MKL (NumPy)
os.environ["OPENBLAS_NUM_THREADS"] = "1"     # OpenBLAS (NumPy)
os.environ["NUMEXPR_NUM_THREADS"] = "1"      # NumExpr
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # macOS Accelerate framework

# Standard library
import random
import sys
import tempfile
import time
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

# Third-party libraries
import numpy as np
from networkx import DiGraph
from rich.console import Console
from rich.traceback import install

# ARIEL framework imports
from ariel.ec import Individual
from myevo.core import FitnessEvaluator, MuLambdaStrategy, TreeGenotype

# Local imports
from myevo.measures.novelty_fitness import KDTreeArchive, extract_morphological_vector
from myevo.measures.morphological_measures import Body, MorphologicalMeasures
from myevo.debugging.memory_profiler import MemoryProfiler, check_temp_cache_status, cleanup_temp_cache

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
DATA = None  # Will be set in main()
SEED = 42
PROFILER = None  # Global memory profiler
CACHE_DIR = None  # Experiment-specific cache directory (set in main())

# Global functions
install(show_locals=False)
console = Console()
RNG = np.random.default_rng(SEED)
random.seed(SEED)


# Helper class for novelty archive (must be at module level for pickling)
class TreeWrapper:
    """Wrapper for tree genotypes to work with novelty archive."""
    def __init__(self, tree):
        self.tree = tree


class TreeLocomotionEvolution:
    """Evolutionary system for optimizing robot morphologies for locomotion.

    This class combines tree genotype evolution with simulation-based fitness
    evaluation. Each morphology is evaluated by simulating it in MuJoCo with
    a neural network controller and measuring locomotion performance.
    """

    def __init__(
        self,
        # Evolution parameters
        mu: int = 10,
        lambda_: int = 50,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.2,
        mutate_after_crossover: bool = False,
        strategy_type: str = "comma",
        selection_method: str = "tournament",
        maximize: bool = True,
        # Genotype parameters
        max_part_limit: int = 25,
        max_actuators: int = 12,
        mutation_strength: int = 1,
        mutation_reps: int = 1,
        mutate_attributes_prob: float = 0.1,
        enable_self_adaptation: bool = False,
        # Simulation parameters
        simulation_duration: float = 10.0,
        controller_hidden_layers: list[int] | None = None,
        controller_activation: str = "tanh",
        sigma_init: float = 1.0,
        # CMA-ES inner loop parameters
        use_cmaes: bool = False,
        cmaes_budget: int = 50,
        cmaes_population_size: int = 10,
        # Novelty search parameters
        use_novelty: bool = False,
        novelty_k_neighbors: int = 5,
        # Lamarckian evolution parameters
        enable_lamarckian: bool = False,
        lamarckian_crossover_mode: str = "average",
        covariance_inheritance_mode: str = "adaptive",
        sigma_inheritance_mode: str = "blend",
        # System parameters
        seed: int = 42,
        num_workers: int = 1,
        verbose: bool = True,
    ):
        """Initialize the tree locomotion evolution system.

        Parameters
        ----------
        mu : int
            Number of parents (population size).
        lambda_ : int
            Number of offspring generated per generation.
        mutation_rate : float
            Proportion of offspring created via mutation (0-1).
        crossover_rate : float
            Proportion of offspring created via crossover (0-1).
        mutate_after_crossover : bool
            Whether to apply mutation after crossover.
        strategy_type : str
            Either 'plus' (μ+λ) or 'comma' (μ,λ).
        selection_method : str
            Parent selection method ('tournament', 'rank', 'proportional').
        maximize : bool
            Whether to maximize (True) or minimize (False) fitness.
        max_part_limit : int
            Maximum number of parts in robot body.
        max_actuators : int
            Maximum number of actuators.
        mutation_strength : int
            Initial mutation strength (tree depth for mutations).
        mutation_reps : int
            Initial number of mutation repetitions.
        mutate_attributes_prob : float
            Probability of mutating self-adaptive parameters.
        enable_self_adaptation : bool
            Whether to enable self-adaptive mutation parameters.
        simulation_duration : float
            Duration of simulation in seconds.
        controller_hidden_layers : list[int] | None
            Hidden layer sizes for neural network controller.
        controller_activation : str
            Activation function for controller ('tanh', 'relu', 'sigmoid').
        sigma_init : float
            Initial standard deviation for weight initialization (uniform in [-sigma_init, sigma_init]).
        use_cmaes : bool
            Whether to use CMA-ES to optimize controller weights (inner loop).
        cmaes_budget : int
            Number of CMA-ES evaluations per morphology.
        cmaes_population_size : int
            Population size for CMA-ES optimizer.
        use_novelty : bool
            Whether to use novelty-adjusted fitness.
        novelty_min_distance : float
            Minimum distance between archived individuals.
        novelty_k_neighbors : int
            Number of nearest neighbors for novelty calculation.
        novelty_adaptive : bool
            Whether to adaptively adjust archive threshold.
        enable_lamarckian : bool
            Whether to enable Lamarckian evolution (offspring inherit learned weights).
        lamarckian_crossover_mode : str
            How to combine parent weights for crossover offspring.
            Options:
            - "average": Average both parents' weights
            - "parent1": Use only first parent's weights
            - "random": Randomly choose one parent's weights
            - "closest_parent": Use weights from parent with smallest tree edit distance to offspring
        covariance_inheritance_mode : str
            How to adapt CMA-ES covariance matrices when morphology changes.
            Options:
            - "adaptive": Preserve learned correlations for existing weights (default)
            - "reset": Always use identity matrix (no inheritance)
            - "preserve": Scale existing matrix to new size (experimental)
        sigma_inheritance_mode : str
            How to adapt CMA-ES step size when morphology changes.
            Options:
            - "blend": Blend parent sigma with initial sigma based on proportion of new weights (default)
            - "reset": Always use sigma_init (no inheritance)
            - "keep": Always use parent sigma (full inheritance)
            - "adaptive": Custom blending logic
        seed : int
            Random seed for reproducibility.
        num_workers : int
            Number of parallel workers for evaluation (currently 1).
        verbose : bool
            Whether to print progress information.
        """
        # Access global state
        global CACHE_DIR, PROFILER

        self.max_part_limit = max_part_limit
        self.max_actuators = max_actuators
        self.simulation_duration = simulation_duration
        self.controller_hidden_layers = controller_hidden_layers or [32, 16, 32]
        self.controller_activation = controller_activation
        self.sigma_init = sigma_init
        self.num_workers = num_workers
        self.verbose = verbose
        self.maximize = maximize
        self.enable_self_adaptation = enable_self_adaptation

        # CMA-ES inner loop parameters
        self.use_cmaes = use_cmaes
        self.cmaes_budget = cmaes_budget
        self.cmaes_population_size = cmaes_population_size

        # Novelty search parameters
        self.use_novelty = use_novelty
        self.novelty_k_neighbors = novelty_k_neighbors

        # Lamarckian evolution parameters (will be passed to strategy)
        self.enable_lamarckian = enable_lamarckian
        self.lamarckian_crossover_mode = lamarckian_crossover_mode
        self.covariance_inheritance_mode = covariance_inheritance_mode
        self.sigma_inheritance_mode = sigma_inheritance_mode

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Initialize tree genotype system
        self.tree_genotype = TreeGenotype(
            max_part_limit=max_part_limit,
            max_actuators=max_actuators,
            mutation_strength=mutation_strength,
            mutation_reps=mutation_reps,
            mutate_attributes_prob=mutate_attributes_prob if enable_self_adaptation else 0.0,
        )

        # Calculate num_mutate and num_crossover from rates
        num_mutate = int(lambda_ * mutation_rate)
        num_crossover = lambda_ - num_mutate

        # Initialize evolution strategy with weight inheritance
        self.strategy = MuLambdaStrategy(
            genotype=self.tree_genotype,
            population_size=mu,
            num_offspring=lambda_,
            num_mutate=num_mutate,
            num_crossover=num_crossover,
            mutate_after_crossover=mutate_after_crossover,
            strategy_type=strategy_type,
            selection_method=selection_method,
            maximize=maximize,
            verbose=True,
            lamarckian_mode=enable_lamarckian,
            weight_crossover_mode=lamarckian_crossover_mode,
            weight_sigma=sigma_init,
            covariance_inheritance_mode=covariance_inheritance_mode,
            sigma_inheritance_mode=sigma_inheritance_mode,
            post_evaluation_callback=self.recalculate_novelty_for_generation if use_novelty else None,
            save_database_per_generation=True,  # Enable per-generation database saving
            cache_dir=CACHE_DIR,  # Experiment-specific cache directory
        )

        # Initialize novelty archive if enabled
        if self.use_novelty:
            self.novelty_archive = KDTreeArchive(
                min_distance=None,  # Store all individuals
                feature_extractor=self._extract_morphological_features,
            )
            console.print(
                f"[bold cyan]Novelty Search:[/bold cyan] Morphological (KDTree), "
                f"k={novelty_k_neighbors}"
            )
        else:
            self.novelty_archive = None

        # Initialize fitness evaluator
        # Note: Profiler can't be pickled for multiprocessing, so only use when num_workers == 1
        use_profiler = PROFILER if num_workers == 1 else None
        self.fitness_evaluator = FitnessEvaluator(
            simulation_duration=simulation_duration,
            controller_hidden_layers=self.controller_hidden_layers,
            controller_activation=controller_activation,
            sigma_init=sigma_init,
            use_cmaes=use_cmaes,
            cmaes_budget=cmaes_budget,
            cmaes_population_size=cmaes_population_size,
            seed=seed,
            cache_dir=CACHE_DIR,
            maximize=maximize,
            profiler=use_profiler,
            enable_lamarckian=enable_lamarckian,
            covariance_inheritance_mode=covariance_inheritance_mode,
            sigma_inheritance_mode=sigma_inheritance_mode,
        )

        # Print configuration
        if self.verbose:
            console.print(
                f"\n[bold cyan]Tree Locomotion Evolution Configuration[/bold cyan]\n"
                f"Strategy: ({strategy_type}) μ={mu}, λ={lambda_}\n"
                f"Morphology: max_parts={max_part_limit}, max_actuators={max_actuators}\n"
                f"Controller: hidden={self.controller_hidden_layers}, activation={controller_activation}\n"
                f"CMA-ES Inner Loop: {'Enabled' if use_cmaes else 'Disabled'}"
                f"{f' (budget={cmaes_budget}, pop={cmaes_population_size})' if use_cmaes else ''}\n"
                f"Simulation: duration={simulation_duration}s\n"
                f"Selection: {selection_method}\n"
                f"Novelty: {'Enabled' if use_novelty else 'Disabled'}\n"
                f"Lamarckian: {'Enabled (inherit optimized weights)' if enable_lamarckian else 'Disabled (inherit initial weights)'}"
                f"{f' (mode={lamarckian_crossover_mode})' if enable_lamarckian else ''}\n"
            )

    def _extract_morphological_features(self, ind):
        """Extract morphological features for novelty calculation.

        This method is used as a feature extractor for the KDTreeArchive.
        It must be a regular method (not a lambda) to support pickling for multiprocessing.

        Parameters
        ----------
        ind : object
            Individual with a .tree attribute.

        Returns
        -------
        np.ndarray
            Morphological feature vector.
        """
        return extract_morphological_vector(
            MorphologicalMeasures(Body(ind.tree, max_part_limit=self.max_part_limit))
        )

    def recalculate_novelty_for_generation(self, population: list[Individual]) -> None:
        """Recalculate novelty for all individuals in a generation.

        This method:
        1. Adds all individuals in the population to the novelty archive
        2. Calculates novelty scores for all individuals based on the updated archive
        3. Updates both the novelty_score tag and the combined fitness value used for selection

        Parameters
        ----------
        population : list[Individual]
            Population of individuals to recalculate novelty for.
        """
        if not self.use_novelty or self.novelty_archive is None:
            return

        from myevo.core import TreeGenotype

        # STEP 1: Add all UNIQUE individuals in the current population to the archive
        # Skip individuals that are already in the archive (for μ+λ where parents carry over)
        for ind in population:
            # Extract tree from genotype
            tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
            wrapper = TreeWrapper(tree)

            # Check if this individual is already in the archive by checking nearest neighbor distance
            if len(self.novelty_archive) > 0:
                feature_vec = self.novelty_archive.feature_extractor(wrapper)
                distances, _ = self.novelty_archive.kdtree.query(feature_vec, k=1)
                nearest_distance = distances if np.isscalar(distances) else distances[0]

                # If nearest neighbor has distance 0, this individual is already in archive
                if nearest_distance < 1e-10:
                    continue  # Skip adding duplicate

            # Add to archive (individual is unique)
            self.novelty_archive.add(wrapper)

        # STEP 2: Calculate novelty for all individuals based on the updated archive
        # Filter out self-matches (distance ~0) when calculating novelty
        for ind in population:
            # Extract tree from genotype
            tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
            wrapper = TreeWrapper(tree)

            # Calculate novelty, filtering out self-matches
            if len(self.novelty_archive) == 0:
                novelty_score = float('inf')
            else:
                # Get feature vector for this individual
                feature_vec = self.novelty_archive.feature_extractor(wrapper)

                # Query more neighbors just in case some are duplicates (self-matches)
                k_query = min(self.novelty_k_neighbors + 5, len(self.novelty_archive))
                distances, _ = self.novelty_archive.kdtree.query(feature_vec, k=k_query)

                # Ensure distances is array-like
                if np.isscalar(distances):
                    distances = np.array([distances])
                else:
                    distances = np.array(distances)

                # Filter out exact matches (distance ~0)
                eps = 1e-10
                mask = distances > eps
                filtered_distances = distances[mask]

                # Calculate novelty from filtered distances
                if len(filtered_distances) == 0:
                    # Only self-matches found - use novelty of 1.0
                    novelty_score = 1.0
                else:
                    # Take mean of k nearest non-duplicate neighbors
                    k_actual = min(self.novelty_k_neighbors, len(filtered_distances))
                    novelty_score = float(np.mean(filtered_distances[:k_actual]))

            # Get locomotion fitness from tags
            locomotion_fitness = ind.tags.get("locomotion_fitness", ind.fitness or 0.0)

            # Store novelty score in tags
            if novelty_score == float('inf'):
                # Archive is empty - use novelty score of 1.0 (neutral multiplier)
                ind.tags["novelty_score"] = 1.0
                ind.fitness = locomotion_fitness
            else:
                # Store novelty and combine with locomotion fitness
                ind.tags["novelty_score"] = float(novelty_score)
                ind.fitness = locomotion_fitness * novelty_score

            # Update fitness_components.json if individual has log_dir
            log_dir = ind.tags.get("log_dir")
            if log_dir is not None:
                import json
                fitness_components_path = Path(log_dir) / "fitness_components.json"
                if fitness_components_path.exists():
                    # Update existing file with novelty score
                    with open(fitness_components_path, 'r') as f:
                        data = json.load(f)
                    data["novelty_score"] = ind.tags.get("novelty_score")
                    with open(fitness_components_path, 'w') as f:
                        json.dump(data, f, indent=2)


    def fitness_function(
        self,
        individual: Individual,
        log_dir: str | None = None,
    ) -> float:
        """Evaluate a tree genotype via simulation.

        Delegates to FitnessEvaluator for all evaluation logic.

        Parameters
        ----------
        individual : Individual
            The Individual object being evaluated.
        log_dir : str | None, optional
            Logging directory for saving results.

        Returns
        -------
        float
            Fitness value (locomotion performance).
        """
        return self.fitness_evaluator.evaluate(individual, log_dir)

    def initialize_diverse_population(self) -> list[DiGraph]:
        """Initialize population with diverse tree depths.

        Creates 30 trees:
        - 10 small trees (depth=1)
        - 10 medium trees (depth=2)
        - 10 large trees (depth=3)

        Returns
        -------
        list[DiGraph]
            Initial population of tree genomes with diverse morphologies.
        """
        population = []

        # Create 10 small trees (depth=1)
        for _ in range(10):
            tree = self.tree_genotype.random_tree(depth=1)
            population.append(tree)

        # Create 10 medium trees (depth=2)
        for _ in range(10):
            tree = self.tree_genotype.random_tree(depth=2)
            population.append(tree)

        # Create 10 large trees (depth=3)
        for _ in range(10):
            tree = self.tree_genotype.random_tree(depth=3)
            population.append(tree)

        return population

    def run(self, num_generations: int) -> list[Individual]:
        """Run the evolutionary algorithm using MuLambdaStrategy.

        Parameters
        ----------
        num_generations : int
            Number of generations to run.

        Returns
        -------
        list[Individual]
            All individuals from all generations.
        """
        global PROFILER

        if self.verbose:
            console.print(
                f"\n[bold cyan]Starting Evolution[/bold cyan]\n"
                f"Generations: {num_generations}\n"
                f"Fitness: {'Locomotion' + (' * Novelty' if self.use_novelty else '')}\n"
            )

        # Profile initial state
        if PROFILER:
            PROFILER.log_memory("evolution_start")
            PROFILER.take_snapshot("evolution_start")

        # Initialize diverse population (returns list of genomes, not Individuals)
        initial_genomes = self.initialize_diverse_population()

        if PROFILER:
            PROFILER.log_memory("after_population_init")

        # Run evolution using MuLambdaStrategy.evolve()
        # This handles ID assignment, weight inheritance, and evaluation automatically
        all_individuals = self.strategy.evolve(
            fitness_function=self.fitness_function,
            num_generations=num_generations,
            engine=None,  # No database engine
            initial_population=initial_genomes,
            log_dir_base=str(DATA),
            num_workers=self.num_workers,
            reevaluate_parents=False,
        )

        # Profile final state
        if PROFILER:
            PROFILER.log_memory("evolution_end")
            PROFILER.take_snapshot("evolution_end")

        # Note: Database is already saved per-generation by the strategy
        # (see save_database_per_generation=True in __init__)

        return all_individuals


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evolutionary morphology optimization using mu-lambda strategy with locomotion fitness"
    )

    # Core experiment parameters
    parser.add_argument(
        "--enable-lamarckian",
        action="store_true",
        help="Enable Lamarckian evolution (offspring inherit optimized weights from parents)"
    )
    parser.add_argument(
        "--use-novelty",
        action="store_true",
        help="Enable novelty search (fitness = locomotion * novelty)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=30,
        help="Number of parallel workers for evaluation (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=50,
        help="Number of generations to evolve (default: 50)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name prefix for experiment directory (default: auto-generated from timestamp)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory for saving experiment data (default: __data__)"
    )
    parser.add_argument(
        "--cmaes-budget",
        type=int,
        default=1000,
        help="CMA-ES evaluations per morphology (default: 1000)"
    )
    parser.add_argument(
        "--cmaes-population-size",
        type=int,
        default=20,
        help="CMA-ES population size (default: 20)"
    )
    parser.add_argument(
        "--covariance-inheritance-mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "reset", "preserve"],
        help="CMA-ES covariance inheritance mode (default: adaptive)"
    )
    parser.add_argument(
        "--sigma-inheritance-mode",
        type=str,
        default="blend",
        choices=["blend", "reset", "keep", "adaptive"],
        help="CMA-ES sigma inheritance mode (default: blend)"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()

    # Create timestamped data directory (once in main process)
    global DATA
    global SEED
    global RNG
    global PROFILER
    SEED = args.seed

    # Re-seed random number generators with command-line seed
    random.seed(SEED)
    np.random.seed(SEED)
    RNG = np.random.default_rng(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        dir_name = f"{args.experiment_name}_{timestamp}"
    else:
        dir_name = f"{SCRIPT_NAME}_{timestamp}"

    # Use custom data directory if specified, otherwise default to __data__
    if args.data_dir:
        base_dir = Path(args.data_dir)
    else:
        base_dir = CWD / "__data__"

    DATA = base_dir / dir_name
    DATA.mkdir(exist_ok=True, parents=True)

    # Set experiment-specific cache directory to avoid conflicts between parallel runs
    global CACHE_DIR
    CACHE_DIR = Path(tempfile.gettempdir()) / f"ariel_cmaes_cache_{timestamp}_{os.getpid()}"
    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    # Initialize memory profiler
    PROFILER = MemoryProfiler(log_dir=DATA, enable_tracemalloc=True)
    PROFILER.log_memory("main_start")

    # Check and clean temp cache at start (ALWAYS clean to ensure fresh start)
    cache_status = check_temp_cache_status(CACHE_DIR)
    if cache_status['exists'] and cache_status['num_files'] > 0:
        console.print(f"[yellow]Found {cache_status['num_files']} files in temp cache ({cache_status['total_mb']:.2f} MB)[/yellow]")
        console.print("[yellow]Cleaning temp cache...[/yellow]")
        files_removed = cleanup_temp_cache(CACHE_DIR)
        console.print(f"[green]Removed {files_removed} cache files[/green]")
    else:
        console.print("[green]Temp cache is clean[/green]")

    PROFILER.log_memory("after_cache_cleanup")

    # Create evolution system with all hyperparameters exposed
    evolution = TreeLocomotionEvolution(
        # Evolution parameters
        mu=30,                           # Population size
        lambda_=30,                      # Offspring per generation
        mutation_rate=0.8,               # 80% mutation, 20% crossover
        crossover_rate=0.2,
        mutate_after_crossover=True,     # Mutate after crossover
        strategy_type="plus",            # (μ+λ) - parents + offspring
        selection_method="tournament",   # Tournament selection
        maximize=True,                   # Maximize fitness (displacement)

        # Genotype parameters
        max_part_limit=25,               # Max robot parts
        max_actuators=12,                # Max actuators
        mutation_strength=1,             # Initial mutation strength
        mutation_reps=1,                 # Initial mutation repetitions
        mutate_attributes_prob=0.1,      # Self-adaptation mutation prob
        enable_self_adaptation=False,    # Enable/disable self-adaptation

        # Simulation parameters
        simulation_duration=35.0,        # Simulation time (seconds)
        controller_hidden_layers=[32, 16, 32],  # NN architecture
        controller_activation="tanh",    # NN activation function
        sigma_init=1.0,                  # Weight initialization range [-sigma, sigma]

        # CMA-ES inner loop parameters
        use_cmaes=True,                  # Enable CMA-ES controller optimization
        cmaes_budget=args.cmaes_budget,  # CMA-ES evaluations per morphology (from command line)
        cmaes_population_size=args.cmaes_population_size,  # CMA-ES population size (from command line)

        # Novelty search parameters
        use_novelty=args.use_novelty,    # Enable novelty search (from command line)
        novelty_k_neighbors=1,           # K-nearest neighbors

        # Lamarckian evolution parameters
        enable_lamarckian=args.enable_lamarckian,  # Enable Lamarckian weight inheritance (from command line)
        lamarckian_crossover_mode="closest_parent",  # Weight combination mode: average, parent1, random, closest_parent
        covariance_inheritance_mode=args.covariance_inheritance_mode,  # CMA-ES covariance inheritance mode (from command line)
        sigma_inheritance_mode=args.sigma_inheritance_mode,  # CMA-ES sigma inheritance mode (from command line)

        # System parameters
        seed=args.seed,                  # Random seed (from command line)
        num_workers=args.num_workers,    # Parallel workers (from command line)
        verbose=True,
    )

    # Run evolution
    PROFILER.log_memory("before_evolution_run")
    all_individuals = evolution.run(num_generations=args.num_generations)
    PROFILER.log_memory("after_evolution_run")

    # Force garbage collection and check memory
    import gc
    gc_stats = PROFILER.force_gc()
    console.print(f"[cyan]Garbage collected {gc_stats['collected']} objects, freed {gc_stats['freed_mb']:.2f} MB[/cyan]")

    PROFILER.log_memory("main_end")

    # Print memory profile summary
    console.print("\n" + "=" * 80)
    PROFILER.print_summary()

    # Save snapshot report
    PROFILER.save_snapshot_report()

    # Check final temp cache status and clean up experiment-specific cache
    final_cache = check_temp_cache_status(CACHE_DIR)
    if final_cache['exists'] and final_cache['num_files'] > 0:
        console.print(f"[yellow]Cleaning up experiment cache: {final_cache['num_files']} files ({final_cache['total_mb']:.2f} MB)[/yellow]")
        cleanup_temp_cache(CACHE_DIR)

    console.print(
        f"\n[bold green]All results saved to:[/bold green] {DATA.absolute()}"
    )
    console.print(f"[bold green]Memory profile saved to:[/bold green] {DATA.absolute() / 'memory_profile.csv'}")


if __name__ == "__main__":
    main()
