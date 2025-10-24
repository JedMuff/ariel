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

# Standard library
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Apply custom config BEFORE importing ARIEL modules
CWD = Path.cwd()
sys.path.insert(0, str(CWD / "myevo"))
from custom_config import ALLOWED_FACES, ALLOWED_ROTATIONS
import ariel.body_phenotypes.robogen_lite.config as ariel_config
ariel_config.ALLOWED_FACES = ALLOWED_FACES
ariel_config.ALLOWED_ROTATIONS = ALLOWED_ROTATIONS

# Third-party libraries
import mujoco as mj
import numpy as np
from networkx import DiGraph
from rich.console import Console
from rich.progress import track
from rich.traceback import install

# ARIEL framework imports
from ariel.ec import Individual, MuLambdaStrategy, TreeGenotype

# Local imports
from neural_network_controller import FlexibleNeuralNetworkController
from novelty import KDTreeArchive, extract_morphological_vector
from morphological_measures import Body, MorphologicalMeasures
from controller_optimizer import optimize_controller_cmaes
from lamarckian_utils import ParentWeightManager
from simulation_utils import (
    create_robot_model,
    create_controller,
    setup_tracker,
    calculate_displacement_fitness,
    simulate_with_controller,
    simulate_with_settling_phase,
)
from visualization_utils import plot_fitness_history, visualize_best_morphology
from data_saving_utils import save_final_database
from record_robot_video import record_robot_video

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
DATA = None  # Will be set in main()
SEED = 42

# Global functions
install(show_locals=False)
console = Console()
RNG = np.random.default_rng(SEED)
random.seed(SEED)


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
        max_depth: int = 4,
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
        novelty_min_distance: float = 3.0,
        novelty_k_neighbors: int = 5,
        novelty_adaptive: bool = False,
        # Lamarckian evolution parameters
        enable_lamarckian: bool = False,
        lamarckian_crossover_mode: str = "average",
        # Video recording parameters
        enable_video_recording: bool = False,
        video_interval: int = 1,
        video_duration: float = 15.0,
        video_platform: str = "macos",
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
        max_depth : int
            Maximum depth for random tree generation.
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
        enable_video_recording : bool
            Whether to record videos of the best robot after each generation.
        video_interval : int
            Record video every N generations (e.g., 1 = every generation, 5 = every 5 generations).
        video_duration : float
            Duration of recorded videos in seconds.
        video_platform : str
            Video codec platform: "macos" (mp4v) or "windows" (avc1).
        seed : int
            Random seed for reproducibility.
        num_workers : int
            Number of parallel workers for evaluation (currently 1).
        verbose : bool
            Whether to print progress information.
        """
        self.max_part_limit = max_part_limit
        self.max_actuators = max_actuators
        self.max_depth = max_depth
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

        # Lamarckian evolution parameters
        self.enable_lamarckian = enable_lamarckian
        self.lamarckian_crossover_mode = lamarckian_crossover_mode
        if self.enable_lamarckian:
            self.weight_manager = ParentWeightManager(crossover_mode=lamarckian_crossover_mode)
        else:
            self.weight_manager = None

        # Non-Lamarckian initial weight inheritance manager
        # Stores initial (pre-optimization) weights for inheritance across generations
        self._initial_weight_manager = ParentWeightManager(crossover_mode=lamarckian_crossover_mode)

        self._current_generation_map: dict = {}  # Temporary genotype → Individual mapping

        # Video recording parameters
        self.enable_video_recording = enable_video_recording
        self.video_interval = video_interval
        self.video_duration = video_duration
        self.video_platform = video_platform

        # These are no longer needed since we save directly during evaluation
        self.learning_curves: dict[int, list[float]] = {}  # For backward compatibility
        self.individual_weights: dict[int, dict[str, np.ndarray]] = {}  # For backward compatibility

        # ID counter for assigning unique IDs to individuals (since not using database)
        self._next_id = 1

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Initialize tree genotype system
        self.tree_genotype = TreeGenotype(
            max_part_limit=max_part_limit,
            max_actuators=max_actuators,
            default_depth=max_depth,
            mutation_strength=mutation_strength,
            mutation_reps=mutation_reps,
            mutate_attributes_prob=mutate_attributes_prob if enable_self_adaptation else 0.0,
        )

        # Calculate num_mutate and num_crossover from rates
        num_mutate = int(lambda_ * mutation_rate)
        num_crossover = lambda_ - num_mutate

        # Initialize evolution strategy
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
            verbose=False,
        )

        # Initialize novelty archive if enabled
        if self.use_novelty:
            self.novelty_archive = KDTreeArchive(
                min_distance=None,  # Store all individuals
                feature_extractor=lambda ind: extract_morphological_vector(
                    MorphologicalMeasures(Body(ind.tree, max_part_limit=self.max_part_limit))
                ),
            )
            console.print(
                f"[bold cyan]Novelty Search:[/bold cyan] Morphological (KDTree), "
                f"k={novelty_k_neighbors}"
            )
        else:
            self.novelty_archive = None

        # Print configuration
        if self.verbose:
            console.print(
                f"\n[bold cyan]Tree Locomotion Evolution Configuration[/bold cyan]\n"
                f"Strategy: ({strategy_type}) μ={mu}, λ={lambda_}\n"
                f"Morphology: max_depth={max_depth}, max_parts={max_part_limit}, "
                f"max_actuators={max_actuators}\n"
                f"Controller: hidden={self.controller_hidden_layers}, activation={controller_activation}\n"
                f"CMA-ES Inner Loop: {'Enabled' if use_cmaes else 'Disabled'}"
                f"{f' (budget={cmaes_budget}, pop={cmaes_population_size})' if use_cmaes else ''}\n"
                f"Simulation: duration={simulation_duration}s\n"
                f"Selection: {selection_method}\n"
                f"Novelty: {'Enabled' if use_novelty else 'Disabled'}\n"
                f"Lamarckian: {'Enabled (inherit optimized weights)' if enable_lamarckian else 'Disabled (inherit initial weights)'}"
                f"{f' (mode={lamarckian_crossover_mode})' if enable_lamarckian else ''}\n"
                f"Video Recording: {'Enabled' if enable_video_recording else 'Disabled'}"
                f"{f' (every {video_interval} gen, {video_duration}s, {video_platform})' if enable_video_recording else ''}\n"
            )

    def optimize_controller_cmaes_wrapper(
        self,
        model: mj.MjModel,
        world_spec: Any,
        initial_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, list[float]]:
        """Wrapper for CMA-ES controller optimization.

        Parameters
        ----------
        model : mj.MjModel
            The MuJoCo model to optimize for.
        world_spec : Any
            The world specification for tracking.
        initial_weights : np.ndarray | None, optional
            Initial weights to start optimization from (for Lamarckian evolution).

        Returns
        -------
        tuple[np.ndarray, float, list[float]]
            Best weights, their fitness value, and learning curve.
        """
        return optimize_controller_cmaes(
            model=model,
            world_spec=world_spec,
            hidden_layers=self.controller_hidden_layers,
            activation=self.controller_activation,
            simulation_duration=self.simulation_duration,
            cmaes_budget=self.cmaes_budget,
            cmaes_population_size=self.cmaes_population_size,
            sigma_init=self.sigma_init,
            initial_weights=initial_weights,
            maximize=self.maximize,
            baseline_time=5.0,
            seed=SEED,
        )

    def fitness_function(
        self,
        genome: DiGraph,
        log_dir: str | None = None,
        individual: Individual | None = None,
    ) -> float:
        """Evaluate a tree genotype via simulation.

        This function:
        1. Builds a robot from the tree genotype
        2. Creates a neural network controller
        3. If use_cmaes is True, optimizes controller weights via CMA-ES inner loop
        4. Otherwise, uses random controller weights (or inherits from parent if Lamarckian)
        5. Simulates in MuJoCo
        6. Returns forward displacement as fitness
        7. Saves all data (body.json, brain weights, learning curves) to log_dir if provided

        Parameters
        ----------
        genome : DiGraph or TreeGenotype
            The tree genome to evaluate.
        log_dir : str | None, optional
            Logging directory provided by evaluate_population.
        individual : Individual | None, optional
            The Individual object being evaluated (for Lamarckian evolution).

        Returns
        -------
        float
            Fitness value (forward displacement in meters).
        """
        from ariel.ec import TreeGenotype

        # Extract tree from TreeGenotype if needed
        tree = genome.tree if isinstance(genome, TreeGenotype) else genome

        # Use log_dir for saving if provided
        save_dir = log_dir

        # Save body.json if we have a save directory
        if save_dir is not None:
            from ariel.body_phenotypes.robogen_lite.decoders import save_graph_as_json
            save_path = Path(save_dir)
            save_graph_as_json(tree, save_path / "body.json")

        # Build robot from tree
        model, data, world_spec = create_robot_model(tree)

        # Skip evaluation if fewer than 4 actuators (not viable for locomotion)
        if model.nu < 4:
            # Save dummy brain files for consistency (empty weights)
            if save_dir is not None:
                import csv
                save_path = Path(save_dir)
                # Save empty weight arrays
                dummy_weights = np.array([])
                np.save(save_path / "initial_brain.npy", dummy_weights)
                np.save(save_path / "optimized_brain.npy", dummy_weights)

                # Save empty learning curve
                with open(save_path / "learning_curve.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['iteration', 'fitness'])
                    # Empty - no controllable actuators
            return 0.0

        # Get controller and determine number of weights
        controller = create_controller(
            model=model,
            hidden_layers=self.controller_hidden_layers,
            activation=self.controller_activation,
            seed=SEED,
        )
        num_weights = controller.get_num_weights()

        # Determine initial weights based on evolution mode
        initial_weights = None

        if self.enable_lamarckian and individual is not None and self.weight_manager is not None:
            # Lamarckian: inherit optimized weights from parent
            parent_weights = self.weight_manager.get_parent_weights(individual, offspring_tree=tree)
            if parent_weights is not None:
                initial_weights = parent_weights
        elif individual is not None and self._initial_weight_manager is not None:
            # Non-Lamarckian: inherit initial weights from parent (if has parent)
            parent_initial_weights = self._initial_weight_manager.get_parent_weights(individual, offspring_tree=tree)
            if parent_initial_weights is not None:
                initial_weights = parent_initial_weights

        # If no inherited weights (generation 0 or missing parent), use random initialization
        if initial_weights is None:
            initial_weights = self.rng.uniform(-self.sigma_init, self.sigma_init, num_weights)

        # Store initial weights for non-Lamarckian inheritance by offspring
        if self._initial_weight_manager is not None:
            self._initial_weight_manager.store_weights(id(tree), initial_weights)

        # Get controller weights - either via CMA-ES or direct use
        if self.use_cmaes:

            # Use CMA-ES to optimize controller weights
            optimized_weights, cmaes_fitness, learning_curve = self.optimize_controller_cmaes_wrapper(
                model, world_spec, initial_weights=initial_weights
            )

            # Store learned weights for potential inheritance by offspring
            if self.weight_manager is not None:
                self.weight_manager.store_weights(id(tree), optimized_weights)

            # Save brain files directly if save_dir is available
            if save_dir is not None:
                import csv
                save_path = Path(save_dir)
                np.save(save_path / "initial_brain.npy", initial_weights)
                np.save(save_path / "optimized_brain.npy", optimized_weights)

                # Save learning curve
                with open(save_path / "learning_curve.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['iteration', 'fitness'])
                    for i, fitness in enumerate(learning_curve):
                        writer.writerow([i, fitness])

            # Return the fitness from CMA-ES optimization
            return cmaes_fitness
        else:
            # No CMA-ES optimization - use initial weights directly
            # (initial_weights already determined above based on inheritance mode)
            controller.set_weights(initial_weights)

            # Reset simulation
            mj.mj_resetData(model, data)

            # Setup tracker
            tracker = setup_tracker(world_spec, data)

            # Run two-phase simulation: 5 seconds settling, then controlled locomotion
            settling_duration = 5.0
            control_duration = self.simulation_duration - settling_duration
            simulate_with_settling_phase(
                model=model,
                data=data,
                controller=controller,
                tracker=tracker,
                settling_duration=settling_duration,
                control_duration=control_duration,
            )

            # Calculate fitness (baseline_time=0 since tracker was reset at start of control phase)
            x_displacement = calculate_displacement_fitness(tracker, baseline_time=0.0, model=model)

            # Store learned weights for Lamarckian inheritance (optimized weights)
            # In non-CMA-ES case, weights don't change, but we still store for Lamarckian mode
            if self.weight_manager is not None:
                self.weight_manager.store_weights(id(tree), initial_weights)

            # Save brain files directly if save_dir is available
            # No optimization in this case, so initial = optimized
            if save_dir is not None:
                import csv
                save_path = Path(save_dir)
                np.save(save_path / "initial_brain.npy", initial_weights)
                np.save(save_path / "optimized_brain.npy", initial_weights)

                # Save empty learning curve (no CMA-ES)
                with open(save_path / "learning_curve.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['iteration', 'fitness'])
                    # Empty - no iterations

            # Handle novelty if enabled
            if self.use_novelty and self.novelty_archive is not None:
                # Create wrapper for archive (needs .tree attribute)
                class TreeWrapper:
                    def __init__(self, tree):
                        self.tree = tree

                wrapper = TreeWrapper(tree)

                # Calculate novelty score
                novelty_score = self.novelty_archive.novelty(wrapper, k=self.novelty_k_neighbors)

                # Add to archive
                self.novelty_archive.add(wrapper)

                # Combine fitness: distance * novelty
                if novelty_score == float('inf'):
                    # First individual - just use distance
                    return float(x_displacement)
                else:
                    # Multiply distance by novelty
                    return float(x_displacement * novelty_score)

            # Return displacement (can be negative if robot moves backward)
            return float(x_displacement)

    def _fitness_function_wrapper(self, genome: DiGraph, log_dir: str | None = None) -> float:
        """Wrapper for fitness_function that looks up Individual object.

        This wrapper enables Lamarckian evolution by providing access to the
        Individual object, which contains parent information needed for weight inheritance.

        Parameters
        ----------
        genome : DiGraph
            The tree genome to evaluate.
        log_dir : str | None, optional
            Logging directory (unused, for compatibility).

        Returns
        -------
        float
            Fitness value.
        """
        # Look up Individual from current generation mapping (may be None in workers)
        individual = self._current_generation_map.get(id(genome))

        # Call actual fitness function with Individual object
        return self.fitness_function(genome, log_dir, individual)

    def _assign_ids(self, population: list[Individual]) -> None:
        """Assign unique IDs to individuals that don't have them.

        Parameters
        ----------
        population : list[Individual]
            Population to assign IDs to.
        """
        for ind in population:
            if ind.id is None:
                ind.id = self._next_id
                self._next_id += 1


    def initialize_diverse_population(self) -> list[Individual]:
        """Initialize population with diverse tree depths.

        Creates 30 individuals:
        - 10 small trees (depth=2)
        - 10 medium trees (depth=3)
        - 10 large trees (depth=4)

        Returns
        -------
        list[Individual]
            Initial population with diverse morphologies.
        """
        from ariel.ec import TreeGenotype

        population = []

        # Create 10 small trees (depth=1)
        for _ in range(10):
            tree = self.tree_genotype.random_tree(depth=1)
            ind = Individual(time_of_birth=0)
            ind.genotype = tree
            population.append(ind)

        # Create 10 medium trees (depth=2)
        for _ in range(10):
            tree = self.tree_genotype.random_tree(depth=2)
            ind = Individual(time_of_birth=0)
            ind.genotype = tree
            population.append(ind)

        # Create 10 large trees (depth=3)
        for _ in range(10):
            tree = self.tree_genotype.random_tree(depth=3)
            ind = Individual(time_of_birth=0)
            ind.genotype = tree
            population.append(ind)

        # Assign IDs to all individuals
        self._assign_ids(population)

        return population

    def _populate_genotype_mapping(
        self,
        population: list[Individual],
        generation: int,
    ) -> None:
        """Populate mapping from genotype ID to Individual object.

        This mapping is used by the fitness function wrapper to access Individual
        objects during evaluation, enabling Lamarckian weight inheritance.

        Parameters
        ----------
        population : list[Individual]
            The population to create mapping for.
        generation : int
            Current generation number.
        """
        from ariel.ec import TreeGenotype

        self._current_generation_map.clear()

        for ind in population:
            tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
            tree_id = id(tree)
            self._current_generation_map[tree_id] = ind


    def run(self, num_generations: int) -> list[Individual]:
        """Run the evolutionary algorithm.

        Parameters
        ----------
        num_generations : int
            Number of generations to run.

        Returns
        -------
        list[Individual]
            All individuals from all generations.
        """
        from ariel.ec.evaluation import evaluate_population

        if self.verbose:
            console.print(
                f"\n[bold cyan]Starting Evolution[/bold cyan]\n"
                f"Generations: {num_generations}\n"
                f"Fitness: {'Locomotion' + (' * Novelty' if self.use_novelty else '')}\n"
            )

        # Initialize population with diverse tree depths (IDs assigned in method)
        population = self.initialize_diverse_population()

        # Evaluate initial population
        console.print("[cyan]Evaluating initial population...[/cyan]")

        # Populate mappings (needed for Lamarckian evolution)
        self._populate_genotype_mapping(population, generation=0)
        fitness_func = self._fitness_function_wrapper

        population = evaluate_population(
            population,
            fitness_func,
            generation=0,
            log_dir_base=str(DATA),
            num_workers=self.num_workers,
        )

        # Track evaluated individuals for Lamarckian evolution
        if self.enable_lamarckian and self.weight_manager is not None:
            for ind in population:
                self.weight_manager.add_evaluated_individual(ind)

        # Track evaluated individuals for non-Lamarckian initial weight inheritance
        if self._initial_weight_manager is not None:
            for ind in population:
                self._initial_weight_manager.add_evaluated_individual(ind)

        # Track all individuals
        all_individuals = population.copy()

        # Print initial stats
        if self.verbose:

            fitnesses = [ind.fitness for ind in population]
            best_fitness = max(fitnesses) if self.maximize else min(fitnesses)
            avg_fitness = np.mean(fitnesses)

            console.print(
                f"G:0 | BestF={best_fitness:.4f}, AvgF={avg_fitness:.4f}\n"
            )

        # Save database after initial generation
        save_final_database(
            all_individuals=all_individuals,
            save_dir=DATA,
            learning_curves=None,
        )

        # Evolution loop
        start_time = time.time()

        for generation in track(
            range(1, num_generations + 1),
            description="Evolving morphologies",
            disable=not self.verbose,
        ):
            gen_start = time.time()

            # Populate mappings before evaluation
            self._populate_genotype_mapping(population, generation=generation)

            # Perform one generation
            population = self.strategy.step(
                population,
                fitness_func,
                generation,
                log_dir_base=str(DATA),
                num_workers=self.num_workers,
            )

            # Track evaluated individuals for Lamarckian evolution
            if self.enable_lamarckian and self.weight_manager is not None:
                # Add new offspring to evaluated individuals list
                for ind in population:
                    self.weight_manager.add_evaluated_individual(ind)

            # Track evaluated individuals for non-Lamarckian initial weight inheritance
            if self._initial_weight_manager is not None:
                for ind in population:
                    self._initial_weight_manager.add_evaluated_individual(ind)

            # Track all individuals
            all_individuals.extend(population)

            # Print progress
            if self.verbose and (generation % 1 == 0 or generation == num_generations):
                fitnesses = [ind.fitness
                            for ind in population]
                best_fitness = max(fitnesses) if self.maximize else min(fitnesses)
                avg_fitness = np.mean(fitnesses)
                gen_time = time.time() - gen_start

                console.print(
                    f"G:{generation} | Time={gen_time:.2f}s, "
                    f"BestF={best_fitness:.4f}, AvgF={avg_fitness:.4f}"
                )

            # Record video of best individual if enabled
            if self.enable_video_recording and generation % self.video_interval == 0:
                if self.verbose:
                    console.print(f"[cyan]Recording video for generation {generation}...[/cyan]")

                from ariel.ec import TreeGenotype

                # Find best individual
                if self.maximize:
                    best_ind = max(population, key=lambda ind: ind.fitness or float('-inf'))
                else:
                    best_ind = min(population, key=lambda ind: ind.fitness or float('inf'))

                if self.verbose:
                    console.print(f"[cyan]Best fitness: {best_ind.fitness:.4f}[/cyan]")

                # Extract tree
                best_tree = best_ind.genotype.tree if isinstance(best_ind.genotype, TreeGenotype) else best_ind.genotype

                # Get brain weights
                weights = None

                # Debug: check weight_manager status
                if self.verbose:
                    if self.weight_manager is not None:
                        console.print(f"[cyan]weight_manager exists, has_weights={self.weight_manager.has_weights(id(best_tree))}[/cyan]")
                    else:
                        console.print(f"[cyan]weight_manager is None[/cyan]")

                # Try 1: Get from weight_manager
                if self.weight_manager is not None and self.weight_manager.has_weights(id(best_tree)):
                    weights = self.weight_manager.get_weights(id(best_tree))
                    if self.verbose:
                        console.print(f"[cyan]Got weights from weight_manager ({len(weights)} params)[/cyan]")

                # Try 2: Load from saved file (most reliable)
                if weights is None and best_ind.id is not None:
                    # Use new directory structure: generation_XX/individual_Y/optimized_brain.npy
                    weights_path = DATA / f"generation_{generation:02d}" / f"individual_{best_ind.id}" / "optimized_brain.npy"
                    if self.verbose:
                        console.print(f"[cyan]Looking for weights at: {weights_path}[/cyan]")
                        console.print(f"[cyan]File exists: {weights_path.exists()}[/cyan]")
                    if weights_path.exists():
                        weights = np.load(weights_path)
                        if self.verbose:
                            console.print(f"[cyan]Loaded weights from file ({len(weights)} params)[/cyan]")
                    else:
                        if self.verbose:
                            console.print(f"[yellow]Weights file not found: {weights_path}[/yellow]")
                            # List what files DO exist in that directory
                            individual_dir = weights_path.parent
                            if individual_dir.exists():
                                files = list(individual_dir.glob("*"))
                                console.print(f"[cyan]Files in directory: {[f.name for f in files]}[/cyan]")

                # Record video if we have valid weights
                if weights is not None and len(weights) > 0:
                    video_dir = DATA / "generation_videos" / f"generation_{generation:04d}"
                    try:
                        if self.verbose:
                            console.print(f"[cyan]Starting video recording...[/cyan]")
                        record_robot_video(
                            body_graph=best_tree,
                            brain_weights=weights,
                            output_path=video_dir,
                            duration=self.video_duration,
                            settling_duration=5.0,  # Same as fitness evaluation
                            controller_hidden_layers=self.controller_hidden_layers,
                            controller_activation=self.controller_activation,
                            platform=self.video_platform,
                            verbose=self.verbose,  # Show video recording progress
                        )
                        if self.verbose:
                            console.print(f"[bold green]✓ Video saved:[/bold green] {video_dir}")
                    except Exception as e:
                        console.print(f"[bold red]✗ Failed to record video:[/bold red] {e}")
                        import traceback
                        console.print(traceback.format_exc())
                else:
                    if self.verbose:
                        console.print(f"[yellow]Skipping video: No valid weights found[/yellow]")

            # Save database after each generation
            save_final_database(
                all_individuals=all_individuals,
                save_dir=DATA,
                learning_curves=None,
            )

        total_time = time.time() - start_time

        if self.verbose:
            console.print(
                f"\n[bold green]Evolution Complete![/bold green]\n"
                f"Total time: {total_time:.2f}s\n"
                f"Avg time/generation: {total_time/num_generations:.2f}s"
            )

        # Final database save (already saved after each generation)
        console.print("\n[cyan]Saving final database...[/cyan]")
        save_final_database(
            all_individuals=all_individuals,
            save_dir=DATA,
            learning_curves=None,
        )

        return all_individuals

    def plot_fitness_history_wrapper(self, population: list[Individual]) -> None:
        """Plot fitness progression over generations."""
        plot_fitness_history(
            population=population,
            maximize=self.maximize,
            save_path=DATA / "fitness_history.png",
            title="Tree Locomotion Evolution: Fitness over Generations",
            ylabel="Fitness (Forward Displacement, m)",
        )

    def visualize_best_wrapper(
        self,
        population: list[Individual],
        mode: str = "viewer",
        duration: float | None = None,
    ) -> None:
        """Visualize the best evolved morphology.

        Parameters
        ----------
        population : list[Individual]
            All individuals from evolution.
        mode : str
            Visualization mode: "viewer", "video", or "frame".
        duration : float | None
            Simulation duration (defaults to self.simulation_duration).
        """
        if duration is None:
            duration = self.simulation_duration

        controller_params = {
            "hidden_layers": self.controller_hidden_layers,
            "activation": self.controller_activation,
            "sigma_init": self.sigma_init,
            "seed": SEED,
            "use_cmaes": self.use_cmaes,
            "cmaes_budget": self.cmaes_budget,
            "cmaes_population_size": self.cmaes_population_size,
        }

        visualize_best_morphology(
            population=population,
            maximize=self.maximize,
            mode=mode,
            controller_params=controller_params,
            save_dir=DATA,
            simulation_duration=duration,
            use_stored_weights=self.enable_lamarckian,
            weight_manager=self.weight_manager,
        )


def main() -> None:
    """Main entry point."""
    # Create timestamped data directory (once in main process)
    global DATA
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA = CWD / "__data__" / f"{SCRIPT_NAME}_{timestamp}"
    DATA.mkdir(exist_ok=True, parents=True)

    # Create evolution system with all hyperparameters exposed
    evolution = TreeLocomotionEvolution(
        # Evolution parameters
        mu=30,                           # Population size
        lambda_=30,                      # Offspring per generation
        mutation_rate=0.8,               # 80% mutation, 20% crossover
        crossover_rate=0.2,
        mutate_after_crossover=True,     # Mutate after crossover
        strategy_type="comma",           # (μ,λ) - offspring only
        selection_method="tournament",   # Tournament selection
        maximize=True,                   # Maximize fitness (displacement)

        # Genotype parameters
        max_depth=3,                     # Max tree depth
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
        cmaes_budget=1000,                 # CMA-ES evaluations per morphology
        cmaes_population_size=20,        # CMA-ES population size

        # Novelty search parameters
        use_novelty=False,               # Enable novelty search
        novelty_min_distance=3.0,        # Archive min distance
        novelty_k_neighbors=1,           # K-nearest neighbors
        novelty_adaptive=False,          # Adaptive archive

        # Lamarckian evolution parameters
        enable_lamarckian=False,          # Enable Lamarckian weight inheritance
        lamarckian_crossover_mode="closest_parent",  # Weight combination mode: average, parent1, random, closest_parent

        # Video recording parameters
        enable_video_recording=True,     # Enable video recording of best robot
        video_interval=1,                # Record video every N generations
        video_duration=35.0,             # Video duration in seconds
        video_platform="macos",          # Video codec: "macos" or "windows"

        # System parameters
        seed=SEED,
        num_workers=4,                   # Parallel workers (currently 1)
        verbose=True,
    )

    # Run evolution
    all_individuals = evolution.run(num_generations=2)

    # Plot fitness history
    evolution.plot_fitness_history_wrapper(all_individuals)

    # Visualize best morphology
    evolution.visualize_best_wrapper(all_individuals, mode="viewer", duration=15.0)

    console.print(
        f"\n[bold green]All results saved to:[/bold green] {DATA.absolute()}"
    )


if __name__ == "__main__":
    main()
