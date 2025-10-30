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
from simulation_utils import (
    create_robot_model,
    create_controller,
    setup_tracker,
    calculate_displacement_fitness,
    simulate_with_controller,
    simulate_with_settling_phase,
)
from visualization_utils import plot_fitness_history, visualize_best_morphology
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
        novelty_k_neighbors: int = 5,
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

        # Lamarckian evolution parameters (will be passed to strategy)
        self.enable_lamarckian = enable_lamarckian
        self.lamarckian_crossover_mode = lamarckian_crossover_mode

        # Video recording parameters
        self.enable_video_recording = enable_video_recording
        self.video_interval = video_interval
        self.video_duration = video_duration
        self.video_platform = video_platform

        # These are no longer needed since we save directly during evaluation
        self.learning_curves: dict[int, list[float]] = {}  # For backward compatibility
        self.individual_weights: dict[int, dict[str, np.ndarray]] = {}  # For backward compatibility

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
            post_evaluation_callback=self.recalculate_novelty_for_generation if use_novelty else None,
            save_database_per_generation=True,  # Enable per-generation database saving
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

        from ariel.ec import TreeGenotype

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
        individual: Individual,
        log_dir: str | None = None,
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
        individual : Individual
            The Individual object being evaluated (provides access to genotype and tags).
        log_dir : str | None, optional
            Logging directory provided by evaluate_population.
            Format: "...base_dir/generation_XX/individual_YY"

        Returns
        -------
        float
            Fitness value (forward displacement in meters).
        """
        from ariel.ec import TreeGenotype

        # Extract tree from genotype
        genome = individual.genotype
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
            # Still create controller and generate random weights for inheritance
            # This ensures offspring can inherit something even if parent wasn't evaluated
            controller = create_controller(
                model=model,
                hidden_layers=self.controller_hidden_layers,
                activation=self.controller_activation,
                seed=SEED,
            )
            num_weights = controller.get_num_weights()
            random_weights = self.rng.uniform(-self.sigma_init, self.sigma_init, num_weights)

            # Store weights in tags for inheritance (even though we didn't evaluate)
            if individual is not None:
                individual.tags["initial_weights"] = random_weights
                individual.tags["optimized_weights"] = random_weights  # Same as initial (no optimization)
                individual.tags["layer_sizes"] = controller.layer_sizes
                individual.tags["locomotion_fitness"] = 0.0  # Store fitness of 0.0

            # Save brain files for consistency
            if save_dir is not None:
                import csv
                import json
                save_path = Path(save_dir)
                np.save(save_path / "initial_brain.npy", random_weights)
                np.save(save_path / "optimized_brain.npy", random_weights)

                # Save metadata
                metadata = {
                    "controller_hidden_layers": self.controller_hidden_layers,
                    "controller_activation": self.controller_activation,
                    "layer_sizes": controller.layer_sizes,
                    "num_actuators": model.nu,
                    "num_weights": len(random_weights),
                    "input_size": model.nq + model.nv,
                    "note": "Robot has < 4 actuators, not evaluated for locomotion",
                }
                with open(save_path / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

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

        # Get initial weights - either inherited from parent (via strategy) or random
        # The strategy populates individual.tags["inherited_weights"] before evaluation
        initial_weights = None

        if individual is not None:
            inherited_weights = individual.tags.get("inherited_weights")
            inherited_layer_sizes = individual.tags.get("inherited_layer_sizes")

            if inherited_weights is not None and inherited_layer_sizes is not None:
                # Strategy provided inherited weights - adapt to offspring morphology if needed
                if inherited_layer_sizes != controller.layer_sizes:
                    # Morphology changed - adapt weights
                    from ariel.ec.strategies.weight_inheritance import adapt_weights_to_morphology
                    initial_weights = adapt_weights_to_morphology(
                        parent_weights=inherited_weights,
                        parent_layer_sizes=inherited_layer_sizes,
                        offspring_layer_sizes=controller.layer_sizes,
                        rng=self.rng,
                        sigma=self.sigma_init,
                    )
                else:
                    # Same morphology - use weights directly
                    initial_weights = inherited_weights

        # Fallback to random initialization if no inherited weights
        if initial_weights is None:
            initial_weights = self.rng.uniform(-self.sigma_init, self.sigma_init, num_weights)

        # Get controller weights - either via CMA-ES or direct use
        if self.use_cmaes:

            # Use CMA-ES to optimize controller weights
            optimized_weights, cmaes_fitness, learning_curve = self.optimize_controller_cmaes_wrapper(
                model, world_spec, initial_weights=initial_weights
            )

            # Store weights in individual tags for strategy to extract
            if individual is not None:
                individual.tags["initial_weights"] = initial_weights
                individual.tags["optimized_weights"] = optimized_weights
                individual.tags["layer_sizes"] = controller.layer_sizes

            # Save brain files directly if save_dir is available
            if save_dir is not None:
                import csv
                import json
                save_path = Path(save_dir)
                np.save(save_path / "initial_brain.npy", initial_weights)
                np.save(save_path / "optimized_brain.npy", optimized_weights)

                # Save metadata about neural network architecture
                metadata = {
                    "controller_hidden_layers": self.controller_hidden_layers,
                    "controller_activation": self.controller_activation,
                    "layer_sizes": controller.layer_sizes,
                    "num_actuators": model.nu,
                    "num_weights": len(optimized_weights),
                    "input_size": model.nq + model.nv,
                }
                with open(save_path / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Save learning curve
                with open(save_path / "learning_curve.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['iteration', 'fitness'])
                    for i, fitness in enumerate(learning_curve):
                        writer.writerow([i, fitness])

            # Use the fitness from CMA-ES
            base_fitness = cmaes_fitness
        else:
            # No CMA-ES optimization - use initial weights directly
            # (initial_weights already determined above based on inheritance mode)
            controller.set_weights(initial_weights)

            # Reset simulation
            mj.mj_resetData(model, data)

            # Setup tracker
            tracker = setup_tracker(world_spec, data)

            # Capture spawn height for penalty calculation
            # Forward kinematics must be computed first to get correct positions
            mj.mj_forward(model, data)

            # Find core geom and get its initial height
            spawn_height = None
            for i in range(model.ngeom):
                geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
                if geom_name and "core" in geom_name.lower():
                    spawn_height = data.geom(i).xpos[2]  # z-coordinate
                    break

            if spawn_height is None:
                raise ValueError("Could not find core geom to determine spawn height")

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
            # Pass spawn_height for the height penalty calculation
            x_displacement = calculate_displacement_fitness(
                tracker,
                baseline_time=0.0,
                model=model,
                spawn_height=spawn_height
            )

            # Store weights in individual tags for strategy to extract
            # In non-CMA-ES case, initial weights = optimized weights (no optimization)
            if individual is not None:
                individual.tags["initial_weights"] = initial_weights
                individual.tags["optimized_weights"] = initial_weights  # Same as initial
                individual.tags["layer_sizes"] = controller.layer_sizes

            # Save brain files directly if save_dir is available
            # No optimization in this case, so initial = optimized
            if save_dir is not None:
                import csv
                import json
                save_path = Path(save_dir)
                np.save(save_path / "initial_brain.npy", initial_weights)
                np.save(save_path / "optimized_brain.npy", initial_weights)

                # Save metadata about neural network architecture
                metadata = {
                    "controller_hidden_layers": self.controller_hidden_layers,
                    "controller_activation": self.controller_activation,
                    "layer_sizes": controller.layer_sizes,
                    "num_actuators": model.nu,
                    "num_weights": len(initial_weights),
                    "input_size": model.nq + model.nv,
                }
                with open(save_path / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Save empty learning curve (no CMA-ES)
                with open(save_path / "learning_curve.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['iteration', 'fitness'])
                    # Empty - no iterations

            # Use displacement as base fitness
            base_fitness = x_displacement

        # Store locomotion fitness in tags for separate tracking
        if individual is not None:
            individual.tags["locomotion_fitness"] = float(base_fitness)

        # NOTE: Do NOT add to archive during evaluation
        # Archive addition happens in recalculate_novelty_for_generation() after all evaluations

        # Save fitness components to individual directory if available
        if save_dir is not None:
            import json
            save_path = Path(save_dir)
            fitness_components = {
                "locomotion_fitness": float(base_fitness),
                "novelty_score": None,  # Will be calculated per-generation
            }
            with open(save_path / "fitness_components.json", 'w') as f:
                json.dump(fitness_components, f, indent=2)

        # Return locomotion fitness only (novelty will be applied per-generation)
        return float(base_fitness)

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
        if self.verbose:
            console.print(
                f"\n[bold cyan]Starting Evolution[/bold cyan]\n"
                f"Generations: {num_generations}\n"
                f"Fitness: {'Locomotion' + (' * Novelty' if self.use_novelty else '')}\n"
            )

        # Initialize diverse population (returns list of genomes, not Individuals)
        initial_genomes = self.initialize_diverse_population()

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

        # Note: Database is already saved per-generation by the strategy
        # (see save_database_per_generation=True in __init__)

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
            weight_manager=None,  # Weights are loaded from files, not from weight manager
        )


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
        default=4,
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

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()

    # Create timestamped data directory (once in main process)
    global DATA
    global SEED
    global RNG
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
    DATA = CWD / "__data__" / dir_name
    DATA.mkdir(exist_ok=True, parents=True)

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
        max_depth=3,                     # Max tree depth
        max_part_limit=25,               # Max robot parts
        max_actuators=12,                # Max actuators
        mutation_strength=2,             # Initial mutation strength
        mutation_reps=2,                 # Initial mutation repetitions
        mutate_attributes_prob=0.1,      # Self-adaptation mutation prob
        enable_self_adaptation=True,    # Enable/disable self-adaptation

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

        # Video recording parameters
        enable_video_recording=False,     # Enable video recording of best robot
        video_interval=1,                # Record video every N generations
        video_duration=35.0,             # Video duration in seconds
        video_platform="macos",          # Video codec: "macos" or "windows"

        # System parameters
        seed=args.seed,                  # Random seed (from command line)
        num_workers=args.num_workers,    # Parallel workers (from command line)
        verbose=True,
    )

    # Run evolution
    all_individuals = evolution.run(num_generations=args.num_generations)

    # Plot fitness history
    evolution.plot_fitness_history_wrapper(all_individuals)

    # Visualize best morphology
    evolution.visualize_best_wrapper(all_individuals, mode="viewer", duration=15.0)

    console.print(
        f"\n[bold green]All results saved to:[/bold green] {DATA.absolute()}"
    )


if __name__ == "__main__":
    main()
