"""Fitness evaluator for robot morphologies.

This module provides a clean, modular fitness evaluation system that separates
concerns and breaks down the evaluation process into manageable steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
from rich.console import Console

from ariel.ec import Individual
from myevo.controllers.controller_optimizer import optimize_controller_cmaes
from myevo.controllers.neural_network_controller import FlexibleNeuralNetworkController
from myevo.measures.locomotion_fitness import (
    calculate_displacement_fitness,
    simulate_with_settling_phase,
)
from myevo.simulation.simulation_utils import (
    create_controller,
    create_robot_model,
    setup_tracker,
)
from myevo.core.services import ResultsPersistence

console = Console()


class FitnessEvaluator:
    """Handles fitness evaluation for robot morphologies.

    This class encapsulates all the logic for evaluating a robot morphology,
    including controller optimization, simulation, and result persistence.
    """

    MIN_ACTUATORS_FOR_EVALUATION = 4
    """Minimum actuators required for locomotion evaluation."""

    def __init__(
        self,
        # Simulation parameters
        simulation_duration: float,
        controller_hidden_layers: list[int],
        controller_activation: str,
        sigma_init: float,
        # CMA-ES parameters
        use_cmaes: bool,
        cmaes_budget: int,
        cmaes_population_size: int,
        # System parameters
        seed: int,
        cache_dir: Path,
        maximize: bool = True,
        profiler: Any | None = None,
        enable_lamarckian: bool = False,
        covariance_inheritance_mode: str = "adaptive",
        sigma_inheritance_mode: str = "blend",
    ):
        """Initialize fitness evaluator.

        Parameters
        ----------
        simulation_duration : float
            Duration of simulation in seconds.
        controller_hidden_layers : list[int]
            Hidden layer sizes for neural network.
        controller_activation : str
            Activation function ('tanh', 'relu', 'sigmoid').
        sigma_init : float
            Initial standard deviation for weights.
        use_cmaes : bool
            Whether to use CMA-ES optimization.
        cmaes_budget : int
            Number of CMA-ES evaluations.
        cmaes_population_size : int
            CMA-ES population size.
        seed : int
            Random seed.
        cache_dir : Path
            Cache directory for temporary files.
        maximize : bool
            Whether to maximize fitness.
        profiler : Any | None
            Memory profiler instance.
        enable_lamarckian : bool
            Whether Lamarckian evolution is enabled (affects CMA-ES state saving).
        covariance_inheritance_mode : str
            How to adapt CMA-ES covariance matrices when morphology changes.
        sigma_inheritance_mode : str
            How to adapt CMA-ES step size when morphology changes.
        """
        self.simulation_duration = simulation_duration
        self.controller_hidden_layers = controller_hidden_layers
        self.controller_activation = controller_activation
        self.sigma_init = sigma_init
        self.use_cmaes = use_cmaes
        self.cmaes_budget = cmaes_budget
        self.cmaes_population_size = cmaes_population_size
        self.seed = seed
        self.cache_dir = cache_dir
        self.maximize = maximize
        self.profiler = profiler
        self.enable_lamarckian = enable_lamarckian
        self.covariance_inheritance_mode = covariance_inheritance_mode
        self.sigma_inheritance_mode = sigma_inheritance_mode
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self, individual: Individual, log_dir: str | None = None
    ) -> float:
        """Evaluate an individual's fitness.

        Parameters
        ----------
        individual : Individual
            Individual to evaluate.
        log_dir : str | None
            Directory to save results to.

        Returns
        -------
        float
            Fitness value.
        """
        # Memory profiling
        self._profile_if_needed(individual, "fitness_start")

        # Extract tree and save body
        tree = self._extract_tree(individual)
        if log_dir is not None:
            self._save_body(tree, log_dir)

        # Build robot model
        model, data, world_spec = create_robot_model(tree)

        # Check if robot is viable for evaluation
        if not self._is_viable(model):
            return self._handle_nonviable_robot(individual, model, log_dir)

        # Get initial weights (with inheritance if available)
        initial_weights = self._get_initial_weights(individual, model)

        # Evaluate robot
        if self.use_cmaes:
            fitness = self._evaluate_with_cmaes(
                model, world_spec, initial_weights, individual, log_dir
            )
        else:
            fitness = self._evaluate_without_cmaes(
                model, data, world_spec, initial_weights, individual, log_dir
            )

        # Save fitness components
        if log_dir is not None:
            ResultsPersistence.save_fitness_components(
                Path(log_dir), float(fitness), novelty_score=None
            )

        # Memory profiling
        self._profile_if_needed(individual, "fitness_end")

        return float(fitness)

    def _extract_tree(self, individual: Individual) -> Any:
        """Extract tree genotype from individual.

        Parameters
        ----------
        individual : Individual
            Individual to extract tree from.

        Returns
        -------
        Any
            Tree genotype.
        """
        from myevo.core import TreeGenotype

        genome = individual.genotype
        return genome.tree if isinstance(genome, TreeGenotype) else genome

    def _save_body(self, tree: Any, log_dir: str) -> None:
        """Save body graph as JSON.

        Parameters
        ----------
        tree : Any
            Tree genotype.
        log_dir : str
            Directory to save to.
        """
        from ariel.body_phenotypes.robogen_lite.decoders import save_graph_as_json

        save_graph_as_json(tree, Path(log_dir) / "body.json")

    def _is_viable(self, model: mj.MjModel) -> bool:
        """Check if robot has enough actuators for evaluation.

        Parameters
        ----------
        model : mj.MjModel
            MuJoCo model.

        Returns
        -------
        bool
            True if viable for evaluation.
        """
        return model.nu >= self.MIN_ACTUATORS_FOR_EVALUATION

    def _handle_nonviable_robot(
        self, individual: Individual, model: mj.MjModel, log_dir: str | None
    ) -> float:
        """Handle robot with insufficient actuators.

        Creates random weights for inheritance but returns zero fitness.

        Parameters
        ----------
        individual : Individual
            Individual being evaluated.
        model : mj.MjModel
            MuJoCo model.
        log_dir : str | None
            Log directory.

        Returns
        -------
        float
            Zero fitness.
        """
        # Create controller for weight generation
        controller = create_controller(
            model=model,
            hidden_layers=self.controller_hidden_layers,
            activation=self.controller_activation,
            seed=self.seed,
        )
        num_weights = controller.get_num_weights()
        random_weights = self.rng.uniform(-self.sigma_init, self.sigma_init, num_weights)

        # Create default CMA-ES state
        from myevo.core.cmaes_inheritance import (
            create_default_cmaes_state,
            save_cmaes_state_to_disk,
        )

        default_state = create_default_cmaes_state(
            controller.layer_sizes, sigma_init=self.sigma_init
        )

        # Store in individual tags
        if individual is not None:
            self._store_weights_in_individual(
                individual,
                random_weights,
                random_weights,
                controller.layer_sizes,
                default_state,
                default_state,
            )
            individual.tags["locomotion_fitness"] = 0.0

        # Save files if log_dir provided
        if log_dir is not None:
            self._save_nonviable_results(
                Path(log_dir), random_weights, controller, model, default_state
            )

        return 0.0

    def _get_initial_weights(
        self, individual: Individual, model: mj.MjModel
    ) -> np.ndarray:
        """Get initial weights (either inherited or random).

        Parameters
        ----------
        individual : Individual
            Individual being evaluated.
        model : mj.MjModel
            MuJoCo model.

        Returns
        -------
        np.ndarray
            Initial weights.
        """
        controller = create_controller(
            model=model,
            hidden_layers=self.controller_hidden_layers,
            activation=self.controller_activation,
            seed=self.seed,
        )

        # Try to inherit weights
        inherited_weights = self._try_inherit_weights(individual, controller)

        # Fallback to random
        if inherited_weights is None:
            return self.rng.uniform(
                -self.sigma_init, self.sigma_init, controller.get_num_weights()
            )

        return inherited_weights

    def _try_inherit_weights(
        self, individual: Individual, controller: FlexibleNeuralNetworkController
    ) -> np.ndarray | None:
        """Try to inherit weights from parent.

        Parameters
        ----------
        individual : Individual
            Individual being evaluated.
        controller : FlexibleNeuralNetworkController
            Neural network controller.

        Returns
        -------
        np.ndarray | None
            Inherited weights, or None if not available.
        """
        if individual is None:
            return None

        inherited_weights = individual.tags.get("inherited_weights")
        inherited_layer_sizes = individual.tags.get("inherited_layer_sizes")

        if inherited_weights is None or inherited_layer_sizes is None:
            return None

        # Adapt if morphology changed
        if inherited_layer_sizes != controller.layer_sizes:
            from myevo.core.weight_inheritance import adapt_weights_to_morphology

            return adapt_weights_to_morphology(
                parent_weights=inherited_weights,
                parent_layer_sizes=inherited_layer_sizes,
                offspring_layer_sizes=controller.layer_sizes,
                rng=self.rng,
                sigma=self.sigma_init,
            )

        return inherited_weights

    def _evaluate_with_cmaes(
        self,
        model: mj.MjModel,
        world_spec: Any,
        initial_weights: np.ndarray,
        individual: Individual,
        log_dir: str | None,
    ) -> float:
        """Evaluate using CMA-ES optimization.

        Parameters
        ----------
        model : mj.MjModel
            MuJoCo model.
        world_spec : Any
            World specification.
        initial_weights : np.ndarray
            Initial weights.
        individual : Individual
            Individual being evaluated.
        log_dir : str | None
            Log directory.

        Returns
        -------
        float
            Optimized fitness.
        """
        # Get inherited CMA-ES state if available
        initial_cmaes_state = self._load_inherited_cmaes_state(individual, model)

        # Optimize
        optimized_weights, fitness, learning_curve, _, metrics = (
            optimize_controller_cmaes(
                model=model,
                world_spec=world_spec,
                hidden_layers=self.controller_hidden_layers,
                activation=self.controller_activation,
                simulation_duration=self.simulation_duration,
                cmaes_budget=self.cmaes_budget,
                cmaes_population_size=self.cmaes_population_size,
                sigma_init=self.sigma_init,
                initial_weights=initial_weights,
                initial_cmaes_state=initial_cmaes_state,
                maximize=self.maximize,
                baseline_time=5.0,
                seed=self.seed,
            )
        )

        # Extract CMA-ES states
        from myevo.core.cmaes_inheritance import (
            create_default_cmaes_state,
            save_cmaes_state_to_disk,
        )

        controller = create_controller(
            model=model,
            hidden_layers=self.controller_hidden_layers,
            activation=self.controller_activation,
            seed=self.seed,
        )

        initial_state = metrics.get("initial_cmaes_state") or create_default_cmaes_state(
            controller.layer_sizes, sigma_init=self.sigma_init
        )
        optimized_state = metrics.get("optimized_cmaes_state") or create_default_cmaes_state(
            controller.layer_sizes, sigma_init=self.sigma_init
        )

        # Store in individual
        if individual is not None:
            self._store_weights_in_individual(
                individual,
                initial_weights,
                optimized_weights,
                controller.layer_sizes,
                initial_state,
                optimized_state,
            )
            individual.tags["locomotion_fitness"] = float(fitness)
            # Store CMA-ES metrics
            individual.tags["cmaes_sigma"] = optimized_state.sigma
            individual.tags["cmaes_condition_number"] = optimized_state.condition_number
            individual.tags["cmaes_mean_fitness"] = metrics.get("mean_fitness")
            individual.tags["cmaes_num_evaluations"] = metrics.get("num_evaluations")

        # Save results
        if log_dir is not None:
            self._save_cmaes_results(
                Path(log_dir),
                initial_weights,
                optimized_weights,
                controller,
                model,
                initial_state,
                optimized_state,
                learning_curve,
            )

        return fitness

    def _evaluate_without_cmaes(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        world_spec: Any,
        initial_weights: np.ndarray,
        individual: Individual,
        log_dir: str | None,
    ) -> float:
        """Evaluate without CMA-ES optimization (use initial weights directly).

        Parameters
        ----------
        model : mj.MjModel
            MuJoCo model.
        data : mj.MjData
            MuJoCo data.
        world_spec : Any
            World specification.
        initial_weights : np.ndarray
            Initial weights.
        individual : Individual
            Individual being evaluated.
        log_dir : str | None
            Log directory.

        Returns
        -------
        float
            Fitness from initial weights.
        """
        controller = create_controller(
            model=model,
            hidden_layers=self.controller_hidden_layers,
            activation=self.controller_activation,
            seed=self.seed,
        )
        controller.set_weights(initial_weights)

        # Reset and setup
        mj.mj_resetData(model, data)
        tracker = setup_tracker(world_spec, data)

        # Get spawn height
        mj.mj_forward(model, data)
        spawn_height = self._get_spawn_height(model, data)

        # Simulate
        settling_duration = 5.0
        control_duration = self.simulation_duration - settling_duration
        contact_count = simulate_with_settling_phase(
            model=model,
            data=data,
            controller=controller,
            tracker=tracker,
            settling_duration=settling_duration,
            control_duration=control_duration,
            track_contacts=True,
        )

        # Calculate fitness
        fitness = calculate_displacement_fitness(
            tracker,
            baseline_time=0.0,
            model=model,
            spawn_height=spawn_height,
            contact_count=contact_count,
        )

        # Create default CMA-ES state for consistency
        from myevo.core.cmaes_inheritance import (
            create_default_cmaes_state,
            save_cmaes_state_to_disk,
        )

        default_state = create_default_cmaes_state(
            controller.layer_sizes, sigma_init=self.sigma_init
        )

        # Store in individual (initial = optimized in this case)
        if individual is not None:
            self._store_weights_in_individual(
                individual,
                initial_weights,
                initial_weights,
                controller.layer_sizes,
                default_state,
                default_state,
            )
            individual.tags["locomotion_fitness"] = float(fitness)
            individual.tags["cmaes_sigma"] = default_state.sigma
            individual.tags["cmaes_condition_number"] = 1.0
            individual.tags["cmaes_mean_fitness"] = None
            individual.tags["cmaes_num_evaluations"] = 0

        # Save results
        if log_dir is not None:
            self._save_simple_results(
                Path(log_dir), initial_weights, controller, model
            )

        return fitness

    def _get_spawn_height(self, model: mj.MjModel, data: mj.MjData) -> float:
        """Get spawn height of robot core.

        Parameters
        ----------
        model : mj.MjModel
            MuJoCo model.
        data : mj.MjData
            MuJoCo data.

        Returns
        -------
        float
            Z-coordinate of core geom.

        Raises
        ------
        ValueError
            If core geom not found.
        """
        for i in range(model.ngeom):
            geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
            if geom_name and "core" in geom_name.lower():
                return data.geom(i).xpos[2]
        raise ValueError("Could not find core geom to determine spawn height")

    def _load_inherited_cmaes_state(
        self, individual: Individual, model: mj.MjModel
    ) -> Any | None:
        """Load inherited CMA-ES state from parent.

        Parameters
        ----------
        individual : Individual
            Individual being evaluated.
        model : mj.MjModel
            MuJoCo model.

        Returns
        -------
        Any | None
            CMA-ES state, or None if not available.
        """
        if individual is None:
            return None

        cache_path_str = individual.tags.get("inherited_cmaes_cache_path")
        layer_sizes = individual.tags.get("inherited_layer_sizes_cmaes")

        if cache_path_str is None or layer_sizes is None:
            return None

        cache_path = Path(cache_path_str)
        if not cache_path.exists():
            return None

        from myevo.core.cmaes_inheritance import (
            adapt_cmaes_state_to_morphology,
            load_cmaes_state_from_disk,
        )

        inherited_state = load_cmaes_state_from_disk(cache_path, layer_sizes)

        # Get current controller architecture
        controller = create_controller(
            model=model,
            hidden_layers=self.controller_hidden_layers,
            activation=self.controller_activation,
            seed=self.seed,
        )

        # Adapt if morphology changed
        if layer_sizes != controller.layer_sizes:
            return adapt_cmaes_state_to_morphology(
                parent_state=inherited_state,
                parent_layer_sizes=layer_sizes,
                offspring_layer_sizes=controller.layer_sizes,
                sigma_init=self.sigma_init,
                covariance_mode=self.covariance_inheritance_mode,
                sigma_mode=self.sigma_inheritance_mode,
                rng=self.rng,
            )

        return inherited_state

    def _store_weights_in_individual(
        self,
        individual: Individual,
        initial_weights: np.ndarray,
        optimized_weights: np.ndarray,
        layer_sizes: list[int],
        initial_state: Any,
        optimized_state: Any,
    ) -> None:
        """Store weights and CMA-ES states in individual tags.

        Parameters
        ----------
        individual : Individual
            Individual to store in.
        initial_weights : np.ndarray
            Pre-optimization weights.
        optimized_weights : np.ndarray
            Post-optimization weights.
        layer_sizes : list[int]
            Neural network architecture.
        initial_state : Any
            Initial CMA-ES state.
        optimized_state : Any
            Optimized CMA-ES state.
        """
        from myevo.core.cmaes_inheritance import save_cmaes_state_to_disk

        individual.tags["initial_weights"] = initial_weights
        individual.tags["optimized_weights"] = optimized_weights
        individual.tags["layer_sizes"] = layer_sizes

        # Save CMA-ES states to cache (only if Lamarckian evolution is enabled)
        if self.enable_lamarckian:
            initial_cache_path = self.cache_dir / f"ind_{individual.id}_initial_state"
            save_cmaes_state_to_disk(initial_state, initial_cache_path)
            individual.tags["initial_cmaes_cache_path"] = str(initial_cache_path)
            individual.tags["initial_cmaes_layer_sizes"] = layer_sizes

            optimized_cache_path = self.cache_dir / f"ind_{individual.id}_optimized_state"
            save_cmaes_state_to_disk(optimized_state, optimized_cache_path)
            individual.tags["optimized_cmaes_cache_path"] = str(optimized_cache_path)
            individual.tags["optimized_cmaes_layer_sizes"] = layer_sizes

    def _save_cmaes_results(
        self,
        save_dir: Path,
        initial_weights: np.ndarray,
        optimized_weights: np.ndarray,
        controller: FlexibleNeuralNetworkController,
        model: mj.MjModel,
        initial_state: Any,
        optimized_state: Any,
        learning_curve: list[float],
    ) -> None:
        """Save CMA-ES optimization results.

        Parameters
        ----------
        save_dir : Path
            Directory to save to.
        initial_weights : np.ndarray
            Pre-optimization weights.
        optimized_weights : np.ndarray
            Post-optimization weights.
        controller : FlexibleNeuralNetworkController
            Neural network controller.
        model : mj.MjModel
            MuJoCo model.
        initial_state : Any
            Initial CMA-ES state.
        optimized_state : Any
            Optimized CMA-ES state.
        learning_curve : list[float]
            Fitness over iterations.
        """
        from myevo.core.cmaes_inheritance import save_cmaes_state_to_disk

        # Save weights
        ResultsPersistence.save_brain_weights(
            save_dir, initial_weights, optimized_weights
        )

        # Save CMA-ES states (only if Lamarckian evolution is enabled)
        if self.enable_lamarckian:
            initial_state_dir = save_dir / "initial_cmaes"
            initial_state_dir.mkdir(exist_ok=True)
            save_cmaes_state_to_disk(initial_state, initial_state_dir)
            save_cmaes_state_to_disk(optimized_state, save_dir)

        # Save metadata
        metadata = {
            "controller_hidden_layers": self.controller_hidden_layers,
            "controller_activation": self.controller_activation,
            "layer_sizes": controller.layer_sizes,
            "num_actuators": model.nu,
            "num_weights": len(optimized_weights),
            "input_size": model.nq + model.nv,
            "cmaes_budget": self.cmaes_budget,
            "cmaes_population_size": self.cmaes_population_size,
            "sigma_init": self.sigma_init,
            "cmaes_sigma_initial": float(initial_state.sigma),
            "cmaes_sigma_final": float(optimized_state.sigma),
            "cmaes_condition_number_initial": (
                float(initial_state.condition_number)
                if initial_state.condition_number
                else None
            ),
            "cmaes_condition_number_final": (
                float(optimized_state.condition_number)
                if optimized_state.condition_number
                else None
            ),
        }
        ResultsPersistence.save_metadata(save_dir, metadata)

        # Save learning curve
        ResultsPersistence.save_learning_curve(save_dir, learning_curve)

    def _save_simple_results(
        self,
        save_dir: Path,
        weights: np.ndarray,
        controller: FlexibleNeuralNetworkController,
        model: mj.MjModel,
    ) -> None:
        """Save results when not using CMA-ES.

        Parameters
        ----------
        save_dir : Path
            Directory to save to.
        weights : np.ndarray
            Neural network weights.
        controller : FlexibleNeuralNetworkController
            Neural network controller.
        model : mj.MjModel
            MuJoCo model.
        """
        # Save weights (initial = optimized)
        ResultsPersistence.save_brain_weights(save_dir, weights, weights)

        # Save metadata
        metadata = {
            "controller_hidden_layers": self.controller_hidden_layers,
            "controller_activation": self.controller_activation,
            "layer_sizes": controller.layer_sizes,
            "num_actuators": model.nu,
            "num_weights": len(weights),
            "input_size": model.nq + model.nv,
        }
        ResultsPersistence.save_metadata(save_dir, metadata)

        # Save empty learning curve
        ResultsPersistence.save_learning_curve(save_dir, [])

    def _save_nonviable_results(
        self,
        save_dir: Path,
        weights: np.ndarray,
        controller: FlexibleNeuralNetworkController,
        model: mj.MjModel,
        state: Any,
    ) -> None:
        """Save results for non-viable robots.

        Parameters
        ----------
        save_dir : Path
            Directory to save to.
        weights : np.ndarray
            Random weights.
        controller : FlexibleNeuralNetworkController
            Controller.
        model : mj.MjModel
            MuJoCo model.
        state : Any
            Default CMA-ES state.
        """
        from myevo.core.cmaes_inheritance import save_cmaes_state_to_disk

        # Save weights
        ResultsPersistence.save_brain_weights(save_dir, weights, weights)

        # Save CMA-ES state (only if Lamarckian evolution is enabled)
        if self.enable_lamarckian:
            save_cmaes_state_to_disk(state, save_dir)

        # Save metadata
        metadata = {
            "controller_hidden_layers": self.controller_hidden_layers,
            "controller_activation": self.controller_activation,
            "layer_sizes": controller.layer_sizes,
            "num_actuators": model.nu,
            "num_weights": len(weights),
            "input_size": model.nq + model.nv,
            "cmaes_budget": 0,
            "cmaes_sigma_final": float(state.sigma),
            "note": f"Robot has < {self.MIN_ACTUATORS_FOR_EVALUATION} actuators, not evaluated",
        }
        ResultsPersistence.save_metadata(save_dir, metadata)

        # Empty learning curve
        ResultsPersistence.save_learning_curve(save_dir, [])

    def _profile_if_needed(self, individual: Individual, tag: str) -> None:
        """Profile memory if profiler enabled and conditions met.

        Parameters
        ----------
        individual : Individual
            Individual being evaluated.
        tag : str
            Profile tag.
        """
        if self.profiler and individual.id is not None and individual.id % 30 == 0:
            self.profiler.log_memory(f"{tag}_ind_{individual.id}")
            if tag == "fitness_end":
                import gc
                gc.collect()
