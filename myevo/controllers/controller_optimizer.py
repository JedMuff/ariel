"""Controller optimization utilities using CMA-ES.

This module provides functions for optimizing neural network controller weights
for robot morphologies using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
algorithm. This is used as an inner loop within morphological evolution.
"""

from __future__ import annotations

from typing import Any
import gc

import mujoco as mj
import nevergrad as ng
import numpy as np

from myevo.simulation.simulation_utils import (
    create_controller,
    create_robot_model,
    setup_tracker,
)
from myevo.measures.locomotion_fitness import (
    calculate_displacement_fitness,
    simulate_with_settling_phase,
)


def initialize_weights(
    num_weights: int,
    sigma: float,
    parent_weights: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Initialize controller weights, optionally from parent weights.

    Parameters
    ----------
    num_weights : int
        Total number of weights needed.
    sigma : float
        Standard deviation for random initialization (uniform in [-sigma, sigma]).
    parent_weights : np.ndarray | None, optional
        Inherited weights from parent (Lamarckian evolution), by default None.
    rng : np.random.Generator | None, optional
        Random number generator, by default None (creates new one).

    Returns
    -------
    np.ndarray
        Initialized weights.
    """
    if parent_weights is not None:
        return parent_weights.copy()

    if rng is None:
        rng = np.random.default_rng()

    return rng.uniform(-sigma, sigma, num_weights)


def optimize_controller_cmaes(
    model: mj.MjModel,
    world_spec: Any,
    hidden_layers: list[int],
    activation: str,
    simulation_duration: float,
    cmaes_budget: int,
    cmaes_population_size: int,
    sigma_init: float = 1.0,
    initial_weights: np.ndarray | None = None,
    initial_cmaes_state: Any | None = None,
    maximize: bool = True,
    baseline_time: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, float, list[float], dict[str, Any] | None, dict[str, Any]]:
    """Optimize controller weights using CMA-ES for a given morphology.

    This function runs an inner optimization loop to find the best controller
    weights for a specific robot morphology. It uses CMA-ES to explore the
    weight space and find weights that maximize locomotion performance.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model to optimize for.
    world_spec : Any
        The world specification for tracking.
    hidden_layers : list[int]
        Neural network hidden layer sizes.
    activation : str
        Activation function name ('tanh', 'relu', 'sigmoid', 'elu').
    simulation_duration : float
        Duration of each simulation in seconds.
    cmaes_budget : int
        Number of CMA-ES evaluations (inner loop iterations).
    cmaes_population_size : int
        Population size for CMA-ES optimizer.
    sigma_init : float, optional
        Initial standard deviation for weight initialization, by default 1.0.
    initial_weights : np.ndarray | None, optional
        Initial weights to start optimization from (for Lamarckian evolution),
        by default None.
    initial_cmaes_state : CMAESState | None, optional
        Initial CMA-ES state (covariance, sigma) to inherit from parent,
        by default None.
    maximize : bool, optional
        Whether to maximize (True) or minimize (False) fitness, by default True.
    baseline_time : float, optional
        Time offset for displacement calculation, by default 5.0 seconds.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    tuple[np.ndarray, float, list[float], dict | None, dict]
        - Best weights found
        - Best fitness value achieved
        - Learning curve (fitness at each iteration)
        - Nevergrad optimizer state dict (for complete restoration)
        - Metrics dict (iterations, budget, initial_fitness, final_fitness, mean_fitness)
    """
    num_actuators = model.nu

    # Create controller for this morphology
    controller = create_controller(
        model=model,
        hidden_layers=hidden_layers,
        activation=activation,
        seed=seed,
    )
    num_weights = controller.get_num_weights()

    # Define evaluation function for CMA-ES
    def evaluate_weights(weights: np.ndarray) -> float:
        """Evaluate a single set of controller weights."""
        # Set controller weights
        controller.set_weights(weights)

        # Reset MuJoCo control callback
        mj.set_mjcb_control(None)

        # Create fresh data
        data = mj.MjData(model)
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
        control_duration = simulation_duration - settling_duration
        contact_count = simulate_with_settling_phase(
            model=model,
            data=data,
            controller=controller,
            tracker=tracker,
            settling_duration=settling_duration,
            control_duration=control_duration,
            track_contacts=True,
        )

        # Calculate fitness with all penalties (height + contact)
        # baseline_time=0 since tracker was reset at start of control phase
        fitness = calculate_displacement_fitness(
            tracker=tracker,
            baseline_time=0.0,
            model=model,
            spawn_height=spawn_height,
            contact_count=contact_count,
        )

        return float(fitness)

    # Initialize CMA-ES optimizer
    instrum = ng.p.Array(shape=(num_weights,))
    optimizer = ng.optimizers.CMA(
        parametrization=instrum,
        budget=cmaes_budget,
        num_workers=1,
    )

    # Set initial guess - use parent weights if available (Lamarckian), otherwise random
    rng = np.random.default_rng(seed)
    if initial_weights is not None:
        initial_guess = initial_weights
    else:
        initial_guess = rng.uniform(-sigma_init, sigma_init, num_weights)
    optimizer.suggest(initial_guess)

    # Initialize with inherited CMA-ES state if provided (Lamarckian CMA-ES)
    if initial_cmaes_state is not None:
        # Access the underlying CMA-ES optimizer
        cma_es = optimizer.optim.es

        # Set inherited covariance matrix and sigma
        cma_es.C = initial_cmaes_state.covariance_matrix.copy()
        cma_es.sigma = initial_cmaes_state.sigma

        # Set mean if available
        if initial_cmaes_state.mean is not None and len(initial_cmaes_state.mean) == num_weights:
            cma_es.mean = initial_cmaes_state.mean.copy()

        # Note: We don't restore full nevergrad state here because the optimizer
        # was already created with potentially different dimensions. We only
        # inherit the learned covariance and sigma.

    # Capture initial CMA-ES state before optimization begins
    # If this fails, we want to know about it (indicates optimizer setup issue)
    from myevo.core.cmaes_inheritance import extract_cmaes_state_from_nevergrad
    input_size = 2 * model.nu + 9
    layer_sizes = [input_size] + hidden_layers + [model.nu]
    initial_cmaes_state_captured = extract_cmaes_state_from_nevergrad(optimizer, layer_sizes)

    # Run CMA-ES optimization
    best_fitness = float('-inf') if maximize else float('inf')
    best_weights = None
    fitness_history = []  # Track fitness at each iteration
    initial_fitness = None  # Track first evaluation

    for i in range(cmaes_budget):
        # Get candidate from CMA-ES
        x = optimizer.ask()
        candidate = x.value

        # Evaluate fitness
        fitness = evaluate_weights(candidate)

        # Track first fitness (before optimization)
        if i == 0:
            initial_fitness = fitness

        # Track fitness history
        fitness_history.append(fitness)

        # Tell optimizer the result (CMA-ES minimizes, so negate for maximization)
        if maximize:
            optimizer.tell(x, -fitness)
        else:
            optimizer.tell(x, fitness)

        # Track best
        if maximize and fitness > best_fitness:
            best_fitness = fitness
            best_weights = candidate.copy()
        elif not maximize and fitness < best_fitness:
            best_fitness = fitness
            best_weights = candidate.copy()

        # Periodic garbage collection to prevent memory accumulation
        # More frequent GC for higher budgets to prevent buildup
        if cmaes_budget >= 500 and i % 100 == 0:
            gc.collect()

    # Extract final CMA-ES state directly from optimizer
    # If this fails, we want to know about it (indicates optimizer corruption)
    from myevo.core.cmaes_inheritance import extract_cmaes_state_from_nevergrad
    # Get layer sizes for the state
    # Note: Input size is 2*num_actuators + 9 (actuator pos+vel + core body state)
    input_size = 2 * model.nu + 9
    layer_sizes = [input_size] + hidden_layers + [model.nu]
    cmaes_state = extract_cmaes_state_from_nevergrad(optimizer, layer_sizes)
    # Note: We don't use nevergrad's dump() because it requires a filepath
    # and we're handling serialization ourselves via the CMAESState object
    nevergrad_state = None

    # Calculate metrics
    metrics = {
        "iterations": len(fitness_history),
        "budget": cmaes_budget,
        "initial_fitness": initial_fitness,
        "final_fitness": best_fitness,
        "mean_fitness": float(np.mean(fitness_history)) if fitness_history else None,
        "num_evaluations": len(fitness_history),
        "initial_cmaes_state": initial_cmaes_state_captured,  # State before optimization
        "optimized_cmaes_state": cmaes_state,  # State after optimization
        "cmaes_state": cmaes_state,  # For backward compatibility (deprecated)
    }

    # Final cleanup after CMA-ES optimization completes
    # This helps prevent memory accumulation across multiple individuals
    gc.collect()

    return best_weights, best_fitness, fitness_history, nevergrad_state, metrics