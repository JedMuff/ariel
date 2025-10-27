"""Simulation utilities for robot morphology evaluation.

This module provides shared functions for setting up and running MuJoCo simulations
of evolved robot morphologies. It eliminates code duplication across different
evolutionary algorithms and visualization scripts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mujoco as mj
import numpy as np
from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker

if TYPE_CHECKING:
    from neural_network_controller import FlexibleNeuralNetworkController


def create_robot_model(tree: DiGraph, check_collisions: bool = False) -> tuple[mj.MjModel, mj.MjData, Any]:
    """Build a MuJoCo model from a robot tree genotype.

    Parameters
    ----------
    tree : DiGraph
        The robot morphology graph (tree genotype).
    check_collisions : bool, optional
        Whether to check for self-collisions during construction, by default False.

    Returns
    -------
    tuple[mj.MjModel, mj.MjData, Any]
        - MuJoCo model
        - MuJoCo data
        - World specification (for tracker setup)
    """
    # Reset MuJoCo control callback
    mj.set_mjcb_control(None)

    # Build robot from tree
    robot_core = construct_mjspec_from_graph(tree, check_collisions=check_collisions)

    # Create world and spawn robot
    world = SimpleFlatWorld()
    world.spawn(robot_core.spec, position=[0, 0, 0.15])

    # Compile model
    model = world.spec.compile()
    data = mj.MjData(model)

    return model, data, world.spec


def create_controller(
    model: mj.MjModel,
    hidden_layers: list[int],
    activation: str,
    seed: int,
) -> FlexibleNeuralNetworkController:
    """Create and initialize a neural network controller for a robot.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model to create controller for.
    hidden_layers : list[int]
        Hidden layer sizes for neural network.
    activation : str
        Activation function name ('tanh', 'relu', 'sigmoid', 'elu').
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    FlexibleNeuralNetworkController
        Initialized controller ready for weight setting.
    """
    from neural_network_controller import FlexibleNeuralNetworkController

    num_actuators = model.nu

    controller = FlexibleNeuralNetworkController(
        num_actuators=num_actuators,
        hidden_layers=hidden_layers,
        activation=activation,
        seed=seed,
    )
    controller.calculate_input_size(model)

    return controller


def setup_tracker(world_spec: Any, data: mj.MjData) -> Tracker:
    """Setup position tracking for the robot's core.

    Parameters
    ----------
    world_spec : Any
        World specification from SimpleFlatWorld.
    data : mj.MjData
        MuJoCo data object.

    Returns
    -------
    Tracker
        Configured tracker for core position.
    """
    tracker = Tracker(
        mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
    )
    tracker.setup(world_spec, data)
    return tracker


def calculate_displacement_fitness(
    tracker: Tracker,
    baseline_time: float,
    model: mj.MjModel,
    spawn_height: float,
) -> float:
    """Calculate fitness as forward displacement from a baseline time.

    This uses the x-axis displacement (forward direction in ARIEL) from
    a baseline time to avoid advantages from initial falling/settling.

    A height penalty is applied: if the robot's spawn height (morphology height)
    is above 0.21m, the spawn height is subtracted from the fitness. This prevents
    tall robots from exploiting falling/settling for free displacement.

    Parameters
    ----------
    tracker : Tracker
        Tracker with recorded position history.
    baseline_time : float
        Time in seconds to use as baseline (e.g., 1.0 for 1 second).
    model : mj.MjModel
        MuJoCo model (needed for timestep).
    spawn_height : float
        The spawn height (z-coordinate) of the robot's core at initialization.
        This is the robot's morphological height used for the penalty.

    Returns
    -------
    float
        Forward displacement in meters from baseline time to end, with height penalty applied.
    """
    # Get timestep information
    dt = model.opt.timestep
    time_steps_per_save = 500  # As defined in Controller
    seconds_per_save = dt * time_steps_per_save

    # Calculate baseline index
    baseline_index = int(baseline_time / seconds_per_save)

    # Ensure we have enough history, otherwise use first available
    initial_pos = tracker.history["xpos"][0][baseline_index]

    final_pos = tracker.history["xpos"][0][-1]

    # X-axis is forward direction in ARIEL
    x_displacement = final_pos[0] - initial_pos[0]

    # Apply height penalty based on spawn height (morphological height)
    # (allows up to 1 core + 3 bricks stacked below it: 0.21m)
    if spawn_height > 0.21:
        fitness = x_displacement - spawn_height
    else:
        fitness = x_displacement

    return float(fitness)


def simulate_with_controller(
    model: mj.MjModel,
    data: mj.MjData,
    controller: FlexibleNeuralNetworkController,
    tracker: Tracker,
    duration: float,
    time_steps_per_ctrl_step: int = 200,
    time_steps_per_save: int = 500,
) -> None:
    """Run a simulation with a neural network controller.

    Parameters
    ----------
    model : mj.MjModel
        MuJoCo model.
    data : mj.MjData
        MuJoCo data (should be reset before calling).
    controller : FlexibleNeuralNetworkController
        Controller with weights already set.
    tracker : Tracker
        Tracker already setup with world_spec and data.
    duration : float
        Simulation duration in seconds.
    time_steps_per_ctrl_step : int, optional
        Control update frequency, by default 200.
    time_steps_per_save : int, optional
        Tracking save frequency, by default 500.
    """
    # Create controller wrapper
    ctrl = Controller(
        controller_callback_function=controller,
        tracker=tracker,
        time_steps_per_ctrl_step=time_steps_per_ctrl_step,
        time_steps_per_save=time_steps_per_save,
    )

    # Set control callback
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Run simulation
    simple_runner(model, data, duration=duration)


def simulate_with_settling_phase(
    model: mj.MjModel,
    data: mj.MjData,
    controller: FlexibleNeuralNetworkController,
    tracker: Tracker,
    settling_duration: float,
    control_duration: float,
    time_steps_per_ctrl_step: int = 200,
    time_steps_per_save: int = 500,
) -> None:
    """Run a two-phase simulation: passive settling, then active control.

    Phase 1: Robot settles passively for settling_duration with no control input.
    Phase 2: Controller takes over from the settled state for control_duration.

    The tracker is reset at the start of Phase 2, so displacement is measured
    only during the controlled phase.

    Parameters
    ----------
    model : mj.MjModel
        MuJoCo model.
    data : mj.MjData
        MuJoCo data (should be reset before calling).
    controller : FlexibleNeuralNetworkController
        Controller with weights already set.
    tracker : Tracker
        Tracker already setup with world_spec and data.
    settling_duration : float
        Duration of passive settling phase in seconds.
    control_duration : float
        Duration of active control phase in seconds.
    time_steps_per_ctrl_step : int, optional
        Control update frequency, by default 200.
    time_steps_per_save : int, optional
        Tracking save frequency, by default 500.
    """
    # Phase 1: Passive settling (no control)
    mj.set_mjcb_control(None)
    simple_runner(model, data, duration=settling_duration)

    # Reset tracker history so we only measure displacement during controlled phase
    tracker.reset()

    # Phase 2: Active control from settled state
    ctrl = Controller(
        controller_callback_function=controller,
        tracker=tracker,
        time_steps_per_ctrl_step=time_steps_per_ctrl_step,
        time_steps_per_save=time_steps_per_save,
    )

    # Set control callback
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Continue simulation with control
    simple_runner(model, data, duration=control_duration)


def evaluate_morphology_fitness(
    tree: DiGraph,
    weights: np.ndarray,
    hidden_layers: list[int],
    activation: str,
    simulation_duration: float,
    baseline_time: float = 1.0,
    seed: int = 42,
    min_actuators: int = 4,
) -> float:
    """Evaluate a single morphology with given controller weights.

    This is a convenience function that combines all steps: building the robot,
    creating the controller, running simulation, and calculating fitness.

    Parameters
    ----------
    tree : DiGraph
        Robot morphology tree genotype.
    weights : np.ndarray
        Controller weights to use.
    hidden_layers : list[int]
        Neural network hidden layer sizes.
    activation : str
        Activation function name.
    simulation_duration : float
        Simulation time in seconds.
    baseline_time : float, optional
        Baseline time for displacement calculation, by default 1.0.
    seed : int, optional
        Random seed, by default 42.
    min_actuators : int, optional
        Minimum actuators required (returns 0.0 if fewer), by default 4.

    Returns
    -------
    float
        Fitness value (forward displacement in meters).
    """
    # Build robot
    model, data, world_spec = create_robot_model(tree)

    # Check actuator count
    if model.nu < min_actuators:
        return 0.0

    # Create and configure controller
    controller = create_controller(model, hidden_layers, activation, seed)
    controller.set_weights(weights)

    # Reset simulation
    mj.mj_resetData(model, data)

    # Setup tracker
    tracker = setup_tracker(world_spec, data)

    # Capture spawn height for penalty calculation
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

    # Run simulation
    simulate_with_controller(
        model, data, controller, tracker, simulation_duration
    )

    # Calculate fitness
    return calculate_displacement_fitness(tracker, baseline_time, model, spawn_height)
