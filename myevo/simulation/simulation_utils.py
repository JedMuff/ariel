"""Simulation utilities for robot model construction and setup.

This module provides functions for building MuJoCo models from robot morphologies
and setting up simulation components (controllers, trackers).

Note: Fitness evaluation functions have been moved to myevo.measures.locomotion_fitness
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mujoco as mj
from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.tracker import Tracker

if TYPE_CHECKING:
    from myevo.controllers.neural_network_controller import FlexibleNeuralNetworkController


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
    from myevo.controllers.neural_network_controller import FlexibleNeuralNetworkController

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
