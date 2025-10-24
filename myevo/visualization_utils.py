"""Visualization utilities for evolutionary robotics.

This module provides functions for plotting fitness histories and visualizing
evolved robot morphologies in various modes (viewer, video, static frame).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from mujoco import viewer
from networkx import DiGraph
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.decoders import draw_graph, save_graph_as_json
from ariel.simulation.controllers.controller import Controller
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

from simulation_utils import create_controller, create_robot_model, setup_tracker

console = Console()


def plot_fitness_history(
    population: list[Any],
    maximize: bool,
    save_path: Path | str,
    title: str = "Fitness over Generations",
    ylabel: str = "Fitness",
) -> None:
    """Plot fitness progression over generations.

    Creates a plot showing best fitness, average fitness, and standard deviation
    across all generations. The plot is saved to the specified path.

    Parameters
    ----------
    population : list[Any]
        All individuals from all generations (must have .time_of_birth and .fitness).
    maximize : bool
        Whether fitness is being maximized (True) or minimized (False).
    save_path : Path | str
        Path to save the plot image.
    title : str, optional
        Plot title, by default "Fitness over Generations".
    ylabel : str, optional
        Y-axis label, by default "Fitness".
    """
    max_gen = max(ind.time_of_birth for ind in population)
    best_fitness_history = []
    avr_fitness_history = []
    std_fitness_history = []

    for gen in range(max_gen + 1):
        gen_inds = [ind for ind in population if ind.time_of_birth == gen]
        if gen_inds:
            if maximize:
                best_fitness = max(ind.fitness or float('-inf') for ind in gen_inds)
            else:
                best_fitness = min(ind.fitness or float('inf') for ind in gen_inds)
            best_fitness_history.append(best_fitness)

            avr_fitness_history.append(
                np.mean([ind.fitness for ind in gen_inds])
            )
            std_fitness_history.append(
                np.std([ind.fitness for ind in gen_inds])
            )

    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, linewidth=2, color="#2E86AB", label="Best Fitness")
    plt.plot(avr_fitness_history, linewidth=2, color="#F6C85F", label="Avg Fitness")
    plt.fill_between(
        range(len(avr_fitness_history)),
        np.array(avr_fitness_history) - np.array(std_fitness_history),
        np.array(avr_fitness_history) + np.array(std_fitness_history),
        color="#F6C85F",
        alpha=0.3,
        label="Std Dev",
    )

    plt.xlabel("Generation", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    console.print(f"\n[green]Fitness plot saved to:[/green] {save_path}")
    plt.close()


def visualize_best_morphology(
    population: list[Any],
    maximize: bool,
    mode: str,
    controller_params: dict[str, Any],
    save_dir: Path | str,
    simulation_duration: float = 10.0,
    use_stored_weights: bool = False,
    weight_manager: Any | None = None,
) -> None:
    """Visualize the best evolved morphology.

    Finds the best individual from the population and visualizes it in the
    specified mode (interactive viewer, video, or static frame).

    Parameters
    ----------
    population : list[Any]
        All individuals from evolution (must have .fitness and .genotype).
    maximize : bool
        Whether fitness is being maximized.
    mode : str
        Visualization mode: "viewer", "video", or "frame".
    controller_params : dict[str, Any]
        Controller configuration with keys:
        - hidden_layers: list[int]
        - activation: str
        - sigma_init: float
        - seed: int
        - use_cmaes: bool
        - cmaes_budget: int (if use_cmaes)
        - cmaes_population_size: int (if use_cmaes)
    save_dir : Path | str
        Directory to save visualization outputs.
    simulation_duration : float, optional
        Duration for simulation/video, by default 10.0 seconds.
    use_stored_weights : bool, optional
        Whether to use stored learned weights, by default False.
    weight_manager : Any | None, optional
        ParentWeightManager instance (if using stored weights), by default None.
    """
    from ariel.ec import TreeGenotype

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Find best individual
    if maximize:
        best_individual = max(population, key=lambda ind: ind.fitness or float('-inf'))
    else:
        best_individual = min(population, key=lambda ind: ind.fitness or float('inf'))

    console.print(
        f"\n[bold cyan]Best Morphology (Fitness: {best_individual.fitness:.4f})[/bold cyan]"
    )

    # Extract tree
    best_genotype = best_individual.genotype
    best_tree = best_genotype.tree if isinstance(best_genotype, TreeGenotype) else best_genotype

    # Save graph visualization
    save_graph_as_json(best_tree, save_dir / "best_tree.json")
    draw_graph(
        best_tree,
        title="Best Evolved Morphology",
        save_file=save_dir / "best_tree.png",
    )

    console.print(f"[green]Tree visualization saved to:[/green] {save_dir / 'best_tree.png'}")

    # Build and simulate robot
    model, data, world_spec = create_robot_model(best_tree)

    # Create controller
    num_actuators = model.nu
    controller = create_controller(
        model=model,
        hidden_layers=controller_params["hidden_layers"],
        activation=controller_params["activation"],
        seed=controller_params["seed"],
    )

    # Get controller weights
    if use_stored_weights and weight_manager is not None and weight_manager.has_weights(id(best_tree)):
        console.print("[cyan]Using learned weights from evolution...[/cyan]")
        weights = weight_manager.get_weights(id(best_tree))
        controller.set_weights(weights)
    elif controller_params.get("use_cmaes", False):
        console.print("[cyan]Optimizing controller with CMA-ES for visualization...[/cyan]")
        from controller_optimizer import optimize_controller_cmaes

        weights, _, _ = optimize_controller_cmaes(
            model=model,
            world_spec=world_spec,
            hidden_layers=controller_params["hidden_layers"],
            activation=controller_params["activation"],
            simulation_duration=simulation_duration,
            cmaes_budget=controller_params["cmaes_budget"],
            cmaes_population_size=controller_params["cmaes_population_size"],
            sigma_init=controller_params["sigma_init"],
            seed=controller_params["seed"],
        )
        controller.set_weights(weights)
    else:
        # Use random weights
        rng = np.random.default_rng(controller_params["seed"])
        num_weights = controller.get_num_weights()
        weights = rng.uniform(-controller_params["sigma_init"], controller_params["sigma_init"], num_weights)
        controller.set_weights(weights)

    # Reset simulation
    mj.mj_resetData(model, data)

    # Setup tracker
    tracker = setup_tracker(world_spec, data)

    # Create controller wrapper
    ctrl = Controller(
        controller_callback_function=controller,
        tracker=tracker,
    )
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Visualize based on mode
    if mode == "frame":
        save_path = save_dir / "best_robot.png"
        single_frame_renderer(model, data, save=True, save_path=str(save_path))
        console.print(f"[green]Frame saved to:[/green] {save_path}")

    elif mode == "video":
        video_recorder = VideoRecorder(output_folder=str(save_dir / "videos"))
        cam_quat = np.zeros(4)
        mj.mju_euler2Quat(cam_quat, np.deg2rad([20, 0, 0]), "XYZ")

        video_renderer(
            model,
            data,
            duration=simulation_duration,
            video_recorder=video_recorder,
            cam_fovy=45,
            cam_pos=[1.5, -1.5, 0.8],
            cam_quat=cam_quat,
        )
        console.print(f"[green]Video saved to:[/green] {save_dir / 'videos'}")

    elif mode == "viewer":
        console.print("[yellow]Close the viewer window when done...[/yellow]")
        viewer.launch(model=model, data=data)
        console.print("[green]Viewer closed[/green]")


def plot_diversity_metrics(
    population: list[Any],
    save_path: Path | str,
    title: str = "Population Diversity over Generations",
) -> None:
    """Plot diversity metrics over generations.

    Shows population diversity using fitness variance and range as proxies.

    Parameters
    ----------
    population : list[Any]
        All individuals from all generations.
    save_path : Path | str
        Path to save the plot.
    title : str, optional
        Plot title, by default "Population Diversity over Generations".
    """
    max_gen = max(ind.time_of_birth for ind in population)
    fitness_variance = []
    fitness_range = []

    for gen in range(max_gen + 1):
        gen_inds = [ind for ind in population if ind.time_of_birth == gen]
        if gen_inds:
            fitnesses = [ind.fitness or 0.0 for ind in gen_inds]
            fitness_variance.append(np.var(fitnesses))
            fitness_range.append(max(fitnesses) - min(fitnesses))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Variance plot
    ax1.plot(fitness_variance, linewidth=2, color="#2E86AB")
    ax1.set_ylabel("Fitness Variance", fontsize=12)
    ax1.set_title("Fitness Variance (Diversity Indicator)", fontsize=12)
    ax1.grid(True)

    # Range plot
    ax2.plot(fitness_range, linewidth=2, color="#F6C85F")
    ax2.set_xlabel("Generation", fontsize=12)
    ax2.set_ylabel("Fitness Range", fontsize=12)
    ax2.set_title("Fitness Range (max - min)", fontsize=12)
    ax2.grid(True)

    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    console.print(f"\n[green]Diversity plot saved to:[/green] {save_path}")
    plt.close()


def save_best_individuals(
    population: list[Any],
    maximize: bool,
    save_dir: Path | str,
    top_n: int = 10,
) -> None:
    """Save visualizations of the top N individuals.

    Parameters
    ----------
    population : list[Any]
        All individuals from evolution.
    maximize : bool
        Whether fitness is being maximized.
    save_dir : Path | str
        Directory to save individual graphs.
    top_n : int, optional
        Number of top individuals to save, by default 10.
    """
    from ariel.ec import TreeGenotype

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Sort by fitness
    sorted_pop = sorted(
        population,
        key=lambda ind: ind.fitness or (float('-inf') if maximize else float('inf')),
        reverse=maximize,
    )

    console.print(f"\n[cyan]Saving top {top_n} individuals...[/cyan]")

    for i, individual in enumerate(sorted_pop[:top_n]):
        tree = individual.genotype.tree if isinstance(individual.genotype, TreeGenotype) else individual.genotype

        # Save graph
        save_path = save_dir / f"rank_{i+1}_fitness_{individual.fitness:.4f}.png"
        draw_graph(
            tree,
            title=f"Rank {i+1} (Fitness: {individual.fitness:.4f})",
            save_file=save_path,
        )

    console.print(f"[green]Saved {top_n} individuals to:[/green] {save_dir}")
