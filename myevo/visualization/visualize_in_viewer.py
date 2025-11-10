"""Interactive MuJoCo viewer for robots with optional controller learning.

This script provides an easy way to visualize robots in the interactive MuJoCo
viewer. You can either:
1. Load a saved robot from a directory (body.json + brain weights)
2. Create a simple test robot and optionally learn a controller

Usage:
    # Load and visualize a saved robot
    python visualize_in_viewer.py --load __data__/mu_lambda_tree_locomotion/individuals/individual_003df036

    # Create a simple robot with random controller
    python visualize_in_viewer.py --create-simple

    # Create a simple robot and learn controller with CMA-ES
    python visualize_in_viewer.py --create-simple --learn --cmaes-budget 100

    # Load saved robot and learn a new controller
    python visualize_in_viewer.py --load <path> --learn --cmaes-budget 200
"""

# Enable modern type annotations
from __future__ import annotations

# Standard library
import argparse
import sys
from pathlib import Path

# Apply custom config BEFORE importing ARIEL modules
CWD = Path.cwd()
sys.path.insert(0, str(CWD / "myevo"))
from myevo.config.custom_config import ALLOWED_FACES, ALLOWED_ROTATIONS
import ariel.body_phenotypes.robogen_lite.config as ariel_config
ariel_config.ALLOWED_FACES = ALLOWED_FACES
ariel_config.ALLOWED_ROTATIONS = ALLOWED_ROTATIONS

# Third-party libraries
import mujoco as mj
import numpy as np
from mujoco import viewer
from rich.console import Console
from rich.traceback import install

# ARIEL framework imports
from ariel.body_phenotypes.robogen_lite.decoders import load_graph_from_json
from ariel.simulation.controllers.controller import Controller

# Local imports
from myevo.simulation.simulation_utils import create_robot_model, create_controller, setup_tracker
from myevo.controllers.controller_optimizer import optimize_controller_cmaes

# Setup
install(show_locals=False)
console = Console()


def create_simple_robot():
    """Create a simple test robot morphology.

    Returns
    -------
    DiGraph
        A simple robot graph with a few modules for testing.
    """
    import networkx as nx
    from ariel.body_phenotypes.robogen_lite.modules import (
        ActiveHingeModule,
        BrickModule,
    )

    # Create a simple snake-like robot
    graph = nx.DiGraph()

    # Add core
    graph.add_node(0, module_type="core", rotation=0)

    # Add a few active hinges and bricks
    graph.add_node(1, module_type="active_hinge", rotation=0)
    graph.add_edge(0, 1, face=0)  # Front face

    graph.add_node(2, module_type="brick", rotation=0)
    graph.add_edge(1, 2, face=1)

    graph.add_node(3, module_type="active_hinge", rotation=1)
    graph.add_edge(2, 3, face=1)

    graph.add_node(4, module_type="brick", rotation=0)
    graph.add_edge(3, 4, face=1)

    console.print("[green]Created simple robot with 5 modules (1 core, 2 hinges, 2 bricks)[/green]")
    return graph


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize robot in interactive MuJoCo viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Robot source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--load",
        type=Path,
        help="Path to individual directory containing body.json and brain weights",
    )
    source_group.add_argument(
        "--create-simple",
        action="store_true",
        help="Create a simple test robot",
    )

    # Controller options
    parser.add_argument(
        "--learn",
        action="store_true",
        help="Learn controller using CMA-ES before visualizing",
    )
    parser.add_argument(
        "--cmaes-budget",
        type=int,
        default=100,
        help="CMA-ES optimization budget (default: 100)",
    )
    parser.add_argument(
        "--cmaes-population",
        type=int,
        default=10,
        help="CMA-ES population size (default: 10)",
    )

    # Controller architecture
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[32, 16, 32],
        help="Hidden layer sizes (default: 32 16 32)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        choices=["tanh", "relu", "sigmoid", "elu"],
        help="Activation function (default: tanh)",
    )

    # Simulation parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=35.0,
        help="Simulation duration in seconds for learning (default: 10.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # ================================================================
    # 1. Load or create robot
    # ================================================================
    if args.load:
        individual_dir = args.load

        if not individual_dir.exists():
            console.print(f"[red]Directory not found:[/red] {individual_dir}")
            return

        if not individual_dir.is_dir():
            console.print(f"[red]Not a directory:[/red] {individual_dir}")
            return

        console.print(f"[cyan]Loading robot from:[/cyan] {individual_dir}")

        body_file = individual_dir / "body.json"
        if not body_file.exists():
            console.print(f"[red]Body file not found:[/red] {body_file}")
            return

        body_graph = load_graph_from_json(body_file)
        console.print(f"[green]Loaded body from {body_file.name}[/green]")

        # Load brain if available (unless we're learning a new one)
        brain_weights = None
        if not args.learn:
            # Try to load optimized brain (prefer .npz format, fallback to .npy)
            brain_file = individual_dir / "optimized_brain.npz"
            if not brain_file.exists():
                brain_file = individual_dir / "optimized_brain.npy"
            if not brain_file.exists():
                brain_file = individual_dir / "initial_brain.npz"
            if not brain_file.exists():
                brain_file = individual_dir / "initial_brain.npy"

            if brain_file.exists():
                # Load weights based on file format
                if brain_file.suffix == '.npz':
                    brain_weights = np.load(brain_file)['weights']  # Extract 'weights' key from npz
                else:
                    brain_weights = np.load(brain_file)
                console.print(f"[green]Loaded brain from {brain_file.name}[/green] ({len(brain_weights)} parameters)")
            else:
                console.print(f"[yellow]No brain weights found, will use random weights[/yellow]")

    else:  # create-simple
        console.print("[cyan]Creating simple test robot...[/cyan]")
        body_graph = create_simple_robot()
        brain_weights = None

    # ================================================================
    # 2. Build MuJoCo model
    # ================================================================
    console.print("\n[cyan]Building MuJoCo model...[/cyan]")
    model, data, world_spec = create_robot_model(body_graph)

    # Override timestep to 0.02 for testing
    model.opt.timestep = 0.002

    if model.nu < 1:
        console.print(f"[red]Robot has no actuators! Cannot visualize with controller.[/red]")
        console.print(f"[yellow]Launching viewer anyway (passive robot)...[/yellow]")
        viewer.launch(model=model, data=data)
        return

    console.print(f"[green]Robot built: {model.nu} actuators, {model.nbody} bodies[/green]")

    # ================================================================
    # 3. Create controller
    # ================================================================
    console.print("\n[cyan]Setting up controller...[/cyan]")
    controller = create_controller(
        model=model,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        seed=args.seed,
    )

    # ================================================================
    # 4. Get controller weights
    # ================================================================
    if args.learn:
        console.print(f"\n[bold cyan]Learning controller with CMA-ES...[/bold cyan]")
        console.print(f"Budget: {args.cmaes_budget}, Population: {args.cmaes_population}")
        console.print(f"Duration: {args.duration}s")

        brain_weights, best_fitness, fitness_history, _, _ = optimize_controller_cmaes(
            model=model,
            world_spec=world_spec,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            simulation_duration=args.duration,
            cmaes_budget=args.cmaes_budget,
            cmaes_population_size=args.cmaes_population,
            sigma_init=1.0,
            initial_weights=brain_weights,  # Use loaded weights as starting point if available
            maximize=True,
            baseline_time=5.0,
            seed=args.seed,
        )

        console.print(f"\n[green]Learning complete! Best fitness: {best_fitness:.4f}[/green]")
        console.print(f"[green]Fitness improved from {fitness_history[0]:.4f} to {fitness_history[-1]:.4f}[/green]")

    elif brain_weights is None:
        # Generate random weights
        console.print("[yellow]Using random controller weights[/yellow]")
        rng = np.random.default_rng(args.seed)
        num_weights = controller.get_num_weights()
        brain_weights = rng.uniform(-1.0, 1.0, num_weights)

    # Set weights
    controller.set_weights(brain_weights)
    console.print(f"[green]Controller ready with {len(brain_weights)} parameters[/green]")

    # ================================================================
    # 5. Run fitness evaluation with debug printing
    # ================================================================
    console.print("\n[bold cyan]Running fitness evaluation with debug info...[/bold cyan]")

    # Reset simulation
    mj.mj_resetData(model, data)
    tracker = setup_tracker(world_spec, data)

    # Phase 1: Settling phase (5 seconds, no control)
    console.print("\n[yellow]Phase 1: Settling phase (5 seconds, no control)[/yellow]")

    # Helper function to get core position from tracker
    def get_core_position(data, model):
        """Get core position by searching for geom with 'core' in name."""
        for i in range(model.ngeom):
            geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
            if geom_name and "core" in geom_name.lower():
                return data.geom(i).xpos.copy()
        raise ValueError("Could not find core geom")

    # Helper function to detect rotor-ground contacts
    def get_rotors_in_contact(model, data):
        """Find which rotor geoms are currently in contact with the ground.

        Returns
        -------
        set[str]
            Set of rotor geom names currently in contact with floor.
        """
        rotors_touching = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, contact.geom2)

            # Check if one geom is a rotor and the other is the floor
            if geom1_name and geom2_name:
                if "rotor" in geom1_name and "floor" in geom2_name:
                    rotors_touching.add(geom1_name)
                elif "rotor" in geom2_name and "floor" in geom1_name:
                    rotors_touching.add(geom2_name)

        return rotors_touching

    # Compute forward kinematics to update positions
    mj.mj_forward(model, data)

    # Get initial position (right after reset) - this is the spawn height
    initial_spawn_pos = get_core_position(data, model)
    spawn_height = initial_spawn_pos[2]  # Store spawn height for penalty calculation
    console.print(f"[cyan]Initial spawn position:[/cyan] x={initial_spawn_pos[0]:.4f}m, y={initial_spawn_pos[1]:.4f}m, z={spawn_height:.4f}m (height)")

    settling_duration = 5.0
    mj.set_mjcb_control(None)
    from ariel.utils.runners import simple_runner
    simple_runner(model, data, duration=settling_duration)

    # Get position after settling
    settling_pos = get_core_position(data, model)
    console.print(f"[cyan]Position after settling:[/cyan] x={settling_pos[0]:.4f}m, y={settling_pos[1]:.4f}m, z={settling_pos[2]:.4f}m (height)")

    # Reset tracker to start measuring from here
    tracker.reset()

    # Phase 2: Control phase
    control_duration = args.duration - settling_duration
    console.print(f"\n[yellow]Phase 2: Control phase ({control_duration:.1f} seconds)[/yellow]")

    # Store the baseline position (same as settling_pos, this is where control phase starts)
    baseline_pos = settling_pos.copy()
    console.print(f"[cyan]Starting position (baseline):[/cyan] x={baseline_pos[0]:.4f}m, y={baseline_pos[1]:.4f}m, z={baseline_pos[2]:.4f}m")

    # Create controller wrapper
    ctrl = Controller(
        controller_callback_function=controller,
        tracker=tracker,
    )
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Run controlled simulation with contact tracking
    # Track unique contact events (when a rotor makes new contact with ground)
    previous_rotors_in_contact = set()
    unique_contact_events = 0
    contact_event_log = []  # Store details of each contact event

    num_steps = int(control_duration / model.opt.timestep)
    for step in range(num_steps):
        mj.mj_step(model, data)

        # Check current contacts
        current_rotors_in_contact = get_rotors_in_contact(model, data)

        # Count new contacts (rotors that are touching now but weren't before)
        new_contacts = current_rotors_in_contact - previous_rotors_in_contact

        # Log each new contact event with timestamp
        if new_contacts:
            sim_time = data.time
            for rotor_name in new_contacts:
                contact_event_log.append((sim_time, rotor_name))
                unique_contact_events += 1

        # Also track when contacts are lost (for debugging)
        lost_contacts = previous_rotors_in_contact - current_rotors_in_contact
        if lost_contacts and len(contact_event_log) < 50:  # Only log first 50 to avoid spam
            sim_time = data.time
            for rotor_name in lost_contacts:
                contact_event_log.append((sim_time, f"LOST: {rotor_name}"))

        # Update tracking set for next iteration
        previous_rotors_in_contact = current_rotors_in_contact

    # Get final position
    final_pos = get_core_position(data, model)
    console.print(f"[cyan]Final position:[/cyan] x={final_pos[0]:.4f}m, y={final_pos[1]:.4f}m, z={final_pos[2]:.4f}m")

    # Print contact statistics
    console.print(f"\n[bold cyan]Contact Statistics:[/bold cyan]")
    console.print(f"[cyan]Hinge rotor ground contacts:[/cyan] {unique_contact_events} unique contact events")

    # Count how many unique rotors made contact
    unique_rotors = set()
    contact_times = []  # Store times of new contacts only
    for time, rotor in contact_event_log:
        if "LOST:" not in str(rotor):
            unique_rotors.add(rotor)
            contact_times.append(time)

    console.print(f"[cyan]Unique rotors that made contact:[/cyan] {len(unique_rotors)}")
    if unique_rotors:
        console.print(f"[dim]  Rotor names: {', '.join(sorted(unique_rotors))}[/dim]")

    # Calculate average time between contacts
    if len(contact_times) > 1:
        time_diffs = [contact_times[i+1] - contact_times[i] for i in range(len(contact_times)-1)]
        avg_time_between = sum(time_diffs) / len(time_diffs)
        console.print(f"[cyan]Average time between contacts:[/cyan] {avg_time_between*1000:.1f} ms ({1.0/avg_time_between:.1f} contacts/sec)")

    # Calculate fitness components using the baseline position
    initial_pos = baseline_pos
    x_displacement = final_pos[0] - initial_pos[0]

    console.print(f"\n[bold yellow]Fitness Calculation:[/bold yellow]")
    console.print(f"[cyan]X-displacement (forward):[/cyan] {x_displacement:.4f}m")
    console.print(f"[cyan]Spawn height (for penalty):[/cyan] {spawn_height:.4f}m")
    console.print(f"[cyan]Settled height:[/cyan] {baseline_pos[2]:.4f}m")

    # Apply height penalty based on spawn height (morphology height)
    if spawn_height > 0.21:
        height_penalty = spawn_height
        fitness = x_displacement - height_penalty
        console.print(f"[red]Height penalty applied:[/red] -{height_penalty:.4f}m (spawn height > 0.21m)")
        console.print(f"[bold green]Final fitness:[/bold green] {fitness:.4f}m = {x_displacement:.4f}m - {height_penalty:.4f}m")
    else:
        fitness = x_displacement
        console.print(f"[green]No height penalty[/green] (spawn height <= 0.21m)")
        console.print(f"[bold green]Final fitness:[/bold green] {fitness:.4f}m")

    # ================================================================
    # 6. Reset for viewer
    # ================================================================
    console.print("\n[cyan]Resetting simulation for interactive viewer...[/cyan]")
    mj.mj_resetData(model, data)
    tracker = setup_tracker(world_spec, data)

    # Create controller wrapper
    # With dt=0.002 and time_steps_per_ctrl_step=100 (default):
    # Controller updates every 100 * 0.002 = 0.2 seconds
    ctrl = Controller(
        controller_callback_function=controller,
        tracker=tracker,
    )
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # ================================================================
    # 7. Launch viewer
    # ================================================================
    console.print("\n[bold green]Launching interactive MuJoCo viewer...[/bold green]")
    console.print("[yellow]Controls:[/yellow]")
    console.print("  - Space: Pause/Resume")
    console.print("  - Right click + drag: Rotate camera")
    console.print("  - Scroll: Zoom")
    console.print("  - Ctrl+R: Reset simulation")
    console.print("[yellow]Close the viewer window when done...[/yellow]\n")

    viewer.launch(model=model, data=data)

    console.print("\n[green]Viewer closed[/green]")


if __name__ == "__main__":
    main()
