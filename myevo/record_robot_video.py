"""Record a video of a robot with tracking camera.

This module provides a simple function to record videos of robots given their
body morphology and brain weights. Can be easily integrated into the evolution
framework to save videos of the best individuals.
"""

# Enable modern type annotations
from __future__ import annotations

# Standard library
import math
from pathlib import Path

# Third-party libraries
import mujoco as mj
import numpy as np
from networkx import DiGraph
from rich.console import Console

# ARIEL framework imports
from ariel.utils.renderers import tracking_video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.controllers.controller import Controller

# Local imports
from simulation_utils import (
    create_robot_model,
    create_controller,
    setup_tracker,
)
from macos_video_recorder import MacOSVideoRecorder

console = Console()


def record_robot_video(
    body_graph: DiGraph,
    brain_weights: np.ndarray,
    output_path: Path | str,
    duration: float = 15.0,
    settling_duration: float = 5.0,
    controller_hidden_layers: list[int] | None = None,
    controller_activation: str = "tanh",
    tracking_distance: float = 1.5,
    tracking_azimuth: float = 135,
    tracking_elevation: float = -30,
    geom_to_track: str = "robot1_core",
    video_width: int = 1280,
    video_height: int = 720,
    video_fps: int = 30,
    seed: int = 42,
    platform: str = "macos",
    verbose: bool = True,
) -> Path:
    """Record a video of a robot with tracking camera.

    The video includes a two-phase simulation:
    - Phase 1 (settling): Robot settles passively for settling_duration
    - Phase 2 (controlled): Controller is active for (duration - settling_duration)

    Parameters
    ----------
    body_graph : DiGraph
        The robot's body morphology graph.
    brain_weights : np.ndarray
        The robot's neural network controller weights.
    output_path : Path | str
        Directory to save the video, or full path including filename.
        If directory, video will be named "robot_video_<timestamp>.mp4".
    duration : float, optional
        Total simulation duration in seconds, by default 15.0.
    settling_duration : float, optional
        Duration of passive settling phase in seconds, by default 5.0.
    controller_hidden_layers : list[int] | None, optional
        Neural network hidden layers, by default [32, 16, 32].
    controller_activation : str, optional
        Activation function, by default "tanh".
    tracking_distance : float, optional
        Camera distance from robot, by default 1.5.
    tracking_azimuth : float, optional
        Camera azimuth angle in degrees, by default 135.
    tracking_elevation : float, optional
        Camera elevation angle in degrees, by default -30.
    geom_to_track : str, optional
        Name of the body to track, by default "robot1_core".
    video_width : int, optional
        Video width in pixels, by default 1280.
    video_height : int, optional
        Video height in pixels, by default 720.
    video_fps : int, optional
        Video frames per second, by default 30.
    seed : int, optional
        Random seed, by default 42.
    platform : str, optional
        Platform codec choice: "macos" (mp4v) or "windows" (avc1), by default "macos".
    verbose : bool, optional
        Print progress information, by default True.

    Returns
    -------
    Path
        Path to the output directory containing the video.

    Examples
    --------
    >>> # Record video of best individual during evolution
    >>> body = best_individual.genotype.tree
    >>> weights = weight_manager.get_weights(id(body))
    >>> record_robot_video(
    ...     body_graph=body,
    ...     brain_weights=weights,
    ...     output_path="videos/generation_10",
    ...     duration=20.0,
    ... )
    """
    if controller_hidden_layers is None:
        controller_hidden_layers = [32, 16, 32]

    output_path = Path(output_path)

    # If output_path is a directory, create it
    if output_path.suffix == "":
        output_dir = output_path
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        # If it's a file path, use its parent as the directory
        output_dir = output_path.parent
        output_dir.mkdir(exist_ok=True, parents=True)

    if verbose:
        console.print(f"\n[cyan]Building MuJoCo model...[/cyan]")

    # Build robot model
    model, data, world_spec = create_robot_model(body_graph)

    # Check if robot has enough actuators
    if model.nu < 1:
        if verbose:
            console.print(f"[red]Robot has no actuators! Cannot create video.[/red]")
        return None

    if verbose:
        console.print(f"[green]Robot built: {model.nu} actuators, {model.nbody} bodies[/green]")

    # Create controller
    controller = create_controller(
        model=model,
        hidden_layers=controller_hidden_layers,
        activation=controller_activation,
        seed=seed,
    )

    # Set weights
    if len(brain_weights) == 0:
        if verbose:
            console.print(f"[yellow]Warning: Empty brain weights provided[/yellow]")
        return None

    controller.set_weights(brain_weights)

    # Reset simulation
    mj.mj_resetData(model, data)

    # Setup tracker
    tracker = setup_tracker(world_spec, data)

    # Create video recorder based on platform
    if platform.lower() == "macos":
        video_recorder = MacOSVideoRecorder(
            file_name="robot_video",
            output_folder=str(output_dir),
            width=video_width,
            height=video_height,
            fps=video_fps,
        )
        codec_info = "mp4v (macOS-compatible)"
    else:  # windows or other
        video_recorder = VideoRecorder(
            file_name="robot_video",
            output_folder=str(output_dir),
            width=video_width,
            height=video_height,
            fps=video_fps,
        )
        codec_info = "avc1 (H.264)"

    if verbose:
        console.print(f"\n[bold cyan]Recording video...[/bold cyan]")
        console.print(f"Total duration: {duration}s ({settling_duration}s settling + {duration - settling_duration}s controlled)")
        console.print(f"Resolution: {video_width}x{video_height} @ {video_fps}fps")
        console.print(f"Codec: {codec_info}")
        console.print(f"Tracking: {geom_to_track} (distance={tracking_distance}, azimuth={tracking_azimuth}, elevation={tracking_elevation})")

    # Find the core body ID for tracking
    try:
        core_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, geom_to_track)
    except ValueError:
        # Try to find any body with "core" in the name
        core_body_id = None
        for i in range(model.nbody):
            body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
            if body_name and "core" in body_name:
                core_body_id = i
                break

    # Enable joint visualization
    scene_option = mj.MjvOption()
    scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True

    # Calculate steps per frame
    options = mj.MjOption()
    steps_per_frame = 1.0 / (options.timestep * video_fps)

    # Render both phases into the same video
    with mj.Renderer(model, width=video_width, height=video_height) as renderer:
        # Phase 1: Record passive settling (no control)
        if verbose:
            console.print(f"[cyan]Phase 1: Passive settling ({settling_duration}s)...[/cyan]")

        mj.set_mjcb_control(None)
        phase1_end_time = data.time + settling_duration

        while data.time < phase1_end_time:
            # Step simulation
            mj.mj_step(model, data, nstep=math.floor(steps_per_frame))

            # Update camera tracking if core body found
            if core_body_id is not None:
                camera = mj.MjvCamera()
                camera.type = mj.mjtCamera.mjCAMERA_TRACKING
                camera.trackbodyid = core_body_id
                camera.distance = tracking_distance
                camera.azimuth = tracking_azimuth
                camera.elevation = tracking_elevation
                renderer.update_scene(data, scene_option=scene_option, camera=camera)
            else:
                renderer.update_scene(data, scene_option=scene_option)

            # Save frame
            video_recorder.write(frame=renderer.render())

        # Phase 2: Record with active controller
        if verbose:
            console.print(f"[cyan]Phase 2: Active control ({duration - settling_duration}s)...[/cyan]")

        # Create controller wrapper
        ctrl = Controller(
            controller_callback_function=controller,
            tracker=tracker,
        )
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        phase2_end_time = data.time + (duration - settling_duration)

        while data.time < phase2_end_time:
            # Step simulation
            mj.mj_step(model, data, nstep=math.floor(steps_per_frame))

            # Update camera tracking if core body found
            if core_body_id is not None:
                camera = mj.MjvCamera()
                camera.type = mj.mjtCamera.mjCAMERA_TRACKING
                camera.trackbodyid = core_body_id
                camera.distance = tracking_distance
                camera.azimuth = tracking_azimuth
                camera.elevation = tracking_elevation
                renderer.update_scene(data, scene_option=scene_option, camera=camera)
            else:
                renderer.update_scene(data, scene_option=scene_option)

            # Save frame
            video_recorder.write(frame=renderer.render())

    # Finalize video only once at the end
    video_recorder.release()

    if verbose:
        console.print(f"\n[bold green]Video saved to:[/bold green] {output_dir}")

    return output_dir
