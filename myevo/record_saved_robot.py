"""Command-line script to record videos from saved robots.

This script loads a saved robot from a directory containing:
- body.json: Robot morphology
- optimized_brain.npy (or initial_brain.npy): Neural network weights
- metadata.json (optional): Neural network architecture info
  - If present, uses saved architecture (controller_hidden_layers, controller_activation)
  - If missing, uses default [32, 16, 32] architecture

Usage:
    python record_saved_robot.py <path_to_individual_directory>

Examples:
    # macOS (default - uses mp4v codec)
    python record_saved_robot.py __data__/mu_lambda_tree_locomotion/individuals/individual_003df036

    # Windows (uses avc1/H.264 codec)
    python record_saved_robot.py __data__/mu_lambda_tree_locomotion/individuals/individual_003df036 --platform windows

    # Custom duration
    python record_saved_robot.py __data__/mu_lambda_tree_locomotion/individuals/individual_003df036 --duration 30 --platform macos
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
from custom_config import ALLOWED_FACES, ALLOWED_ROTATIONS
import ariel.body_phenotypes.robogen_lite.config as ariel_config
ariel_config.ALLOWED_FACES = ALLOWED_FACES
ariel_config.ALLOWED_ROTATIONS = ALLOWED_ROTATIONS

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.traceback import install

# ARIEL framework imports
from ariel.body_phenotypes.robogen_lite.decoders import load_graph_from_json

# Local imports
from record_robot_video import record_robot_video

# Setup
install(show_locals=False)
console = Console()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Record video of a saved robot with tracking camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input directory
    parser.add_argument(
        "individual_dir",
        type=Path,
        help="Path to individual directory containing body.json and brain weights",
    )

    # Video parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Simulation duration in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS (default: 30)",
    )

    # Platform parameters
    parser.add_argument(
        "--platform",
        type=str,
        default="macos",
        choices=["macos", "windows"],
        help="Video codec platform (default: macos)",
    )

    # Camera parameters
    parser.add_argument(
        "--distance",
        type=float,
        default=1.5,
        help="Camera distance (default: 1.5)",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=135,
        help="Camera azimuth angle (default: 135)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-30,
        help="Camera elevation angle (default: -30)",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="robot1_core",
        help="Body name to track (default: robot1_core)",
    )

    args = parser.parse_args()

    # Get individual directory
    individual_dir = args.individual_dir

    if not individual_dir.exists():
        console.print(f"[red]Directory not found:[/red] {individual_dir}")
        return

    if not individual_dir.is_dir():
        console.print(f"[red]Not a directory:[/red] {individual_dir}")
        return

    # Load body and brain
    console.print(f"[cyan]Loading from:[/cyan] {individual_dir}")

    body_file = individual_dir / "body.json"
    if not body_file.exists():
        console.print(f"[red]Body file not found:[/red] {body_file}")
        return

    body_graph = load_graph_from_json(body_file)

    # Load metadata if available
    metadata_file = individual_dir / "metadata.json"
    controller_hidden_layers = None
    controller_activation = None

    if metadata_file.exists():
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        controller_hidden_layers = metadata.get("controller_hidden_layers")
        controller_activation = metadata.get("controller_activation")
        console.print(f"[green]Loaded metadata:[/green] hidden_layers={controller_hidden_layers}, activation={controller_activation}")
    else:
        console.print(f"[yellow]No metadata.json found, will use default or command-line parameters[/yellow]")

    # Load brain (prefer optimized)
    brain_file = individual_dir / "optimized_brain.npy"
    if not brain_file.exists():
        brain_file = individual_dir / "initial_brain.npy"

    if not brain_file.exists():
        console.print(f"[red]No brain weights found in:[/red] {individual_dir}")
        return

    brain_weights = np.load(brain_file)
    console.print(f"[green]Loaded {brain_file.name}[/green] ({len(brain_weights)} parameters)")

    # Record video
    output_path = individual_dir / "videos"

    record_robot_video(
        body_graph=body_graph,
        brain_weights=brain_weights,
        output_path=output_path,
        duration=args.duration,
        controller_hidden_layers=controller_hidden_layers,  # From metadata if available
        controller_activation=controller_activation,  # From metadata if available
        video_width=args.width,
        video_height=args.height,
        video_fps=args.fps,
        platform=args.platform,
        tracking_distance=args.distance,
        tracking_azimuth=args.azimuth,
        tracking_elevation=args.elevation,
        geom_to_track=args.track,
    )


if __name__ == "__main__":
    main()
