"""Visualize morphological novelty search results.

This script provides comprehensive visualization of evolution results including:
- Novelty progression over generations
- Morphological descriptor trends
- Final generation morphology grid
- MuJoCo isometric renders of robots
"""

import sys
from pathlib import Path

# Setup paths
CWD = Path.cwd()
sys.path.insert(0, str(CWD / "myevo"))

# Monkey-patch config
from config_no_rotation import ALLOWED_ROTATIONS
import ariel.body_phenotypes.robogen_lite.config as ariel_config
ariel_config.ALLOWED_ROTATIONS = ALLOWED_ROTATIONS

import matplotlib.pyplot as plt
import numpy as np
import mujoco as mj
from myevo.core import TreeGenotype
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders import draw_graph
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer
from myevo.measures.novelty import PoissonArchive, euclidean_distance


def select_diverse_individuals(all_individuals, measures_dict, n_individuals=12,
                               min_distance=None):
    """Select the most spread-out individuals from the archive using Poisson disk sampling.

    Parameters
    ----------
    all_individuals : list
        All individuals from evolution
    measures_dict : dict
        Dictionary mapping tree id to morphological measures
    n_individuals : int
        Target number of individuals to select
    min_distance : float or None
        Minimum distance between selected individuals. If None, automatically determined.

    Returns
    -------
    list
        Selected diverse individuals
    """
    from myevo.core import TreeGenotype
    from morphological_measures import MorphologicalMeasures

    # Filter to only individuals with measures
    valid_individuals = []
    for ind in all_individuals:
        tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
        if id(tree) in measures_dict:
            valid_individuals.append(ind)

    if len(valid_individuals) <= n_individuals:
        return valid_individuals

    # Auto-determine min_distance if not provided
    # Start with a large distance and reduce until we get enough individuals
    if min_distance is None:
        # Calculate pairwise distances to estimate good min_distance
        import random
        sample_size = min(100, len(valid_individuals))
        sample = random.sample(valid_individuals, sample_size)

        distances = []
        for i, ind1 in enumerate(sample):
            tree1 = ind1.genotype.tree if isinstance(ind1.genotype, TreeGenotype) else ind1.genotype
            m1 = measures_dict[id(tree1)]
            for ind2 in sample[i+1:]:
                tree2 = ind2.genotype.tree if isinstance(ind2.genotype, TreeGenotype) else ind2.genotype
                m2 = measures_dict[id(tree2)]
                distances.append(euclidean_distance(m1, m2))

        # Use median distance as starting point
        median_dist = np.median(distances)
        min_distance = median_dist * 1.5  # Start higher to get spread

    # Try Poisson sampling with decreasing min_distance until we get enough
    best_selection = []
    current_min_dist = min_distance

    for attempt in range(10):
        # Create Poisson archive
        poisson = PoissonArchive(
            min_distance=current_min_dist,
            distance_fn=lambda ind1, ind2: euclidean_distance(
                measures_dict[id(ind1.genotype.tree if isinstance(ind1.genotype, TreeGenotype) else ind1.genotype)],
                measures_dict[id(ind2.genotype.tree if isinstance(ind2.genotype, TreeGenotype) else ind2.genotype)]
            ),
            use_cache=True
        )

        # Add individuals in random order to avoid bias
        import random
        shuffled = valid_individuals.copy()
        random.shuffle(shuffled)

        for ind in shuffled:
            poisson.add(ind)
            if len(poisson) >= n_individuals:
                break

        selected = poisson.get_archive()

        if len(selected) >= n_individuals:
            best_selection = selected[:n_individuals]
            break
        elif len(selected) > len(best_selection):
            best_selection = selected

        # Reduce min_distance for next attempt
        current_min_dist *= 0.8

    # If still not enough, just take the best we got plus random extras
    if len(best_selection) < n_individuals:
        remaining = [ind for ind in valid_individuals if ind not in best_selection]
        import random
        extra = random.sample(remaining, min(n_individuals - len(best_selection), len(remaining)))
        best_selection.extend(extra)

    return best_selection


def plot_novelty_history(all_individuals, output_dir):
    """Plot novelty scores over generations."""
    max_gen = max(ind.time_of_birth for ind in all_individuals)
    best_novelty = []
    avg_novelty = []

    for gen in range(max_gen + 1):
        gen_inds = [ind for ind in all_individuals if ind.time_of_birth == gen]
        if gen_inds:
            best_novelty.append(max(ind.fitness or 0.0 for ind in gen_inds))
            avg_novelty.append(np.mean([ind.fitness or 0.0 for ind in gen_inds]))

    plt.figure(figsize=(10, 6))
    plt.plot(best_novelty, linewidth=2, color="#2E86AB", label="Best Novelty")
    plt.plot(avg_novelty, linewidth=2, color="#F6C85F", label="Avg Novelty")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Novelty Score", fontsize=12)
    plt.title("Morphological Novelty Search: Novelty over Generations", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = output_dir / "novelty_history.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Novelty plot saved to: {plot_path}")


def plot_morphological_descriptors(all_individuals, measures_dict, output_dir):
    """Plot each morphological descriptor over generations."""
    max_gen = max(ind.time_of_birth for ind in all_individuals)

    # Descriptors to track (8 total)
    descriptors = ['branching', 'limbs', 'length_of_limbs', 'coverage', 'joints', 'proportion', 'symmetry', 'size']
    descriptor_data = {desc: {'mean': [], 'std': []} for desc in descriptors}

    for gen in range(max_gen + 1):
        gen_inds = [ind for ind in all_individuals if ind.time_of_birth == gen]

        for desc in descriptors:
            values = []
            for ind in gen_inds:
                tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
                tree_id = id(tree)
                if tree_id in measures_dict:
                    m = measures_dict[tree_id]
                    # Handle proportion specially (can be 2d or 3d)
                    if desc == 'proportion':
                        prop = m.proportion_2d if m.is_2d else m.proportion_3d
                        values.append(prop)
                    else:
                        values.append(getattr(m, desc))

            if values:
                descriptor_data[desc]['mean'].append(np.mean(values))
                descriptor_data[desc]['std'].append(np.std(values))

    # Create subplot grid (3 rows, 3 columns for 8 descriptors)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Morphological Descriptors over Generations", fontsize=16)

    for idx, desc in enumerate(descriptors):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        means = descriptor_data[desc]['mean']
        stds = descriptor_data[desc]['std']
        generations = range(len(means))

        ax.plot(generations, means, linewidth=2, color="#2E86AB", label="Mean")
        ax.fill_between(generations,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.3, color="#2E86AB", label="±1 Std Dev")

        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel(desc.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots (we have 8 descriptors in a 3x3 grid = 1 unused)
    for idx in range(len(descriptors), 9):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')

    plt.tight_layout()

    plot_path = output_dir / "morphological_descriptors.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Morphological descriptors plot saved to: {plot_path}")


def render_mujoco_isometric(tree, save_path, cam_distance=0.5,
                           elevation=35, azimuth=45, cam_fovy=30):
    """Render an isometric view of a robot in MuJoCo.

    Parameters
    ----------
    tree : DiGraph
        Robot tree structure
    save_path : Path
        Where to save the rendered image
    cam_distance : float
        Distance of camera from robot center in meters (default: 0.5)
        Smaller = closer/more zoom. Try 0.3-0.5 for small robots.
    elevation : float
        Camera elevation angle in degrees (default: 35 for isometric)
    azimuth : float
        Camera azimuth angle in degrees (default: 45 for isometric)
    cam_fovy : float
        Camera field of view (vertical) in degrees (default: 30)
        Smaller = more zoom (telephoto), larger = wider view

    Returns
    -------
    bool
        True if successful, False otherwise

    Notes
    -----
    Standard isometric view uses elevation=35.264° and azimuth=45°
    Adjust these angles to change the view:
    - elevation: 0° = side view, 90° = top view
    - azimuth: 0° = front, 90° = right side, 180° = back, 270° = left side
    """
    try:
        # Build robot (skip phenotype collision checking since genotype-level checking is done)
        robot_core = construct_mjspec_from_graph(tree, check_collisions=False)
        world = SimpleFlatWorld()
        robot_spawn_pos = np.array([0, 0, 10.0])
        world.spawn(robot_core.spec, position=robot_spawn_pos.tolist())

        # Compile model
        model = world.spec.compile()
        data = mj.MjData(model)

        # Calculate camera position from spherical coordinates
        # Robot is at [0, 0, 0.15], so camera should look at approximately that point
        lookat_point = robot_spawn_pos + np.array([0, 0, 0.3])

        # Convert angles to radians
        el_rad = np.deg2rad(elevation)
        az_rad = np.deg2rad(azimuth)

        # Spherical to Cartesian conversion (relative to lookat point)
        cam_pos = lookat_point + np.array([
            cam_distance * np.cos(el_rad) * np.cos(az_rad),
            cam_distance * np.cos(el_rad) * np.sin(az_rad),
            cam_distance * np.sin(el_rad)
        ])

        # Calculate "lookat" quaternion
        # Forward vector: from camera to lookat point
        forward = lookat_point - cam_pos
        forward = forward / np.linalg.norm(forward)

        # Up vector (world up)
        world_up = np.array([0, 0, 1])

        # Right vector: cross product of forward and up
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            # Handle singularity when looking straight up/down
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)

        # Recalculate up vector to be orthogonal
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Build rotation matrix (camera coordinate system)
        # MuJoCo camera convention: -Z forward, Y up, X right
        rotation_matrix = np.array([
            right,      # X axis
            up,         # Y axis
            -forward    # Z axis (negated because camera looks along -Z)
        ]).T

        # Convert rotation matrix to quaternion
        cam_quat = np.zeros(4)
        mj.mju_mat2Quat(cam_quat, rotation_matrix.flatten())

        # Render single frame
        single_frame_renderer(
            model, data,
            save=True,
            save_path=str(save_path),
            cam_pos=cam_pos.tolist(),
            cam_quat=cam_quat.tolist(),
            cam_fovy=cam_fovy,  # Field of view (smaller = more zoom)
            width=640,
            height=640,
        )

        return True

    except Exception as e:
        print(f"Error rendering MuJoCo: {e}")
        import traceback
        traceback.print_exc()
        return False


def plot_final_morphologies_grid(all_individuals, measures_dict, output_dir,
                                 n_display=12, font_size=10):
    """Plot grid of final generation morphologies with detailed info."""
    # Get final generation
    final_gen = max(ind.time_of_birth for ind in all_individuals)
    final_individuals = [ind for ind in all_individuals if ind.time_of_birth == final_gen]
    final_individuals.sort(key=lambda ind: ind.fitness or 0.0, reverse=True)

    n_display = min(n_display, len(final_individuals))
    rows = 3
    cols = 4

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle("Final Generation Morphologies (Sorted by Novelty)", fontsize=18, fontweight='bold')

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        if idx < n_display:
            ind = final_individuals[idx]
            tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
            save_path = output_dir / f"final_morph_{idx}.png"

            # Draw and save the graph
            graph_drawn = False
            try:
                draw_graph(tree, title=f"Robot {idx} - Novelty: {ind.fitness:.3f}",
                          save_file=save_path)
                plt.close('all')
                graph_drawn = True
            except ValueError:
                # NetworkX bug with simple graphs - use text instead
                num_nodes = len(tree.nodes())
                num_edges = len(tree.edges())
                ax.text(0.5, 0.6,
                       f"Robot {idx}\n{num_nodes} nodes, {num_edges} edges",
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=font_size + 2, fontweight='bold')
            except Exception as e:
                print(f"Error drawing graph {idx}: {e}")

            # Load and display image if drawn
            if graph_drawn and save_path.exists():
                from PIL import Image
                img = Image.open(save_path)
                ax.imshow(img)

            ax.axis('off')

            # Add detailed morphological info below
            tree_id = id(tree)
            if tree_id in measures_dict:
                m = measures_dict[tree_id]
                # Determine proportion
                prop = m.proportion_2d if m.is_2d else m.proportion_3d

                info_text = (
                    f"Novelty: {ind.fitness:.3f}\n"
                    f"Branching: {m.branching:.2f} | Limbs: {m.limbs:.2f}\n"
                    f"Length: {m.length_of_limbs:.2f} | Coverage: {m.coverage:.2f}\n"
                    f"Joints: {m.joints:.2f} | Proportion: {prop:.2f}\n"
                    f"Symmetry: {m.symmetry:.2f}"
                )

                ax.text(0.5, -0.02, info_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=font_size,
                       family='monospace', bbox=dict(boxstyle='round',
                       facecolor='wheat', alpha=0.3))
        else:
            ax.axis('off')

    plt.tight_layout()
    grid_path = output_dir / "final_morphologies_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Morphology grid saved to: {grid_path}")


def plot_mujoco_renders_grid(all_individuals, measures_dict, output_dir,
                             n_display=12, font_size=10, cam_distance=0.5,
                             elevation=35, azimuth=45, cam_fovy=30):
    """Plot grid of MuJoCo isometric renders of final generation.

    Parameters
    ----------
    cam_distance : float
        Distance of camera from robot in meters (default: 0.5)
    elevation : float
        Camera elevation angle in degrees (default: 35 for isometric)
    azimuth : float
        Camera azimuth angle in degrees (default: 45 for isometric)
    cam_fovy : float
        Camera field of view in degrees (default: 30, smaller = more zoom)
    """
    # Get final generation
    final_gen = max(ind.time_of_birth for ind in all_individuals)
    final_individuals = [ind for ind in all_individuals if ind.time_of_birth == final_gen]
    final_individuals.sort(key=lambda ind: ind.fitness or 0.0, reverse=True)

    n_display = min(n_display, len(final_individuals))
    rows = 3
    cols = 4

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle("Final Generation - MuJoCo Renders (Isometric View)", fontsize=18, fontweight='bold')

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        if idx < n_display:
            ind = final_individuals[idx]
            tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
            render_path = output_dir / f"mujoco_render_{idx}.png"

            # Render MuJoCo view
            success = render_mujoco_isometric(tree, render_path,
                                             cam_distance=cam_distance,
                                             elevation=elevation,
                                             azimuth=azimuth,
                                             cam_fovy=cam_fovy)

            if success and render_path.exists():
                from PIL import Image
                img = Image.open(render_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Render\nFailed",
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=font_size)

            ax.axis('off')

            # Add info
            tree_id = id(tree)
            if tree_id in measures_dict:
                m = measures_dict[tree_id]
                prop = m.proportion_2d if m.is_2d else m.proportion_3d

                info_text = (
                    f"Robot {idx} - Novelty: {ind.fitness:.3f}\n"
                    f"B:{m.branching:.2f} L:{m.limbs:.2f} C:{m.coverage:.2f} "
                    f"J:{m.joints:.2f} S:{m.symmetry:.2f}"
                )

                ax.text(0.5, -0.02, info_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=font_size,
                       family='monospace', bbox=dict(boxstyle='round',
                       facecolor='lightblue', alpha=0.3))
        else:
            ax.axis('off')

    plt.tight_layout()
    grid_path = output_dir / "mujoco_renders_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"MuJoCo renders grid saved to: {grid_path}")


def plot_diverse_representatives_grid(all_individuals, measures_dict, output_dir,
                                     n_display=12, font_size=10, cam_distance=0.5,
                                     elevation=35, azimuth=45, cam_fovy=30):
    """Plot grid of most diverse individuals across the entire search space.

    Uses Poisson disk sampling to select spread-out representatives.
    """
    print("Selecting diverse representatives...")
    diverse_individuals = select_diverse_individuals(all_individuals, measures_dict,
                                                     n_individuals=n_display)

    print(f"Selected {len(diverse_individuals)} diverse representatives")

    rows = 3
    cols = 4

    # Graph visualization grid
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle("Diverse Representatives Across Search Space (Poisson Sampled)",
                fontsize=18, fontweight='bold')

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        if idx < len(diverse_individuals):
            ind = diverse_individuals[idx]
            tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
            save_path = output_dir / f"diverse_morph_{idx}.png"

            # Draw graph
            graph_drawn = False
            try:
                draw_graph(tree, title=f"Individual {idx} - Gen {ind.time_of_birth}",
                          save_file=save_path)
                plt.close('all')
                graph_drawn = True
            except ValueError:
                num_nodes = len(tree.nodes())
                num_edges = len(tree.edges())
                ax.text(0.5, 0.6,
                       f"Individual {idx}\n{num_nodes} nodes, {num_edges} edges",
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=font_size + 2, fontweight='bold')
            except Exception as e:
                print(f"Error drawing graph {idx}: {e}")

            if graph_drawn and save_path.exists():
                from PIL import Image
                img = Image.open(save_path)
                ax.imshow(img)

            ax.axis('off')

            # Add detailed info
            tree_id = id(tree)
            if tree_id in measures_dict:
                m = measures_dict[tree_id]
                prop = m.proportion_2d if m.is_2d else m.proportion_3d

                info_text = (
                    f"Gen: {ind.time_of_birth} | Novelty: {ind.fitness:.3f}\n"
                    f"Branching: {m.branching:.2f} | Limbs: {m.limbs:.2f}\n"
                    f"Length: {m.length_of_limbs:.2f} | Coverage: {m.coverage:.2f}\n"
                    f"Joints: {m.joints:.2f} | Proportion: {prop:.2f}\n"
                    f"Symmetry: {m.symmetry:.2f}"
                )

                ax.text(0.5, -0.02, info_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=font_size,
                       family='monospace', bbox=dict(boxstyle='round',
                       facecolor='lightgreen', alpha=0.3))
        else:
            ax.axis('off')

    plt.tight_layout()
    grid_path = output_dir / "diverse_representatives_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Diverse representatives grid saved to: {grid_path}")

    # MuJoCo renders grid
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle("Diverse Representatives - MuJoCo Renders",
                fontsize=18, fontweight='bold')

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        if idx < len(diverse_individuals):
            ind = diverse_individuals[idx]
            tree = ind.genotype.tree if isinstance(ind.genotype, TreeGenotype) else ind.genotype
            render_path = output_dir / f"diverse_render_{idx}.png"

            # Render MuJoCo view
            success = render_mujoco_isometric(tree, render_path,
                                             cam_distance=cam_distance,
                                             elevation=elevation,
                                             azimuth=azimuth,
                                             cam_fovy=cam_fovy)

            if success and render_path.exists():
                from PIL import Image
                img = Image.open(render_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Render\nFailed",
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=font_size)

            ax.axis('off')

            # Add info
            tree_id = id(tree)
            if tree_id in measures_dict:
                m = measures_dict[tree_id]
                prop = m.proportion_2d if m.is_2d else m.proportion_3d

                info_text = (
                    f"Individual {idx} - Gen {ind.time_of_birth}\n"
                    f"B:{m.branching:.2f} L:{m.limbs:.2f} C:{m.coverage:.2f} "
                    f"J:{m.joints:.2f} S:{m.symmetry:.2f}"
                )

                ax.text(0.5, -0.02, info_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=font_size,
                       family='monospace', bbox=dict(boxstyle='round',
                       facecolor='lightgreen', alpha=0.3))
        else:
            ax.axis('off')

    plt.tight_layout()
    grid_path = output_dir / "diverse_representatives_mujoco.png"
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Diverse representatives MuJoCo grid saved to: {grid_path}")


def visualize_all(results, font_size=10, n_display=12,
                  cam_distance=0.5, elevation=35, azimuth=45, cam_fovy=30):
    """Create all visualizations.

    Parameters
    ----------
    results : dict
        Results from run_novelty_evolution
    font_size : int
        Font size for annotations
    n_display : int
        Number of individuals to display in grids
    cam_distance : float
        Camera distance for MuJoCo renders in meters (default: 0.5)
        Decrease for closer view of small robots
    elevation : float
        Camera elevation angle in degrees (default: 35 for isometric)
    azimuth : float
        Camera azimuth angle in degrees (default: 45 for isometric)
    cam_fovy : float
        Camera field of view in degrees (default: 30)
        Smaller values = more zoom (try 20-25 for small robots)
    """
    all_individuals = results['all_individuals']
    measures_dict = results['measures_dict']
    output_dir = results['output_dir']

    print("\nGenerating visualizations...")

    # 1. Novelty history
    plot_novelty_history(all_individuals, output_dir)

    # 2. Morphological descriptors over time
    plot_morphological_descriptors(all_individuals, measures_dict, output_dir)

    # 3. Final morphologies grid (graph visualizations)
    plot_final_morphologies_grid(all_individuals, measures_dict, output_dir,
                                 n_display=n_display, font_size=font_size)

    # 4. MuJoCo renders grid
    plot_mujoco_renders_grid(all_individuals, measures_dict, output_dir,
                             n_display=n_display, font_size=font_size,
                             cam_distance=cam_distance, elevation=elevation,
                             azimuth=azimuth, cam_fovy=cam_fovy)

    # 5. Diverse representatives across search space
    plot_diverse_representatives_grid(all_individuals, measures_dict, output_dir,
                                      n_display=n_display, font_size=font_size,
                                      cam_distance=cam_distance, elevation=elevation,
                                      azimuth=azimuth, cam_fovy=cam_fovy)

    print(f"\n{'='*60}")
    print("All visualizations complete!")
    print(f"Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    from run_novelty_evolution import run_novelty_evolution

    # Run evolution
    print("Running evolution...")
    results = run_novelty_evolution(
        mu=100,
        lambda_=100,
        num_generations=20,
        max_depth=3,
        max_parts=25,
        use_kdtree=True,
        min_distance=0.5,
        k_neighbors=3,
        seed=42,
    )

    # Generate all visualizations
    visualize_all(results,
                  font_size=11,
                  n_display=12,
                  cam_distance=0.4,   # Camera distance from robot (meters, smaller = closer)
                  elevation=15,       # Camera elevation (0=side, 90=top)
                  azimuth=45,         # Camera azimuth (0=front, 90=right, 180=back, 270=left)
                  cam_fovy=1.5)        # Field of view (degrees, smaller = more zoom)
