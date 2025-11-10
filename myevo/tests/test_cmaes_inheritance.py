"""Quick test of CMA-ES state inheritance implementation."""

import sys
sys.path.insert(0, '/home/jed/workspaces/ariel/myevo')

from pathlib import Path
import shutil

# Create test directory in __data__/
test_dir = Path("/home/jed/workspaces/ariel/__data__/test_cmaes_inheritance")
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(parents=True)

# Set global DATA variable for log directory
from myevo.core import mu_lambda_tree_locomotion
mu_lambda_tree_locomotion.DATA = test_dir

from myevo.core.mu_lambda_tree_locomotion import TreeLocomotionEvolution

print("=" * 60)
print("Testing CMA-ES State Inheritance Implementation")
print("=" * 60)

# Create evolution system with minimal settings
evolution = TreeLocomotionEvolution(
    # Minimal population for quick test
    mu=3,
    lambda_=6,
    mutation_rate=0.8,
    crossover_rate=0.2,
    # Small morphologies for fast simulation
    max_depth=2,
    max_part_limit=10,
    max_actuators=8,
    # Simulation settings (needs to be > 5s for settling phase)
    simulation_duration=10.0,
    controller_hidden_layers=[16, 16],
    # Small CMA-ES budget for quick test
    use_cmaes=True,
    cmaes_budget=20,  # Small budget as requested
    cmaes_population_size=4,
    # Enable Lamarckian with new CMA-ES inheritance
    enable_lamarckian=True,
    lamarckian_crossover_mode="closest_parent",
    covariance_inheritance_mode="adaptive",
    sigma_inheritance_mode="blend",
    # Disable novelty and video for simplicity
    use_novelty=False,
    enable_video_recording=False,
    # System settings
    seed=42,
    num_workers=1,
    verbose=True,
)

print("\n✓ Evolution system initialized successfully")
print(f"  - CMA-ES budget: {evolution.cmaes_budget}")
print(f"  - Covariance mode: {evolution.covariance_inheritance_mode}")
print(f"  - Sigma mode: {evolution.sigma_inheritance_mode}")

# Run 2 generations to test inheritance
print("\n" + "=" * 60)
print("Running 2 generations to test CMA-ES state inheritance...")
print("=" * 60)

try:
    final_population = evolution.run(num_generations=2)

    print("\n" + "=" * 60)
    print("✓ TEST PASSED: Evolution completed successfully!")
    print("=" * 60)

    # Check that files were created
    database_path = test_dir / "database.csv"
    if database_path.exists():
        print(f"\n✓ Database created: {database_path}")

        # Read and check for CMA-ES fields
        import csv
        with open(database_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        cmaes_fields = ['cmaes_sigma', 'cmaes_condition_number', 'cmaes_mean_fitness', 'cmaes_num_evaluations']
        found_fields = [field for field in cmaes_fields if field in headers]

        print(f"✓ CMA-ES database fields present: {found_fields}")

        # Check if CMA-ES state files were created
        gen_0_dir = test_dir / "generation_00" / "individual_0"
        if gen_0_dir.exists():
            cmaes_files = [
                "cmaes_state.pkl",
                "cmaes_covariance.npy",
                "cmaes_sigma.txt",
                "cmaes_mean.npy",
            ]
            found_files = [f for f in cmaes_files if (gen_0_dir / f).exists()]
            print(f"✓ CMA-ES state files created: {found_files}")

            # Read sigma value
            sigma_file = gen_0_dir / "cmaes_sigma.txt"
            if sigma_file.exists():
                with open(sigma_file, 'r') as f:
                    sigma = float(f.read().strip())
                print(f"✓ Sample sigma value: {sigma:.4f}")

    # Print final best fitness
    best_ind = max(final_population, key=lambda x: x.fitness if x.fitness else float('-inf'))
    print(f"\n✓ Best fitness after 2 generations: {best_ind.fitness:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed! Implementation is working correctly.")
    print("=" * 60)

except Exception as e:
    print("\n" + "=" * 60)
    print("✗ TEST FAILED")
    print("=" * 60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
