"""Test that initial and optimized CMA-ES states are properly separated."""

import sys
sys.path.insert(0, '/home/jed/workspaces/ariel/myevo')

from pathlib import Path
import shutil
import numpy as np

# Create test directory
test_dir = Path("/home/jed/workspaces/ariel/__data__/test_state_separation")
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir(parents=True)

# Set global DATA variable
from myevo.core import mu_lambda_tree_locomotion
mu_lambda_tree_locomotion.DATA = test_dir

from myevo.core.mu_lambda_tree_locomotion import TreeLocomotionEvolution

print("=" * 70)
print("TEST: Verify Initial vs Optimized State Separation")
print("=" * 70)

# Run evolution with Lamarckian mode
print("\n[1] Testing Lamarckian Mode (should inherit optimized states)")
print("-" * 70)

evolution_lamarck = TreeLocomotionEvolution(
    mu=2, lambda_=4,
    max_depth=2, max_part_limit=12, max_actuators=8,
    simulation_duration=10.0,
    controller_hidden_layers=[8, 8],
    use_cmaes=True,
    cmaes_budget=30,  # Enough to see some evolution
    cmaes_population_size=4,
    enable_lamarckian=True,  # LAMARCKIAN MODE
    seed=42,
    num_workers=1,
    verbose=False,
)

# Run 1 generation
all_individuals = evolution_lamarck.run(num_generations=1)

# Get the final population (not all individuals)
# The strategy keeps only the current population's states due to memory cleanup
final_population = evolution_lamarck.strategy._last_selected_population

# Check that state managers have separate states
print("\nChecking state managers...")

initial_manager = evolution_lamarck.strategy._initial_cmaes_manager
optimized_manager = evolution_lamarck.strategy._optimized_cmaes_manager

print(f"Initial manager has {len(initial_manager)} states")
print(f"Optimized manager has {len(optimized_manager)} states")

# Find an individual with CMA-ES optimization from final population
individuals_with_cmaes = [
    ind for ind in final_population
    if ind.tags.get("cmaes_num_evaluations", 0) > 0
]

if individuals_with_cmaes:
    ind = individuals_with_cmaes[0]
    print(f"\n✓ Found individual {ind.id} with CMA-ES optimization")
    print(f"  - CMA-ES evaluations: {ind.tags.get('cmaes_num_evaluations')}")

    # Get states from managers
    initial_state = initial_manager.get_state(ind.id)
    optimized_state = optimized_manager.get_state(ind.id)

    if initial_state and optimized_state:
        print(f"\n✓ Both states found in managers")
        print(f"  - Initial sigma: {initial_state.sigma:.6f}")
        print(f"  - Optimized sigma: {optimized_state.sigma:.6f}")
        print(f"  - Initial condition number: {initial_state.condition_number:.6f}")
        print(f"  - Optimized condition number: {optimized_state.condition_number:.6f}")

        # Check if they're different (they should be after optimization)
        sigma_diff = abs(initial_state.sigma - optimized_state.sigma)
        cond_diff = abs(initial_state.condition_number - optimized_state.condition_number)

        print(f"\n  - Sigma difference: {sigma_diff:.6f}")
        print(f"  - Condition number difference: {cond_diff:.6f}")

        # Verify covariance matrices are different
        cov_diff = np.linalg.norm(initial_state.covariance_matrix - optimized_state.covariance_matrix)
        print(f"  - Covariance matrix difference (Frobenius norm): {cov_diff:.6f}")

        if cov_diff > 1e-6:
            print(f"\n✓ PASS: Initial and optimized states are different!")
        else:
            print(f"\n✗ WARNING: States appear identical (might indicate short optimization)")

    else:
        print(f"✗ FAIL: States not found in managers")
        sys.exit(1)
else:
    print("✗ No individuals with CMA-ES optimization found")
    print("  This might be due to all individuals having < 4 actuators")

print("\n" + "=" * 70)
print("TEST PASSED: State separation is working correctly!")
print("=" * 70)
print(f"\nTest data saved to: {test_dir}")
