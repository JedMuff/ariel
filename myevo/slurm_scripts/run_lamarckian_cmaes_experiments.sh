#!/bin/bash

#SBATCH --output=slurm_out/lamarckian_cmaes-%A_%a.out
#SBATCH --error=slurm_out/lamarckian_cmaes-%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=32
#SBATCH --array=0-39

# Virtual environment path - UPDATE THIS to match your cluster setup
VENV_PATH=/home/jed/workspace/ariel/.venv

echo "==============================================="
echo "SLURM Job Information"
echo "==============================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Using Python from: $VENV_PATH"
echo "==============================================="

# Setup Python environment
export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$VENV_PATH/lib/python3.10/site-packages:$PYTHONPATH"

# CRITICAL: Prevent nested parallelism - limit threads per worker
# With 30 workers, each worker should use only 1 thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Verify Python installation
which python3
python3 --version

# Calculate experiment type and repetition from array task ID
# Organization: all experiments run in parallel for each repetition
# Tasks 0-3:   Repetition 0 (all 4 experiments)
# Tasks 4-7:   Repetition 1 (all 4 experiments)
# Tasks 8-11:  Repetition 2 (all 4 experiments)
# ...
# Tasks 36-39: Repetition 9 (all 4 experiments)

REPETITION=$((SLURM_ARRAY_TASK_ID / 4))
EXPERIMENT_ID=$((SLURM_ARRAY_TASK_ID % 4))
# Make seed unique for each experiment AND repetition
# Formula: base_seed + (repetition * 10) + experiment_id
# This ensures each experiment has a unique seed
SEED=$((17 + (REPETITION * 10) + EXPERIMENT_ID))

echo "Experiment ID: $EXPERIMENT_ID"
echo "Repetition: $REPETITION"
echo "Seed: $SEED"

# Set experiment-specific parameters
if [ $EXPERIMENT_ID -eq 0 ]; then
    # Lamarckian + Covariance Inheritance (sigma reset) + Locomotion
    EXPERIMENT_NAME="lamarck_cov_sigmareset_locomotion_rep${REPETITION}"
    FLAGS="--enable-lamarckian --covariance-inheritance-mode adaptive --sigma-inheritance-mode reset"
    echo "Running: Lamarckian (Cov=adaptive, Sigma=reset) + Locomotion"
elif [ $EXPERIMENT_ID -eq 1 ]; then
    # Lamarckian + Covariance + Sigma Inheritance + Locomotion
    EXPERIMENT_NAME="lamarck_cov_sigma_locomotion_rep${REPETITION}"
    FLAGS="--enable-lamarckian --covariance-inheritance-mode adaptive --sigma-inheritance-mode blend"
    echo "Running: Lamarckian (Cov=adaptive, Sigma=blend) + Locomotion"
elif [ $EXPERIMENT_ID -eq 2 ]; then
    # Lamarckian + Covariance Inheritance (sigma reset) + Novelty*Locomotion
    EXPERIMENT_NAME="lamarck_cov_sigmareset_novelty_rep${REPETITION}"
    FLAGS="--enable-lamarckian --covariance-inheritance-mode adaptive --sigma-inheritance-mode reset --use-novelty"
    echo "Running: Lamarckian (Cov=adaptive, Sigma=reset) + Novelty*Locomotion"
elif [ $EXPERIMENT_ID -eq 3 ]; then
    # Lamarckian + Covariance + Sigma Inheritance + Novelty*Locomotion
    EXPERIMENT_NAME="lamarck_cov_sigma_novelty_rep${REPETITION}"
    FLAGS="--enable-lamarckian --covariance-inheritance-mode adaptive --sigma-inheritance-mode blend --use-novelty"
    echo "Running: Lamarckian (Cov=adaptive, Sigma=blend) + Novelty*Locomotion"
else
    echo "ERROR: Invalid experiment ID: $EXPERIMENT_ID"
    exit 1
fi

echo "==============================================="
echo "Experiment Configuration"
echo "==============================================="
echo "Name: $EXPERIMENT_NAME"
echo "Flags: $FLAGS"
echo "Seed: $SEED"
echo "Workers: 30"
echo "Generations: 50"
echo "==============================================="

# Run the experiment
echo "Starting experiment at $(date)"
srun python3 myevo/mu_lambda_tree_locomotion.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --seed $SEED \
    --num-workers 30 \
    $FLAGS

EXIT_CODE=$?

echo "==============================================="
echo "Experiment finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "==============================================="

exit $EXIT_CODE
