#!/bin/bash

#SBATCH --output=slurm_out/darwinian-%A_%a.out
#SBATCH --error=slurm_out/darwinian-%A_%a.err
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
# Now organized so all experiments run in parallel for each repetition:
# Tasks 0-1:   Repetition 0 (both Darwinian experiments)
# Tasks 2-3:   Repetition 1 (both Darwinian experiments)
# Tasks 4-5:   Repetition 2 (both Darwinian experiments)
# ...
# Tasks 38-39: Repetition 19 (both Darwinian experiments)

REPETITION=$((SLURM_ARRAY_TASK_ID / 2))
EXPERIMENT_ID=$((SLURM_ARRAY_TASK_ID % 2))
# Make seed unique for each experiment AND repetition
# Formula: base_seed + (repetition * 10) + experiment_id
# This ensures each experiment has a unique seed
SEED=$((27 + (REPETITION * 10) + EXPERIMENT_ID))

echo "Experiment ID: $EXPERIMENT_ID"
echo "Repetition: $REPETITION"
echo "Seed: $SEED"

# Set experiment-specific parameters
# Only Darwinian experiments (no Lamarckian evolution)
if [ $EXPERIMENT_ID -eq 0 ]; then
    # Darwinian + Locomotion (baseline)
    EXPERIMENT_NAME="darwinian_locomotion_rep${REPETITION}"
    FLAGS=""
    echo "Running: Darwinian Evolution + Locomotion (baseline)"
elif [ $EXPERIMENT_ID -eq 1 ]; then
    # Darwinian + Novelty*Locomotion
    EXPERIMENT_NAME="darwinian_novelty_rep${REPETITION}"
    FLAGS="--use-novelty"
    echo "Running: Darwinian Evolution + Novelty*Locomotion"
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
echo "Total Repetitions: 20"
echo "==============================================="

# Run the experiment
echo "Starting experiment at $(date)"
srun python3 myevo/core/mu_lambda_tree_locomotion.py \
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
