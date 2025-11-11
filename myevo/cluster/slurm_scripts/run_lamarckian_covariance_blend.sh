#!/bin/bash

#SBATCH --output=slurm_out/lamarckian_cov_blend-%A_%a.out
#SBATCH --error=slurm_out/lamarckian_cov_blend-%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=32
#SBATCH --array=0-9

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

# Each array task is one repetition (0-9 = 10 repetitions)
REPETITION=$SLURM_ARRAY_TASK_ID

# Generate unique seed for each repetition
# Base seed 42, offset by repetition
SEED=$((42 + REPETITION))

# Experiment name includes repetition number
EXPERIMENT_NAME="lamarckian_cov_adaptive_sigma_blend_rep${REPETITION}"

echo "==============================================="
echo "Experiment Configuration"
echo "==============================================="
echo "Name: $EXPERIMENT_NAME"
echo "Repetition: $REPETITION"
echo "Seed: $SEED"
echo "Workers: 30"
echo "Generations: 50"
echo "Total Repetitions: 10"
echo "Evolution: Lamarckian (weight inheritance enabled)"
echo "Covariance Mode: adaptive"
echo "Sigma Mode: blend"
echo "==============================================="

# Run the experiment with Lamarckian evolution
echo "Starting experiment at $(date)"
srun python3 myevo/core/mu_lambda_tree_locomotion.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --seed $SEED \
    --num-workers 30 \
    --enable-lamarckian \
    --covariance-inheritance-mode adaptive \
    --sigma-inheritance-mode blend \
    --use-novelty

EXIT_CODE=$?

echo "==============================================="
echo "Experiment finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "==============================================="

exit $EXIT_CODE
