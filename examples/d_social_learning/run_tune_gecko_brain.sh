#!/bin/bash
#
# NSGA-II tuning of gecko brain/CMA-ES hyperparameters (100 trials, inner
# parallelization only). Single long-running job — trials run sequentially,
# each trial's CMA-ES generations parallelized across the allocated CPUs.
#
# Usage:
#   sbatch examples/d_social_learning/run_tune_gecko_brain.sh
#
#SBATCH --job-name=gecko-tune
#SBATCH --output=out_files/gecko-tune-%j.out
#SBATCH --error=out_files/gecko-tune-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REPO_ROOT=/home/jed/workspaces/ariel
VENV_PATH=$REPO_ROOT/.venv

echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Date:       $(date)"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_TRIALS=100
WORKERS=$SLURM_CPUS_PER_TASK
STUDY_NAME="gecko_nsga2_$(date +%Y%m%d_%H%M%S)"
STORAGE="sqlite:///__data__/tuning/${STUDY_NAME}.db"
WEIGHTS_DIR="__data__/tuning/${STUDY_NAME}_weights"

echo "Trials: $N_TRIALS  Workers: $WORKERS  Study: $STUDY_NAME"

mkdir -p out_files

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/ariel/tune_gecko_brain.py \
    --n-trials "$N_TRIALS" \
    --workers "$WORKERS" \
    --study-name "$STUDY_NAME" \
    --storage "$STORAGE" \
    --weights-dir "$WEIGHTS_DIR"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished gecko brain tuning"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
