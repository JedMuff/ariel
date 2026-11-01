#!/bin/bash
#
# Batch script for run_scratch_learning.py
# Runs controller learning from scratch on the last generation of morphologies.
#
# Usage:
#   sbatch examples/d_social_learning/run_scratch_batch.sh [ariel|evogym]
#

#SBATCH --job-name=scratch-learning
#SBATCH --output=out_files/scratch-learning-%A_%a.out
#SBATCH --error=out_files/scratch-learning-%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=genoa
#SBATCH --array=0-239

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REPO_ROOT=$HOME/ariel   # <-- edit this if your repo lives elsewhere
VENV_PATH=$REPO_ROOT/.venv

echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Date:       $(date)"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Map array index -> (scheme, x, rep)
# ---------------------------------------------------------------------------

SCHEMES=(darwinian lamarckian random random_many best best_many similar_STRUCT similar_many_STRUCT)
X_VALS=(00 05 10)
N_REPS=10

IDX=$SLURM_ARRAY_TASK_ID

REP=$(( IDX / 24 ))                 # 8 schemes x 3 x-values = 24 combos per rep
REM=$(( IDX % 24 ))
SCHEME_IDX=$(( REM / 3 ))
X_IDX=$(( REM % 3 ))

SCHEME="${SCHEMES[$SCHEME_IDX]}"
X="${X_VALS[$X_IDX]}"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

INNER_GENS=100
INNER_POP=10
SIGMA=0.45
HIDDEN=16
WORKERS=32

echo "Processing experiment: Scheme=$SCHEME, X=$X, Rep=$REP, Platform=$PLATFORM"
echo "Params: inner-gens=$INNER_GENS inner-pop=$INNER_POP sigma=$SIGMA hidden=$HIDDEN workers=$WORKERS"

mkdir -p out_files

START_TIME=$(date +%s)

# Run the scratch learning script passing parameters instead of path.
srun "$VENV_PATH/bin/python" examples/d_social_learning/run_scratch_learning.py \
    --platform evogym \
    --scheme "$SCHEME" \
    --x "$X" \
    --rep "$REP" \
    --inner-gens "$INNER_GENS" \
    --inner-pop "$INNER_POP" \
    --sigma "$SIGMA" \
    --hidden "$HIDDEN" \
    --workers "$WORKERS"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished scratch learning for Scheme=$SCHEME, X=$X, Rep=$REP"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
