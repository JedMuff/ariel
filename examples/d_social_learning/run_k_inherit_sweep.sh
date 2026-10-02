#!/bin/bash
#
# k_inherit pilot (array version): best_many at x=1.0 only, sweeping
# --k-inherit over a few values to decide which one to use in the main
# experiment. Otherwise identical to run_social_parta_only.sh.
#
# experiment.py already puts k in the output path
# (__data__/social/<platform>/best_many/x10/k<K>/rep_<R>_nov<METRIC>), so the
# different k values never collide with each other or with earlier runs.
# The run dir is printed as `RUN_DIR=<path>` in this job's log
# (out_files/k-inherit-sweep-<jobid>_<taskid>.out).
#
# Usage:
#   sbatch examples/d_social_learning/run_k_inherit_sweep.sh
#
# 3 k-values x 5 reps = 15 combos -> array indices 0-14.
# Ordering: rep outermost, then k.
#
#SBATCH --job-name=k-inherit-sweep
#SBATCH --output=out_files/k-inherit-sweep-%A_%a.out
#SBATCH --error=out_files/k-inherit-sweep-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --partition=genoa
#SBATCH --array=0-14

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
# Map array index -> (k, rep)
# ---------------------------------------------------------------------------

SCHEME=best_many
X=1.0
K_VALUES=(3 5 8)
N_REPS=5

IDX=$SLURM_ARRAY_TASK_ID

REP=$(( IDX / ${#K_VALUES[@]} ))
K="${K_VALUES[$(( IDX % ${#K_VALUES[@]} ))]}"

# ---------------------------------------------------------------------------
# Parameters (same as run_social_parta_only.sh)
# ---------------------------------------------------------------------------

GENS=30
INNER_GENS=50
INNER_POP=20
SIGMA=0.45
HIDDEN=16
WORKERS=24

# Selection strategy: true for (mu,lambda), false for (mu+lambda).
# See run_social_parta_only.sh for the POP/LAM rationale.
COMMA_SELECTION=false

if [ "$COMMA_SELECTION" = true ]; then
    POP=10
    LAM=20
    SELECTION_FLAG=(--comma-selection)
else
    POP=20
    LAM=20
    SELECTION_FLAG=()
fi

NOVELTY_METRIC=STRUCT

echo "Scheme: $SCHEME  x=$X  k=$K  rep=$REP  (array idx=$IDX)"
echo "Params: gens=$GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP sigma=$SIGMA hidden=$HIDDEN workers=$WORKERS comma_selection=$COMMA_SELECTION novelty_metric=$NOVELTY_METRIC k_inherit=$K"

mkdir -p out_files

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/experiment.py \
    --scheme "$SCHEME" --x "$X" --rep "$REP" \
    --gens "$GENS" --pop "$POP" --lam "$LAM" \
    --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
    --sigma "$SIGMA" --hidden "$HIDDEN" \
    --k-inherit "$K" \
    --workers "$WORKERS" --novelty-metric "$NOVELTY_METRIC" "${SELECTION_FLAG[@]}"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished: scheme=$SCHEME x=$X k=$K rep=$REP"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
