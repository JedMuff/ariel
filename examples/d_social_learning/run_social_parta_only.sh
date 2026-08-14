#!/bin/bash
#
# Part A (array version): first 20 generations of each experiment.
# Self-contained — does NOT need run_social_submit.sh to launch it, and does
# NOT submit part B. Run part B yourself afterwards for whichever
# scheme/x/rep combos you want to continue.
#
# experiment.py timestamps its own output directory per run (so re-runs never
# clobber each other) and prints it as `RUN_DIR=<path>` in this job's log
# (out_files/social-parta-<jobid>_<taskid>.out). To continue a run manually,
# grep that file for RUN_DIR= and pass it to experiment.py's --resume-dir.
#
# Usage:
#   sbatch examples/d_social_learning/run_social_parta_array.sh
#
# 8 schemes x 3 x-values x 10 reps = 240 combos -> array indices 0-239.
# Ordering matches run_social_submit.sh: rep outermost, then scheme, then x.
#
#SBATCH --job-name=social-parta
#SBATCH --output=out_files/social-parta-%A_%a.out
#SBATCH --error=out_files/social-parta-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --partition=genoa
#SBATCH --array=0-119

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

SCHEMES=(darwinian lamarckian random random_many best best_many similar similar_many)
X_VALUES=(0 0.5 1)
N_REPS=5

IDX=$SLURM_ARRAY_TASK_ID

REP=$(( IDX / 24 ))                 # 8 schemes x 3 x-values = 24 combos per rep
REM=$(( IDX % 24 ))
SCHEME_IDX=$(( REM / 3 ))
X_IDX=$(( REM % 3 ))

SCHEME="${SCHEMES[$SCHEME_IDX]}"
X="${X_VALUES[$X_IDX]}"

# ---------------------------------------------------------------------------
# Parameters (same as run_social_parta.sh)
# ---------------------------------------------------------------------------

GENS=30
INNER_GENS=25
INNER_POP=20
SIGMA=0.45
HIDDEN=16
WORKERS=24

# Selection strategy: set to true for (mu,lambda) (survivors drawn from
# offspring only), false for the default (mu+lambda) (survivors drawn from
# parents+offspring). (mu,lambda) requires LAM >= POP.
#
# POP/LAM are chosen per-strategy so both evaluate the same number of
# individuals per generation (LAM=20 evals/gen either way). (mu+lambda) uses
# POP=LAM=20 (20 survivors picked from 40 parents+offspring). (mu,lambda)
# shrinks POP to 10 so it retains real selection pressure (10 survivors
# picked from 20 offspring only) instead of all offspring surviving.
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

echo "Scheme: $SCHEME  x=$X  rep=$REP  (array idx=$IDX)"
echo "Params: gens=$GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP sigma=$SIGMA hidden=$HIDDEN workers=$WORKERS comma_selection=$COMMA_SELECTION"

mkdir -p out_files

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/ariel/experiment.py \
    --scheme "$SCHEME" --x "$X" --rep "$REP" \
    --gens "$GENS" --pop "$POP" --lam "$LAM" \
    --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
    --sigma "$SIGMA" --hidden "$HIDDEN" \
    --workers "$WORKERS" "${SELECTION_FLAG[@]}"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished part A: scheme=$SCHEME x=$X rep=$REP"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"