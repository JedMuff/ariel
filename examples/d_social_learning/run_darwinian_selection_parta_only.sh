#!/bin/bash
#
# Darwinian selection-scheme comparison (array version): first 30 generations
# of the darwinian scheme, x=1.0, across 4 survivor-selection variants:
#   0. (mu+lambda) + elitist      (today's default)
#   1. (mu+lambda) + tournament(4)
#   2. (mu,lambda) + elitist      (existing --comma-selection behavior)
#   3. (mu,lambda) + tournament(4)
#
# Self-contained — does NOT need run_social_submit.sh to launch it, and does
# NOT submit part B. Run part B yourself afterwards for whichever
# variant/rep combos you want to continue.
#
# experiment.py timestamps its own output directory per run (so re-runs never
# clobber each other) and prints it as `RUN_DIR=<path>` in this job's log
# (out_files/darwinian-selection-<jobid>_<taskid>.out). To continue a run
# manually, grep that file for RUN_DIR= and pass it to experiment.py's
# --resume-dir.
#
# Usage:
#   sbatch examples/d_social_learning/run_darwinian_selection_parta_only.sh
#
# 4 selection variants x 5 reps = 20 combos -> array indices 0-19.
#
#SBATCH --job-name=darwinian-selection
#SBATCH --output=out_files/darwinian-selection-%A_%a.out
#SBATCH --error=out_files/darwinian-selection-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=20G
#SBATCH --partition=genoa
#SBATCH --array=0-19

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
# Map array index -> (rep, selection variant)
# ---------------------------------------------------------------------------

SCHEME=darwinian
X=1.0
N_REPS=5

# variant index -> "COMMA_SELECTION SELECTION_METHOD TOURNAMENT_SIZE"
VARIANTS=(
    "false elitist    4"
    "false tournament 4"
    "true  elitist    4"
    "true  tournament 4"
)
N_VARIANTS=${#VARIANTS[@]}

IDX=$SLURM_ARRAY_TASK_ID

REP=$(( IDX / N_VARIANTS ))
VARIANT_IDX=$(( IDX % N_VARIANTS ))

read -r COMMA_SELECTION SELECTION_METHOD TOURNAMENT_SIZE <<< "${VARIANTS[$VARIANT_IDX]}"

# ---------------------------------------------------------------------------
# Parameters (same fixed budget as run_social_parta_only.sh)
# ---------------------------------------------------------------------------

GENS=30
INNER_GENS=50
INNER_POP=20
SIGMA=0.45
HIDDEN=16
WORKERS=24

# POP/LAM are chosen per-strategy so both evaluate the same number of
# individuals per generation (LAM=20 evals/gen either way). (mu+lambda) uses
# POP=LAM=20 (20 survivors picked from 40 parents+offspring). (mu,lambda)
# shrinks POP to 10 so it retains real selection pressure (10 survivors
# picked from 20 offspring only) instead of all offspring surviving.
SELECTION_FLAGS=(--selection "$SELECTION_METHOD")
if [ "$SELECTION_METHOD" = "tournament" ]; then
    SELECTION_FLAGS+=(--tournament-size "$TOURNAMENT_SIZE")
fi

if [ "$COMMA_SELECTION" = true ]; then
    POP=10
    LAM=20
    SELECTION_FLAGS+=(--comma-selection)
else
    POP=20
    LAM=20
fi

echo "Scheme: $SCHEME  x=$X  rep=$REP  variant=$VARIANT_IDX (array idx=$IDX)"
echo "Params: gens=$GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP sigma=$SIGMA hidden=$HIDDEN workers=$WORKERS comma_selection=$COMMA_SELECTION selection=$SELECTION_METHOD tournament_size=$TOURNAMENT_SIZE"

mkdir -p out_files

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/ariel/experiment.py \
    --scheme "$SCHEME" --x "$X" --rep "$REP" \
    --gens "$GENS" --pop "$POP" --lam "$LAM" \
    --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
    --sigma "$SIGMA" --hidden "$HIDDEN" \
    --workers "$WORKERS" "${SELECTION_FLAGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished part A: scheme=$SCHEME x=$X rep=$REP variant=$VARIANT_IDX"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
