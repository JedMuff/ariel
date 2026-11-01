#!/bin/bash
#
# Learn controllers from scratch on the final populations of the zoo fe10 runs
# (__data__/social/ariel_zoo_fe10). run_scratch_learning.py takes the zoo preset,
# --final-evals 10 and the CMA-ES settings from each run's run_config.json.
#
# Usage:
#   sbatch examples/d_social_learning/run_scratch_zoo.sh
#
# Every morphology learns for SCRATCH_GENS = 100 inner generations (2000 evals at
# population 20), whatever budget it evolved under. Each value of SCRATCH_GENS has
# its own output directory. To use another, check --time and --eval-timeout first:
#   sbatch --export=ALL,SCRATCH_GENS=200 --time=04:00:00 examples/d_social_learning/run_scratch_zoo.sh
#
# 20 reps x 2 budgets x 8 schemes x 3 x-values = 960 indices, of which 720 have a
# run: ig20 has no x = 0 runs, and ig10's x = 0 runs stop at rep 9. The other
# tasks exit straight away. Ordering: rep outermost, then budget, scheme, x:
#     index = 48 * rep + 24 * budget_idx + 3 * scheme_idx + x_idx
# Runs that already have results are skipped, so resubmitting the full range is
# safe and only fills in what is missing.
#
#SBATCH --job-name=scratch-zoo
#SBATCH --output=out_files/scratch-zoo-%A_%a.out
#SBATCH --error=out_files/scratch-zoo-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --partition=genoa
#SBATCH --array=0-959

REPO_ROOT=$HOME/ariel   # <-- edit this if your repo lives elsewhere
VENV_PATH=$REPO_ROOT/.venv

echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Date:       $(date)"

cd "$REPO_ROOT"

SCHEMES=(darwinian lamarckian similar_TOPDOWN best_many random_many random best similar_many_TOPDOWN)
X_DIRS=(00 05 10)
BUDGETS=(10 20)
K_INHERIT=5
WORKERS=24

IDX=$SLURM_ARRAY_TASK_ID
REP=$(( IDX / 48 ))
REM=$(( IDX % 48 ))
BUDGET_IDX=$(( REM / 24 ))
REM=$(( REM % 24 ))
SCHEME_IDX=$(( REM / 3 ))
X_IDX=$(( REM % 3 ))

SCHEME="${SCHEMES[$SCHEME_IDX]}"
X="${X_DIRS[$X_IDX]}"
BUDGET="${BUDGETS[$BUDGET_IDX]}"
SCRATCH_GENS=${SCRATCH_GENS:-100}

DATA_DIR=__data__/social/ariel_zoo_fe10/ig${BUDGET}
OUT_DIR=__data__/scratch_learning/ariel_zoo_fe10/ig${BUDGET}/sg${SCRATCH_GENS}

echo "Scheme: $SCHEME  x=$X  rep=$REP  budget=$BUDGET  scratch-gens=$SCRATCH_GENS  (array idx=$IDX)"

if ! compgen -G "$DATA_DIR/$SCHEME/x$X/k$K_INHERIT/rep_${REP}_*/database.db" > /dev/null; then
    echo "No run for this combination, skipping"
    exit 0
fi

mkdir -p out_files

srun "$VENV_PATH/bin/python" examples/d_social_learning/run_scratch_learning.py \
    --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" \
    --scheme "$SCHEME" --x "$X" --rep "$REP" --k-inherit "$K_INHERIT" \
    --inner-gens "$SCRATCH_GENS" --workers "$WORKERS"

echo "done"
