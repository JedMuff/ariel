#!/bin/bash
#
# Full social-learning experiment: ARIEL only
# 8 schemes × 3 x-values × 10 reps = 240 array jobs
#
# Usage:
#   sbatch examples/d_social_learning/run_social_experiment.slurm
#
#SBATCH --job-name=social-full
#SBATCH --output=out_files/social-%A_%a.out
#SBATCH --error=out_files/social-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=21
#SBATCH --mem=17G
#SBATCH --array=0-239

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REPO_ROOT=/home/jed/workspaces/ariel
VENV_PATH=$REPO_ROOT/.venv

echo "Node:       $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Date:       $(date)"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Index → scheme / x / rep
# Layout: scheme(8) × x(3) × rep(10), rep is innermost
# ---------------------------------------------------------------------------

SCHEMES=(darwinian lamarckian random random_many best best_many similar similar_many)
X_VALUES=(0 0.5 1)

GENS=20
POP=20
LAM=20
INNER_GENS=50
INNER_POP=20
WORKERS=20

TASK=$SLURM_ARRAY_TASK_ID

SCHEME_IDX=$(( TASK / 30 ))          # 0-7
REMAINDER=$(( TASK % 30 ))
X_IDX=$(( REMAINDER / 10 ))          # 0-2
REP=$(( REMAINDER % 10 ))            # 0-9

SCHEME="${SCHEMES[$SCHEME_IDX]}"
X="${X_VALUES[$X_IDX]}"

echo "Scheme: $SCHEME  x=$X  rep=$REP"
echo "Params: gens=$GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP workers=$WORKERS"

mkdir -p out_files

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/ariel/experiment.py \
    --scheme "$SCHEME" --x "$X" --rep "$REP" \
    --gens "$GENS" --pop "$POP" --lam "$LAM" \
    --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
    --workers "$WORKERS"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished: scheme=$SCHEME x=$X rep=$REP"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
