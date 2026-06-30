#!/bin/bash
#
# Part B: second 20 generations of one experiment, resumed from part A's DB.
# Submitted individually by run_social_submit.sh with --dependency=afterok:<partA_jobid>.
#
#SBATCH --job-name=social-partb
#SBATCH --output=out_files/social-%j.out
#SBATCH --error=out_files/social-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=17G

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
# Parameters passed by submit script
# ---------------------------------------------------------------------------

SCHEME="$1"
X="$2"
REP="$3"

GENS=20
POP=20
LAM=20
INNER_GENS=50
INNER_POP=20
WORKERS=20

echo "Scheme: $SCHEME  x=$X  rep=$REP"
echo "Params: gens=$GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP workers=$WORKERS"

mkdir -p out_files

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/ariel/experiment.py \
    --scheme "$SCHEME" --x "$X" --rep "$REP" \
    --gens "$GENS" --pop "$POP" --lam "$LAM" \
    --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
    --workers "$WORKERS" \
    --resume

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished part B: scheme=$SCHEME x=$X rep=$REP"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
