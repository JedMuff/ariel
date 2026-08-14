#!/bin/bash
#
# Part A: first 20 generations of one experiment.
# Submitted individually by run_social_submit.sh — not as an array.
#
#SBATCH --job-name=social-parta
#SBATCH --output=out_files/social-%j.out
#SBATCH --error=out_files/social-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=17G

# Propagate experiment.py's exit code through the `| tee` pipe below (used to
# capture the RUN_DIR= line), so this job's own exit status still reflects
# whether the run actually succeeded -- part B's sbatch --dependency=afterok
# relies on that.
set -o pipefail

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
INNER_GENS=25
INNER_POP=20
SIGMA=0.45
HIDDEN=16
WORKERS=20

echo "Scheme: $SCHEME  x=$X  rep=$REP"
echo "Params: gens=$GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP sigma=$SIGMA hidden=$HIDDEN workers=$WORKERS"

mkdir -p out_files

# experiment.py now timestamps its own output directory per run (so re-runs
# never clobber each other) and prints it as `RUN_DIR=<path>`. Part B doesn't
# know that timestamp, so capture it here into a marker file part B reads.
RUN_DIR_FILE="out_files/rundir_${SCHEME}_${X}_${REP}.txt"

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/ariel/experiment.py \
    --scheme "$SCHEME" --x "$X" --rep "$REP" \
    --gens "$GENS" --pop "$POP" --lam "$LAM" \
    --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
    --sigma "$SIGMA" --hidden "$HIDDEN" \
    --workers "$WORKERS" \
    | tee >(grep '^RUN_DIR=' | cut -d= -f2- > "$RUN_DIR_FILE")
STATUS=${PIPESTATUS[0]}

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

if [[ $STATUS -ne 0 ]]; then
    echo ""
    echo "experiment.py failed (exit $STATUS): scheme=$SCHEME x=$X rep=$REP"
    exit $STATUS
fi

echo ""
echo "Finished part A: scheme=$SCHEME x=$X rep=$REP -> $(cat "$RUN_DIR_FILE")"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
