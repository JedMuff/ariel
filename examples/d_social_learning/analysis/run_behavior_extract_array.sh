#!/bin/bash
#
# Array job: run behavior_extract.py over all 80 social-learning experiments
# with behaviour (8 schemes x 2 x-values [x05, x10] x 5 reps), one array task
# per experiment. x00 is excluded: with no social learning, there's no
# behaviour to extract.
#
# Reads each experiment's database.db from $SCRATCH_IN and writes
# behavior.npz back to $SCRATCH_OUT, mirroring the scheme/xVV/rep_N
# directory layout.
#
# Fill in SCRATCH_IN / SCRATCH_OUT below before submitting.
#
# Usage (run from repo root):
#   sbatch examples/d_social_learning/analysis/run_behavior_extract_array.sh
#
# Dry-run a single task locally (no sbatch):
#   SLURM_ARRAY_TASK_ID=0 bash examples/d_social_learning/analysis/run_behavior_extract_array.sh
#
#SBATCH --job-name=behavior-extract
#SBATCH --output=out_files/behavior-extract-%A_%a.out
#SBATCH --error=out_files/behavior-extract-%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=13G
#SBATCH --array=0-79

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REPO_ROOT=/home/jed/workspaces/ariel
VENV_PATH=$REPO_ROOT/.venv

# TODO: fill in before submitting.
SCRATCH_IN="/scratch/jed/social/ariel"      # holds {scheme}/x{XX}/rep_{N}/database.db
SCRATCH_OUT="/scratch/jed/social/behavior"  # mirrors the same layout

WORKERS=16
RECORD_EVERY_N=100

echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_ARRAY_JOB_ID"
echo "Task ID:    $SLURM_ARRAY_TASK_ID"
echo "Date:       $(date)"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID (0-119) -> (scheme, x_dir, rep)
#
# Order must match run_social_submit.sh's experiment grid: 8 schemes x
# 3 x-values x 5 reps. On-disk x directories are x00/x05/x10 (not the
# str(x).replace(".","") used to *create* them, which gives x0/x1 for
# x=0/1 -- verified against actual output layout).
# ---------------------------------------------------------------------------

SCHEMES=(darwinian lamarckian random random_many best best_many similar similar_many)
X_DIRS=(x05 x10)
N_REPS=5

combos=()
for scheme in "${SCHEMES[@]}"; do
    for x_dir in "${X_DIRS[@]}"; do
        for (( rep=0; rep<N_REPS; rep++ )); do
            combos+=("${scheme} ${x_dir} ${rep}")
        done
    done
done

if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "SLURM_ARRAY_TASK_ID not set (run via sbatch, or export it for a local dry-run)"
    exit 1
fi

read -r SCHEME X_DIR REP <<< "${combos[$SLURM_ARRAY_TASK_ID]}"

echo "Experiment: scheme=$SCHEME x=$X_DIR rep=$REP  (${#combos[@]} total combos)"

DB_PATH="$SCRATCH_IN/$SCHEME/$X_DIR/rep_$REP/database.db"
OUT_DIR="$SCRATCH_OUT/$SCHEME/$X_DIR/rep_$REP"

if [[ ! -f "$DB_PATH" ]]; then
    echo "No database.db at $DB_PATH -- skipping"
    exit 1
fi

mkdir -p "$OUT_DIR" out_files

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/analysis/behavior_extract.py \
    --db-path "$DB_PATH" \
    --all \
    --workers "$WORKERS" \
    --record-every-n "$RECORD_EVERY_N" \
    --out-dir "$OUT_DIR"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished: scheme=$SCHEME x=$X_DIR rep=$REP"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
