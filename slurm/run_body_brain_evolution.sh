#!/bin/bash

#SBATCH --job-name=ariel-body-brain
#SBATCH --output=out_files/ariel-bb-%A_%a.out
#SBATCH --error=out_files/ariel-bb-%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-4

set -euo pipefail

REPO=/home/jed/workspaces/ariel
VENV_PATH=$REPO/.venv

TMP_DIR=/tmp/${USER}/ariel_bb_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=$REPO/__data__/body_brain_evolution
RUN_TAG=body_brain_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
SEED=$((42 + SLURM_ARRAY_TASK_ID))

cd "$REPO"
mkdir -p out_files
mkdir -p "$TMP_DIR"

# Always move whatever the job produced — successful or not — so partial
# results are preserved on failure / timeout / preemption.
cleanup() {
    rc=$?
    if [ -d "$TMP_DIR" ] && [ -n "$(ls -A "$TMP_DIR" 2>/dev/null || true)" ]; then
        echo "Moving results from $TMP_DIR to $FINAL_DIR/$RUN_TAG ..."
        mkdir -p "$FINAL_DIR"
        mv "$TMP_DIR" "$FINAL_DIR/$RUN_TAG"
        echo "Results saved to $FINAL_DIR/$RUN_TAG"
    else
        echo "No results in $TMP_DIR to move (rc=$rc)"
        rm -rf "$TMP_DIR" || true
    fi
    exit $rc
}
trap cleanup EXIT

source "$VENV_PATH/bin/activate"

echo "Node:    $(hostname)"
echo "Job:     $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID  Seed: $SEED"
echo "Python:  $(which python)"
python --version
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "Tmp dir: $TMP_DIR"
echo "Final:   $FINAL_DIR/$RUN_TAG"
echo "Started: $(date)"

# The script writes to ./__data__/<script_name>/. We cd into the scratch dir
# so all outputs land there; the cleanup trap moves them to FINAL_DIR.
cd "$TMP_DIR"

srun python "$REPO/examples/re_book/6_body_brain_randomized_waypoints.py" \
    --strategy across \
    --budget 50 \
    --pop 16 --lam 16 \
    --brain-budget 75 --brain-pop 20 \
    --brain-workers $SLURM_CPUS_PER_TASK \
    --dur 45.0 --reach-radius 0.35 --num-waypoints 3 --arena-radius 3.0 \
    --max-modules 12 --max-depth 12 \
    --seed $SEED \
    --no-video

echo "Finished: $(date)"
echo "done"
