#!/bin/bash

#SBATCH --job-name=ariel-bb-lamarckian
#SBATCH --output=out_files/ariel-bbl-%A_%a.out
#SBATCH --error=out_files/ariel-bbl-%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=31G
#SBATCH --array=0-4
# Request 1 GPU per task: we don't compute on it, but MuJoCo's EGL renderer
# needs the DRI render nodes (/dev/dri/renderD*) which are only accessible
# when a GPU is allocated. Without this, EGL fails with 'Permission denied'.
#SBATCH --gres=gpu:1

set -euo pipefail

REPO=/home/jed/workspaces/ariel
VENV_PATH=$REPO/.venv

TMP_DIR=/tmp/${USER}/ariel_bbl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=/scratch/jed/ariel_experiments_01052026
RUN_TAG=body_brain_lamarckian_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
SEED=$((42 + SLURM_ARRAY_TASK_ID))

cd "$REPO"
mkdir -p out_files

source "$VENV_PATH/bin/activate"

# Headless rendering: compute nodes have no X display, so MuJoCo's default
# GLFW backend fails. Use EGL (GPU-accelerated, no display needed). Requires
# a GPU allocation (see #SBATCH --gres=gpu:1 above) for /dev/dri permissions.
export MUJOCO_GL=egl

echo "Node:       $(hostname)"
echo "Job:        $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID  Seed: $SEED"
echo "Python:     $(which python)"
python --version
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "MUJOCO_GL:  $MUJOCO_GL"
echo "Tmp dir:    $TMP_DIR"
echo "Final:      $FINAL_DIR/$RUN_TAG"
echo "Started:    $(date)"

# Run the experiment via srun. Wrap in bash -c so the compute node creates
# its own /tmp dir, cds in, runs the experiment, and moves results to scratch
# all on the same node. This avoids any batch-vs-step node mismatch and
# ensures partial results are preserved on failure / timeout / preemption.
srun bash -c "
    set -uo pipefail
    mkdir -p \"$TMP_DIR\"
    finalize() {
        rc=\$?
        if [ -d \"$TMP_DIR\" ] && [ -n \"\$(ls -A \"$TMP_DIR\" 2>/dev/null || true)\" ]; then
            echo \"Moving results from $TMP_DIR to $FINAL_DIR/$RUN_TAG ...\"
            mkdir -p \"$FINAL_DIR\"
            mv \"$TMP_DIR\" \"$FINAL_DIR/$RUN_TAG\"
            echo \"Results saved to $FINAL_DIR/$RUN_TAG\"
        else
            echo \"No results in $TMP_DIR to move (rc=\$rc)\"
            rm -rf \"$TMP_DIR\" || true
        fi
        exit \$rc
    }
    trap finalize EXIT
    cd \"$TMP_DIR\"
    python \"$REPO/examples/re_book/6_body_brain_lamarckian.py\" \
        --strategy across \
        --budget 50 \
        --pop 16 --lam 16 \
        --brain-budget 75 --brain-pop 20 \
        --brain-workers $SLURM_CPUS_PER_TASK \
        --dur 45.0 --reach-radius 0.20 --num-waypoints 10 --arena-radius 3.0 \
        --max-modules 12 --max-depth 12 \
        --seed $SEED \
        --no-video
"

echo "Finished: $(date)"
echo "done"
