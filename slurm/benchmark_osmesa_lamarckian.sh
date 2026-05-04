#!/bin/bash

#SBATCH --job-name=ariel-bbl-bench
#SBATCH --output=out_files/ariel-bbl-bench-%A.out
#SBATCH --error=out_files/ariel-bbl-bench-%A.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=31G

set -euo pipefail

REPO=/home/jed/workspaces/ariel
VENV_PATH=$REPO/.venv

TMP_DIR=/tmp/${USER}/ariel_bbl_bench_${SLURM_JOB_ID}
FINAL_DIR=/scratch/jed/ariel_experiments_01052026
RUN_TAG=body_brain_lamarckian_bench_${SLURM_JOB_ID}
SEED=42

cd "$REPO"
mkdir -p out_files

source "$VENV_PATH/bin/activate"

# Headless software rendering: no GPU, no display. Slower per frame than EGL,
# but doesn't need /dev/dri access — so we don't need --gres=gpu:1.
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

echo "Node:       $(hostname)"
echo "Job:        $SLURM_JOB_ID  Seed: $SEED"
echo "Python:     $(which python)"
python --version
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "MUJOCO_GL:  $MUJOCO_GL"
echo "Tmp dir:    $TMP_DIR"
echo "Final:      $FINAL_DIR/$RUN_TAG"
echo "Started:    $(date)"

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
        --budget 1 \
        --pop 16 --lam 16 \
        --brain-budget 75 --brain-pop 20 \
        --brain-workers $SLURM_CPUS_PER_TASK \
        --dur 45.0 --reach-radius 0.20 --num-waypoints 3 --arena-radius 3.0 \
        --max-modules 12 --max-depth 12 \
        --seed $SEED \
        --no-video
"

echo "Finished: $(date)"
echo "done"
