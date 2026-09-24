#!/bin/bash
#
# Symmetry-pressure investigation — integration test run.
#
# Five locomotion jobs (one per strategy + new-fitness plus), using the
# parameters from INTEGRATION_TEST_PLAN.md.  Each runs for ~20-25 min on
# 32 CPUs.
#
# Submit:
#   cd /home/jed/workspaces/ariel
#   sbatch slurm/run_symmetry_integration_test.sh
#
# Array index → strategy:
#   0 = plus               (original fitness)
#   1 = comma              (original fitness)
#   2 = nsga2-efficiency   (original fitness)
#   3 = nsga2-symmetry     (original fitness)
#   4 = plus-new-fitness   (new fitness: x_disp - init_height - 0.005*c_hinge)

#SBATCH --job-name=ariel-sym-inttest
#SBATCH --output=out_files/ariel-sym-inttest-%A_%a.out
#SBATCH --error=out_files/ariel-sym-inttest-%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=31G
#SBATCH --array=0-4
#SBATCH --gres=gpu:1

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

STRATEGIES=(plus comma nsga2-efficiency nsga2-symmetry plus)
SEEDS=(42 43 44 45 42)

STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Tag distinguishes the new-fitness run from the original plus run
if [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    RUN_LABEL=plus-new-fitness
else
    RUN_LABEL=$STRATEGY
fi

TMP_DIR=/tmp/${USER}/ariel_inttest_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=/scratch/jed/ariel_symmetry_integration_test
RUN_TAG=inttest_${RUN_LABEL}_${SLURM_JOB_ID}

# ── Environment ───────────────────────────────────────────────────────────────

cd "$REPO"
mkdir -p out_files

source "$VENV_PATH/bin/activate"

export MUJOCO_GL=egl

# ── Diagnostics ───────────────────────────────────────────────────────────────

echo "========================================================"
echo "Node:           $(hostname)"
echo "Job:            $SLURM_JOB_ID   Array: $SLURM_ARRAY_TASK_ID"
echo "Strategy:       $STRATEGY  (label: $RUN_LABEL)"
echo "Seed:           $SEED"
echo "CPUs:           $SLURM_CPUS_PER_TASK"
echo "MUJOCO_GL:      $MUJOCO_GL"
echo "Tmp dir:        $TMP_DIR"
echo "Final dir:      $FINAL_DIR/$RUN_TAG"
echo "Python:         $(which python)  ($(python --version 2>&1))"
echo "Started:        $(date)"
echo "========================================================"

# ── Run ───────────────────────────────────────────────────────────────────────

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
    python \"$SCRIPTS_DIR/gecko_locomotion.py\" \
        --strategy-type $STRATEGY \
        --budget 15 \
        --pop 10 --lam 10 \
        --brain-budget 100 \
        --brain-pop 20 \
        --brain-workers $SLURM_CPUS_PER_TASK \
        --dur 10.0 \
        --max-modules 25 --max-depth 25 \
        --num-evals 1 \
        --seed $SEED \
        --no-video
"

echo "Finished: $(date)"
echo "done"
