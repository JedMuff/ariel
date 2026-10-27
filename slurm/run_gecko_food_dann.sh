#!/bin/bash
#
# Food-collection benchmark for the distributed ANN on the fixed zoo gecko,
# trained from scratch: one DANN (dann_pos_hinge_h16 + the camera's three green
# fractions broadcast to every hinge, 753 params) drives the body directly, with
# no skills and no SkillController (compare the food task in
# slurm/run_symmetry_pressure_sweep.sh). See
# examples/symmetry_pressure/gecko_food_dann.py.
#
# Task (deterministic): SUBTASKS separate episodes from the same start, each
# with one food 2 m away, the foods evenly spaced on the circle (the first
# straight ahead of the gecko); each ends when its food is collected or after
# SUBTASK_TIME s. Fitness: each collected food scores 1 + its share of time
# left; of the missed foods, only the first in order (straight ahead, then
# clockwise) adds its progress towards it; plus COVERAGE_WEIGHT x each
# subtask's food share of the camera image, averaged over SUBTASK_TIME. The best
# candidate is then replayed on the training subtasks and on held-out angles
# halfway between them (final_eval.json).
#
# CMA-ES: x0 = 0, sigma 0.5, popsize 20, 300 generations at 20 Hz control
# (DSOCIAL_ZOO).
#
# Timing (local, 20 CPUs): ~2 s per generation with 20 s subtasks, so
# ~10 min for 300 generations.
#
# Usage:
#   sbatch slurm/run_gecko_food_dann.sh
#   sbatch --array=0 slurm/run_gecko_food_dann.sh     # one seed first
#   GENS=1000 sbatch slurm/run_gecko_food_dann.sh     # longer budget
#   OPTIMIZER=sepcma POP=50 sbatch slurm/run_gecko_food_dann.sh

#SBATCH --job-name=ariel-food-dann
#SBATCH --output=out_files/ariel-food-dann-%A_%a.out
#SBATCH --error=out_files/ariel-food-dann-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --array=0-9

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

REP=$SLURM_ARRAY_TASK_ID
SEED=$((42 + REP))
GENS=${GENS:-300}
OPTIMIZER=${OPTIMIZER:-cma}   # cma | sepcma | oes
POP=${POP:-20}
SUBTASKS=${SUBTASKS:-5}
SUBTASK_TIME=${SUBTASK_TIME:-20}
COVERAGE_WEIGHT=${COVERAGE_WEIGHT:-1}
VISION_NORM=${VISION_NORM:-sum}   # sum (direction only) | raw

TMP_DIR=/tmp/${USER}/ariel_food_dann_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=/scratch/jed/ariel_gecko_food_dann
RUN_TAG=food_dann_gecko_${OPTIMIZER}_p${POP}_t${SUBTASKS}x${SUBTASK_TIME}s_g${GENS}_rep${REP}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# ── Environment ───────────────────────────────────────────────────────────────

cd "$REPO"
mkdir -p out_files

source "$VENV_PATH/bin/activate"

# Headless EGL rendering (no X display on compute nodes).
export MUJOCO_GL=egl

# ── Diagnostics ───────────────────────────────────────────────────────────────

echo "========================================================"
echo "Node:           $(hostname)"
echo "Array job/task: $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Body / brain:   gecko (zoo) / dann + vision"
echo "Optimizer:      $OPTIMIZER (pop $POP)"
echo "Gens:           $GENS"
echo "Subtasks:       $SUBTASKS x $SUBTASK_TIME s, coverage weight $COVERAGE_WEIGHT, vision $VISION_NORM"
echo "Rep:            $REP"
echo "Seed:           $SEED"
echo "CPUs:           $SLURM_CPUS_PER_TASK"
echo "MUJOCO_GL:      $MUJOCO_GL"
echo "Tmp dir:        $TMP_DIR"
echo "Final dir:      $FINAL_DIR/$RUN_TAG"
echo "Python:         $(which python)  ($(python --version 2>&1))"
echo "Started:        $(date)"
echo "========================================================"

# ── Run (tmp → scratch on exit) ───────────────────────────────────────────────

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

    python \"$SCRIPTS_DIR/gecko_food_dann.py\" \
        --gens $GENS \
        --optimizer $OPTIMIZER \
        --pop $POP \
        --subtasks $SUBTASKS \
        --subtask-time $SUBTASK_TIME \
        --coverage-weight $COVERAGE_WEIGHT \
        --vision-norm $VISION_NORM \
        --workers $SLURM_CPUS_PER_TASK \
        --seed $SEED \
        --time-limit 3000
"

echo "Finished: $(date)"
echo "done"
