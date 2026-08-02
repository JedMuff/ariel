#!/bin/bash
#
# Symmetry-pressure investigation — food task with a state-based skill
# controller brain (loco / turn-left / turn-right trained separately via
# CMA-ES, driven at eval time by the vision-gated SkillController).
#
# Single-configuration run — no command-strategy or brain-budget sweep.
# Skill-pipeline hyperparameters (ANN size, CMA-ES sigma/popsize, loco/turn
# budgets, commit-steps, centre-fwd-thresh) are fixed constants inside
# gecko_food_skills.py; only the outer body-evolution knobs are set here.
#
# Usage:
#   sbatch slurm/run_gecko_food_skills.sh

#SBATCH --job-name=ariel-food-skills
#SBATCH --output=out_files/ariel-food-skills-%j.out
#SBATCH --error=out_files/ariel-food-skills-%j.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
# MuJoCo's EGL renderer requires /dev/dri/renderD* nodes (only exposed when
# a GPU is allocated), even though we do no GPU compute.
#SBATCH --gres=gpu:1
#SBATCH --exclude=node01

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

TMP_DIR=/tmp/${USER}/ariel_food_skills_${SLURM_JOB_ID}
FINAL_DIR=/scratch/jed/ariel_food_skills
RUN_TAG=food_skills_${TIMESTAMP}_${SLURM_JOB_ID}

STRATEGY=plus
# y_zero mirrors RIGHT<->LEFT (true left-right bilateral symmetry, the
# biologically correct axis for a walking gecko); x_equals_y is a 45-degree
# diagonal mirror with no locomotion relevance.
SYMMETRY_AXIS=y_zero  # none | y_zero | x_equals_y
SEED=$RANDOM$RANDOM

# ── Environment ───────────────────────────────────────────────────────────────

cd "$REPO"
mkdir -p out_files

source "$VENV_PATH/bin/activate"

# Headless EGL rendering (no X display on compute nodes).
export MUJOCO_GL=egl

# ── Diagnostics ───────────────────────────────────────────────────────────────

echo "========================================================"
echo "Node:           $(hostname)"
echo "Job:            $SLURM_JOB_ID"
echo "Strategy:       $STRATEGY"
echo "Symmetry axis:  $SYMMETRY_AXIS"
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
    python \"$SCRIPTS_DIR/gecko_food_skills.py\" \
        --strategy-type $STRATEGY \
        --budget 30 \
        --pop 10 --lam 20 \
        --repeat-evals \
        --brain-workers $SLURM_CPUS_PER_TASK \
        --max-modules 25 --max-depth 25 \
        --symmetry-axis $SYMMETRY_AXIS \
        --seed $SEED \
        --no-video \
        --time-limit 259200
"

echo "Finished: $(date)"
echo "done"
