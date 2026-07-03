#!/bin/bash
#
# Symmetry-pressure investigation — full experiment sweep.
#
# Design:
#   task           × [locomotion, food]                              = 2
#   strategy-type  × [plus, comma, nsga2-efficiency, nsga2-symmetry] = 4
#   brain-budget   × [200, 500, 1000]                                = 3
#   repeat-eval    × [off, on]                                        = 2
#   ─────────────────────────────────────────────────────────────────
#   unique conditions = 2 × 4 × 3 × 2 = 48
#   × 5 reps = 240 total jobs
#
# Array index encoding — rep is slowest so one pass over all conditions
# completes before any condition gets its second rep:
#   TASK_IDX     = SLURM_ARRAY_TASK_ID % 2              (0–1)
#   STRATEGY_IDX = (SLURM_ARRAY_TASK_ID / 2) % 4        (0–3)
#   BUDGET_IDX   = (SLURM_ARRAY_TASK_ID / 8) % 3        (0–2)
#   REEVAL_IDX   = (SLURM_ARRAY_TASK_ID / 24) % 2       (0–1)
#   REP          = (SLURM_ARRAY_TASK_ID / 48) % 5       (0–4)

#SBATCH --job-name=ariel-sym
#SBATCH --output=out_files/ariel-sym-%A_%a.out
#SBATCH --error=out_files/ariel-sym-%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G
#SBATCH --array=0-239
# MuJoCo's EGL renderer requires /dev/dri/renderD* nodes (only exposed when
# a GPU is allocated), even though we do no GPU compute.
#SBATCH --gres=gpu:1

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

TMP_DIR=/tmp/${USER}/ariel_sym_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=/scratch/jed/ariel_symmetry_investigation
RUN_TAG=sym_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# ── Decode array index ────────────────────────────────────────────────────────

ID=$SLURM_ARRAY_TASK_ID

TASK_IDX=$(( ID % 2 ))
STRATEGY_IDX=$(( (ID / 2) % 4 ))
BUDGET_IDX=$(( (ID / 8) % 3 ))
REEVAL_IDX=$(( (ID / 24) % 2 ))
REP=$(( (ID / 48) % 5 ))

TASKS=(gecko_locomotion.py gecko_food.py)
STRATEGIES=(plus comma nsga2-efficiency nsga2-symmetry)
BUDGETS=(200 500 1000)
REEVALS=(0 1)   # 0 = off, 1 = on (re-evaluate parents each generation)

SCRIPT=${TASKS[$TASK_IDX]}
STRATEGY=${STRATEGIES[$STRATEGY_IDX]}
BRAIN_BUDGET=${BUDGETS[$BUDGET_IDX]}
REPEAT_EVALS=${REEVALS[$REEVAL_IDX]}
SEED=$((42 + REP))

# ── Environment ───────────────────────────────────────────────────────────────

cd "$REPO"
mkdir -p out_files

source "$VENV_PATH/bin/activate"

# Headless EGL rendering (no X display on compute nodes).
export MUJOCO_GL=egl

# ── Diagnostics ───────────────────────────────────────────────────────────────

echo "========================================================"
echo "Node:           $(hostname)"
echo "Job:            $SLURM_JOB_ID   Array: $SLURM_ARRAY_TASK_ID"
echo "Script:         $SCRIPT"
echo "Strategy:       $STRATEGY"
echo "Brain budget:   $BRAIN_BUDGET"
echo "Repeat evals:   $REPEAT_EVALS"
echo "Rep / Seed:     $REP / $SEED"
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
    python \"$SCRIPTS_DIR/$SCRIPT\" \
        --strategy-type $STRATEGY \
        --budget 30 \
        --pop 20 --lam 20 \
        --brain-budget $BRAIN_BUDGET \
        --brain-pop 20 \
        --brain-workers $SLURM_CPUS_PER_TASK \
        --max-modules 25 --max-depth 25 \
        $([ "$REPEAT_EVALS" = "1" ] && echo "--repeat-evals") \
        --seed $SEED \
        --no-video
"

echo "Finished: $(date)"
echo "done"
