#!/bin/bash
#
# Control-stride sweep on the forward locomotion task — 3 control strides x
# 3 genome representations x N_REPS repetitions, at the same production scale
# as slurm/run_symmetry_pressure_sweep.sh. Stride 9 (~55.6Hz, the shared.py
# default) is not rerun: the forward runs of that sweep (job 37568, seeds
# 42-44) are the baseline.
#
# Stride = physics steps between controller updates. Physics runs at MuJoCo's
# default 500Hz, so 20 = 25Hz, 50 = 10Hz, 100 = 5Hz.
#
# Note: CTRL_ALPHA smoothing and the jerk penalty (JERK_THRESHOLD, mean
# |delta ctrl| per update) in shared.py are defined per controller update, not
# per second, so the stride also changes the physical smoothing time constant
# and what the jerk hurdle means.
#
# Runtime: stride-9 forward runs took ~22-29h (tree), ~32-36h (tree_symmetric)
# and ~49-69h (cppn) for 31 generations. Physics steps per episode don't depend
# on the stride; only network updates get rarer, so these are upper bounds.
# Genomes are ordered slowest-first so the cppn jobs start early.
#
# Dry-run a few indices first, e.g.:
#   sbatch --array=0-2 slurm/run_control_stride_sweep.sh   # cppn, all 3 strides, rep 0
#
# Usage:
#   sbatch slurm/run_control_stride_sweep.sh

#SBATCH --job-name=ariel-ctrl-stride
#SBATCH --output=out_files/ariel-ctrl-stride-%A_%a.out
#SBATCH --error=out_files/ariel-ctrl-stride-%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-26

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

# ── Factorial index decoding ─────────────────────────────────────────────────
# rep is the slowest-varying index, so one full pass over stride x genome
# completes before any condition repeats.

N_REPS=3
STRIDES=(20 50 100)
GENOME_TYPES=(cppn tree_symmetric tree)

ID=$SLURM_ARRAY_TASK_ID
STRIDE_IDX=$(( ID % 3 ))
GENOME_IDX=$(( (ID / 3) % 3 ))
REP=$(( (ID / 9) % N_REPS ))

STRIDE=${STRIDES[$STRIDE_IDX]}
GENOME_TYPE=${GENOME_TYPES[$GENOME_IDX]}
SEED=$((42 + REP))   # same seeds as the stride-9 baseline runs

TMP_DIR=/tmp/${USER}/ariel_ctrlstride_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=/scratch/jed/ariel_control_stride_sweep
RUN_TAG=ctrlstride_forward_${GENOME_TYPE}_s${STRIDE}_rep${REP}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

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
echo "Task:           forward"
echo "Control stride: $STRIDE steps (~$(awk "BEGIN{printf \"%.1f\", 500/$STRIDE}")Hz)"
echo "Genome type:    $GENOME_TYPE"
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

    python \"$SCRIPTS_DIR/gecko_skill_tasks.py\" \
        --task forward \
        --control-step-freq $STRIDE \
        --strategy-type plus \
        --budget 30 \
        --pop 10 --lam 20 \
        --repeat-evals \
        --brain-workers $SLURM_CPUS_PER_TASK \
        --max-modules 25 --max-depth 25 \
        --genome-type $GENOME_TYPE \
        --seed $SEED \
        --time-limit 259200
"

echo "Finished: $(date)"
echo "done"
