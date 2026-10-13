#!/bin/bash
#
# Rowing-penalty experiment: forward locomotion at 20Hz control (stride 25)
# with the ANN brain, on bodies built with kgd's zoo modules: module sizes,
# masses and hinge (kp, kv, armature, +-90 deg joint limit)
# (gecko_skill_tasks.py --zoo-modules, shared.zoo_modules_spec).
#
# Penalty arm: a skill's forward displacement is scaled by 1 - rowing_frac
# (--rowing-penalty). rowing_frac is the share of the stroke that
# rowing_detect.py calls rowing (limbs on both sides swept in phase; see
# examples/symmetry_pressure/ROWING.md), computed on the episode's 50Hz body
# poses. A full rower scores ~0; a backward displacement is never scaled.
# No-penalty arm: identical but without --rowing-penalty, the baseline. (The
# stride sweep's own stride-25 ANN runs used the local modules, so they are
# not a baseline for this.)
#
# 2 genome representations (tree_symmetric = symmetry enforced, tree = not) x
# 2 arms x N_REPS repetitions = 12 jobs. Everything else matches
# slurm/run_control_stride_sweep.sh, seeds included.
#
# Results go to FINAL_DIR=/scratch/jed/ariel_rowing_penalty_ann/<arm> (arm =
# penalty | nopenalty) with the same run tags as the stride sweep
# (ctrlstride_forward_<genome>_s25_rep<r>_...), so each arm directory works
# unchanged as --stride-root for behavior_extract.py, rowing_detect.py and the
# plot scripts.
#
# Dry-run one index per arm first:
#   sbatch --array=0,2 slurm/run_rowing_penalty.sh
#
# Usage:
#   sbatch slurm/run_rowing_penalty.sh

#SBATCH --job-name=ariel-rowing-pen
#SBATCH --output=out_files/ariel-rowing-pen-%A_%a.out
#SBATCH --error=out_files/ariel-rowing-pen-%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-11

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

# ── Factorial index decoding ─────────────────────────────────────────────────
# rep is the slowest-varying index, so one full pass over genome x arm
# completes before any condition repeats.

N_REPS=3
STRIDE=25                            # 500Hz physics / 25 = 20Hz control
GENOME_TYPES=(tree_symmetric tree)
ARMS=(penalty nopenalty)
N_GENOMES=${#GENOME_TYPES[@]}
N_ARMS=${#ARMS[@]}

ID=$SLURM_ARRAY_TASK_ID
GENOME_IDX=$(( ID % N_GENOMES ))
ARM_IDX=$(( (ID / N_GENOMES) % N_ARMS ))
REP=$(( (ID / (N_GENOMES * N_ARMS)) % N_REPS ))

GENOME_TYPE=${GENOME_TYPES[$GENOME_IDX]}
ARM=${ARMS[$ARM_IDX]}
SEED=$((42 + REP))   # same seeds as the stride sweep
BRAIN=ann
PENALTY_FLAG=""
if [ "$ARM" = penalty ]; then PENALTY_FLAG="--rowing-penalty"; fi

TMP_DIR=/tmp/${USER}/ariel_rowingpen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=/scratch/jed/ariel_rowing_penalty_${BRAIN}/${ARM}
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
echo "Brain:          $BRAIN"
echo "Arm:            $ARM"
echo "Modules:        zoo (kgd sizes, masses, hinge)"
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
        --brain $BRAIN \
        --zoo-modules $PENALTY_FLAG \
        --time-limit 259200
"

echo "Finished: $(date)"
echo "done"
