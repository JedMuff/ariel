#!/bin/bash
#
# Symmetry-pressure task x genome sweep — 4 tasks x 3 genome representations
# x N_REPS repetitions, run at production scale (same outer-EA/skill-training
# budgets as slurm/run_gecko_food_skills.sh) so all four tasks are directly
# comparable.
#
# Tasks:   forward | multidirection | turn_avg  (examples/symmetry_pressure/gecko_skill_tasks.py)
#          food                                 (examples/symmetry_pressure/gecko_food_skills.py)
# Genomes: tree | tree_symmetric | cppn          (examples/symmetry_pressure/genome_adapter.py)
#
# WARNING: this is 4 * 3 * N_REPS production-scale jobs, each up to 3 days
# (--time-limit 259200 below). Before submitting the full array, dry-run a
# couple of indices first, e.g.:
#   sbatch --array=0-3 slurm/run_symmetry_pressure_sweep.sh   # one rep, all 4 tasks x genome 0
#
# Usage:
#   sbatch slurm/run_symmetry_pressure_sweep.sh

#SBATCH --job-name=ariel-sympress-sweep
#SBATCH --output=out_files/ariel-sympress-sweep-%A_%a.out
#SBATCH --error=out_files/ariel-sympress-sweep-%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-59
#SBATCH --nodelist=node02,node03

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

# ── Factorial index decoding ─────────────────────────────────────────────────
# rep is the slowest-varying index, so one full pass over task x genome
# completes before any condition repeats (matches the convention established
# in slurm/archive/run_symmetry_investigation.sh).

N_REPS=5
TASKS=(forward multidirection turn_avg food)
GENOME_TYPES=(tree tree_symmetric cppn)

ID=$SLURM_ARRAY_TASK_ID
TASK_IDX=$(( ID % 4 ))
GENOME_IDX=$(( (ID / 4) % 3 ))
REP=$(( (ID / 12) % N_REPS ))

TASK=${TASKS[$TASK_IDX]}
GENOME_TYPE=${GENOME_TYPES[$GENOME_IDX]}
SEED=$((42 + REP))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

TMP_DIR=/tmp/${USER}/ariel_sympress_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=/scratch/jed/ariel_symmetry_pressure_sweep
RUN_TAG=sympress_${TASK}_${GENOME_TYPE}_rep${REP}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

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
echo "Task:           $TASK"
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

    if [ \"$TASK\" = \"food\" ]; then
        python \"$SCRIPTS_DIR/gecko_food_skills.py\" \
            --strategy-type plus \
            --budget 30 \
            --pop 10 --lam 20 \
            --repeat-evals \
            --brain-workers $SLURM_CPUS_PER_TASK \
            --max-modules 25 --max-depth 25 \
            --genome-type $GENOME_TYPE \
            --seed $SEED \
            --no-video \
            --time-limit 259200
    else
        python \"$SCRIPTS_DIR/gecko_skill_tasks.py\" \
            --task $TASK \
            --strategy-type plus \
            --budget 30 \
            --pop 10 --lam 20 \
            --repeat-evals \
            --brain-workers $SLURM_CPUS_PER_TASK \
            --max-modules 25 --max-depth 25 \
            --genome-type $GENOME_TYPE \
            --seed $SEED \
            --time-limit 259200
    fi
"

echo "Finished: $(date)"
echo "done"
