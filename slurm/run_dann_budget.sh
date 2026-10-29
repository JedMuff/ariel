#!/bin/bash
#
# DANN inner-budget series on forward: does the symmetric advantage need a
# well-trained controller? 2 skill budgets x 2 genomes x N_REPS repetitions.
#
# Same setup as the forward runs of slurm/run_symmetry_pressure_sweep_dann.sh
# (--brain dann --eval-profile dsocial_zoo: zoo modules and hinge, 15 s
# episodes at 20 Hz, CMA-ES popsize 20 from x0 = 0, outer mu 10 lambda 20,
# budget 30, --repeat-evals, seeds 42 + rep), except the inner CMA-ES budget:
# --skill-budget 40 (800 evaluations per body) or 80 (1600, the original
# sweep profile's 100 x 16). The DANN sweep is the 10-generation (200
# evaluations) point. In the zoo benchmark (__data__/ariel_zoo_benchmark/
# zoo_fitness_aggregate.png) the DANN is at ~0.08 m/s after 200 evaluations
# and ~0.21 m/s after 1600, so 200 compares bodies early in learning.
#
# Timing: ~43 min per run at 10 generations (local, 20 workers), so ~3 h at
# 40 and ~6 h at 80. --time-limit stops after the generation in progress at 12 h.
#
# Index decoding: genome fastest, then skill budget, rep slowest.
#
# Usage:
#   sbatch slurm/run_dann_budget.sh
#   SKILL_BUDGETS="80" sbatch --array=0-9 slurm/run_dann_budget.sh   # 1600 evals only
#
# Results go to FINAL_DIR/sb<budget>/ with the main sweep's run tags, so the
# sweep analysis scripts read each budget with
#   ARIEL_SWEEP_ROOT=/scratch/jed/ariel_dann_budget/sb80

#SBATCH --job-name=ariel-dann-budget
#SBATCH --output=out_files/ariel-dann-budget-%A_%a.out
#SBATCH --error=out_files/ariel-dann-budget-%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --array=0-19

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

# ── Factorial index decoding ─────────────────────────────────────────────────

read -r -a SKILL_BUDGETS <<< "${SKILL_BUDGETS:-40 80}"
GENOME_TYPES=(tree tree_symmetric)
N_REPS=${N_REPS:-5}
N_BUDGETS=${#SKILL_BUDGETS[@]}

ID=$SLURM_ARRAY_TASK_ID
GENOME_IDX=$(( ID % 2 ))
BUDGET_IDX=$(( (ID / 2) % N_BUDGETS ))
REP=$(( ID / (2 * N_BUDGETS) ))
if [ "$REP" -ge "$N_REPS" ]; then
    echo "Array index $ID is past 2 genomes x $N_BUDGETS budgets x $N_REPS reps; nothing to do"
    exit 0
fi

TASK=forward
GENOME_TYPE=${GENOME_TYPES[$GENOME_IDX]}
SKILL_BUDGET=${SKILL_BUDGETS[$BUDGET_IDX]}
SEED=$((42 + REP))

TMP_DIR=/tmp/${USER}/ariel_dannbudget_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=${FINAL_DIR:-/scratch/jed/ariel_dann_budget}/sb${SKILL_BUDGET}
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
echo "Brain:          dann (eval profile dsocial_zoo)"
echo "Skill budget:   $SKILL_BUDGET CMA-ES generations x 20"
echo "Rep / seed:     $REP / $SEED"
echo "CPUs:           $SLURM_CPUS_PER_TASK"
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
        --task $TASK \
        --strategy-type plus \
        --budget 30 \
        --pop 10 --lam 20 \
        --repeat-evals \
        --brain-workers $SLURM_CPUS_PER_TASK \
        --max-modules 25 --max-depth 25 \
        --genome-type $GENOME_TYPE \
        --seed $SEED \
        --brain dann \
        --eval-profile dsocial_zoo \
        --skill-budget $SKILL_BUDGET \
        --time-limit 43200
"

echo "Finished: $(date)"
echo "done"
