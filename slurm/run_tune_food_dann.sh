#!/bin/bash
#
# Hyperparameter search for the DANN food benchmark on the fixed zoo gecko
# (examples/symmetry_pressure/tune_food_dann.py): Optuna TPE over the fitness
# weights, training foods (number, ring rotation, distance, time), DANN hidden size and
# optimiser settings, minimising the training compute (simulated seconds)
# until the controller collects all 8 probe foods. Runs stop early when solved,
# out of budget, or hopeless (no food by HOPELESS_GEN, no new best for
# STALL_GENS).
#
# Every array task runs TRIALS trials one after another against one shared
# study (a journal file in STUDY_DIR), each trial training SEEDS seeds (42, 43,
# ...: the same for every trial) side by side, splitting the task's CPUs
# between them. Array task 0 also queues the current defaults as the
# reference trial. Default: 20 tasks x 10 trials = 200 trials, 5 seeds each.
#
# Timing (measured locally: ~64 simulated seconds per CPU-second, the same for
# hidden 16 and 32): with 5 seeds on 40 CPUs (8 workers per run) a trial takes
# ~15-26 min (a run stopped as hopeless ~5 min), so 10 trials take ~2.5-4.5 h.
# 200 trials are ~2,000-3,500 CPU-hours.
#
# Usage:
#   sbatch slurm/run_tune_food_dann.sh                 # 20 tasks x 10 trials = 200
#   sbatch --array=0-9 slurm/run_tune_food_dann.sh     # 100 trials
#   SEARCH=subtasks,ring_offset_frac GRID=1 STUDY_DIR=/scratch/jed/ariel_food_dann_ring \
#     sbatch --array=0-4 slurm/run_tune_food_dann.sh   # 6 x 5 grid, the rest at the defaults
# Report (any time, also while running):
#   python examples/symmetry_pressure/tune_food_dann.py --study-dir $STUDY_DIR --report

#SBATCH --job-name=ariel-food-tune
#SBATCH --output=out_files/ariel-food-tune-%A_%a.out
#SBATCH --error=out_files/ariel-food-tune-%A_%a.err
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --array=0-19

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

TRIALS=${TRIALS:-10}
SEEDS=${SEEDS:-5}
BUDGET=${BUDGET:-8e5}
HOPELESS_GEN=${HOPELESS_GEN:-60}
STALL_GENS=${STALL_GENS:-150}
STUDY_DIR=${STUDY_DIR:-/scratch/jed/ariel_food_dann_tune}
SEARCH=${SEARCH:-}   # comma-separated parameters; empty = all
GRID=${GRID:-0}      # 1 = every combination of SEARCH's grid values
WORKERS=$(( SLURM_CPUS_PER_TASK / SEEDS ))

EXTRA=()
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    EXTRA+=(--enqueue-baseline)
fi
if [ -n "$SEARCH" ]; then
    EXTRA+=(--search "$SEARCH")
fi
if [ "$GRID" = "1" ]; then
    EXTRA+=(--grid)
fi

cd "$REPO"
mkdir -p out_files "$STUDY_DIR"
source "$VENV_PATH/bin/activate"
export MUJOCO_GL=egl

echo "========================================================"
echo "Node:           $(hostname)"
echo "Array job/task: $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Study dir:      $STUDY_DIR"
echo "Trials:         $TRIALS x $SEEDS seeds, $WORKERS workers per run"
echo "Search:         ${SEARCH:-all} (grid $GRID)"
echo "Budget:         $BUDGET sim-s, hopeless gen $HOPELESS_GEN, stall $STALL_GENS"
echo "Started:        $(date)"
echo "========================================================"

srun python "$SCRIPTS_DIR/tune_food_dann.py" \
    --study-dir "$STUDY_DIR" \
    --trials "$TRIALS" \
    --jobs 1 \
    --seeds-per-trial "$SEEDS" \
    --workers-per-run "$WORKERS" \
    --budget-sim-s "$BUDGET" \
    --hopeless-gen "$HOPELESS_GEN" \
    --stall-gens "$STALL_GENS" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "Finished: $(date)"
