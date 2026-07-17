#!/bin/bash
#
# Multi-objective HPO: CMA-ES + ANN on gecko tasks.
#
# Tunes: sigma, population size, budget, weight init scale,
#        number of hidden layers (1-3), hidden layer sizes (1-64 each).
# Objectives: mean best-fitness and mean wall-clock time (both minimised).
# Optuna manages all parallelism internally (n_jobs=64, JournalStorage).
#
# Set TASK=food or TASK=loco before submitting:
#   TASK=food sbatch slurm/run_tune_cma_ann.sh
#   TASK=loco sbatch slurm/run_tune_cma_ann.sh

#SBATCH --job-name=ariel-tune
#SBATCH --output=out_files/tune-%j.out
#SBATCH --error=out_files/tune-%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
# MuJoCo's EGL renderer requires /dev/dri/renderD* nodes (only exposed when
# a GPU is allocated), even though we do no GPU compute.
#SBATCH --gres=gpu:1
#SBATCH --exclude=node01

set -euo pipefail

TASK=${TASK:-food}   # override with: TASK=loco sbatch ...

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR=$REPO/__data__/tune_cma_ann_${TASK}_${TIMESTAMP}

# ── Environment ───────────────────────────────────────────────────────────────

cd "$REPO"
mkdir -p out_files

source "$VENV_PATH/bin/activate"

# Headless EGL rendering (no X display on compute nodes).
export MUJOCO_GL=egl

# ── Diagnostics ───────────────────────────────────────────────────────────────

echo "========================================================"
echo "Node:      $(hostname)"
echo "Job:       $SLURM_JOB_ID"
echo "Task:      $TASK"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "MUJOCO_GL: $MUJOCO_GL"
echo "Data dir:  $DATA_DIR"
echo "Python:    $(which python)  ($(python --version 2>&1))"
echo "Started:   $(date)"
echo "========================================================"

# ── Run (tmp → scratch on exit) ───────────────────────────────────────────────

mkdir -p "$DATA_DIR"

srun python "$SCRIPTS_DIR/tune_cma_ann.py" \
    --task $TASK \
    --n-trials 200 \
    --runs-per-trial 5 \
    --workers $SLURM_CPUS_PER_TASK \
    --trial-time-limit 300 \
    --seed 42 \
    --dur 60.0 \
    $([ "$TASK" = "food" ] && echo "--num-waypoints 10 --reach-radius 0.20 --arena-radius 2.0") \
    --out-dir "$DATA_DIR"

echo "Finished: $(date)"
echo "done"
