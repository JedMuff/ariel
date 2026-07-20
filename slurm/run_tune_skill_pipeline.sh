#!/bin/bash
#
# Multi-objective HPO for the gecko skill-controller pipeline.
#
# Tunes: network architecture (layers, hidden sizes), CMA-ES sigma / pop /
#        loco budget / turn budget, and SkillController commit-steps.
# Objectives: maximise mean food-task score AND minimise total wall-clock time.
# Each trial is capped at TRIAL_TIME_LIMIT seconds (default 600 = 10 min).
#
# Usage:
#   sbatch slurm/run_tune_skill_pipeline.sh

#SBATCH --job-name=ariel-tune-skills
#SBATCH --output=slurm_out/tune-skills-%j.out
#SBATCH --error=slurm_out/tune-skills-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
# MuJoCo's EGL renderer requires /dev/dri/renderD* nodes (only exposed when
# a GPU is allocated), even though we do no GPU compute.
#SBATCH --gres=gpu:1
#SBATCH --exclude=node01

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/skill_controller_hpo

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR=$REPO/__data__/tune_skill_pipeline_${TIMESTAMP}

# ── Environment ───────────────────────────────────────────────────────────────

cd "$REPO"
mkdir -p slurm_out
mkdir -p "$DATA_DIR"

source "$VENV_PATH/bin/activate"

# Headless EGL rendering (no X display on compute nodes).
export MUJOCO_GL=egl

# ── Diagnostics ───────────────────────────────────────────────────────────────

echo "========================================================"
echo "Node:             $(hostname)"
echo "Job:              $SLURM_JOB_ID"
echo "CPUs:             $SLURM_CPUS_PER_TASK"
echo "MUJOCO_GL:        $MUJOCO_GL"
echo "Data dir:         $DATA_DIR"
echo "Python:           $(which python)  ($(python --version 2>&1))"
echo "Started:          $(date)"
echo "========================================================"

# ── Run ───────────────────────────────────────────────────────────────────────

srun python "$SCRIPTS_DIR/tune_skill_pipeline.py" \
    --n-trials 100 \
    --n-eval-runs 5 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --eval-duration 120.0 \
    --num-waypoints 10 \
    --reach-radius 0.20 \
    --collect-radius 0.40 \
    --loco-budget-max 500 \
    --turn-budget-max 300 \
    --trial-time-limit 600 \
    --out-dir "$DATA_DIR"

echo "Finished: $(date)"
echo "done"
