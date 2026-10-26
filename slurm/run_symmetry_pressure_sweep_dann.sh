#!/bin/bash
#
# Symmetry-pressure task x genome sweep with the distributed ANN (DANN) brain:
# 3 tasks x 3 genome representations x N_REPS repetitions.
#
# Same outer EA as slurm/run_symmetry_pressure_sweep.sh (mu=10, lambda=20,
# budget 30, --repeat-evals), but each skill trains the DANN (--brain dann,
# dann_pos_hinge_h16: one 705-param MLP shared by every hinge) under
# --eval-profile dsocial_zoo (shared.DSOCIAL_ZOO): d_social_learning's zoo
# preset (zoo modules, 15 s episodes at 20 Hz control; 5 s for turns) and inner CMA-ES
# (10 gens, popsize 20, sigma 0.5 from x0 = 0), with translation scored in
# [0, 2] (2 x displacement over 0.5 m/s x duration) and turns in [0, 1] (net
# yaw over 540 deg, ending the episode at 540 deg).
#
# Tasks:   forward | multidirection | turn_avg  (examples/symmetry_pressure/gecko_skill_tasks.py)
# Genomes: tree | tree_symmetric | cppn          (examples/symmetry_pressure/genome_adapter.py)
#
# Timing (local, tree_symmetric, 20 workers per run with three runs side by
# side): 30 generations took 30 min (forward); turn_avg took 57 min for 20,
# so ~1.4 h for 30; multidirection ran ~5.4 min per generation, so ~2.7 h.
# --time-limit stops after the generation in progress at 7 h.
#
# Usage:
#   sbatch slurm/run_symmetry_pressure_sweep_dann.sh
#   sbatch --array=0-2 slurm/run_symmetry_pressure_sweep_dann.sh   # one rep, 3 tasks x tree
#
# Results go to FINAL_DIR with the main sweep's run tags; point the analysis
# scripts at it with ARIEL_SWEEP_ROOT=/scratch/jed/ariel_symmetry_pressure_sweep_dann.

#SBATCH --job-name=ariel-sympress-dann
#SBATCH --output=out_files/ariel-sympress-dann-%A_%a.out
#SBATCH --error=out_files/ariel-sympress-dann-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --array=0-44

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

# ── Factorial index decoding ─────────────────────────────────────────────────
# rep is the slowest-varying index, so one full pass over task x genome
# completes before any condition repeats (matches the convention established
# in slurm/archive/run_symmetry_investigation.sh).

N_REPS=5
TASKS=(forward multidirection turn_avg)
GENOME_TYPES=(tree tree_symmetric cppn)

ID=$SLURM_ARRAY_TASK_ID
TASK_IDX=$(( ID % 3 ))
GENOME_IDX=$(( (ID / 3) % 3 ))
REP=$(( (ID / 9) % N_REPS ))

TASK=${TASKS[$TASK_IDX]}
GENOME_TYPE=${GENOME_TYPES[$GENOME_IDX]}
SEED=$((42 + REP))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

TMP_DIR=/tmp/${USER}/ariel_sympress_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=/scratch/jed/ariel_symmetry_pressure_sweep_dann
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
        --time-limit 25200
"

echo "Finished: $(date)"
echo "done"
