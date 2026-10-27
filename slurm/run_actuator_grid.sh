#!/bin/bash
#
# Hinge-actuator grid: where does the symmetric advantage on forward come from?
# kp x kv x armature (18 cells, joint limit on) x genome {tree, tree_symmetric}
# x brain {dann, bang_bang} x N_REPS repetitions, all at 20 Hz control.
#
# Everything but the hinge is the DANN sweep's forward setup
# (slurm/run_symmetry_pressure_sweep_dann.sh): --eval-profile dsocial_zoo, so
# zoo module sizes and masses, stride 25 (20 Hz), 15 s episodes, inner CMA-ES
# 10 gens x 20 (sigma 0.5; x0 = 0 for dann, 0.5 for bang_bang, see
# shared.BRAIN_X0), outer mu 10 lambda 20, budget 30, --repeat-evals. The hinge
# is set with --hinge-kp/--hinge-kv/--hinge-armature/--[no-]hinge-limit
# (shared.HingeActuator) and recorded in run_config.json.
#
# The grid holds the zoo hinge (kp 1.36, kv 0.356, armature 0.081, limit;
# exact values below) and the old local hinge's kp 1, kv 1, armature 0, but
# with the joint limit on: the old hinge has none. On zoo bodies the limit is
# a near no-op (square-wave swing within 7 deg with or without it, every
# cell), so the limit axis is off by default; LIMS="1 0" brings it back (36
# cells).
# Seeds are SEED_BASE + rep in every cell (common random numbers).
#
# Index decoding: cell fastest, then genome, then brain, rep slowest, so one
# pass over cells x genome x brain finishes before any rep repeats.
# Array size = n_cells x 2 x 2 x N_REPS (18 x 4 x 3 = 216 by default).
#
# Usage:
#   sbatch slurm/run_actuator_grid.sh                       # full grid, reps 0-2
#   sbatch --array=0-215%60 slurm/run_actuator_grid.sh      # throttled
#   LIMS="1 0" sbatch --array=0-431 slurm/run_actuator_grid.sh   # with the limit axis
#   # confirmation stage: chosen cells (indices into the KPS x KVS x ARMS x LIMS
#   # product, limit fastest), 5 fresh seeds:
#   CELLS="4 13 30" SEED_BASE=45 N_REPS=5 sbatch --array=0-59 slurm/run_actuator_grid.sh
#
# Timing (local, 20 workers): see the plan; ~1 min per outer generation, so
# ~30-40 min per run. --time-limit stops after the generation in progress at 7 h.
#
# Results go to FINAL_DIR; analyse with
#   examples/symmetry_pressure/plot_actuator_grid.py --root $FINAL_DIR

#SBATCH --job-name=ariel-actgrid
#SBATCH --output=out_files/ariel-actgrid-%A_%a.out
#SBATCH --error=out_files/ariel-actgrid-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --array=0-215

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

# ── Grid (each axis overridable as a space-separated env var) ────────────────

# The zoo hinge's exact values (shared.ZOO_HINGE_*), so its cell is the DANN sweep's hinge.
ZOO_KP=1.3605113362038874
ZOO_KV=0.3555497840137177
ZOO_ARM=0.08117047736037042
read -r -a KPS    <<< "${KPS:-1 $ZOO_KP 2.5}"
read -r -a KVS    <<< "${KVS:-$ZOO_KV 1 2}"
read -r -a ARMS   <<< "${ARMS:-0 $ZOO_ARM}"
read -r -a LIMS   <<< "${LIMS:-1}"
GENOME_TYPES=(tree tree_symmetric)
BRAINS=(dann bang_bang)
N_REPS=${N_REPS:-3}
SEED_BASE=${SEED_BASE:-42}

# All cells of the product (limit fastest, kp slowest), or the CELLS subset.
ALL_CELLS=()
for kp in "${KPS[@]}"; do for kv in "${KVS[@]}"; do for arm in "${ARMS[@]}"; do for lim in "${LIMS[@]}"; do
    ALL_CELLS+=("$kp $kv $arm $lim")
done; done; done; done
if [ -n "${CELLS:-}" ]; then
    read -r -a CELL_IDS <<< "$CELLS"
else
    CELL_IDS=($(seq 0 $(( ${#ALL_CELLS[@]} - 1 ))))
fi
N_CELLS=${#CELL_IDS[@]}

ID=$SLURM_ARRAY_TASK_ID
CELL_POS=$(( ID % N_CELLS ))
GENOME_IDX=$(( (ID / N_CELLS) % 2 ))
BRAIN_IDX=$(( (ID / (N_CELLS * 2)) % 2 ))
REP=$(( ID / (N_CELLS * 4) ))
if [ "$REP" -ge "$N_REPS" ]; then
    echo "Array index $ID is past $N_CELLS cells x 2 genomes x 2 brains x $N_REPS reps; nothing to do"
    exit 0
fi

read -r KP KV ARM LIM <<< "${ALL_CELLS[${CELL_IDS[$CELL_POS]}]}"
GENOME_TYPE=${GENOME_TYPES[$GENOME_IDX]}
BRAIN=${BRAINS[$BRAIN_IDX]}
SEED=$((SEED_BASE + REP))
if [ "$LIM" = 1 ]; then LIMIT_FLAG=--hinge-limit; else LIMIT_FLAG=--no-hinge-limit; fi
LABEL=$(LC_ALL=C printf "kp%g_kv%g_arm%g_lim%s" "$KP" "$KV" "$ARM" "$LIM")  # = shared.HingeActuator.label

TMP_DIR=/tmp/${USER}/ariel_actgrid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=${FINAL_DIR:-/scratch/jed/ariel_actuator_grid}
RUN_TAG=actgrid_${BRAIN}_${GENOME_TYPE}_${LABEL}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

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
echo "Hinge:          kp=$KP kv=$KV armature=$ARM limit=$LIM  (cell ${CELL_IDS[$CELL_POS]})"
echo "Genome type:    $GENOME_TYPE"
echo "Brain:          $BRAIN (eval profile dsocial_zoo)"
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
        --task forward \
        --strategy-type plus \
        --budget 30 \
        --pop 10 --lam 20 \
        --repeat-evals \
        --brain-workers $SLURM_CPUS_PER_TASK \
        --max-modules 25 --max-depth 25 \
        --genome-type $GENOME_TYPE \
        --seed $SEED \
        --brain $BRAIN \
        --eval-profile dsocial_zoo \
        --hinge-kp $KP --hinge-kv $KV --hinge-armature $ARM $LIMIT_FLAG \
        --time-limit 25200
"

echo "Finished: $(date)"
echo "done"
