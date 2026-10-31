#!/bin/bash
#
# E0 hinge-actuator grid (R1): kp x kv x armature (18 cells, joint limit on)
# x genome {tree, tree_symmetric} x brain {dann, bang_bang} x N_REPS reps,
# forward, 200 evaluations per body. Reproduces the investigation's
# slurm/run_actuator_grid.sh (results in __data__/ariel_actuator_grid);
# only rerun it to regenerate that data from this branch.
#
# v1 protocol (normalised fitness, (10+20) with --repeat-evals): set
# FINAL_ROOT=/scratch/$USER/ariel_paper to write next to the other v1 runs.
# The paper's v2 grid is e0_dann_grid.sh.
#
# Index decoding: cell fastest, then genome, then brain, rep slowest.
# Array size = 18 x 2 x 2 x N_REPS (216 for 3 reps). Seeds 42 + rep.
#
# Usage:
#   sbatch slurm/symmetry_paper/e0_actuator_grid.sh
#   CELLS="4 13" SEED_BASE=45 N_REPS=5 sbatch --array=0-19 slurm/symmetry_paper/e0_actuator_grid.sh
# Analyse:
#   python examples/symmetry_paper/analyse_e0.py --grid-root $FINAL_ROOT/e0_grid

#SBATCH --job-name=paper-e0-grid
#SBATCH --output=out_files/paper-e0-grid-%A_%a.out
#SBATCH --error=out_files/paper-e0-grid-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-215

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/_common.sh"

# The zoo hinge's exact values (sim.ZOO_HINGE_*).
ZOO_KP=1.3605113362038874
ZOO_KV=0.3555497840137177
ZOO_ARM=0.08117047736037042
read -r -a KPS  <<< "${KPS:-1 $ZOO_KP 2.5}"
read -r -a KVS  <<< "${KVS:-$ZOO_KV 1 2}"
read -r -a ARMS <<< "${ARMS:-0 $ZOO_ARM}"
GENOME_TYPES=(tree tree_symmetric)
BRAINS=(dann bang_bang)
N_REPS=${N_REPS:-3}
SEED_BASE=${SEED_BASE:-42}

ALL_CELLS=()
for kp in "${KPS[@]}"; do for kv in "${KVS[@]}"; do for arm in "${ARMS[@]}"; do
    ALL_CELLS+=("$kp $kv $arm")
done; done; done
if [ -n "${CELLS:-}" ]; then read -r -a CELL_IDS <<< "$CELLS"; else CELL_IDS=($(seq 0 $(( ${#ALL_CELLS[@]} - 1 )))); fi
N_CELLS=${#CELL_IDS[@]}

ID=$SLURM_ARRAY_TASK_ID
CELL_POS=$(( ID % N_CELLS ))
GENOME=${GENOME_TYPES[$(( (ID / N_CELLS) % 2 ))]}
BRAIN=${BRAINS[$(( (ID / (N_CELLS * 2)) % 2 ))]}
REP=$(( ID / (N_CELLS * 4) ))
if [ "$REP" -ge "$N_REPS" ]; then echo "Index $ID past the grid; nothing to do"; exit 0; fi
read -r KP KV ARM <<< "${ALL_CELLS[${CELL_IDS[$CELL_POS]}]}"
SEED=$((SEED_BASE + REP))
LABEL=$(LC_ALL=C printf "kp%g_kv%g_arm%g_lim1" "$KP" "$KV" "$ARM")  # = sim.HingeActuator.label
TAG=actgrid_${BRAIN}_${GENOME}_${LABEL}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${ID}

setup_env
banner "Hinge:          kp=$KP kv=$KV armature=$ARM limit=1" "Genome / brain: $GENOME / $BRAIN" "Seed:           $SEED" \
       "Final dir:      $FINAL_ROOT/e0_grid/$TAG"
run_in_tmp "/tmp/$USER/paper_e0_grid_${SLURM_ARRAY_JOB_ID}_$ID" "$FINAL_ROOT/e0_grid/$TAG" \
    python "$SCRIPTS_DIR/evolve.py" --genome-type "$GENOME" --brain "$BRAIN" --seed "$SEED" \
        --budget 30 --pop 10 --lam 20 --repeat-evals --skill-budget 10 \
        --brain-workers "$SLURM_CPUS_PER_TASK" --max-modules 25 --max-depth 25 \
        --hinge-kp "$KP" --hinge-kv "$KV" --hinge-armature "$ARM" --hinge-limit --time-limit 25200
