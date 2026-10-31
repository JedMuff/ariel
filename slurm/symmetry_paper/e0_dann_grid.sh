#!/bin/bash
#
# E0 hinge grid, paper v2 (R1): DANN at the largest inner budget (80 CMA-ES
# generations = 1,600 evaluations per body), kp {1, default, 2.5} x kv
# {default, 1, 2} at the default armature (joint limit on) x genome {tree,
# tree_symmetric} x N_REPS reps, forward. v2 protocol (_common.sh): fitness in
# metres, (20+20) without re-evaluation.
#
# The default-hinge cell is the 1,600-evaluation arm of e0_dann_budget.sh
# (same seeds 42 + rep), so it is skipped here unless SKIP_DEFAULT=0; analyse
# the grid with both roots.
#
# Index decoding: cell fastest, then genome, rep slowest.
# Array size = n_cells x 2 x N_REPS (8 x 2 x 5 = 80 with the default cell
# skipped; 9 x 2 x 5 = 90 with SKIP_DEFAULT=0). Seeds 42 + rep.
#
# Usage:
#   sbatch slurm/symmetry_paper/e0_dann_grid.sh
#   sbatch --array=0-79%20 slurm/symmetry_paper/e0_dann_grid.sh      # throttled
# Analyse:
#   python examples/symmetry_paper/analyse_e0.py --grid-root $FINAL_ROOT/e0_grid $FINAL_ROOT/e0_budget/sb80 \
#       --budget-roots $FINAL_ROOT/e0_budget/sb{10,40,80}

#SBATCH --job-name=paper-e0-dgrid
#SBATCH --output=out_files/paper-e0-dgrid-%A_%a.out
#SBATCH --error=out_files/paper-e0-dgrid-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-79

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/_common.sh"

# The default hinge's exact values (sim.ZOO_HINGE_*).
DEF_KP=1.3605113362038874
DEF_KV=0.3555497840137177
DEF_ARM=0.08117047736037042
read -r -a KPS <<< "${KPS:-1 $DEF_KP 2.5}"
read -r -a KVS <<< "${KVS:-$DEF_KV 1 2}"
ARM=${ARM:-$DEF_ARM}
SKILL_BUDGET=${SKILL_BUDGET:-80}
GENOME_TYPES=(tree tree_symmetric)
N_REPS=${N_REPS:-5}
SEED_BASE=${SEED_BASE:-42}

CELLS=()
for kp in "${KPS[@]}"; do for kv in "${KVS[@]}"; do
    if [ "${SKIP_DEFAULT:-1}" = 1 ] && [ "$kp" = "$DEF_KP" ] && [ "$kv" = "$DEF_KV" ]; then continue; fi
    CELLS+=("$kp $kv")
done; done
N_CELLS=${#CELLS[@]}

ID=$SLURM_ARRAY_TASK_ID
GENOME=${GENOME_TYPES[$(( (ID / N_CELLS) % 2 ))]}
REP=$(( ID / (N_CELLS * 2) ))
if [ "$REP" -ge "$N_REPS" ]; then echo "Index $ID past the grid; nothing to do"; exit 0; fi
read -r KP KV <<< "${CELLS[$(( ID % N_CELLS ))]}"
SEED=$((SEED_BASE + REP))
LABEL=$(LC_ALL=C printf "kp%g_kv%g_arm%g_lim1" "$KP" "$KV" "$ARM")  # = sim.HingeActuator.label
TAG=dgrid_dann_${GENOME}_${LABEL}_sb${SKILL_BUDGET}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${ID}

if [ -n "${DRY_RUN:-}" ]; then echo "$TAG"; exit 0; fi
setup_env
banner "Hinge:          kp=$KP kv=$KV armature=$ARM limit=1" "Genome / brain: $GENOME / dann" \
       "Skill budget:   $SKILL_BUDGET generations x 20" "Seed:           $SEED" "Final dir:      $FINAL_ROOT/e0_grid/$TAG"
run_in_tmp "/tmp/$USER/paper_e0_dgrid_${SLURM_ARRAY_JOB_ID}_$ID" "$FINAL_ROOT/e0_grid/$TAG" \
    python "$SCRIPTS_DIR/evolve.py" --genome-type "$GENOME" --brain dann --seed "$SEED" "${V2_EVOLVE_ARGS[@]}" \
        --skill-budget "$SKILL_BUDGET" --brain-workers "$SLURM_CPUS_PER_TASK" \
        --hinge-kp "$KP" --hinge-kv "$KV" --hinge-armature "$ARM" --hinge-limit --time-limit 82800
