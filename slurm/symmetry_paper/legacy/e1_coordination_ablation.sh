#!/bin/bash
#
# E1 evolution at the default hinge (R1, R6): brain {bang_bang,
# bang_bang_drift, bang_bang_free} x genome {tree, tree_symmetric, tree_tied}
# x N_REPS reps, forward, 200 evaluations per body (SKILL_BUDGET, default 10). v2
# protocol (_common.sh): fitness in metres, (20+20) for 30 generations without
# re-evaluation. The DANN runs at each budget, with and without retraining, are
# submitted by submit_longer_learning.sh.
#
# bang_bang_drift is the control for bang_bang_free: shared frequency plus
# small per-hinge offsets, i.e. free's extra genes without its out-of-step
# start. BRAINS overrides the brain list.
#
# Index decoding: genome fastest, then brain, rep slowest, so all arms of a
# seed run before the next seed. Array size = 3 x n_brains x N_REPS (180 for
# the default three brains x 20). Seeds 42 + rep, so runs pair by seed.
#
# E1_NAME names the output folder under $FINAL_ROOT (default e1). RETRAIN=1
# retrains every survivor every generation (about twice the compute). TIME_LIMIT
# is evolve.py's limit in seconds (default 32400); raise --time with it.
#
# Usage:
#   sbatch slurm/symmetry_paper/legacy/e1_coordination_ablation.sh
#   sbatch --array=0-179%40 slurm/symmetry_paper/legacy/e1_coordination_ablation.sh    # throttled
# Then:
#   E3_ROOTS=$FINAL_ROOT/e1 E3_NAME=e1 sbatch --array=0-179 slurm/symmetry_paper/legacy/e3_behaviour.sh
#   python examples/symmetry_paper/legacy/legacy/analyse_e1.py --root $FINAL_ROOT/e1 $FINAL_ROOT/e0_budget/sb10 \
#       --behaviour $FINAL_ROOT/e3/e1

#SBATCH --job-name=paper-e1
#SBATCH --output=out_files/paper-e1-%A_%a.out
#SBATCH --error=out_files/paper-e1-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-179

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/legacy/_common.sh"

GENOME_TYPES=(tree tree_symmetric tree_tied)
read -r -a BRAINS <<< "${BRAINS:-bang_bang bang_bang_drift bang_bang_free}"
N_BRAINS=${#BRAINS[@]}
N_REPS=${N_REPS:-20}
E1_NAME=${E1_NAME:-e1}
SKILL_BUDGET=${SKILL_BUDGET:-10}
TIME_LIMIT=${TIME_LIMIT:-32400}
RETRAIN_ARGS=()
if [ -n "${RETRAIN:-}" ]; then RETRAIN_ARGS=(--retrain-survivors); fi

ID=$SLURM_ARRAY_TASK_ID
GENOME=${GENOME_TYPES[$(( ID % 3 ))]}
BRAIN=${BRAINS[$(( (ID / 3) % N_BRAINS ))]}
REP=$(( ID / (3 * N_BRAINS) ))
if [ "$REP" -ge "$N_REPS" ]; then echo "Index $ID past the ablation; nothing to do"; exit 0; fi
SEED=$((42 + REP))
TAG=e1_${BRAIN}_${GENOME}_sb${SKILL_BUDGET}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${ID}

if [ -n "${DRY_RUN:-}" ]; then echo "$TAG"; exit 0; fi
setup_env
banner "Genome / brain: $GENOME / $BRAIN" "Seed:           $SEED" "Final dir:      $FINAL_ROOT/$E1_NAME/$TAG"
run_in_tmp "/tmp/$USER/paper_e1_${SLURM_ARRAY_JOB_ID}_$ID" "$FINAL_ROOT/$E1_NAME/$TAG" \
    python "$SCRIPTS_DIR/evolve.py" --genome-type "$GENOME" --brain "$BRAIN" --seed "$SEED" \
        "${V2_EVOLVE_ARGS[@]}" "${RETRAIN_ARGS[@]}" --skill-budget "$SKILL_BUDGET" \
        --brain-workers "$SLURM_CPUS_PER_TASK" --time-limit "$TIME_LIMIT"
