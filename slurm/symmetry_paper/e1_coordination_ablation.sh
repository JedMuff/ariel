#!/bin/bash
#
# E1 coordination ablation (R6): remove bang_bang's shared frequency, keep
# everything else. Brain {bang_bang, bang_bang_free} x genome {tree,
# tree_symmetric, tree_tied} x N_REPS reps, zoo hinge, forward, 200
# evaluations per body (skill budget 10), outer (10+20) for 30 generations
# with --repeat-evals.
#
# BRAINS overrides the brain list, e.g. BRAINS=bang_bang_drift for the control
# arm (shared frequency plus small per-hinge offsets: bang_bang_free's extra
# genes without its out-of-step start).
#
# Index decoding: genome fastest, then brain, rep slowest, so all arms of a
# seed run before the next seed. Array size = 3 x n_brains x N_REPS (60 for
# the default two brains x 10, 30 for one brain). Seeds 42 + rep, so runs pair
# by seed.
#
# Usage:
#   sbatch slurm/symmetry_paper/e1_coordination_ablation.sh
#   sbatch --array=0-59%30 slurm/symmetry_paper/e1_coordination_ablation.sh    # throttled
#   BRAINS=bang_bang_drift sbatch --array=0-29 slurm/symmetry_paper/e1_coordination_ablation.sh
# Then:
#   E3_ROOTS=$FINAL_ROOT/e1 E3_NAME=e1 sbatch --array=0-59 slurm/symmetry_paper/e3_behaviour.sh
#   python examples/symmetry_paper/analyse_e1.py --root $FINAL_ROOT/e1 --behaviour $FINAL_ROOT/e3/e1

#SBATCH --job-name=paper-e1
#SBATCH --output=out_files/paper-e1-%A_%a.out
#SBATCH --error=out_files/paper-e1-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-59

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/_common.sh"

GENOME_TYPES=(tree tree_symmetric tree_tied)
read -r -a BRAINS <<< "${BRAINS:-bang_bang bang_bang_free}"
N_BRAINS=${#BRAINS[@]}
N_REPS=${N_REPS:-10}

ID=$SLURM_ARRAY_TASK_ID
GENOME=${GENOME_TYPES[$(( ID % 3 ))]}
BRAIN=${BRAINS[$(( (ID / 3) % N_BRAINS ))]}
REP=$(( ID / (3 * N_BRAINS) ))
if [ "$REP" -ge "$N_REPS" ]; then echo "Index $ID past the ablation; nothing to do"; exit 0; fi
SEED=$((42 + REP))
TAG=e1_${BRAIN}_${GENOME}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${ID}

setup_env
banner "Genome / brain: $GENOME / $BRAIN" "Seed:           $SEED" "Final dir:      $FINAL_ROOT/e1/$TAG"
run_in_tmp "/tmp/$USER/paper_e1_${SLURM_ARRAY_JOB_ID}_$ID" "$FINAL_ROOT/e1/$TAG" \
    python "$SCRIPTS_DIR/evolve.py" --genome-type "$GENOME" --brain "$BRAIN" --seed "$SEED" \
        --budget 30 --pop 10 --lam 20 --repeat-evals --skill-budget 10 \
        --brain-workers "$SLURM_CPUS_PER_TASK" --max-modules 25 --max-depth 25 --time-limit 32400
