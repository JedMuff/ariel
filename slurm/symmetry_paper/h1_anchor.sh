#!/bin/bash
#
# Head-mass experiment, step 1: the anchor gate. Does a fixed random body row
# on the old physics at all? 40 symmetric + 40 tree E2 random bodies, old
# modules and hinge, 1,600 evaluations, 1 seed, under the old sweep's profile
# (legacy) and the paper's (dsocial_zoo).
#
# Index decoding: body fastest, then profile, brain slowest.
# Array size = 80 x 2 x n_brains (default BRAINS="ann": 160).
#
# Body lines: the first 40 of each genome in bodies_random.jsonl (80 + 80).
#
#   export REPO=$PWD; sbatch --array=0-159 slurm/symmetry_paper/h1_anchor.sh
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash slurm/symmetry_paper/h1_anchor.sh
#
#SBATCH --job-name=paper-h1
#SBATCH --output=out_files/paper-h1-%A_%a.out
#SBATCH --error=out_files/paper-h1-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --array=0-159

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/_common.sh"

BODIES=$FINAL_ROOT/e2/bodies_random.jsonl
[ -f "$BODIES" ] || { echo "No $BODIES"; exit 1; }
read -r -a BRAINS <<< "${BRAINS:-ann}"
PROFILES=(legacy dsocial_zoo)

ID=$SLURM_ARRAY_TASK_ID
BODY=$(( ID % 80 ))
REST=$(( ID / 80 ))
PROFILE=${PROFILES[$(( REST % 2 ))]}
BRAIN_IDX=$(( REST / 2 ))
if [ "$BRAIN_IDX" -ge "${#BRAINS[@]}" ]; then echo "Index $ID past the design; nothing to do"; exit 0; fi
BRAIN=${BRAINS[$BRAIN_IDX]}
# bodies_random.jsonl holds 80 tree, then 80 tree_symmetric.
if [ "$BODY" -lt 40 ]; then LINE=$BODY; else LINE=$(( BODY - 40 + 80 )); fi
OUT=$FINAL_ROOT/head_mass/h1/$PROFILE

if [ -n "${DRY_RUN:-}" ]; then echo "line $LINE brain $BRAIN profile $PROFILE -> $OUT"; exit 0; fi
setup_env
banner "Line:           $LINE" "Brain:          $BRAIN" "Profile:        $PROFILE" "Out root:       $OUT"
srun python "$SCRIPTS_DIR/train_fixed_body.py" --bodies "$BODIES" --line "$LINE" --brain "$BRAIN" \
    --evals 1600 --seeds 42 --workers "$SLURM_CPUS_PER_TASK" --module-set local --hinge local \
    --profile "$PROFILE" "${V2_TRAIN_ARGS[@]}" -o "$OUT"
echo "Finished: $(date)"
