#!/bin/bash
#SBATCH --job-name=sym-e2
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Experiment 2: per array task, PER_TASK consecutive lines of LIST
# (conditions.py e2: body-line brain), each trained with E2_SEEDS on body line
# of BODIES, into $OUT_ROOT/$E2_DIR/<body_id>__<brain> (E2_DIR: e2, or e2m for the
# shape-matched bodies). Finished seeds are
# skipped, so resubmitting completes a set. Submit with submit.sh.

set -euo pipefail
source "${REPO:-$(dirname "$0")/../..}/slurm/symmetry_paper/_common.sh"
setup_env

: "${LIST:?set LIST to a conditions.py e2 list}"
: "${BODIES:?set BODIES to the sample_bodies.py file}"
PER_TASK=${PER_TASK:-1}
EVALS=${EVALS:-800}
SEEDS=${E2_SEEDS:-42-61}  # not SEEDS: that is submit.sh's E1 seed count, and sbatch passes the environment on
FIRST=$(( ${LIST_OFFSET:-0} + SLURM_ARRAY_TASK_ID * PER_TASK ))

banner "Lines:          $FIRST..$(( FIRST + PER_TASK - 1 )) of $LIST" "Bodies:         $BODIES" \
       "Evals / seeds:  $EVALS / $SEEDS"
for (( i = FIRST; i < FIRST + PER_TASK; i++ )); do
    LINE=$(line_of "$LIST" "$i")
    [ -z "$LINE" ] && break
    read -r BODY BRAIN <<< "$LINE"
    python "$CODE/train_fixed_body.py" --bodies "$BODIES" --line "$BODY" --brain "$BRAIN" --evals "$EVALS" \
        --seeds "$SEEDS" --workers "${SLURM_CPUS_PER_TASK:-32}" --out-root "$OUT_ROOT/${E2_DIR:-e2}"
done
echo "Finished: $(date)"
