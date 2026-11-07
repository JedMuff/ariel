#!/bin/bash
#SBATCH --job-name=sym-e1
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# One Experiment 1 run per array task: line (LIST_OFFSET + task id) of LIST
# (conditions.py e1 / pilot: genome brain evals seed out-subdir), into
# $OUT_ROOT/<out-subdir>. A run that stops at the time limit resumes from its
# last generation when the same line is submitted again; finished runs (DONE)
# are skipped. Submit with submit.sh, which sets LIST, LIST_OFFSET, --time and
# the array range. EXTRA_ARGS is appended to evolve.py's arguments (testing).

set -euo pipefail
source "${REPO:-$(dirname "$0")/../..}/slurm/symmetry_paper/_common.sh"
setup_env

: "${LIST:?set LIST to a conditions.py list}"
IDX=$(( ${LIST_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
read -r GENOME BRAIN EVALS SEED SUB <<< "$(line_of "$LIST" "$IDX")"
OUT=$OUT_ROOT/$SUB

banner "Run:            $GENOME $BRAIN e$EVALS s$SEED (line $IDX of $LIST)" "Out:            $OUT"
if [ -f "$OUT/DONE" ]; then
    echo "already done"
    exit 0
fi
python "$CODE/evolve.py" --genome "$GENOME" --brain "$BRAIN" --evals "$EVALS" --seed "$SEED" \
    --workers "${SLURM_CPUS_PER_TASK:-32}" --out "$OUT" --time-limit "$(seconds_left 1200)" ${EXTRA_ARGS:-}
echo "Finished: $(date)"
