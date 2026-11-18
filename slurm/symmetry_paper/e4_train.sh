#!/bin/bash
#SBATCH --job-name=sym-e4
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Experiment 4: long retraining of Experiment 1's final-generation bodies. Each array task
# trains PER_TASK consecutive lines of BODIES (export_final_bodies.py), from line
# (LIST_OFFSET + task) * PER_TASK, each with its own controller kind for E4_EVALS evaluations
# (one seed, E4_SEED), E4_CONCURRENT bodies at a time, into $OUT_ROOT/e4. Finished bodies
# are skipped, so resubmitting completes the set. Submit with submit.sh e4.

set -euo pipefail
source "${REPO:-$(dirname "$0")/../..}/slurm/symmetry_paper/_common.sh"
setup_env

: "${BODIES:?set BODIES to the export_final_bodies.py file}"
PER_TASK=${PER_TASK:-48}
EVALS=${E4_EVALS:-10000}
FIRST=$(( ( ${LIST_OFFSET:-0} + SLURM_ARRAY_TASK_ID ) * PER_TASK ))  # LIST_OFFSET counts tasks (submit.sh chunks)
# Start no new body in the last 75 minutes: a 10,000-evaluation training can take that long
# on a large body, and one cut off by the time limit is lost (and redone on resubmission).
LIMIT=$(seconds_left "${E4_MARGIN:-4500}")

banner "Lines:          $FIRST..$(( FIRST + PER_TASK - 1 )) of $BODIES" \
       "Evals / seed:   $EVALS / ${E4_SEED:-42}" "Concurrent:     ${E4_CONCURRENT:-8}" "Time limit:     ${LIMIT}s"
python "$CODE/train_long.py" --bodies "$BODIES" --first "$FIRST" --count "$PER_TASK" --evals "$EVALS" \
    --seed "${E4_SEED:-42}" --workers "${SLURM_CPUS_PER_TASK:-32}" --concurrent "${E4_CONCURRENT:-8}" \
    --time-limit "$LIMIT" --out-root "$OUT_ROOT/e4"
echo "Finished: $(date)"
