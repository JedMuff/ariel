#!/bin/bash
#
# Compare outer- vs inner-parallelization for gecko brain training.
# Runs both schemes back to back in one job with the same settings.
#
# Usage:
#   sbatch examples/d_social_learning/run_bench_parallelization.sh
#
#SBATCH --job-name=gecko-bench-parallel
#SBATCH --output=out_files/gecko-bench-parallel-%j.out
#SBATCH --error=out_files/gecko-bench-parallel-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REPO_ROOT=/home/jed/workspaces/ariel
VENV_PATH=$REPO_ROOT/.venv

echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Date:       $(date)"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_INDIVIDUALS=16
INNER_GENS=20
POP_SIZE=16
SIGMA=0.5
HIDDEN=32
WORKERS=$SLURM_CPUS_PER_TASK

echo "n_individuals=$N_INDIVIDUALS inner_gens=$INNER_GENS pop_size=$POP_SIZE sigma=$SIGMA hidden=$HIDDEN workers=$WORKERS"

mkdir -p out_files

START_TIME=$(date +%s)

for SCHEME in outer inner; do
    echo ""
    echo "=== scheme=$SCHEME ==="
    srun "$VENV_PATH/bin/python" examples/d_social_learning/ariel/bench_parallelization.py \
        --scheme "$SCHEME" \
        --n-individuals "$N_INDIVIDUALS" \
        --inner-gens "$INNER_GENS" \
        --pop-size "$POP_SIZE" \
        --sigma "$SIGMA" \
        --hidden "$HIDDEN" \
        --workers "$WORKERS"
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished. Results written under __data__/benchmarks/"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
