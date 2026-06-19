#!/bin/bash
#
# Smoke-test: 8 schemes × 2 domains = 16 array jobs
# x=0 (no novelty), 1 rep, 10 gens, mu=10 lambda=10
# Initial generation: mu=10 unevaluated individuals produced, then lam=10
# offspring each subsequent gen → 20 evals in gen-0, 10 per gen after.
#
# Usage:
#   sbatch run_smoke_test.slurm
#
#SBATCH --job-name=social-smoke
#SBATCH --output=out_files/smoke-%A_%a.out
#SBATCH --error=out_files/smoke-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --array=0-15

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REPO_ROOT=/home/jed/workspace/ariel-evogym
VENV_PATH=/home/jed/workspace/ariel-evogym/.venv
EVOGYM_PYTHON=/home/jed/workspace/ariel-evogym/evogym-venv/bin/python

echo "Node:       $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Date:       $(date)"

export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$VENV_PATH/lib/python3.10/site-packages:$PYTHONPATH"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Scheme / domain mapping
# Task IDs 0-7  → ariel  (darwinian, lamarckian, random, random_many, best, best_many, similar, similar_many)
# Task IDs 8-15 → evogym (same scheme order)
# ---------------------------------------------------------------------------

SCHEMES=(darwinian lamarckian random random_many best best_many similar similar_many)

GENS=10
POP=10
LAM=10
INNER_GENS=5
INNER_POP=8
WORKERS=5
X=0
REP=0

if (( SLURM_ARRAY_TASK_ID < 8 )); then
    DOMAIN="ariel"
    SCHEME_IDX=$SLURM_ARRAY_TASK_ID
else
    DOMAIN="evogym"
    SCHEME_IDX=$(( SLURM_ARRAY_TASK_ID - 8 ))
fi

SCHEME="${SCHEMES[$SCHEME_IDX]}"

echo "Domain: $DOMAIN  Scheme: $SCHEME  x=$X  rep=$REP"
echo "Params: gens=$GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP workers=$WORKERS"

mkdir -p out_files

START_TIME=$(date +%s)

if [[ "$DOMAIN" == "ariel" ]]; then
    srun "$VENV_PATH/bin/python" examples/d_social_learning/ariel/experiment.py \
        --scheme "$SCHEME" --x "$X" --rep "$REP" \
        --gens "$GENS" --pop "$POP" --lam "$LAM" \
        --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
        --workers "$WORKERS"
elif [[ "$DOMAIN" == "evogym" ]]; then
    srun "$EVOGYM_PYTHON" examples/d_social_learning/evogym/experiment.py \
        --scheme "$SCHEME" --x "$X" --rep "$REP" \
        --gens "$GENS" --pop "$POP" --lam "$LAM" \
        --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
        --workers "$WORKERS"
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished: domain=$DOMAIN scheme=$SCHEME"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
