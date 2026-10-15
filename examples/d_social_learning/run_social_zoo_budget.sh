#!/bin/bash
#
# Zoo-physics learning-budget pilot (array version). Same outer loop as
# run_social_parta_only.sh, with these differences:
#   - --sim-preset zoo: kgd's zoo module physics, 15 s episodes at 20 Hz
#     control, displacement measured from the start position (see
#     sim_ariel/evaluator.py PRESETS)
#   - two lifetime-learning budgets, 10 and 20 inner CMA-ES generations
#   - the part A report's subset: 5 schemes, x in {0.5, 1}
#   - 10 reps
#
# Each budget gets its own data tree, __data__/social/ariel_zoo/ig{10,20}, laid
# out like __data__/social/ariel, so the part A analysis can be pointed at it
# with --data-dir. As in part A, experiment.py prints each run's directory as
# `RUN_DIR=<path>` in out_files/social-zoo-<jobid>_<taskid>.out, and the seed
# is a stable hash of (scheme, x, k_inherit, rep), so both budgets share seeds.
#
# Usage:
#   sbatch examples/d_social_learning/run_social_zoo_budget.sh
#
# 10 reps x 2 budgets x 5 schemes x 2 x-values = 200 combos -> array indices
# 0-199. Ordering: rep outermost, then budget, then scheme, then x.
#
#SBATCH --job-name=social-zoo
#SBATCH --output=out_files/social-zoo-%A_%a.out
#SBATCH --error=out_files/social-zoo-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --partition=genoa
#SBATCH --array=0-199

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REPO_ROOT=$HOME/ariel   # <-- edit this if your repo lives elsewhere
VENV_PATH=$REPO_ROOT/.venv

echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Date:       $(date)"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Map array index -> (rep, budget, scheme, x)
# ---------------------------------------------------------------------------

SCHEMES=(darwinian lamarckian similar_TOPDOWN best_many random_many)
X_VALUES=(0.5 1)
BUDGETS=(10 20)   # inner CMA-ES generations
N_REPS=10

IDX=$SLURM_ARRAY_TASK_ID
N_X=${#X_VALUES[@]}
N_SCHEMES=${#SCHEMES[@]}
PER_BUDGET=$(( N_SCHEMES * N_X ))
PER_REP=$(( ${#BUDGETS[@]} * PER_BUDGET ))

REP=$(( IDX / PER_REP ))
REM=$(( IDX % PER_REP ))
BUDGET_IDX=$(( REM / PER_BUDGET ))
REM=$(( REM % PER_BUDGET ))
SCHEME_IDX=$(( REM / N_X ))
X_IDX=$(( REM % N_X ))

if [ "$REP" -ge "$N_REPS" ]; then
    echo "array index $IDX is out of range (max $(( N_REPS * PER_REP - 1 )))" >&2
    exit 1
fi

SCHEME="${SCHEMES[$SCHEME_IDX]}"
X="${X_VALUES[$X_IDX]}"
INNER_GENS="${BUDGETS[$BUDGET_IDX]}"

# ---------------------------------------------------------------------------
# Parameters (as run_social_parta_only.sh, except INNER_GENS and SIM_PRESET)
# ---------------------------------------------------------------------------

GENS=30
POP=20
LAM=20
INNER_POP=20
SIGMA=0.45
HIDDEN=16
WORKERS=24
SIM_PRESET=zoo
# Exact top-down tree edit distance, for both the novelty term and the
# similar_TOPDOWN scheme.
NOVELTY_METRIC=TOPDOWN
K_INHERIT=5
OUT_ROOT=__data__/social/ariel_zoo/ig${INNER_GENS}

echo "Scheme: $SCHEME  x=$X  rep=$REP  inner-gens=$INNER_GENS  (array idx=$IDX)"
echo "Params: gens=$GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP sigma=$SIGMA hidden=$HIDDEN workers=$WORKERS sim_preset=$SIM_PRESET novelty_metric=$NOVELTY_METRIC k_inherit=$K_INHERIT out_root=$OUT_ROOT"

mkdir -p out_files

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/experiment.py \
    --scheme "$SCHEME" --x "$X" --rep "$REP" \
    --gens "$GENS" --pop "$POP" --lam "$LAM" \
    --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
    --sigma "$SIGMA" --hidden "$HIDDEN" \
    --workers "$WORKERS" --novelty-metric "$NOVELTY_METRIC" \
    --k-inherit "$K_INHERIT" --sim-preset "$SIM_PRESET" --out-root "$OUT_ROOT"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished: scheme=$SCHEME x=$X rep=$REP inner-gens=$INNER_GENS"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
