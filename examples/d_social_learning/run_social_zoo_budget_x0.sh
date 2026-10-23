#!/bin/bash
#
# x = 0 (novelty-only selection) companion to run_social_zoo_budget.sh: same
# parameters and code, restricted to the ig10 budget, x = 0 and reps 0-9 (the
# subset the final results use). Part 1 of 2: generations 0-30. Extend to 60
# with run_social_zoo_budget_x0_extend.sh once these finish.
#
# Usage:
#   sbatch examples/d_social_learning/run_social_zoo_budget_x0.sh
#
# 10 reps x 8 schemes = 80 combos -> array indices 0-79, with
#     index = 8 * rep + scheme_idx
# (run_social_zoo_budget.sh's mapping with one budget and one x value). Runs
# go to __data__/social/ariel_zoo_fe10/ig10/<scheme>/x00/. As there, a task
# skips its combo if a complete run already exists (FORCE=1 to override).
#
#SBATCH --job-name=social-zoo-x0
#SBATCH --output=out_files/social-zoo-x0-%A_%a.out
#SBATCH --error=out_files/social-zoo-x0-%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --partition=genoa
#SBATCH --array=0-79

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

# The first 5 are the first batch's; new schemes go at the end so existing
# indices keep their meaning.
SCHEMES=(darwinian lamarckian similar_TOPDOWN best_many random_many random best similar_many_TOPDOWN)
X_VALUES=(0)
BUDGETS=(10)   # inner CMA-ES generations
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
FINAL_EVALS=10
OUT_ROOT=__data__/social/ariel_zoo_fe${FINAL_EVALS}/ig${INNER_GENS}

echo "Scheme: $SCHEME  x=$X  rep=$REP  inner-gens=$INNER_GENS  (array idx=$IDX)"
echo "Params: gens=$GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP sigma=$SIGMA hidden=$HIDDEN workers=$WORKERS sim_preset=$SIM_PRESET novelty_metric=$NOVELTY_METRIC k_inherit=$K_INHERIT final_evals=$FINAL_EVALS out_root=$OUT_ROOT"

mkdir -p out_files

# Skip combos that already have a complete run (generation $GENS evaluated),
# so overlapping or repeated submissions don't duplicate work.
X_DIR=$(printf "x%02d" "$(awk -v x="$X" 'BEGIN { printf "%d", x * 10 + 0.5 }')")
if [ "${FORCE:-0}" != "1" ] && "$VENV_PATH/bin/python" - "$OUT_ROOT/$SCHEME/$X_DIR/k$K_INHERIT" "$REP" "$NOVELTY_METRIC" "$GENS" <<'EOF'
import sqlite3, sys
from pathlib import Path
root, rep, nov, gens = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
for db in root.glob(f"rep_{rep}_*nov{nov}_*/database.db"):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    n = con.execute("SELECT count(*) FROM individual WHERE time_of_birth = ? AND fitness_ IS NOT NULL",
                    (gens,)).fetchone()[0]
    con.close()
    if n:
        print(f"complete run exists: {db.parent}")
        sys.exit(0)
sys.exit(1)
EOF
then
    echo "Skipping (set FORCE=1 to run anyway)"
    exit 0
fi

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/experiment.py \
    --scheme "$SCHEME" --x "$X" --rep "$REP" \
    --gens "$GENS" --pop "$POP" --lam "$LAM" \
    --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
    --sigma "$SIGMA" --hidden "$HIDDEN" \
    --workers "$WORKERS" --novelty-metric "$NOVELTY_METRIC" \
    --k-inherit "$K_INHERIT" --sim-preset "$SIM_PRESET" --final-evals "$FINAL_EVALS" \
    --out-root "$OUT_ROOT"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished: scheme=$SCHEME x=$X rep=$REP inner-gens=$INNER_GENS"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
