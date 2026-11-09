#!/bin/bash
#
# EvoGym learning-budget experiment (array version). Same outer loop and array
# layout as run_social_zoo_budget.sh, on --platform evogym:
#   - (mu+lambda) with mu = lambda = 100, 60 outer generations
#   - two lifetime-learning budgets of 50 and 100 episodes: inner CMA-ES with
#     population 10 for 5 or 10 generations
#   - all 8 schemes, with similar_*_STRUCT (aligned Hamming distance on the
#     voxel grid) as the morphological-similarity schemes, and the same
#     STRUCT distance for the novelty term
#   - x in {0, 0.5, 1}, k_inherit = 8
#   - 10 reps (raise N_REPS to 20 and resubmit the extra range to add more)
#   - no --final-evals: each robot's score is CMA-ES's best episode. EvoGym is
#     deterministic for a fixed controller, so repeating it would add nothing.
# Distance and novelty are min-max normalised to [0, 1] per generation before
# the x-blend (experiment.py evaluate_and_select).
#
# Episodes per run: 100 offspring x 60 gens x budget (+ the initial 100), i.e.
# ~610k at budget 100 and ~305k at budget 50.
#
# Each budget gets its own data tree, __data__/social/evogym_budget/ib{50,100},
# laid out like __data__/social/ariel, so the analysis can be pointed at it
# with --data-dir. experiment.py prints each run's directory as
# `RUN_DIR=<path>` in out_files/social-evogym-<jobid>_<taskid>.out, and the
# seed is a stable hash of (scheme, x, k_inherit, rep), so both budgets share
# seeds.
#
# EvoGym needs its own Python 3.10 venv (EVOGYM_VENV below); ariel is put on
# PYTHONPATH from src/ rather than installed. Setup, once, from the repo root:
#   uv venv evogym-venv --python 3.10
#   uv pip install --python evogym-venv/bin/python evogym gymnasium "numpy<2" \
#       nevergrad rich sqlmodel sqlalchemy pydantic pydantic-settings "setuptools<81"
#   uv pip install --python evogym-venv/bin/python torch \
#       --index-url https://download.pytorch.org/whl/cpu
#
# Usage:
#   sbatch examples/d_social_learning/run_social_evogym_budget.sh
#
# 10 reps x 2 budgets x 8 schemes x 3 x-values = 480 combos -> array indices
# 0-479. Ordering: rep outermost, then budget, then scheme, then x:
#     index = 48 * rep + 24 * budget_idx + 3 * scheme_idx + x_idx
# A task skips its combo if a complete run of it already exists under
# OUT_ROOT (set FORCE=1 to run anyway), so resubmitting the full range is safe
# and only fills in what is missing. To rerun a single combo, override the
# array, e.g. `sbatch --array=337 ...`; the analysis keeps the newest run dir
# per (scheme, x, rep).
#
# Each individual's inner learning runs in one worker, so a generation uses at
# most lambda = 100 cores; 120 is the smallest 24-core multiple above that.
#
#SBATCH --job-name=social-evogym
#SBATCH --output=out_files/social-evogym-%A_%a.out
#SBATCH --error=out_files/social-evogym-%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=120
#SBATCH --mem=120G
#SBATCH --partition=genoa
#SBATCH --array=0-479

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REPO_ROOT=$HOME/ariel   # <-- edit this if your repo lives elsewhere
EVOGYM_VENV=$REPO_ROOT/evogym-venv
export PYTHONPATH=$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}

echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Date:       $(date)"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Map array index -> (rep, budget, scheme, x)
# ---------------------------------------------------------------------------

SCHEMES=(darwinian lamarckian random random_many best best_many similar_STRUCT similar_many_STRUCT)
X_VALUES=(0 0.5 1)
INNER_POP=10
BUDGETS=(5 10)   # inner CMA-ES generations: 5 x 10 = 50 and 10 x 10 = 100 episodes
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
BUDGET=$(( INNER_GENS * INNER_POP ))

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

GENS=60
POP=100
LAM=100
SIGMA=0.45
HIDDEN=16
WORKERS=$(( SLURM_CPUS_PER_TASK < LAM ? SLURM_CPUS_PER_TASK : LAM ))
NOVELTY_METRIC=STRUCT
K_INHERIT=8
# A large robot's 100 episodes can take well over the default 30 min.
EVAL_TIMEOUT=7200
OUT_ROOT=__data__/social/evogym_budget/ib${BUDGET}

echo "Scheme: $SCHEME  x=$X  rep=$REP  budget=$BUDGET (inner-gens=$INNER_GENS)  (array idx=$IDX)"
echo "Params: gens=$GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP sigma=$SIGMA hidden=$HIDDEN workers=$WORKERS novelty_metric=$NOVELTY_METRIC k_inherit=$K_INHERIT out_root=$OUT_ROOT"

mkdir -p out_files

# Skip combos that already have a complete run (generation $GENS evaluated),
# so overlapping or repeated submissions don't duplicate work.
X_DIR=$(printf "x%02d" "$(awk -v x="$X" 'BEGIN { printf "%d", x * 10 + 0.5 }')")
if [ "${FORCE:-0}" != "1" ] && "$EVOGYM_VENV/bin/python" - "$OUT_ROOT/$SCHEME/$X_DIR/k$K_INHERIT" "$REP" "$NOVELTY_METRIC" "$GENS" <<'EOF'
import sqlite3, sys
from pathlib import Path
root, rep, nov, gens = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
for db in root.glob(f"rep_{rep}_nov{nov}_*/database.db"):
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

srun "$EVOGYM_VENV/bin/python" examples/d_social_learning/experiment.py \
    --platform evogym \
    --scheme "$SCHEME" --x "$X" --rep "$REP" \
    --gens "$GENS" --pop "$POP" --lam "$LAM" \
    --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
    --sigma "$SIGMA" --hidden "$HIDDEN" \
    --workers "$WORKERS" --novelty-metric "$NOVELTY_METRIC" \
    --k-inherit "$K_INHERIT" --eval-timeout "$EVAL_TIMEOUT" \
    --out-root "$OUT_ROOT"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished: scheme=$SCHEME x=$X rep=$REP budget=$BUDGET"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
