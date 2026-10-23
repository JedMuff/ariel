#!/bin/bash
#
# Extend the x = 0 runs of run_social_zoo_budget_x0.sh from 30 to TARGET_GENS
# generations (default 60). Same as run_social_zoo_budget_extend.sh (see there
# for how the resume works), restricted to ig10, x = 0 and reps 0-9.
#
# Usage:
#   sbatch examples/d_social_learning/run_social_zoo_budget_x0_extend.sh
#
# Same array indices as run_social_zoo_budget_x0.sh:
#     index = 8 * rep + scheme_idx
# Resubmitting after a timeout picks up where a run stopped and skips runs
# already at TARGET_GENS.
#
#SBATCH --job-name=social-zoo-x0-ext
#SBATCH --output=out_files/social-zoo-x0-ext-%A_%a.out
#SBATCH --error=out_files/social-zoo-x0-ext-%A_%a.err
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --partition=genoa
#SBATCH --array=0-79

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REPO_ROOT=${REPO_ROOT:-$HOME/ariel}   # <-- edit this if your repo lives elsewhere
VENV_PATH=$REPO_ROOT/.venv

echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Date:       $(date)"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Map array index -> (rep, budget, scheme, x), as run_social_zoo_budget.sh
# ---------------------------------------------------------------------------

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
# Parameters (exactly as run_social_zoo_budget.sh, except the generations)
# ---------------------------------------------------------------------------

TARGET_GENS=${TARGET_GENS:-60}
POP=20
LAM=20
INNER_POP=20
SIGMA=0.45
HIDDEN=16
WORKERS=24
SIM_PRESET=zoo
NOVELTY_METRIC=TOPDOWN
K_INHERIT=5
FINAL_EVALS=10
OUT_ROOT=${OUT_ROOT:-__data__/social/ariel_zoo_fe${FINAL_EVALS}/ig${INNER_GENS}}

X_DIR=$(printf "x%02d" "$(awk -v x="$X" 'BEGIN { printf "%d", x * 10 + 0.5 }')")
REP_ROOT=$OUT_ROOT/$SCHEME/$X_DIR/k$K_INHERIT

echo "Scheme: $SCHEME  x=$X  rep=$REP  inner-gens=$INNER_GENS  (array idx=$IDX)"
echo "Params: target_gens=$TARGET_GENS pop=$POP lam=$LAM inner-gens=$INNER_GENS inner-pop=$INNER_POP sigma=$SIGMA hidden=$HIDDEN workers=$WORKERS sim_preset=$SIM_PRESET novelty_metric=$NOVELTY_METRIC k_inherit=$K_INHERIT final_evals=$FINAL_EVALS out_root=$OUT_ROOT"

mkdir -p out_files

# ---------------------------------------------------------------------------
# Find the run to extend and the last generation it completed
# ---------------------------------------------------------------------------

# Prints "<run dir> <last generation>" for the newest run dir of this combo
# (the one the analysis uses). The last generation is the highest generation
# with evaluated births in the run's newest database file.
read -r RUN_DIR LAST_GEN < <("$VENV_PATH/bin/python" - "$REP_ROOT" "$REP" "$FINAL_EVALS" "$NOVELTY_METRIC" <<'EOF'
import re, sqlite3, sys
from pathlib import Path
root, rep, fe, nov = Path(sys.argv[1]), *sys.argv[2:]
runs = sorted(d for d in root.glob(f"rep_{rep}_fe{fe}_nov{nov}_*") if (d / "database.db").exists())
if not runs:
    sys.exit(f"no run of rep {rep} under {root}")
run = runs[-1]
dbs = [run / "database.db"] + sorted(run.glob("database_part*.db"),
                                     key=lambda p: int(re.search(r"part(\d+)", p.name)[1]))
con = sqlite3.connect(f"file:{dbs[-1]}?mode=ro", uri=True)
last = con.execute("SELECT max(time_of_birth) FROM individual WHERE fitness_ IS NOT NULL").fetchone()[0]
con.close()
if last is None:
    sys.exit(f"{dbs[-1]} has no evaluated individuals; move it aside and resubmit")
print(run, last)
EOF
)
if [ -z "$RUN_DIR" ]; then
    echo "could not find a run to extend" >&2
    exit 1
fi

GENS_LEFT=$(( TARGET_GENS - LAST_GEN ))
echo "Run dir: $RUN_DIR  (at generation $LAST_GEN, $GENS_LEFT to go)"
if [ "$GENS_LEFT" -le 0 ]; then
    echo "Already at generation $TARGET_GENS, skipping"
    exit 0
fi

START_TIME=$(date +%s)

srun "$VENV_PATH/bin/python" examples/d_social_learning/experiment.py \
    --scheme "$SCHEME" --x "$X" --rep "$REP" \
    --gens "$GENS_LEFT" --pop "$POP" --lam "$LAM" \
    --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" \
    --sigma "$SIGMA" --hidden "$HIDDEN" \
    --workers "$WORKERS" --novelty-metric "$NOVELTY_METRIC" \
    --k-inherit "$K_INHERIT" --sim-preset "$SIM_PRESET" --final-evals "$FINAL_EVALS" \
    --resume-dir "$RUN_DIR"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Finished: scheme=$SCHEME x=$X rep=$REP inner-gens=$INNER_GENS (to generation $TARGET_GENS)"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
