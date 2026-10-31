#!/bin/bash
#
# E0 DANN budget series (R1): does the symmetric advantage need a well-trained
# controller? Skill budgets x genome {tree, tree_symmetric} x N_REPS reps, DANN,
# default hinge, forward. Skill budget 10 / 40 / 80 inner CMA-ES generations =
# 200 / 800 / 1600 evaluations per body. v2 protocol (_common.sh): fitness in
# metres, (20+20) without re-evaluation. The
# 1,600-evaluation arm doubles as the default cell of e0_dann_grid.sh, and the
# 200-evaluation arm as E1's DANN arm. (V1=1 reproduces the v1 runs: normalised
# fitness, (10+20) with --repeat-evals, as the investigation's
# slurm/run_dann_budget.sh; set FINAL_ROOT to the v1 root as well.)
#
# Index decoding: genome fastest, then skill budget, rep slowest.
# Array size = 2 x n_budgets x N_REPS (60 for "10 40 80" x 10). Seeds 42 + rep.
#
# Usage:
#   sbatch slurm/symmetry_paper/e0_dann_budget.sh
#   SKILL_BUDGETS="80" sbatch --array=0-19 slurm/symmetry_paper/e0_dann_budget.sh
# Analyse:
#   python examples/symmetry_paper/analyse_e0.py --budget-roots $FINAL_ROOT/e0_budget/sb{10,40,80}

#SBATCH --job-name=paper-e0-budget
#SBATCH --output=out_files/paper-e0-budget-%A_%a.out
#SBATCH --error=out_files/paper-e0-budget-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-59

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/_common.sh"

read -r -a SKILL_BUDGETS <<< "${SKILL_BUDGETS:-10 40 80}"
GENOME_TYPES=(tree tree_symmetric)
N_REPS=${N_REPS:-10}
N_BUDGETS=${#SKILL_BUDGETS[@]}

ID=$SLURM_ARRAY_TASK_ID
GENOME=${GENOME_TYPES[$(( ID % 2 ))]}
SKILL_BUDGET=${SKILL_BUDGETS[$(( (ID / 2) % N_BUDGETS ))]}
REP=$(( ID / (2 * N_BUDGETS) ))
if [ "$REP" -ge "$N_REPS" ]; then echo "Index $ID past the series; nothing to do"; exit 0; fi
SEED=$((42 + REP))
TAG=budget_dann_${GENOME}_sb${SKILL_BUDGET}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${ID}

if [ -n "${DRY_RUN:-}" ]; then echo "$TAG"; exit 0; fi
if [ -n "${V1:-}" ]; then
    EVOLVE_ARGS=(--budget 30 --pop 10 --lam 20 --repeat-evals --fitness normalised --max-modules 25 --max-depth 25)
    LIMIT=43200
else
    EVOLVE_ARGS=("${V2_EVOLVE_ARGS[@]}")
    LIMIT=82800
fi
setup_env
banner "Genome / brain: $GENOME / dann" "Skill budget:   $SKILL_BUDGET generations x 20" "Seed:           $SEED" \
       "Final dir:      $FINAL_ROOT/e0_budget/sb$SKILL_BUDGET/$TAG"
run_in_tmp "/tmp/$USER/paper_e0_budget_${SLURM_ARRAY_JOB_ID}_$ID" "$FINAL_ROOT/e0_budget/sb$SKILL_BUDGET/$TAG" \
    python "$SCRIPTS_DIR/evolve.py" --genome-type "$GENOME" --brain dann --seed "$SEED" \
        "${EVOLVE_ARGS[@]}" --skill-budget "$SKILL_BUDGET" \
        --brain-workers "$SLURM_CPUS_PER_TASK" --time-limit "$LIMIT"
