#!/bin/bash
#
# Submit the follow-up experiments and their replays. Run on the login node from
# the checkout, with the code committed and pulled:
#
#   export REPO=$PWD
#   bash slurm/symmetry_paper/legacy/submit_longer_learning.sh            # everything
#   bash slurm/symmetry_paper/legacy/submit_longer_learning.sh dann       # DANN evolution, parents keep their scores
#   bash slurm/symmetry_paper/legacy/submit_longer_learning.sh retrain    # DANN evolution, survivors retrained
#   bash slurm/symmetry_paper/legacy/submit_longer_learning.sh x3         # x3 fixed bodies
#   BUDGETS="10" bash slurm/symmetry_paper/legacy/submit_longer_learning.sh retrain   # one budget only
#   DRY_RUN=1 bash slurm/symmetry_paper/legacy/submit_longer_learning.sh  # print, submit nothing
#
# dann     DANN evolution at 200 / 800 / 1,600 evaluations per body (BUDGETS: CMA-ES
#          generations 10 / 40 / 80), tree / symmetric / tied x 20 seeds, in
#          $FINAL_ROOT/e1_dann/sb<budget> (60 runs per budget). About 880 / 3,200 /
#          6,200 CPU-h. These supersede e0_budget.
# retrain  The same runs, but every survivor is retrained every generation with a
#          fresh seed, in $FINAL_ROOT/e1_dann_retrain/sb<budget>. About 1,800 /
#          6,600 / 12,900 CPU-h.
# x3       Fixed-body training at three times the budget (bang-bang family 4,800
#          evaluations, DANN 15,000), every body set, same seeds, in
#          $FINAL_ROOT/e2_x3 (1,120 tasks, about 3,200 CPU-h). Runs do not save the
#          CMA-ES state, so these restart from evaluation 0 and retrace the old curves.
#
# Gait replays are queued to start when their experiment ends. Afterwards:
#   python examples/symmetry_paper/legacy/legacy/compare_retraining.py --plain $FINAL_ROOT/e1_dann/sb* \
#       --retrained $FINAL_ROOT/e1_dann_retrain/sb*
#   python examples/symmetry_paper/legacy/legacy/verify_rerun.py --short $FINAL_ROOT/e2 --long $FINAL_ROOT/e2_x3
#   E2_NAME=e2_x3 OUT=<local copy>/paper_x3 V2=<local copy> bash examples/symmetry_paper/legacy/legacy/paper/make_figures.sh

set -euo pipefail
: "${REPO:?export REPO=\$PWD (the checkout to run from)}"
source "$REPO/slurm/symmetry_paper/legacy/_common.sh"
WHAT=${1:-all}
SLURM_DIR=$REPO/slurm/symmetry_paper/legacy
read -r -a BUDGETS <<< "${BUDGETS:-10 40 80}"
EVAL_SCALE=${EVAL_SCALE:-3}
E2_NAME=${E2_NAME:-e2_x${EVAL_SCALE}}
# The slowest case (retraining, 1,600 evaluations, tied) should take about 15 h.
EVOLUTION_TIME=24:00:00
EVOLUTION_LIMIT=82800

# submit VAR=value ... -- sbatch-args: prints the job id; with DRY_RUN prints the command.
submit() {
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    if [ -n "${DRY_RUN:-}" ]; then
        echo "[dry run] ${envs[*]} sbatch $*" >&2
        echo 0
    else
        env "${envs[@]}" sbatch --parsable "$@" | cut -d";" -f1
    fi
}

# evolution NAME RETRAIN_VALUE: one array and its replays per budget.
evolution() {
    local name=$1 retrain=$2 sb job
    for sb in "${BUDGETS[@]}"; do
        job=$(submit BRAINS=dann SKILL_BUDGET="$sb" TIME_LIMIT="$EVOLUTION_LIMIT" RETRAIN="$retrain" E1_NAME="$name/sb$sb" REPO="$REPO" -- \
                     --time="$EVOLUTION_TIME" --array=0-59 "$SLURM_DIR/e1_coordination_ablation.sh")
        echo "$name, skill budget $sb: job $job"
        submit E3_ROOTS="$FINAL_ROOT/$name/sb$sb" E3_NAME="${name}_sb$sb" REPO="$REPO" -- \
            --dependency=afterany:"$job" --array=0-59 "$SLURM_DIR/e3_behaviour.sh" > /dev/null
    done
}

if [ "$WHAT" = all ] || [ "$WHAT" = dann ]; then
    evolution e1_dann ""
fi

if [ "$WHAT" = all ] || [ "$WHAT" = retrain ]; then
    evolution e1_dann_retrain 1
fi

if [ "$WHAT" = all ] || [ "$WHAT" = x3 ]; then
    for SET in random tied shape_matched leg_matched; do
        [ -f "$FINAL_ROOT/e2/bodies_$SET.jsonl" ] || { echo "Missing $FINAL_ROOT/e2/bodies_$SET.jsonl"; exit 1; }
    done
    RANDOM_JOB=$(submit BODY_SET=random BRAINS="bang_bang bang_bang_drift bang_bang_free dann" EVAL_SCALE="$EVAL_SCALE" \
                        E2_NAME="$E2_NAME" REPO="$REPO" -- --array=0-639 "$SLURM_DIR/e2_fixed_body_curves.sh")
    echo "Fixed bodies, random: job $RANDOM_JOB"
    for SET in tied shape_matched leg_matched; do
        JOB=$(submit BODY_SET="$SET" BRAINS="bang_bang dann" EVAL_SCALE="$EVAL_SCALE" E2_NAME="$E2_NAME" REPO="$REPO" -- \
                     --array=0-159 "$SLURM_DIR/e2_fixed_body_curves.sh")
        echo "Fixed bodies, $SET: job $JOB"
    done
    # 160 bodies x 4 controllers x 3 seeds = 1,920 trainings on the random bodies.
    submit E3_ROOTS="$FINAL_ROOT/$E2_NAME/random" E3_NAME="${E2_NAME}_random" REPO="$REPO" -- \
        --dependency=afterany:"$RANDOM_JOB" --array=0-1919%100 "$SLURM_DIR/e3_behaviour.sh" > /dev/null
    submit PS_ROOTS="$FINAL_ROOT/$E2_NAME/random" PS_NAME="${E2_NAME}_random" REPO="$REPO" -- \
        --dependency=afterany:"$RANDOM_JOB" --array=0-47 "$SLURM_DIR/e3_phase_shift.sh" > /dev/null
fi
