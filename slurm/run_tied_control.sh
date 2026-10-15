#!/bin/bash
#
# Dimensionality control for symmetry enforcement: does tree_symmetric win
# because its body genome has fewer free genes, or because it is symmetric?
#
# tree_tied (gecko_skill_tasks.py --genome-type tree_tied, see
# src/ariel/ec/genotypes/tree/tying.py) ties modules to randomly placed twins:
# when a module is placed, with probability --tie-prob (0.47) a twin with the
# same type and rotation goes to a random free site (never its mirror
# position), and every later mutation of one applies to both. About 64% of
# modules end up tied, as mirrored in tree_symmetric, but the body has no
# geometric symmetry.
#   tree_tied ~ tree_symmetric  -> the advantage is dimensionality
#   tree_tied ~ tree            -> symmetry itself matters
#
# Settings match the no-penalty arm of slurm/run_rowing_penalty.sh: forward
# task, stride 25 (20Hz), ANN brain, zoo modules, budget 30, pop 10 + 20.
# Those runs (seeds 42-44, /scratch/jed/ariel_rowing_penalty_ann/nopenalty)
# are reps 0-2 of tree and tree_symmetric, so this array adds:
#   tree_tied       reps 0-5 (seeds 42-47)
#   tree            reps 3-5 (seeds 45-47)
#   tree_symmetric  reps 3-5 (seeds 45-47)
# = 12 jobs, 6 reps per genome in total.
#
# Results go to FINAL_DIR=/scratch/jed/ariel_tied_control_ann with the stride
# sweep's run tags (ctrlstride_forward_<genome>_s25_rep<r>_...).
#
# Dry-run one job first:
#   sbatch --array=0 slurm/run_tied_control.sh
#
# Usage:
#   sbatch slurm/run_tied_control.sh

#SBATCH --job-name=ariel-tied-ctrl
#SBATCH --output=out_files/ariel-tied-ctrl-%A_%a.out
#SBATCH --error=out_files/ariel-tied-ctrl-%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-11

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure

# ── Job list: (genome, rep) ───────────────────────────────────────────────────

JOBS=(
    "tree_tied 0" "tree_tied 1" "tree_tied 2" "tree_tied 3" "tree_tied 4" "tree_tied 5"
    "tree 3" "tree_symmetric 3"
    "tree 4" "tree_symmetric 4"
    "tree 5" "tree_symmetric 5"
)

ID=$SLURM_ARRAY_TASK_ID
read -r GENOME_TYPE REP <<< "${JOBS[$ID]}"

STRIDE=25                            # 500Hz physics / 25 = 20Hz control
SEED=$((42 + REP))                   # reps 0-2 share seeds with the rowing-penalty runs
BRAIN=ann
TIE_PROB=0.47

TMP_DIR=/tmp/${USER}/ariel_tiedctrl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=/scratch/jed/ariel_tied_control_${BRAIN}
RUN_TAG=ctrlstride_forward_${GENOME_TYPE}_s${STRIDE}_rep${REP}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# ── Environment ───────────────────────────────────────────────────────────────

cd "$REPO"
mkdir -p out_files

source "$VENV_PATH/bin/activate"

# Headless EGL rendering (no X display on compute nodes).
export MUJOCO_GL=egl

# ── Diagnostics ───────────────────────────────────────────────────────────────

echo "========================================================"
echo "Node:           $(hostname)"
echo "Array job/task: $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Task:           forward"
echo "Control stride: $STRIDE steps (~$(awk "BEGIN{printf \"%.1f\", 500/$STRIDE}")Hz)"
echo "Genome type:    $GENOME_TYPE"
echo "Tie prob:       $TIE_PROB (tree_tied only)"
echo "Brain:          $BRAIN"
echo "Modules:        zoo (kgd sizes, masses, hinge)"
echo "Rep:            $REP"
echo "Seed:           $SEED"
echo "CPUs:           $SLURM_CPUS_PER_TASK"
echo "MUJOCO_GL:      $MUJOCO_GL"
echo "Tmp dir:        $TMP_DIR"
echo "Final dir:      $FINAL_DIR/$RUN_TAG"
echo "Python:         $(which python)  ($(python --version 2>&1))"
echo "Started:        $(date)"
echo "========================================================"

# ── Run (tmp → scratch on exit) ───────────────────────────────────────────────

srun bash -c "
    set -uo pipefail
    mkdir -p \"$TMP_DIR\"
    finalize() {
        rc=\$?
        if [ -d \"$TMP_DIR\" ] && [ -n \"\$(ls -A \"$TMP_DIR\" 2>/dev/null || true)\" ]; then
            echo \"Moving results from $TMP_DIR to $FINAL_DIR/$RUN_TAG ...\"
            mkdir -p \"$FINAL_DIR\"
            mv \"$TMP_DIR\" \"$FINAL_DIR/$RUN_TAG\"
            echo \"Results saved to $FINAL_DIR/$RUN_TAG\"
        else
            echo \"No results in $TMP_DIR to move (rc=\$rc)\"
            rm -rf \"$TMP_DIR\" || true
        fi
        exit \$rc
    }
    trap finalize EXIT

    cd \"$TMP_DIR\"

    python \"$SCRIPTS_DIR/gecko_skill_tasks.py\" \
        --task forward \
        --control-step-freq $STRIDE \
        --strategy-type plus \
        --budget 30 \
        --pop 10 --lam 20 \
        --repeat-evals \
        --brain-workers $SLURM_CPUS_PER_TASK \
        --max-modules 25 --max-depth 25 \
        --genome-type $GENOME_TYPE \
        --tie-prob $TIE_PROB \
        --seed $SEED \
        --brain $BRAIN \
        --zoo-modules \
        --time-limit 259200
"

echo "Finished: $(date)"
echo "done"
