#!/bin/bash
#
# E2 fixed-body learning curves (R2, R4, R5): train each brain on fixed
# bodies, no outer loop, logging every generation and checkpointing at
# 50 / 200 / 800 / 1600 / 5000 evaluations (train_fixed_body.py).
#
# BODY_SET picks the bodies file $FINAL_ROOT/e2/bodies_<BODY_SET>.jsonl, made
# beforehand with sample_bodies.py (seconds; login node or locally):
#   random   40 + 40 hinge-matched bodies from the initial distribution
#            (can run now, in parallel with E1)
#   evolved  the best last-generation body of every tree / tree_symmetric E1
#            run (after E1 finishes)
# One task per (body, brain), running the 3 training seeds (42 43 44) in turn;
# bang_bang and bang_bang_free train for 1600 evaluations, dann for 5000.
# Finished seeds are skipped, so a task can be resubmitted.
#
# Index decoding: body fastest, brain slowest.
# Array size = n_bodies x 3 (random: 80 x 3 = 240; evolved: 40 x 3 = 120).
#
# Usage:
#   python examples/symmetry_paper/sample_bodies.py random --n 40 -o $FINAL_ROOT/e2/bodies_random.jsonl
#   BODY_SET=random sbatch --array=0-239 slurm/symmetry_paper/e2_fixed_body_curves.sh
#   python examples/symmetry_paper/sample_bodies.py evolved --runs-root $FINAL_ROOT/e1 -o $FINAL_ROOT/e2/bodies_evolved.jsonl
#   BODY_SET=evolved sbatch --array=0-119 slurm/symmetry_paper/e2_fixed_body_curves.sh
# Then:
#   E3_ROOTS=$FINAL_ROOT/e2/random E3_NAME=e2_random sbatch --array=0-719%100 slurm/symmetry_paper/e3_behaviour.sh
#   python examples/symmetry_paper/analyse_e2.py --roots $FINAL_ROOT/e2/random $FINAL_ROOT/e2/evolved

#SBATCH --job-name=paper-e2
#SBATCH --output=out_files/paper-e2-%A_%a.out
#SBATCH --error=out_files/paper-e2-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --array=0-239

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/_common.sh"

BODY_SET=${BODY_SET:?set BODY_SET=random or BODY_SET=evolved}
BODIES=$FINAL_ROOT/e2/bodies_${BODY_SET}.jsonl
[ -f "$BODIES" ] || { echo "No $BODIES: run sample_bodies.py first"; exit 1; }
N_BODIES=$(wc -l < "$BODIES")
BRAINS=(bang_bang bang_bang_free dann)
declare -A EVALS=([bang_bang]=1600 [bang_bang_free]=1600 [dann]=5000)

ID=$SLURM_ARRAY_TASK_ID
LINE=$(( ID % N_BODIES ))
BRAIN_IDX=$(( ID / N_BODIES ))
if [ "$BRAIN_IDX" -ge 3 ]; then echo "Index $ID past $N_BODIES bodies x 3 brains; nothing to do"; exit 0; fi
BRAIN=${BRAINS[$BRAIN_IDX]}

setup_env
banner "Body set:       $BODY_SET (line $LINE of $N_BODIES)" "Brain:          $BRAIN (${EVALS[$BRAIN]} evaluations)" \
       "Out root:       $FINAL_ROOT/e2/$BODY_SET"
srun python "$SCRIPTS_DIR/train_fixed_body.py" --bodies "$BODIES" --line "$LINE" --brain "$BRAIN" \
    --evals "${EVALS[$BRAIN]}" --seeds 42 43 44 --workers "$SLURM_CPUS_PER_TASK" -o "$FINAL_ROOT/e2/$BODY_SET"
echo "Finished: $(date)"
