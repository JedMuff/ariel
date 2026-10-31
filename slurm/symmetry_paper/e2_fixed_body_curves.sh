#!/bin/bash
#
# E2 fixed-body learning curves (R2, R4, R5): train each brain on fixed
# bodies, no outer loop, logging every generation and checkpointing at
# 50 / 200 / 800 / 1600 / 5000 evaluations (train_fixed_body.py).
#
# v2 protocol (_common.sh): fitness in metres.
#
# BODY_SET picks the bodies file $FINAL_ROOT/e2/bodies_<BODY_SET>.jsonl, made
# beforehand with sample_bodies.py (seconds; login node or locally):
#   random   80 + 80 hinge-matched bodies from the initial distribution
#            (can run now, in parallel with E1)
#   evolved  the best last-generation body of every tree / tree_symmetric E1
#            run (after E1 finishes)
#   tied           80 tree_tied bodies with the random symmetric bodies' hinge counts
#   shape_matched  80 tree bodies matched to the random symmetric bodies on rest
#                  shape (from a pool of 24000; ~4 min on 20 CPUs)
#   leg_matched    E5: 80 tree bodies matched to the tied bodies on hinge count
#                  and leg hinges
# One task per (body, brain), running the 3 training seeds (42 43 44) in turn;
# bang_bang, bang_bang_free and bang_bang_drift train for 1600 evaluations,
# dann for 5000. BRAINS overrides the brain list (default: bang_bang
# bang_bang_free dann), e.g. BRAINS=bang_bang_drift with --array=0-79 for the
# drift control on the random bodies.
# Finished seeds are skipped, so a task can be resubmitted.
#
# Index decoding: body fastest, brain slowest.
# Array size = n_bodies x n_brains (random: 160 x 4 = 640 with
# BRAINS="bang_bang bang_bang_drift bang_bang_free dann"; tied, shape_matched,
# leg_matched: 80 x 2 = 160 with BRAINS="bang_bang dann").
#
# Usage:
#   python examples/symmetry_paper/sample_bodies.py random --n 80 --pool 800 -o $FINAL_ROOT/e2/bodies_random.jsonl
#   BODY_SET=random BRAINS="bang_bang bang_bang_drift bang_bang_free dann" \
#       sbatch --array=0-639 slurm/symmetry_paper/e2_fixed_body_curves.sh
#   python examples/symmetry_paper/sample_bodies.py evolved --runs-root $FINAL_ROOT/e1 -o $FINAL_ROOT/e2/bodies_evolved.jsonl
#   BODY_SET=evolved sbatch --array=0-119 slurm/symmetry_paper/e2_fixed_body_curves.sh
#   python examples/symmetry_paper/sample_bodies.py tied --like $FINAL_ROOT/e2/bodies_random.jsonl -o $FINAL_ROOT/e2/bodies_tied.jsonl
#   python examples/symmetry_paper/sample_bodies.py shape_matched --like $FINAL_ROOT/e2/bodies_random.jsonl --pool 24000 \
#       -o $FINAL_ROOT/e2/bodies_shape_matched.jsonl
#   python examples/symmetry_paper/sample_bodies.py shape_matched --like $FINAL_ROOT/e2/bodies_tied.jsonl --pool 24000 \
#       --target-genome tree_tied --measures n_leg --body-source leg_matched --id-prefix legmatch_tree \
#       -o $FINAL_ROOT/e2/bodies_leg_matched.jsonl
#   BODY_SET=tied BRAINS="bang_bang dann" sbatch --array=0-159 slurm/symmetry_paper/e2_fixed_body_curves.sh
#   BODY_SET=shape_matched BRAINS="bang_bang dann" sbatch --array=0-159 slurm/symmetry_paper/e2_fixed_body_curves.sh
#   BODY_SET=leg_matched BRAINS="bang_bang dann" sbatch --array=0-159 slurm/symmetry_paper/e2_fixed_body_curves.sh
# Then:
#   python examples/symmetry_paper/analyse_matched.py --roots $FINAL_ROOT/e2/{random,tied,shape_matched} \
#       --bodies $FINAL_ROOT/e2/bodies_{random,tied,shape_matched}.jsonl --out $FINAL_ROOT/e2/analysis_matched
#   E3_ROOTS=$FINAL_ROOT/e2/random E3_NAME=e2_random sbatch --array=0-719%100 slurm/symmetry_paper/e3_behaviour.sh
#   python examples/symmetry_paper/analyse_e2.py --roots $FINAL_ROOT/e2/random $FINAL_ROOT/e2/evolved

#SBATCH --job-name=paper-e2
#SBATCH --output=out_files/paper-e2-%A_%a.out
#SBATCH --error=out_files/paper-e2-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --array=0-639

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/_common.sh"

BODY_SET=${BODY_SET:?set BODY_SET=random, evolved, tied, shape_matched or leg_matched}
BODIES=$FINAL_ROOT/e2/bodies_${BODY_SET}.jsonl
[ -f "$BODIES" ] || { echo "No $BODIES: run sample_bodies.py first"; exit 1; }
N_BODIES=$(wc -l < "$BODIES")
read -r -a BRAINS <<< "${BRAINS:-bang_bang bang_bang_drift bang_bang_free dann}"
declare -A EVALS=([bang_bang]=1600 [bang_bang_free]=1600 [bang_bang_drift]=1600 [dann]=5000)

ID=$SLURM_ARRAY_TASK_ID
LINE=$(( ID % N_BODIES ))
BRAIN_IDX=$(( ID / N_BODIES ))
if [ "$BRAIN_IDX" -ge "${#BRAINS[@]}" ]; then echo "Index $ID past $N_BODIES bodies x ${#BRAINS[@]} brains; nothing to do"; exit 0; fi
BRAIN=${BRAINS[$BRAIN_IDX]}

if [ -n "${DRY_RUN:-}" ]; then echo "$BODY_SET line $LINE brain $BRAIN"; exit 0; fi
setup_env
banner "Body set:       $BODY_SET (line $LINE of $N_BODIES)" "Brain:          $BRAIN (${EVALS[$BRAIN]} evaluations)" \
       "Out root:       $FINAL_ROOT/e2/$BODY_SET"
srun python "$SCRIPTS_DIR/train_fixed_body.py" --bodies "$BODIES" --line "$LINE" --brain "$BRAIN" \
    --evals "${EVALS[$BRAIN]}" --seeds 42 43 44 --workers "$SLURM_CPUS_PER_TASK" "${V2_TRAIN_ARGS[@]}" -o "$FINAL_ROOT/e2/$BODY_SET"
echo "Finished: $(date)"
