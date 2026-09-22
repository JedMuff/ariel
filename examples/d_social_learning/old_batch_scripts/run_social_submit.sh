#!/bin/bash
#
# Submit 480 jobs (240 experiments × 2 parts) to SLURM.
#
# Ordering: rep is outermost, then scheme, then x — so all rep-0 experiments
# are queued before rep-1, etc. Within each rep, all 24 scheme/x combos are
# submitted as partA+partB pairs. partB depends on partA via afterok.
#
# Usage (run from repo root):
#   bash examples/d_social_learning/run_social_submit.sh
#
# Dry-run (print commands without submitting):
#   DRY_RUN=1 bash examples/d_social_learning/run_social_submit.sh

SCHEMES=(darwinian lamarckian random random_many best best_many similar similar_many)
X_VALUES=(0 0.5 1)
N_REPS=10

SCRIPT_DIR="examples/d_social_learning"
PARTA="$SCRIPT_DIR/run_social_parta.sh"
PARTB="$SCRIPT_DIR/run_social_partb.sh"

mkdir -p out_files

total=0

# Outermost loop: rep — ensures rep-0 of every experiment is queued before rep-1
for (( rep=0; rep<N_REPS; rep++ )); do
    for scheme in "${SCHEMES[@]}"; do
        for x in "${X_VALUES[@]}"; do

            if [[ "${DRY_RUN:-0}" == "1" ]]; then
                echo "DRY sbatch $PARTA $scheme $x $rep"
                echo "DRY sbatch --dependency=afterok:<partA_id> $PARTB $scheme $x $rep"
            else
                partA_id=$(sbatch --parsable "$PARTA" "$scheme" "$x" "$rep")
                sbatch --parsable \
                    --dependency=afterok:"$partA_id" \
                    --qos=partb_high \
                    "$PARTB" "$scheme" "$x" "$rep"
                echo "Submitted: scheme=$scheme x=$x rep=$rep  partA=$partA_id"
            fi

            (( total += 2 ))
        done
    done
done

echo ""
echo "Total jobs submitted: $total"
