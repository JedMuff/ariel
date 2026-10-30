#!/bin/bash
#
# E3 behaviour (R3, R5, behaviour readouts of R6): replay checkpoints and
# measure their gaits (replay.py: gait class, phase relations, phase quality,
# gait cycle). One task per run directory under E3_ROOTS:
#   evolution runs (E0, E1)      the last generation's bodies (GENS=last), or
#                                GENS=all / a comma list
#   fixed-body trainings (E2)    every checkpoint along training
# Rollouts only, minutes per run.
#
# Environment:
#   E3_ROOTS  space-separated roots holding the run dirs (required)
#   E3_NAME   output name: results go to $FINAL_ROOT/e3/$E3_NAME/<run>/ (required)
#   GENS      evolution runs: last (default), all, or e.g. 0,10,29
# The array index picks the index-th run of `replay.py --list-runs $E3_ROOTS`;
# size the array to that count (indices past it exit).
#
# Usage:
#   python examples/symmetry_paper/replay.py --list-runs $FINAL_ROOT/e1 | wc -l
#   E3_ROOTS=$FINAL_ROOT/e1 E3_NAME=e1 sbatch --array=0-59 slurm/symmetry_paper/e3_behaviour.sh
#   E3_ROOTS="$FINAL_ROOT/e0_budget/sb10 $FINAL_ROOT/e0_budget/sb40 $FINAL_ROOT/e0_budget/sb80" E3_NAME=budget \
#       sbatch --array=0-29 slurm/symmetry_paper/e3_behaviour.sh
#   E3_ROOTS=$FINAL_ROOT/e2/random E3_NAME=e2_random sbatch --array=0-719%100 slurm/symmetry_paper/e3_behaviour.sh
# Then:
#   python examples/symmetry_paper/gait_species.py --root $FINAL_ROOT/e3/e1
#   python examples/symmetry_paper/analyse_e3.py --roots $FINAL_ROOT/e3/e1 $FINAL_ROOT/e3/budget \
#       --training-roots $FINAL_ROOT/e3/e2_random $FINAL_ROOT/e3/e2_evolved

#SBATCH --job-name=paper-e3
#SBATCH --output=out_files/paper-e3-%A_%a.out
#SBATCH --error=out_files/paper-e3-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --array=0-59

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/_common.sh"

read -r -a ROOTS <<< "${E3_ROOTS:?set E3_ROOTS to the run roots}"
NAME=${E3_NAME:?set E3_NAME (output under $FINAL_ROOT/e3/)}
GENS=${GENS:-last}

setup_env
mapfile -t RUNS < <(python "$SCRIPTS_DIR/replay.py" --list-runs "${ROOTS[@]}")
ID=$SLURM_ARRAY_TASK_ID
if [ "$ID" -ge "${#RUNS[@]}" ]; then echo "Index $ID past the ${#RUNS[@]} runs; nothing to do"; exit 0; fi
RUN=${RUNS[$ID]}

banner "Run:            $RUN (${ID} of ${#RUNS[@]})" "Generations:    $GENS" "Out:            $FINAL_ROOT/e3/$NAME"
srun python "$SCRIPTS_DIR/replay.py" --run-dir "$RUN" --gens "$GENS" --workers "$SLURM_CPUS_PER_TASK" \
    -o "$FINAL_ROOT/e3/$NAME"
echo "Finished: $(date)"
