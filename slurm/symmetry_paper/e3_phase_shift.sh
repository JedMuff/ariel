#!/bin/bash
#
# Phase-shift replay (R5/R6 causal check): replay trained controllers with the
# left side's timing shifted or detuned, against the same number of random
# lateral hinges (phase_shift.py). Rollouts only, 25 episodes per checkpoint,
# ~15 s per run on 20 CPUs, so each task takes a chunk of RUNS_PER_TASK runs.
#
# Environment:
#   PS_ROOTS        space-separated roots holding the run dirs (required)
#   PS_NAME         output name: results go to $FINAL_ROOT/e3/phase_shift_$PS_NAME/<run>/ (required)
#   RUNS_PER_TASK   runs per array task (default 40)
#   CKPTS           fixed-body trainings: last (default) or all checkpoints
# Size the array to ceil(n_runs / RUNS_PER_TASK), n_runs from
# `replay.py --list-runs $PS_ROOTS | wc -l`; indices past the end exit.
#
# Usage:
#   PS_ROOTS=$FINAL_ROOT/e2/random PS_NAME=e2_random sbatch --array=0-23 slurm/symmetry_paper/e3_phase_shift.sh
#   PS_ROOTS=$FINAL_ROOT/e1 PS_NAME=e1 sbatch --array=0-2 slurm/symmetry_paper/e3_phase_shift.sh
# Then:
#   python examples/symmetry_paper/analyse_phase_shift.py \
#       --roots $FINAL_ROOT/e3/phase_shift_e2_random $FINAL_ROOT/e3/phase_shift_e1

#SBATCH --job-name=paper-ps
#SBATCH --output=out_files/paper-ps-%A_%a.out
#SBATCH --error=out_files/paper-ps-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --array=0-23

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/_common.sh"

read -r -a ROOTS <<< "${PS_ROOTS:?set PS_ROOTS to the run roots}"
NAME=${PS_NAME:?set PS_NAME (output under $FINAL_ROOT/e3/phase_shift_<name>)}
PER=${RUNS_PER_TASK:-40}
CKPTS=${CKPTS:-last}

setup_env
mapfile -t RUNS < <(python "$SCRIPTS_DIR/replay.py" --list-runs "${ROOTS[@]}")
START=$((SLURM_ARRAY_TASK_ID * PER))
if [ "$START" -ge "${#RUNS[@]}" ]; then echo "Task $SLURM_ARRAY_TASK_ID past the ${#RUNS[@]} runs; nothing to do"; exit 0; fi
CHUNK=("${RUNS[@]:$START:$PER}")

banner "Runs:           $START..$((START + ${#CHUNK[@]} - 1)) of ${#RUNS[@]}" "Checkpoints:    $CKPTS" \
       "Out:            $FINAL_ROOT/e3/phase_shift_$NAME"
srun python "$SCRIPTS_DIR/phase_shift.py" --run-dir "${CHUNK[@]}" --ckpts "$CKPTS" \
    --workers "$SLURM_CPUS_PER_TASK" -o "$FINAL_ROOT/e3/phase_shift_$NAME"
echo "Finished: $(date)"
