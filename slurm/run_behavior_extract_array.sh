#!/bin/bash
#
# Array job: run examples/symmetry_pressure/behavior_extract.py over every
# locomotion sweep run (sympress_{forward,multidirection,turn_avg}_* and
# ctrlstride_forward_*), one array task per run. Food runs are excluded
# (different rollout; see gecko_food_skills.py).
#
# Each task replays every checkpointed body x trained skill of its run and
# writes behavior.npz + descriptors.csv to <sweep_root>/behavior/<RUN_TAG>/.
# Measured locally: ~0.1 s/episode amortized at 16 workers, i.e. a few
# minutes per run (~720 checkpoints x 1/5/3 skills).
#
# Run tags contain the producing job's IDs, so the task -> run mapping is the
# sorted directory listing, not a factorial decode. Size --array to match:
#   python examples/symmetry_pressure/behavior_extract.py --list-runs \
#       /scratch/jed/ariel_symmetry_pressure_sweep /scratch/jed/ariel_control_stride_sweep | wc -l
# and submit with --array=0-<N-1>, e.g.
#   sbatch --array=0-56 slurm/run_behavior_extract_array.sh
#
# Dry-run a single task locally (no sbatch; set SWEEP_ROOTS to local copies):
#   SLURM_ARRAY_TASK_ID=0 SWEEP_ROOTS="__data__/ariel_symmetry_pressure_sweep __data__/ariel_control_stride_sweep" \
#       REPO=$PWD bash slurm/run_behavior_extract_array.sh

#SBATCH --job-name=ariel-behavior-extract
#SBATCH --output=out_files/ariel-behavior-extract-%A_%a.out
#SBATCH --error=out_files/ariel-behavior-extract-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

set -euo pipefail

REPO=${REPO:-/home/jed/workspaces/ariel-symmetry/ariel}
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure
SWEEP_ROOTS=${SWEEP_ROOTS:-"/scratch/jed/ariel_symmetry_pressure_sweep /scratch/jed/ariel_control_stride_sweep"}

WORKERS=${SLURM_CPUS_PER_TASK:-16}
RECORD_EVERY_N=100   # 5Hz trace rows at dt=0.002s; descriptors always use every step

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID not set (run via sbatch, or export it for a local dry-run)"
    exit 1
fi

cd "$REPO"
mkdir -p out_files
source "$VENV_PATH/bin/activate"

# ── Task -> run dir ───────────────────────────────────────────────────────────

# shellcheck disable=SC2086  # SWEEP_ROOTS is a space-separated list
mapfile -t RUNS < <(python "$SCRIPTS_DIR/behavior_extract.py" --list-runs $SWEEP_ROOTS)

if (( SLURM_ARRAY_TASK_ID >= ${#RUNS[@]} )); then
    echo "Task $SLURM_ARRAY_TASK_ID out of range (${#RUNS[@]} runs) -- nothing to do"
    exit 0
fi

RUN_DIR=${RUNS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=$(dirname "$RUN_DIR")/behavior/$(basename "$RUN_DIR")

echo "========================================================"
echo "Node:           $(hostname)"
echo "Array job/task: ${SLURM_ARRAY_JOB_ID:-local} / $SLURM_ARRAY_TASK_ID  (${#RUNS[@]} runs total)"
echo "Run dir:        $RUN_DIR"
echo "Out dir:        $OUT_DIR"
echo "Workers:        $WORKERS"
echo "Python:         $(which python)  ($(python --version 2>&1))"
echo "Started:        $(date)"
echo "========================================================"

START_TIME=$(date +%s)

LAUNCH=()
[[ -n "${SLURM_JOB_ID:-}" ]] && LAUNCH=(srun)

"${LAUNCH[@]}" python "$SCRIPTS_DIR/behavior_extract.py" \
    --run-dir "$RUN_DIR" \
    --gens all \
    --workers "$WORKERS" \
    --record-every-n "$RECORD_EVERY_N" \
    --out-dir "$OUT_DIR"

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "Finished: $(basename "$RUN_DIR")"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
