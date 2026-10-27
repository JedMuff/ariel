#!/bin/bash
#
# Behaviour rollouts for the DANN symmetry-pressure sweep
# (slurm/run_symmetry_pressure_sweep_dann.sh): one array task per run.
#
# Each task
#   1. replays every checkpointed body x trained skill of its run
#      (behavior_extract.py, all generations, 50Hz traces) under the run's
#      eval profile, checking each replay against the stored fitness;
#   2. runs the rowing / gait-class detector on that run (rowing_detect.py).
# Everything for a run lands in <sweep_root>/behavior_50hz/<RUN_TAG>/:
#   behavior.npz           traces + episode_meta + descriptors (~1GB for multidirection)
#   descriptors.csv        per-episode behavioural descriptors
#   rowing/runs/<run>.csv  per-episode gait labels (rowing, alternating, ...)
#   rowing/pairs/<run>.csv per left/right module pair
# so tasks never write the same file, and the CSVs can be fetched without the npz:
#   rsync -avz --exclude behavior.npz jed@10.0.0.1:/scratch/jed/ariel_symmetry_pressure_sweep_dann/behavior_50hz/ \
#       __data__/ariel_symmetry_pressure_sweep_dann/behavior_50hz/
#
# Measured locally (multidirection, the largest task, 16 workers): extraction
# ~5 min (3490 episodes, ~24GB peak RAM in the collecting process), rowing
# detection ~5 min single-threaded. forward and turn_avg runs are smaller.
#
# The array index is the run's position in behavior_extract.py --list-runs order
# (30 runs: 3 tasks x tree / tree_symmetric x 5 reps):
#   python examples/symmetry_pressure/behavior_extract.py --list-runs /scratch/jed/ariel_symmetry_pressure_sweep_dann
#
# Usage:
#   sbatch slurm/run_behavior_extract_dann.sh
#   sbatch --array=0 slurm/run_behavior_extract_dann.sh        # one run first

#SBATCH --job-name=ariel-behavior-dann
#SBATCH --output=out_files/ariel-behavior-dann-%A_%a.out
#SBATCH --error=out_files/ariel-behavior-dann-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=0-29

set -euo pipefail

REPO=${REPO:-/home/jed/workspaces/ariel-symmetry/ariel}
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/symmetry_pressure
SWEEP_ROOT=${SWEEP_ROOT:-/scratch/jed/ariel_symmetry_pressure_sweep_dann}
BEHAVIOR_SUBDIR=behavior_50hz
RECORD_EVERY_N=10   # 50Hz trace rows at dt=0.002s, as rowing_detect.py was validated on

WORKERS=${SLURM_CPUS_PER_TASK:-16}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID not set (run via sbatch, or export it for a local dry-run)"
    exit 1
fi

cd "$REPO"
mkdir -p out_files
source "$VENV_PATH/bin/activate"

# ── Task -> run dir ───────────────────────────────────────────────────────────

mapfile -t RUNS < <(python "$SCRIPTS_DIR/behavior_extract.py" --list-runs "$SWEEP_ROOT")

if (( SLURM_ARRAY_TASK_ID >= ${#RUNS[@]} )); then
    echo "Task $SLURM_ARRAY_TASK_ID out of range (${#RUNS[@]} runs) -- nothing to do"
    exit 0
fi

RUN_DIR=${RUNS[$SLURM_ARRAY_TASK_ID]}
RUN=$(basename "$RUN_DIR")
OUT_DIR=$SWEEP_ROOT/$BEHAVIOR_SUBDIR/$RUN

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

echo ""
echo "Rowing detection ($(( $(date +%s) - START_TIME ))s in)..."
"${LAUNCH[@]}" python "$SCRIPTS_DIR/rowing_detect.py" \
    --stride-root "$SWEEP_ROOT" \
    --behavior-subdir "$BEHAVIOR_SUBDIR" \
    --runs "$RUN" \
    --workers 1 \
    -o "$OUT_DIR/rowing"

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "Finished: $RUN"
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "done"
