#!/bin/bash
#
# Replay every robot ever born in the zoo learning-budget runs
# (__data__/social/ariel_zoo_fe10: 640 runs x ~1220 robots) with its learned
# (EPISODES perturbed episodes) and its starting weights (INIT_EPISODES), and
# save the gait metrics and rowing labels per episode, plus each episode's
# 50 Hz state as a compact trace (see behavior_replay.py --select all). Each array task takes every N-th run
# (--shard TASK/N); a run whose episodes.csv exists is skipped, so
# resubmitting is safe and only fills in what is missing.
#
# (run_behavior_extract_array.sh is for the old __data__/social/ariel layout and
# behavior_extract.py, which cannot replay zoo-preset runs.)
#
# Measured locally (3 + 1 episodes): one run (4,880 episodes) takes ~40 s on
# 32 cores, i.e. ~0.35 CPU-h, so ~7 min per task and ~230 CPU-h for all 640
# runs. Traces: ~105 MB per run, ~70 GB in all.
#
# Usage (from the repo root):
#   sbatch examples/d_social_learning/analysis/run_behavior_replay.sh
#   TRACES=none sbatch examples/d_social_learning/analysis/run_behavior_replay.sh   # metrics only
#
# Dry run of one run locally:
#   SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=640 bash examples/d_social_learning/analysis/run_behavior_replay.sh
#
# Afterwards, merge and copy the small CSVs back (traces stay on scratch):
#   .venv/bin/python examples/d_social_learning/analysis/behavior_replay.py --merge --out-dir "$OUT_ROOT"
#   rsync -av --exclude traces.zip snellius:"$OUT_ROOT"/ __data__/social/behavior_zoo_all/
#
#SBATCH --job-name=behavior-replay
#SBATCH --output=out_files/behavior-replay-%A_%a.out
#SBATCH --error=out_files/behavior-replay-%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --partition=genoa
#SBATCH --array=0-63

REPO_ROOT=${REPO_ROOT:-$HOME/ariel}   # <-- edit this if your repo lives elsewhere
VENV_PATH=$REPO_ROOT/.venv
DATA_ROOT=${DATA_ROOT:-__data__/social/ariel_zoo_fe10}
OUT_ROOT=${OUT_ROOT:-/scratch-shared/$USER/behavior_zoo_all}
TRACES=${TRACES:-all}
EPISODES=${EPISODES:-3}
INIT_EPISODES=${INIT_EPISODES:-1}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

echo "Node:  $(hostname)"
echo "Job:   ${SLURM_ARRAY_JOB_ID:-local} shard $SHARD/$N_SHARDS"
echo "Date:  $(date)"
echo "Data:  $DATA_ROOT -> $OUT_ROOT (traces=$TRACES, episodes=$EPISODES learned + $INIT_EPISODES init)"

cd "$REPO_ROOT" || exit 1
mkdir -p out_files "$OUT_ROOT"

START_TIME=$(date +%s)

"$VENV_PATH/bin/python" examples/d_social_learning/analysis/behavior_replay.py \
    --root "$DATA_ROOT" \
    --out-dir "$OUT_ROOT" \
    --select all \
    --episodes "$EPISODES" \
    --init-episodes "$INIT_EPISODES" \
    --traces "$TRACES" \
    --shard "$SHARD/$N_SHARDS" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
