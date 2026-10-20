#!/bin/bash
#
# Replay every robot ever born in the zoo learning-budget runs
# (__data__/social/ariel_zoo_fe10: 640 runs x ~1220 robots) with its learned
# and its starting weights, EPISODES perturbed episodes each, and save the gait
# metrics and rowing labels per episode, plus 50 Hz traces (see
# behavior_replay.py --select all). Each array task takes every N-th run
# (--shard TASK/N); a run whose episodes.csv exists is skipped, so
# resubmitting is safe and only fills in what is missing.
#
# (run_behavior_extract_array.sh is for the old __data__/social/ariel layout and
# behavior_extract.py, which cannot replay zoo-preset runs.)
#
# Measured locally: one run (7,320 episodes) takes 100-200 s on 32 cores, so a
# task of 10 runs takes 20-35 min. Traces per run: ~1.9 GB with TRACES=all,
# ~0.6 GB with ep0 (episode 0 of both weight sets), so ~1.2 TB / ~0.4 TB for
# all 640 runs: check the quota of OUT_ROOT (`myquota`) first.
#
# Usage (from the repo root):
#   sbatch examples/d_social_learning/analysis/run_behavior_replay.sh
#   TRACES=all sbatch examples/d_social_learning/analysis/run_behavior_replay.sh
#
# Dry run of one run locally:
#   SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=640 bash examples/d_social_learning/analysis/run_behavior_replay.sh
#
# Afterwards, merge and copy the small CSVs back (traces stay on scratch):
#   .venv/bin/python examples/d_social_learning/analysis/behavior_replay.py --merge --out-dir "$OUT_ROOT"
#   rsync -av --exclude traces.npz snellius:"$OUT_ROOT"/ __data__/social/behavior_zoo_all/
#
#SBATCH --job-name=behavior-replay
#SBATCH --output=out_files/behavior-replay-%A_%a.out
#SBATCH --error=out_files/behavior-replay-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --partition=genoa
#SBATCH --array=0-63

REPO_ROOT=${REPO_ROOT:-$HOME/ariel}   # <-- edit this if your repo lives elsewhere
VENV_PATH=$REPO_ROOT/.venv
DATA_ROOT=${DATA_ROOT:-__data__/social/ariel_zoo_fe10}
OUT_ROOT=${OUT_ROOT:-/scratch-shared/$USER/behavior_zoo_all}
TRACES=${TRACES:-ep0}
EPISODES=${EPISODES:-3}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

echo "Node:  $(hostname)"
echo "Job:   ${SLURM_ARRAY_JOB_ID:-local} shard $SHARD/$N_SHARDS"
echo "Date:  $(date)"
echo "Data:  $DATA_ROOT -> $OUT_ROOT (traces=$TRACES, episodes=$EPISODES)"

cd "$REPO_ROOT" || exit 1
mkdir -p out_files "$OUT_ROOT"

START_TIME=$(date +%s)

"$VENV_PATH/bin/python" examples/d_social_learning/analysis/behavior_replay.py \
    --root "$DATA_ROOT" \
    --out-dir "$OUT_ROOT" \
    --select all \
    --episodes "$EPISODES" \
    --traces "$TRACES" \
    --shard "$SHARD/$N_SHARDS" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "Elapsed time: ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
