#!/bin/bash
#
# Zoo brain benchmark on one machine (no Slurm): every canonical body x BRAINS x
# N_REPS, the same protocol, seeds and run-dir naming as
# slurm/run_zoo_benchmark.sh, so the runs sit next to the cluster ones and
# aggregate_zoo_benchmark.py picks them up. JOBS runs go at once with WORKERS
# evaluation processes each (default 4 x 16 = 64 cores).
#
# Each run writes to a .tmp dir and is renamed when it finishes; finished runs
# (a summary.json under OUT_ROOT for that body/brain/rep) are skipped, so the
# script can be restarted after an interruption.
#
# Usage (from anywhere; detach so it survives closing the terminal):
#   setsid nohup examples/zoo_benchmark/run_local_benchmark.sh > out_files/zoo_local.log 2>&1 &
#   BRAINS="distributed_ann ann" N_REPS=2 examples/zoo_benchmark/run_local_benchmark.sh
#
# Aggregate afterwards:
#   python examples/zoo_benchmark/aggregate_zoo_benchmark.py __data__/ariel_zoo_benchmark

set -euo pipefail

SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPTS_DIR/../.." && pwd)

BODIES=${BODIES:-"ant babya babyb blokky garrix gecko insect linkin longleg park penguin
        pentapod queen salamander snake spider spider45 squarish stingray
        tinlicker turtle ww zappa"}
BRAINS=${BRAINS:-distributed_ann}
N_REPS=${N_REPS:-10}
BUDGET=${BUDGET:-10000}
DURATION=${DURATION:-15}
CONTROL_FREQ=${CONTROL_FREQ:-20}
JOBS=${JOBS:-4}
WORKERS=${WORKERS:-16}
OUT_ROOT=${OUT_ROOT:-$REPO/__data__/ariel_zoo_benchmark}
LOG_DIR=${LOG_DIR:-$REPO/out_files/zoo_local}
TAG=${TAG:-local$(date +%Y%m%d)}

# One BLAS thread per evaluation process: the distributed ANN's small matmuls
# otherwise spawn a thread pool in every worker and oversubscribe the cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl
export REPO SCRIPTS_DIR OUT_ROOT LOG_DIR BUDGET DURATION CONTROL_FREQ WORKERS
mkdir -p "$OUT_ROOT" "$LOG_DIR"

run_one() {
    local body=$1 brain=$2 rep=$3 idx=$4
    local seed=$((42 + rep))
    if compgen -G "$OUT_ROOT/zoo_${body}_${brain}_rep${rep}_*/summary.json" > /dev/null; then
        echo "skip  $body $brain rep$rep (already finished)"
        return 0
    fi
    local tag="zoo_${body}_${brain}_rep${rep}_${TAG}_${idx}"
    local tmp="$OUT_ROOT/.tmp_$tag"
    rm -rf "$tmp"
    echo "start $tag  $(date +%T)"
    if (cd "$SCRIPTS_DIR" && "$REPO/.venv/bin/python" zoo_benchmark.py \
            --body "$body" --brain "$brain" --seed "$seed" --budget "$BUDGET" \
            --duration "$DURATION" --control-freq "$CONTROL_FREQ" \
            --workers "$WORKERS" --out-dir "$tmp") > "$LOG_DIR/$tag.log" 2>&1; then
        mv "$tmp" "$OUT_ROOT/$tag"
        echo "done  $tag  $(date +%T)"
    else
        echo "FAIL  $tag  (see $LOG_DIR/$tag.log)"
    fi
}
export -f run_one
export TAG

echo "Bodies: $(echo $BODIES | wc -w)  brains: $BRAINS  reps: $N_REPS  budget: $BUDGET"
echo "Jobs: $JOBS x $WORKERS workers  ->  $OUT_ROOT   started $(date)"

# Rep slowest, then brain, then body, as the Slurm script orders its array.
idx=0
for rep in $(seq 0 $((N_REPS - 1))); do
    for brain in $BRAINS; do
        for body in $BODIES; do
            echo "$body $brain $rep $idx"
            idx=$((idx + 1))
        done
    done
done | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _

echo "Finished $(date)"
