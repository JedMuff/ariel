#!/bin/bash
#
# Benchmark: how long E1, E2 and E3 take on a compute node, from short slices
# of each, with the same resources as the real jobs (20 CPUs). About an hour
# of wall time per task. Every step's wall time goes to
# $BENCH_ROOT/<task>/bench_times.jsonl; benchmark_report.py turns them into
# per-experiment estimates.
#
# Tasks:
#   0-5   E1: one arm each, genome {tree, tree_symmetric, tree_tied} fastest,
#         brain {bang_bang, bang_bang_free}, seed 42, the E1 command but
#         --budget 3 (generations 0-3 of 30); then E3 on its last generation.
#         The bang_bang tree / sym arms repeat the first generations of the
#         hinge grid's zoo-cell seed-42 runs exactly, so they also measure the
#         node's speed against those runs.
#   6-8   E2 random bodies: brain {bang_bang, bang_bang_free, dann}, 3 + 3
#         hinge-matched random bodies (sample_bodies.py random --n 3), one
#         training seed, the full E2 budget (1600 / 1600 / 5000 evals); then
#         E3 on each training, one replay.py call each as in e3_behaviour.sh.
#   9-11  E2 evolved bodies: as 6-8 on bench/bodies_evolved.jsonl, the
#         champions of the hinge grid's zoo-cell bang_bang runs (3 tree, 3
#         sym), standing in for the E1 champions.
#
# Usage:
#   sbatch slurm/symmetry_paper/benchmark.sh
# Then, locally after copying $BENCH_ROOT (needs __data__/ariel_actuator_grid):
#   python examples/symmetry_paper/benchmark_report.py --bench-root <copied BENCH_ROOT>

#SBATCH --job-name=paper-bench
#SBATCH --output=out_files/paper-bench-%A_%a.out
#SBATCH --error=out_files/paper-bench-%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --array=0-11

set -euo pipefail
source "${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}/slurm/symmetry_paper/_common.sh"

BENCH_ROOT=${BENCH_ROOT:-$FINAL_ROOT/bench}
ID=$SLURM_ARRAY_TASK_ID
CPUS=$SLURM_CPUS_PER_TASK

# timed STEP KEY=VALUE... -- command [args...]: runs the command and appends
# {"step", <key/values>, "wall_s"} to bench_times.jsonl in the current dir.
timed() {
    local step=$1 extra="" t0 t1
    shift
    while [ "$1" != "--" ]; do extra="$extra, \"${1%%=*}\": \"${1#*=}\""; shift; done
    shift
    t0=$(date +%s%N)
    "$@"
    t1=$(date +%s%N)
    printf '{"step": "%s"%s, "node": "%s", "cpus": %s, "wall_s": %s}\n' \
        "$step" "$extra" "$(hostname)" "$CPUS" "$(( (t1 - t0) / 1000000 ))e-3" >> bench_times.jsonl
}

setup_env

if [ "$ID" -lt 6 ]; then
    GENOME_TYPES=(tree tree_symmetric tree_tied)
    BRAINS=(bang_bang bang_bang_free)
    GENOME=${GENOME_TYPES[$(( ID % 3 ))]}
    BRAIN=${BRAINS[$(( ID / 3 ))]}
    OUT=$BENCH_ROOT/e1_${BRAIN}_${GENOME}
    rm -rf "$OUT" && mkdir -p "$OUT" && cd "$OUT"
    banner "Benchmark:      E1 $GENOME / $BRAIN, generations 0-3" "Out:            $OUT"
    timed e1_evolve genome=$GENOME brain=$BRAIN -- \
        srun python "$SCRIPTS_DIR/evolve.py" --genome-type "$GENOME" --brain "$BRAIN" --seed 42 \
            --budget 3 --pop 10 --lam 20 --repeat-evals --skill-budget 10 \
            --brain-workers "$CPUS" --max-modules 25 --max-depth 25
    timed e3_replay_evolution genome=$GENOME brain=$BRAIN -- \
        srun python "$SCRIPTS_DIR/replay.py" --run-dir "$OUT" --gens last --workers "$CPUS" -o "$OUT/e3"
else
    BRAINS=(bang_bang bang_bang_free dann)
    declare -A EVALS=([bang_bang]=1600 [bang_bang_free]=1600 [dann]=5000)
    BRAIN=${BRAINS[$(( (ID - 6) % 3 ))]}
    BODY_SET=$([ "$ID" -lt 9 ] && echo random || echo evolved)
    OUT=$BENCH_ROOT/e2_${BODY_SET}_${BRAIN}
    rm -rf "$OUT" && mkdir -p "$OUT" && cd "$OUT"
    banner "Benchmark:      E2 $BODY_SET bodies / $BRAIN (${EVALS[$BRAIN]} evaluations)" "Out:            $OUT"
    if [ "$BODY_SET" = random ]; then
        timed e2_sample_bodies -- python "$SCRIPTS_DIR/sample_bodies.py" random --n 3 -o bodies.jsonl
    else
        cp "$REPO/slurm/symmetry_paper/bench/bodies_evolved.jsonl" bodies.jsonl
    fi
    for LINE in $(seq 0 $(( $(wc -l < bodies.jsonl) - 1 ))); do
        BODY_ID=$(sed -n "$((LINE + 1))p" bodies.jsonl | python -c "import json, sys; print(json.load(sys.stdin)['body_id'])")
        timed e2_train body_set=$BODY_SET brain=$BRAIN body_id=$BODY_ID -- \
            srun python "$SCRIPTS_DIR/train_fixed_body.py" --bodies bodies.jsonl --line "$LINE" --brain "$BRAIN" \
                --evals "${EVALS[$BRAIN]}" --seeds 42 --workers "$CPUS" -o "$OUT/trainings"
        timed e3_replay_training body_set=$BODY_SET brain=$BRAIN body_id=$BODY_ID -- \
            srun python "$SCRIPTS_DIR/replay.py" --run-dir "$OUT/trainings/${BODY_ID}__${BRAIN}__s42" \
                --workers "$CPUS" -o "$OUT/e3"
    done
fi
echo "Finished: $(date)"
