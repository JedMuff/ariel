#!/bin/bash
# Submit the symmetry paper's jobs (run on the login node, from anywhere).
#
#   submit.sh bench [PARTITION ...]      cost per evaluation per partition (benchmark.sh)
#   submit.sh pilot                      Experiment 1, one seed per condition (60 runs)
#   submit.sh e1 [EVALS ...]             Experiment 1, SEEDS seeds (default all three budgets)
#   submit.sh bodies                     Experiment 2 bodies: sample_bodies.py (login node, a minute)
#   submit.sh e2                         Experiment 2 trainings of $OUT_ROOT/e2/bodies.jsonl
#   submit.sh replay ROOT [ROOT ...]     gait measures of the finished runs under the roots
#   submit.sh replay_gens ROOT [ROOT ...]  the same for the survivors of 16 generations (REPLAY_GENS),
#                                        with mechanics and rest measures (MAX_TASKS=1: a pilot)
#
# Experiment 1 runs that are already DONE are left out, so rerunning a command
# resubmits only what is missing; runs stopped by the time limit resume where
# they stopped. Arrays are split into chunks of MAX_ARRAY tasks and throttled
# to THROTTLE running tasks.
#
# Environment (defaults in brackets; see _common.sh for REPO and OUT_ROOT):
#   ACCOUNT    Slurm account, if not the default
#   PARTITION  [rome]       CPUS  [32]      THROTTLE [100]   MAX_ARRAY [1000]
#   GENOMES    [all five] (pilot, e1) space-separated subset, e.g. "tree tree_symmetric"
#   SEEDS      [20] (e1)    TIME_E100 [08:00:00]  TIME_E200 [14:00:00]  TIME_E800 [48:00:00]
#   E2_PER_TASK [4] (body x brain jobs per task)  TIME_E2 [12:00:00]  E2_SEEDS [42-61]
#   DRY_RUN=1  print the sbatch commands, submit nothing

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
export REPO OUT_ROOT VENV

PARTITION=${PARTITION:-rome}
CPUS=${CPUS:-32}
THROTTLE=${THROTTLE:-100}
MAX_ARRAY=${MAX_ARRAY:-1000}
SEEDS=${SEEDS:-20}
LISTS=$OUT_ROOT/lists
mkdir -p "$LISTS" "$OUT_ROOT/logs"

sb() {  # sbatch with the common options; prints the command under DRY_RUN
    local cmd=(sbatch --parsable --chdir "$OUT_ROOT" --partition "$PARTITION" --cpus-per-task "$CPUS"
               --output "$OUT_ROOT/logs/%x-%A_%a.out" --error "$OUT_ROOT/logs/%x-%A_%a.out"
               --export "ALL,REPO=$REPO,OUT_ROOT=$OUT_ROOT,VENV=$VENV${EXTRA_EXPORT:-}" ${ACCOUNT:+--account "$ACCOUNT"} "$@")
    if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

array() {  # array SCRIPT LIST N_TASKS [sbatch options...]: chunks of MAX_ARRAY
    local script=$1 list=$2 n=$3
    shift 3
    for (( off = 0; off < n; off += MAX_ARRAY )); do
        local size=$(( n - off < MAX_ARRAY ? n - off : MAX_ARRAY ))
        EXTRA_EXPORT="${EXTRA_EXPORT:-},LIST=$list,LIST_OFFSET=$off" \
            sb --array "0-$(( size - 1 ))%$THROTTLE" "$@" "$SLURM_DIR/$script"
    done
}

undone() {  # keep E1 list lines whose run has no DONE
    while IFS=$'\t' read -r g b e s sub; do
        [ -f "$OUT_ROOT/$sub/DONE" ] || printf '%s\t%s\t%s\t%s\t%s\n' "$g" "$b" "$e" "$s" "$sub"
    done
}

submit_e1() {  # submit_e1 KIND EVALS...
    local kind=$1
    shift
    for e in "$@"; do
        local list="$LISTS/${kind}_e$e.tsv" time_var="TIME_E$e"
        (cd "$CODE" && python conditions.py "$kind" --seeds "$SEEDS" --evals "$e" ${GENOMES:+--genomes $GENOMES}) | undone > "$list"
        local n
        n=$(wc -l < "$list")
        echo "$kind e$e: $n runs to do ($list)"
        [ "$n" -gt 0 ] && array e1_evolve.sh "$list" "$n" --job-name "sym-$kind-e$e" --time "${!time_var:-12:00:00}"
    done
    return 0
}

TIME_E100=${TIME_E100:-08:00:00}
TIME_E200=${TIME_E200:-14:00:00}
TIME_E800=${TIME_E800:-48:00:00}

setup_env
case "${1:-}" in
    bench)
        shift
        for p in "${@:-$PARTITION}"; do PARTITION=$p sb --job-name "sym-bench-$p" "$SLURM_DIR/benchmark.sh"; done
        ;;
    pilot)
        submit_e1 pilot 100 200 800
        ;;
    e1)
        shift
        if [ $# -gt 0 ]; then submit_e1 e1 "$@"; else submit_e1 e1 100 200 800; fi
        ;;
    bodies)
        (cd "$CODE" && python sample_bodies.py --n 80 -o "$OUT_ROOT/e2/bodies.jsonl")
        ;;
    e2)
        BODIES=$OUT_ROOT/e2/bodies.jsonl
        [ -f "$BODIES" ] || { echo "no $BODIES: run 'submit.sh bodies' first"; exit 1; }
        (cd "$CODE" && python conditions.py e2 --bodies "$BODIES") > "$LISTS/e2.tsv"
        per=${E2_PER_TASK:-4}
        n=$(( ( $(wc -l < "$LISTS/e2.tsv") + per - 1 ) / per ))
        echo "e2: $(wc -l < "$LISTS/e2.tsv") body x brain jobs in $n tasks"
        EXTRA_EXPORT=",BODIES=$BODIES,PER_TASK=$per" \
            array e2_train.sh "$LISTS/e2.tsv" "$n" --job-name sym-e2 --time "${TIME_E2:-12:00:00}"
        ;;
    replay)
        shift
        [ $# -gt 0 ] || { echo "usage: submit.sh replay ROOT [ROOT ...]"; exit 1; }
        (cd "$CODE" && python replay.py --list-runs "$@") > "$LISTS/replay.txt"
        per=20
        n=$(( ( $(wc -l < "$LISTS/replay.txt") + per - 1 ) / per ))
        echo "replay: $(wc -l < "$LISTS/replay.txt") runs in $n tasks"
        EXTRA_EXPORT=",PER_TASK=$per" array replay.sh "$LISTS/replay.txt" "$n" --job-name sym-replay
        ;;
    replay_gens)
        shift
        [ $# -gt 0 ] || { echo "usage: submit.sh replay_gens ROOT [ROOT ...]"; exit 1; }
        (cd "$CODE" && python replay.py --list-runs "$@") > "$LISTS/replay_gens.txt"
        # mechanics.py runs one process per run, so one run per CPU fills a task.
        per=${REPLAY_GENS_PER_TASK:-$CPUS}
        n=$(( ( $(wc -l < "$LISTS/replay_gens.txt") + per - 1 ) / per ))
        [ -z "${MAX_TASKS:-}" ] || n=$(( n < MAX_TASKS ? n : MAX_TASKS ))
        echo "replay_gens: $(wc -l < "$LISTS/replay_gens.txt") runs, $n tasks of $per"
        export WHICH=${REPLAY_GENS:-s:0,1,2,3,4,5,7,10,15,20,25,30,35,40,45,50}
        export MECH_WHICH=$WHICH REPLAY_DIR=replay_gens MECH_DIR=mechanics_gens REST_DIR=rest_gens
        EXTRA_EXPORT=",PER_TASK=$per" array replay.sh "$LISTS/replay_gens.txt" "$n" --job-name sym-replay-gens
        ;;
    *)
        sed -n '2,27p' "$0"
        exit 1
        ;;
esac
