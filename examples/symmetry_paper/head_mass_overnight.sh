#!/bin/bash
#
# Head-mass experiment, local overnight run: which physics brings about rowing
# under evolution? Bang-bang, short evolution (10+20, parents retrained, 160-200
# evaluations per body, as the old run that rowed: symmetric reached 50-70%
# rowing by generations 3-5).
#
# Queue, one full pass per seed so a partial night stays balanced:
#   factorial  symmetric, head {0.25, 1.0, 1.363, 2.0} x hinge x modules, legacy protocol (16 cells)
#   tree       old and current physics, legacy protocol (2 cells)
#   protocol   symmetric, old and current physics, paper protocol (2 cells)
# then, if time remains, head 0.5 at old physics. Finished runs are skipped, so
# rerunning resumes the queue (a run cut off mid-way starts again).
#
# Usage: bash examples/symmetry_paper/head_mass_overnight.sh [ROOT]
#   env: SEEDS, GENS, SKILL_BUDGET, JOBS, WORKERS, REPLAY_WORKERS
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT=${1:-$REPO/__data__/ariel_head_mass/evolution}
SEEDS=${SEEDS:-43 44 45}; GENS=${GENS:-10}; SKILL_BUDGET=${SKILL_BUDGET:-10}
JOBS=${JOBS:-4}; WORKERS=${WORKERS:-14}; REPLAY_WORKERS=${REPLAY_WORKERS:-48}
PY=$REPO/.venv/bin/python; SCRIPTS=$REPO/examples/symmetry_paper
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 MUJOCO_GL=egl
mkdir -p "$ROOT"

run_one() {  # genome head hinge modules profile seed
    local genome=$1 head=$2 hinge=$3 modules=$4 profile=$5 seed=$6
    local name=${genome}_m${head}_hinge-${hinge}_mod-${modules}_${profile}_s${seed}
    local dir=$ROOT/runs/$name
    [ -f "$dir/DONE" ] && { echo "$name: done, skipped"; return 0; }
    rm -rf "$dir"; mkdir -p "$dir"; cd "$dir" || return 1
    local args=(--genome-type "$genome" --brain bang_bang --budget "$GENS" --pop 10 --lam 20 --skill-budget "$SKILL_BUDGET"
                --repeat-evals --brain-workers "$WORKERS" --seed "$seed" --max-modules 25 --max-depth 25
                --module-set "$modules" --core-mass "$head" --profile "$profile" --fitness displacement)
    [ "$hinge" = old ] && args+=(--hinge-kp 1 --hinge-kv 1 --hinge-armature 0 --no-hinge-limit)
    local t0=$SECONDS
    "$PY" "$SCRIPTS/evolve.py" "${args[@]}" > run.log 2>&1
    local rc=$?
    # A run that died still has its finished generations, so it is labelled too.
    if [ "$rc" -eq 0 ]; then touch DONE; fi
    echo "$(date +%H:%M) $name: rc=$rc in $(( (SECONDS - t0) / 60 )) min"
}
export -f run_one
export ROOT GENS SKILL_BUDGET WORKERS PY SCRIPTS

{
for seed in $SEEDS; do
    for head in 1.0 1.363 0.25 2.0; do for hinge in old zoo; do for modules in local zoo; do
        echo tree_symmetric $head $hinge $modules legacy $seed
    done; done; done
    echo tree 1.0 old local legacy $seed
    echo tree 1.363 zoo zoo legacy $seed
    echo tree_symmetric 1.0 old local dsocial_zoo $seed
    echo tree_symmetric 1.363 zoo zoo dsocial_zoo $seed
done
for seed in $SEEDS; do echo tree_symmetric 0.5 old local legacy $seed; done
} > "$ROOT/queue.txt"
echo "$(date): $(wc -l < "$ROOT/queue.txt") runs, $JOBS at a time"
xargs -a "$ROOT/queue.txt" -P "$JOBS" -L1 bash -c 'run_one "$0" "$@"'

echo "$(date): labelling"
cd "$REPO" || exit 1
LAST=$(seq -s, $(( GENS > 4 ? GENS - 4 : 0 )) "$GENS")
"$PY" "$SCRIPTS/replay.py" --run-dir "$ROOT"/runs/* --gens "0,$LAST" --workers "$REPLAY_WORKERS" -o "$ROOT/labels" 2>&1 | tail -n +1
"$PY" "$SCRIPTS/analyse_head_mass.py" --root "$ROOT" | tee "$ROOT/summary.txt"
touch "$ROOT/ALL_DONE"
echo "$(date): finished"
