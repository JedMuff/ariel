#!/bin/bash
#
# Zoo brain benchmark: 23 canonical bodies x 4 brains x N_REPS repetitions.
# Each task optimises one brain on one fixed body with CMA-ES (no body
# evolution), using kgd's apets-ariel zoo protocol so results are comparable
# with theirs (see examples/zoo_benchmark/zoo_benchmark.py):
#   - kgd's canonical bodies and hinge actuator (kp=1.36, kv=0.36,
#     armature=0.081, +-90 deg), vendored in examples/zoo_benchmark/
#   - 15 s episodes, 20 Hz control, action = brain output in +-pi/2 written
#     straight to ctrl (no smoothing, penalties or settle phase)
#   - fitness = signed x-speed of the core
#   - CMA-ES: x0=0.5 (kgd), sigma0=0.5, default popsize, tolfun=0,
#     tolflatfitness=10, BUDGET evaluations (kgd: 10000); the same for every
#     brain except the ANN, which starts at x0=0 (at 0.5 its outputs all
#     saturate and fitness is flat; see DEFAULT_INITIAL_MEAN in zoo_benchmark.py)
#
# Brains: ann (MLP), sine (open-loop sine CPG), revolve_cpg (kgd's RevolveCPG),
# matsuoka (Matsuoka oscillator network).
#
# Runtime: a smoke test on gecko was ~1-2 s per 100 evaluations with 16
# workers, so a 10000-evaluation run should take a few minutes; the time limit
# leaves a wide margin for the 14-hinge bodies.
#
# Index decoding: body fastest, then brain, then rep (slowest), so a full
# body x brain pass completes before any condition repeats.
#
# Dry-run a few indices first, e.g.:
#   sbatch --array=0-3 slurm/run_zoo_benchmark.sh     # 4 bodies, matsuoka, rep 0
#
# Usage:
#   sbatch slurm/run_zoo_benchmark.sh
#   INITIAL_MEAN=0 sbatch slurm/run_zoo_benchmark.sh   # force one CMA-ES x0 for all brains
#
# Aggregate afterwards:
#   python examples/zoo_benchmark/aggregate_zoo_benchmark.py /scratch/jed/ariel_zoo_benchmark

#SBATCH --job-name=ariel-zoo-bench
#SBATCH --output=out_files/ariel-zoo-bench-%A_%a.out
#SBATCH --error=out_files/ariel-zoo-bench-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --array=0-919%100

set -euo pipefail

REPO=/home/jed/workspaces/ariel-symmetry/ariel
VENV_PATH=$REPO/.venv
SCRIPTS_DIR=$REPO/examples/zoo_benchmark

# ── Factorial index decoding ─────────────────────────────────────────────────

N_REPS=10
BODIES=(ant babya babyb blokky garrix gecko insect linkin longleg park penguin
        pentapod queen salamander snake spider spider45 squarish stingray
        tinlicker turtle ww zappa)
BRAINS=(matsuoka ann revolve_cpg sine)   # slowest first
N_BODIES=${#BODIES[@]}
N_BRAINS=${#BRAINS[@]}

ID=$SLURM_ARRAY_TASK_ID
BODY_IDX=$(( ID % N_BODIES ))
BRAIN_IDX=$(( (ID / N_BODIES) % N_BRAINS ))
REP=$(( (ID / (N_BODIES * N_BRAINS)) % N_REPS ))

BODY=${BODIES[$BODY_IDX]}
BRAIN=${BRAINS[$BRAIN_IDX]}
SEED=$((42 + REP))

BUDGET=${BUDGET:-10000}
DURATION=${DURATION:-15}
CONTROL_FREQ=${CONTROL_FREQ:-20}
INITIAL_MEAN=${INITIAL_MEAN:-}   # empty = per-brain default
INITIAL_MEAN_ARG=${INITIAL_MEAN:+--initial-mean $INITIAL_MEAN}

TMP_DIR=/tmp/${USER}/ariel_zoo_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=/scratch/jed/ariel_zoo_benchmark
RUN_TAG=zoo_${BODY}_${BRAIN}_rep${REP}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# ── Environment ───────────────────────────────────────────────────────────────

cd "$REPO"
mkdir -p out_files

source "$VENV_PATH/bin/activate"

# Headless EGL rendering (no X display on compute nodes).
export MUJOCO_GL=egl

# ── Diagnostics ───────────────────────────────────────────────────────────────

echo "========================================================"
echo "Node:           $(hostname)"
echo "Array job/task: $SLURM_ARRAY_JOB_ID / $SLURM_ARRAY_TASK_ID"
echo "Body:           $BODY"
echo "Brain:          $BRAIN"
echo "Rep:            $REP"
echo "Seed:           $SEED"
echo "Budget:         $BUDGET evaluations"
echo "Episode:        ${DURATION}s at ${CONTROL_FREQ}Hz control"
echo "CMA-ES x0:      ${INITIAL_MEAN:-per-brain default}"
echo "CPUs:           $SLURM_CPUS_PER_TASK"
echo "Tmp dir:        $TMP_DIR"
echo "Final dir:      $FINAL_DIR/$RUN_TAG"
echo "Python:         $(which python)  ($(python --version 2>&1))"
echo "Started:        $(date)"
echo "========================================================"

# ── Run (tmp → scratch on exit) ───────────────────────────────────────────────

srun bash -c "
    set -uo pipefail
    mkdir -p \"$TMP_DIR\"
    finalize() {
        rc=\$?
        if [ -d \"$TMP_DIR\" ] && [ -n \"\$(ls -A \"$TMP_DIR\" 2>/dev/null || true)\" ]; then
            echo \"Moving results from $TMP_DIR to $FINAL_DIR/$RUN_TAG ...\"
            mkdir -p \"$FINAL_DIR\"
            mv \"$TMP_DIR\" \"$FINAL_DIR/$RUN_TAG\"
            echo \"Results saved to $FINAL_DIR/$RUN_TAG\"
        else
            echo \"No results in $TMP_DIR to move (rc=\$rc)\"
            rm -rf \"$TMP_DIR\" || true
        fi
        exit \$rc
    }
    trap finalize EXIT

    cd \"$SCRIPTS_DIR\"

    python zoo_benchmark.py \
        --body $BODY \
        --brain $BRAIN \
        --seed $SEED \
        --budget $BUDGET \
        --duration $DURATION \
        --control-freq $CONTROL_FREQ \
        $INITIAL_MEAN_ARG \
        --workers $SLURM_CPUS_PER_TASK \
        --out-dir \"$TMP_DIR\"
"

echo "Finished: $(date)"
echo "done"
