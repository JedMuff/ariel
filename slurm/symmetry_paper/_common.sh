# Shared setup for the symmetry paper's Slurm scripts (sourced, not submitted).
#
# Environment overrides:
#   REPO        checkout of the symmetry-paper branch (default below)
#   FINAL_ROOT  where results go (default /scratch/$USER/ariel_paper_v2; the v1
#               runs, normalised fitness and (10+20) with --repeat-evals, are in
#               /scratch/$USER/ariel_paper)
#
# Paper v2 protocol: fitness is forward displacement in m (unclipped, no
# penalties), outer EA (20+20) without re-evaluation; V2_EVOLVE_ARGS /
# V2_TRAIN_ARGS carry it.

REPO=${REPO:-/home/jed/workspaces/ariel-symmetry/ariel_paper}
FINAL_ROOT=${FINAL_ROOT:-/scratch/$USER/ariel_paper_v2}
V2_EVOLVE_ARGS=(--budget 30 --pop 20 --lam 20 --fitness displacement --max-modules 25 --max-depth 25)
V2_TRAIN_ARGS=(--fitness displacement)
SCRIPTS_DIR=$REPO/examples/symmetry_paper

setup_env() {
    cd "$REPO"
    mkdir -p out_files
    # shellcheck disable=SC1091
    source "$REPO/.venv/bin/activate"
    export MUJOCO_GL=egl          # headless rendering on compute nodes
    export PYTHONHASHSEED=0       # nothing should depend on it; fixed anyway
}

banner() {  # banner "key: value" ...
    echo "========================================================"
    echo "Node:           $(hostname)"
    echo "Array job/task: ${SLURM_ARRAY_JOB_ID:-?} / ${SLURM_ARRAY_TASK_ID:-?}"
    for line in "$@"; do echo "$line"; done
    echo "CPUs:           ${SLURM_CPUS_PER_TASK:-?}"
    echo "Python:         $(which python)  ($(python --version 2>&1))"
    echo "Git commit:     $(git -C "$SCRIPTS_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "Started:        $(date)"
    echo "========================================================"
}

# run_in_tmp TMP_DIR FINAL_DIR command [args...]
# Runs the command inside TMP_DIR (node-local) and moves TMP_DIR to FINAL_DIR
# when it ends, also on failure or when Slurm signals the time limit
# (submit with --signal=B:TERM@120 so there is time to move).
run_in_tmp() {
    _TMP=$1 _FINAL=$2
    shift 2
    mkdir -p "$_TMP"
    trap _finalize EXIT
    trap 'exit 143' TERM
    (cd "$_TMP" && srun "$@")
}

_finalize() {
    rc=$?
    if [ -d "$_TMP" ] && [ -n "$(ls -A "$_TMP" 2>/dev/null)" ]; then
        mkdir -p "$(dirname "$_FINAL")"
        mv "$_TMP" "$_FINAL"
        echo "Results saved to $_FINAL (exit $rc)"
    else
        echo "No results in $_TMP to move (exit $rc)"
        rm -rf "$_TMP"
    fi
    echo "Finished: $(date)"
}
