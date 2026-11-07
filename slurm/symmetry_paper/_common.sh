# Shared setup for the symmetry paper's Slurm scripts (sourced, not submitted).
#
# Environment (set before sbatch, or in ~/.symmetry_paper.env, which is sourced if present):
#   REPO       checkout to run from (default: the one this file is in)
#   OUT_ROOT   where results go (default /scratch-shared/$USER/symmetry_paper).
#              Snellius purges /scratch-shared after 14 days: copy finished runs
#              to project space, or point OUT_ROOT there.
#   VENV       virtualenv to activate (default $REPO/.venv)

# shellcheck disable=SC1090
[ -f "$HOME/.symmetry_paper.env" ] && source "$HOME/.symmetry_paper.env"

_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(git -C "$_here" rev-parse --show-toplevel 2>/dev/null || echo "$_here/../..")}
OUT_ROOT=${OUT_ROOT:-/scratch-shared/$USER/symmetry_paper}
VENV=${VENV:-$REPO/.venv}
CODE=$REPO/examples/symmetry_paper
SLURM_DIR=$REPO/slurm/symmetry_paper

setup_env() {
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export MUJOCO_GL=egl          # headless on compute nodes
    export PYTHONHASHSEED=0       # nothing should depend on it; fixed anyway
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    mkdir -p "$OUT_ROOT"
}

banner() {  # banner "key: value" ...
    echo "========================================================"
    echo "Node:           $(hostname)  ($(lscpu | sed -n 's/^Model name: *//p'))"
    echo "Job/task:       ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-?}} / ${SLURM_ARRAY_TASK_ID:-?}"
    for line in "$@"; do echo "$line"; done
    echo "CPUs:           ${SLURM_CPUS_PER_TASK:-?}"
    echo "Python:         $(command -v python)  ($(python --version 2>&1))"
    echo "Git commit:     $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "Started:        $(date)"
    echo "========================================================"
}

# line_of LIST INDEX: line INDEX (0-based) of a list file
line_of() {
    sed -n "$(( $2 + 1 ))p" "$1"
}

# seconds left before this job's time limit, minus a margin (for --time-limit)
seconds_left() {
    local margin=${1:-900} left=""
    [ -n "${SLURM_JOB_ID:-}" ] && left=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null || true)
    python - "$left" "$margin" <<'EOF'
import sys
s, margin = sys.argv[1].strip(), int(sys.argv[2])
days, _, rest = s.rpartition("-")
parts = [int(x) for x in rest.split(":")] if rest else []
while len(parts) < 3:
    parts.insert(0, 0)
h, m, sec = parts
print(max(60, (int(days or 0) * 24 + h) * 3600 + m * 60 + sec - margin) if rest else 10**9)
EOF
}
