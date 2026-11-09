#!/bin/bash
#SBATCH --job-name=sym-replay
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Gait measures of finished runs, PER_TASK run directories of LIST (one per
# line, e.g. from `python replay.py --list-runs ...`) per array task:
# replay.py (gait class, phase quality, cycles) over WHICH (default final:
# the final population, every seed of an E2 training) into
# $OUT_ROOT/replay/<run name>, and mechanics.py --mode measure on the
# champion into $OUT_ROOT/mechanics/<run name> (runs.run_name: for E1
# <genome>__<brain>__e<evals>__s<seed>). Run it on the machine type that
# trained the runs, or the replayed scores will not reproduce the stored ones.
#
# Optional environment: REPLAY_DIR [replay], MECH_DIR [mechanics], MECH_WHICH
# [champion], and REST_DIR [none]: if set, rest_measures.py of WHICH's bodies
# into $OUT_ROOT/$REST_DIR/<run name>.

set -euo pipefail
source "${REPO:-$(dirname "$0")/../..}/slurm/symmetry_paper/_common.sh"
setup_env

: "${LIST:?set LIST to a file of run directories}"
PER_TASK=${PER_TASK:-20}
WHICH=${WHICH:-final}
REPLAY_DIR=${REPLAY_DIR:-replay}
MECH_DIR=${MECH_DIR:-mechanics}
MECH_WHICH=${MECH_WHICH:-champion}
FIRST=$(( ${LIST_OFFSET:-0} + SLURM_ARRAY_TASK_ID * PER_TASK ))
mapfile -t RUNS < <(sed -n "$(( FIRST + 1 )),$(( FIRST + PER_TASK ))p" "$LIST")

banner "Runs:           ${#RUNS[@]} from line $FIRST of $LIST"
for run in "${RUNS[@]}"; do
    name=$(cd "$CODE" && python -c "import runs, sys; print(runs.run_name(sys.argv[1]))" "$run")
    [ -f "$OUT_ROOT/$REPLAY_DIR/$name/cycles.npz" ] && continue    # written last
    python "$CODE/replay.py" --run-dir "$run" --which "$WHICH" --workers "${SLURM_CPUS_PER_TASK:-32}" \
        -o "$OUT_ROOT/$REPLAY_DIR"
done
python "$CODE/mechanics.py" --mode measure --run-dir "${RUNS[@]}" --which "$MECH_WHICH" \
    --workers "${SLURM_CPUS_PER_TASK:-32}" -o "$OUT_ROOT/$MECH_DIR"
if [ -n "${REST_DIR:-}" ]; then
    python "$CODE/rest_measures.py" --run-dir "${RUNS[@]}" --which "$WHICH" \
        --workers "${SLURM_CPUS_PER_TASK:-32}" -o "$OUT_ROOT/$REST_DIR"
fi
echo "Finished: $(date)"
