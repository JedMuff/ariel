#!/bin/bash
#SBATCH --job-name=sym-bench
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Cost per evaluation for every genome and brain on this node type, and the
# projected core-hours of Experiments 1 and 2 (benchmark.py). Submit once per
# partition you consider (sbatch -p genoa ... for the other).

set -euo pipefail
source "${REPO:-$(dirname "$0")/../..}/slurm/symmetry_paper/_common.sh"
setup_env
banner "Partition:      ${SLURM_JOB_PARTITION:-?}"
python "$CODE/benchmark.py" --workers "${SLURM_CPUS_PER_TASK:-32}" --bodies 4 --evals 60 \
    -o "$OUT_ROOT/benchmark/${SLURM_JOB_PARTITION:-local}_${SLURM_CPUS_PER_TASK:-32}"
