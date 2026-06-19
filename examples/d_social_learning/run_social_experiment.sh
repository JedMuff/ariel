#!/usr/bin/env bash
# Orchestrate all social-learning experiment runs.
#
# Full grid: 8 schemes × 3 x-values × 10 reps = 240 ARIEL + 240 EvoGym runs.
#
# Usage:
#   ./run_social_experiment.sh [options]
#
# Options (all optional):
#   --gens N          outer generations (default 100)
#   --inner-gens N    inner CMA-ES generations (default 20)
#   --pop N           mu (default 20)
#   --lam N           lambda (default 100)
#   --workers N       parallel workers (default cpu count)
#   --reps N          repetitions (default 10)
#   --schemes LIST    comma-separated scheme names (default all 8)
#   --x-values LIST   comma-separated x values (default 0,0.5,1)
#   --domain LIST     comma-separated domains: ariel,evogym (default both)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
GENS=100
INNER_GENS=20
INNER_POP=16
POP=20
LAM=100
WORKERS=""
REPS=10
SCHEMES="darwinian,lamarckian,random,random_many,best,best_many,similar,similar_many"
X_VALUES="0,0.5,1"
DOMAIN="ariel,evogym"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gens)        GENS="$2";        shift 2 ;;
        --inner-gens)  INNER_GENS="$2";  shift 2 ;;
        --inner-pop)   INNER_POP="$2";   shift 2 ;;
        --pop)         POP="$2";         shift 2 ;;
        --lam)         LAM="$2";         shift 2 ;;
        --workers)     WORKERS="$2";     shift 2 ;;
        --reps)        REPS="$2";        shift 2 ;;
        --schemes)     SCHEMES="$2";     shift 2 ;;
        --x-values)    X_VALUES="$2";    shift 2 ;;
        --domain)      DOMAIN="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

WORKERS_FLAG=""
if [[ -n "$WORKERS" ]]; then
    WORKERS_FLAG="--workers $WORKERS"
fi

IFS=',' read -ra SCHEME_LIST <<< "$SCHEMES"
IFS=',' read -ra X_LIST <<< "$X_VALUES"
IFS=',' read -ra DOMAIN_LIST <<< "$DOMAIN"

total_ariel=0
total_evogym=0

cd "$REPO_ROOT"

for domain in "${DOMAIN_LIST[@]}"; do
    for scheme in "${SCHEME_LIST[@]}"; do
        for x in "${X_LIST[@]}"; do
            for (( rep=0; rep<REPS; rep++ )); do
                if [[ "$domain" == "ariel" ]]; then
                    echo "[ariel] scheme=$scheme x=$x rep=$rep"
                    uv run examples/d_social_learning/ariel/experiment.py \
                        --scheme "$scheme" --x "$x" --rep "$rep" \
                        --gens "$GENS" --pop "$POP" --lam "$LAM" \
                        --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" $WORKERS_FLAG
                    (( total_ariel++ )) || true
                elif [[ "$domain" == "evogym" ]]; then
                    echo "[evogym] scheme=$scheme x=$x rep=$rep"
                    evogym-venv/bin/python examples/d_social_learning/evogym/experiment.py \
                        --scheme "$scheme" --x "$x" --rep "$rep" \
                        --gens "$GENS" --pop "$POP" --lam "$LAM" \
                        --inner-gens "$INNER_GENS" --inner-pop "$INNER_POP" $WORKERS_FLAG
                    (( total_evogym++ )) || true
                fi
            done
        done
    done
done

echo ""
echo "Done. ARIEL runs: $total_ariel  EvoGym runs: $total_evogym"
