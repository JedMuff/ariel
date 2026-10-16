#!/bin/bash
#
# Rebuild all zoo learning-budget figures from $DATA/ig{10,20}
# (run_social_zoo_budget.sh) into $OUT (defaults below), per budget B:
#   $OUT/report/B/                        multi-panel overview + CSVs (parta_report.py)
#   $OUT/paper/{pdf,png}/B/               publication figures (paper_figures.py)
#   $OUT/learning_budget/B/               retrospective truncation cache + CSVs,
#   $OUT/learning_budget/{pdf,png}/B/     and its figures (learning_budget.py)
# and across budgets:
#   $OUT/compare/                         CSVs (zoo_budget_compare.py)
#   $OUT/paper/{pdf,png}/compare/         its figures
#
# and finally copies the figures the reports use into the Overleaf folder
# __data__/social/figures/overleaf (build_overleaf.py), which also holds the
# report, zoo_budget_report.tex.
#
# learning_budget.py caches the per-offspring curves in budget_curves.npz;
# delete it to re-extract after the data changes.
#
# Usage (from the repo root):
#   bash examples/d_social_learning/make_zoo_budget_figures.sh
set -euo pipefail

# Defaults: the --final-evals 10 runs (run_social_zoo_budget.sh). For the first
# pilot batch (best-of-N scoring, 5 schemes x 10 reps) run with
#   DATA=__data__/social/ariel_zoo OUT=__data__/social/figures/zoo_budget
DATA=${DATA:-__data__/social/ariel_zoo_fe10}
OUT=${OUT:-__data__/social/figures/zoo_budget_fe10}
ANALYSIS=examples/d_social_learning/analysis
# All 8 part A schemes, in part A's order (darwinian first).
SCHEMES=(darwinian lamarckian random random_many best best_many similar_TOPDOWN similar_many_TOPDOWN)

for B in ig10 ig20; do
    uv run "$ANALYSIS/parta_report.py" --data-dir "$DATA/$B" --run-novelty TOPDOWN \
        --schemes "${SCHEMES[@]}" --x-values 0.5 1 --out-dir "$OUT/report/$B"
    uv run "$ANALYSIS/paper_figures.py" --report-dir "$OUT/report/$B" --out-dir "$OUT/paper" \
        --prefix "$B" --run-novelty TOPDOWN --schemes "${SCHEMES[@]}"
    uv run "$ANALYSIS/learning_budget.py" --report-dir "$OUT/report/$B" --out-dir "$OUT/learning_budget" \
        --prefix "$B" --run-novelty TOPDOWN --schemes "${SCHEMES[@]}"
done
uv run "$ANALYSIS/zoo_budget_compare.py" --budgets ig10 ig20 --report-root "$OUT/report" \
    --budget-root "$OUT/learning_budget" --paper-dir "$OUT/paper" --out-dir "$OUT/compare"
uv run "$ANALYSIS/build_overleaf.py"
