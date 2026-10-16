#!/bin/bash
#
# Rebuild all zoo learning-budget pilot figures from __data__/social/ariel_zoo/ig{10,20}
# (run_social_zoo_budget.sh), per budget B:
#   __data__/social/figures/zoo_budget/report/B/              multi-panel overview + CSVs (parta_report.py)
#   __data__/social/figures/zoo_budget/paper/{pdf,png}/B/     publication figures (paper_figures.py)
#   __data__/social/figures/zoo_budget/learning_budget/B/     retrospective truncation cache + CSVs,
#   __data__/social/figures/zoo_budget/learning_budget/{pdf,png}/B/   and its figures (learning_budget.py)
# and across budgets:
#   __data__/social/figures/zoo_budget/compare/               CSVs (zoo_budget_compare.py)
#   __data__/social/figures/zoo_budget/paper/{pdf,png}/compare/   its figures
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

DATA=__data__/social/ariel_zoo
OUT=__data__/social/figures/zoo_budget
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
uv run "$ANALYSIS/zoo_budget_compare.py" --budgets ig10 ig20
uv run "$ANALYSIS/build_overleaf.py"
