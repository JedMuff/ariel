#!/usr/bin/env bash
# Rebuild every paper figure and number from __data__, then copy the figures
# into paper/figures/ (the folder uploaded to Overleaf).
#
#   bash examples/symmetry_paper/paper/make_figures.sh        # from the repo root
#   FAST=1 bash ...                                            # skip the two settle-simulating scripts
#   V2=/path/to/ariel_paper_v2 bash ...                        # the v2 runs (copied from the cluster)
#
# Without V2 the v1 data are used (normalised fitness; the paper's current
# numbers). With V2 the same layout is read from that root: e0_grid,
# e0_budget/sb{10,40,80}, e1, e2/{random,tied,shape_matched}, e3/....
#
# Outputs (CSVs, logs, inline numbers) stay in $OUT; only PNGs are copied.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
S=examples/symmetry_paper
if [[ -n "${V2:-}" ]]; then
    D=$V2
    GRID=("$D/e0_grid" "$D/e0_budget/sb80")
    BUDGET=("$D/e0_budget/sb10" "$D/e0_budget/sb40" "$D/e0_budget/sb80")
    E1=("$D/e1" "$D/e0_budget/sb10")
else
    D=__data__/ariel_paper
    GRID=(__data__/ariel_actuator_grid)
    BUDGET=(__data__/ariel_symmetry_pressure_sweep_dann __data__/ariel_dann_budget/sb40 __data__/ariel_dann_budget/sb80)
    E1=("$D/e1" __data__/ariel_symmetry_pressure_sweep_dann)
fi
OUT=${OUT:-$D/paper}
FIG=$S/paper/figures
mkdir -p "$OUT/logs" "$FIG"
run() { local name=$1; shift; echo "== $name"; uv run python "$@" > "$OUT/logs/$name.txt" 2>&1 || { tail -20 "$OUT/logs/$name.txt"; exit 1; }; }

run e0 $S/analyse_e0.py --grid-roots "${GRID[@]}" --budget-roots "${BUDGET[@]}" --out $OUT/e0
run e1 $S/analyse_e1.py --root "${E1[@]}" --behaviour $D/e3/e1 --out $OUT/e1
run e2 $S/analyse_e2.py --roots $D/e2/random --out $OUT/e2
run e3 $S/analyse_e3.py --roots $D/e3/e1 --training-roots $D/e3/e2_random --out $OUT/e3
run ps $S/analyse_phase_shift.py --roots $D/e3/phase_shift_e2_random --out $OUT/phase_shift
run ps_e1 $S/analyse_phase_shift.py --roots $D/e3/phase_shift_e1 --out $OUT/phase_shift_e1
if [[ -z "${FAST:-}" || ! -f $OUT/matched/matched_bodies.csv || ! -f $OUT/bodies/body_features.csv ]]; then
    run matched $S/analyse_matched.py --roots $D/e2/{random,tied,shape_matched} \
        --bodies $D/e2/bodies_{random,tied,shape_matched}.jsonl --out $OUT/matched
    run bodies $S/analyse_bodies.py --bodies $D/e2/bodies_random.jsonl --replays $D/e3/e2_random --out $OUT/bodies
fi
run extra $S/paper_figures.py --data $D --analysis $OUT --grid-root "${GRID[0]}" --out $OUT/extra

cp $OUT/e0/fig1_grid_budget.png             $FIG/r1_grid_budget.png
cp $OUT/e1/fig_e1_curves_genome.png         $FIG/r1_e1_curves_genome.png
cp $OUT/e1/fig_e1_curves_controller.png     $FIG/r1_e1_curves_controller.png
cp $OUT/extra/fig_curves_sets.png           $FIG/r2_curves_sets.png
cp $OUT/e2/fig5_learning_curves.png         $FIG/r2_curves_variants.png
cp $OUT/e2/fig2_bodies.png                  $FIG/r3_body_counts.png
cp $OUT/extra/fig_shape_measures.png        $FIG/r3_shape_measures.png
cp $OUT/matched/fig9_matched.png            $FIG/r3_matched.png
cp $OUT/e3/fig3_gait_classes.png            $FIG/r4_gait_classes.png
cp $OUT/e3/fig6_phase_training_random.png   $FIG/r5_phase_training.png
cp $OUT/extra/fig_locked.png                $FIG/r5_locked.png
cp $OUT/extra/fig_locked_seeds.png          $FIG/r5_locked_seeds.png
cp $OUT/phase_shift/fig8_phase_shift_relative.png $FIG/r5_phase_shift_relative.png
cp $OUT/phase_shift/fig8_phase_shift.png    $FIG/r5_phase_shift.png
cp $OUT/phase_shift_e1/fig8_phase_shift.png $FIG/r5_phase_shift_e1.png
echo "figures -> $FIG; numbers -> $OUT/logs, $OUT/extra/inline_numbers.txt"
