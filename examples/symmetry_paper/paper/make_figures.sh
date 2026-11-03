#!/usr/bin/env bash
# Rebuild every paper figure and number from the v2 data, then copy the figures
# into paper/figures/ (the folder uploaded to Overleaf).
#
#   bash examples/symmetry_paper/paper/make_figures.sh        # from the repo root
#   FAST=1 bash ...                                            # skip the two body-settle scripts
#   V2=/path/to/ariel_paper_v2 bash ...                        # another copy of the v2 runs
#   E2_NAME=e2_x3 OUT=$V2/paper_x3 bash ...                    # the x3 fixed-body runs
#
# Reads e1 (and e1_dann/sb10 if present), $E2_NAME/{random,tied,shape_matched,leg_matched},
# the body files in e2/, and e3/... under $V2. E2_NAME=e2_x3 reads the replays
# e3/e2_x3_random and e3/phase_shift_e2_x3_random.
# Outputs (CSVs, logs, inline numbers) stay in $OUT; only PNGs are copied.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
S=examples/symmetry_paper
D=${V2:-__data__/ariel_paper_v2}
OUT=${OUT:-$D/paper}
FIG=$S/paper/figures
E2=${E2_NAME:-e2}
E3_E2=${E2}_random
E1_ROOTS=("$D/e1"); E1_REPLAYS=("$D/e3/e1")
if [[ -d $D/e1_dann/sb10 ]]; then E1_ROOTS+=("$D/e1_dann/sb10"); E1_REPLAYS+=("$D/e3/e1_dann_sb10"); fi
mkdir -p "$OUT/logs" "$FIG"
run() { local name=$1; shift; echo "== $name"; uv run python "$@" > "$OUT/logs/$name.txt" 2>&1 || { tail -20 "$OUT/logs/$name.txt"; exit 1; }; }

run e1 $S/analyse_e1.py --root "${E1_ROOTS[@]}" --behaviour "${E1_REPLAYS[@]}" --out $OUT/e1
run e2 $S/analyse_e2.py --roots $D/$E2/random --out $OUT/e2
run e3 $S/analyse_e3.py --roots "${E1_REPLAYS[@]}" --training-roots $D/e3/$E3_E2 --out $OUT/e3
run ps $S/analyse_phase_shift.py --roots $D/e3/phase_shift_$E3_E2 --out $OUT/phase_shift
if [[ -z "${FAST:-}" || ! -f $OUT/matched/matched_bodies.csv || ! -f $OUT/bodies/body_features.csv ]]; then
    run matched $S/analyse_matched.py --roots $D/$E2/{random,tied,shape_matched,leg_matched} \
        --bodies $D/e2/bodies_{random,tied,shape_matched,leg_matched}.jsonl --out $OUT/matched
    run bodies $S/analyse_bodies.py --bodies $D/e2/bodies_random.jsonl --replays $D/e3/$E3_E2 --out $OUT/bodies
fi
run extra $S/paper_figures.py --data $D --e2-name $E2 --e3-training-name $E3_E2 --analysis $OUT --out $OUT/extra
run stats $S/paper_stats.py --data $D --e2-name $E2 --phase-shift-name phase_shift_$E3_E2 --analysis $OUT --out $OUT/stats

rm -f $FIG/*.png
cp $OUT/e1/fig_e1_curves_genome.png         $FIG/evolution_genomes.png
cp $OUT/extra/fig_curves_sets.png           $FIG/fixed_body_curves.png
cp $OUT/e2/fig5_learning_curves.png         $FIG/fixed_body_controllers.png
cp $OUT/extra/fig_shape_measures.png        $FIG/shape_measures.png
cp $OUT/matched/fig9_matched.png            $FIG/body_sets.png
cp $OUT/extra/fig_locked.png                $FIG/locked.png
cp $OUT/phase_shift/fig8_phase_shift_relative.png $FIG/phase_shift.png
echo "figures -> $FIG; numbers -> $OUT/logs, $OUT/extra/inline_numbers.txt, $OUT/stats/stats.txt"
