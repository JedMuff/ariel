# Paper handoff: data, figures, numbers and methods

Oct 31, 2026 · for drafting the LaTeX paper from `PAPER_PLAN.md`

This file gives everything needed to write the paper without re-deriving it from the code: where each dataset lives, how to rebuild each figure, where each number in the plan comes from, and the method parameters.

## 0. Read first

- **`PAPER_PLAN.md` is the authority** for the story, section structure (R1–R6), figure list and numbers. Revised Oct 31 to a body-advantage thesis.
- **`SYMMETRY_ADVANTAGE.md` is out of date.** It records the investigation branch's results under the old framing (coordination, learnability) and refers to that branch's scripts. Use it only for grid details that the plan quotes (gait shares across the hinge grid). Where it disagrees with the plan, the plan wins.
- **Scope rules:** the paper uses two brains, `bang_bang` and `dann`; `bang_bang_free` and `bang_bang_drift` are variants of bang\_bang, not further brains. Do not cite results from the ANN brain, the old `sweep` simulator profile, the local-module runs, or the food task. Rowing is a gait class, not a hypothesis. The hinge grid is reported by its parameter ranges.
- **Branch:** `symmetry-paper`. Code: `examples/symmetry_paper/`; Slurm: `slurm/symmetry_paper/`. Every command below runs from the repo root as `uv run python examples/symmetry_paper/<script> …`.
- **Never run plain `ruff check`** here: the repo config rewrites files and deletes `print` calls. Use `ruff check --no-fix`.

## 1. Data map

All paths relative to the repo root. `__data__` is not in git.

| data | path | what |
| --- | --- | --- |
| Hinge grid (E0) | `__data__/ariel_actuator_grid/` | 216 evolution runs: kp {1, 1.36, 2.5} × kv {0.356, 1, 2} × armature {0, 0.081} × {bang\_bang, dann} × {tree, sym} × 3 seeds |
| Grid gait labels | `__data__/ariel_actuator_grid/rowing_grid/individuals.csv` | last 5 generations of every grid run replayed, gait class per episode (investigation-branch format) |
| DANN budget, 200 evals | `__data__/ariel_symmetry_pressure_sweep_dann/` | DANN evolution, skill budget 10 (200 evals per body), tree vs sym × 5 |
| DANN budget, 800 / 1600 evals | `__data__/ariel_dann_budget/sb40/`, `…/sb80/` | same at skill budget 40 / 80, × 5 |
| E1 evolution | `__data__/ariel_paper/e1/` | 90 runs: {bang\_bang, bang\_bang\_free, bang\_bang\_drift} × {tree, tree\_symmetric, tree\_tied} × seeds 42–51, zoo hinge |
| E2 bodies | `__data__/ariel_paper/e2/bodies_{random,tied,shape_matched}.jsonl` | 40 + 40 hinge-matched random tree / sym bodies; 40 tied; 40 tree bodies shape-matched to the sym ones |
| E2 trainings | `__data__/ariel_paper/e2/{random,tied,shape_matched}/` | random: 80 bodies × 4 brains × 3 seeds = 960; tied and shape\_matched: 40 × {bang\_bang, dann} × 3 = 240 each |
| E3 replays | `__data__/ariel_paper/e3/e1/`, `…/e3/e2_random/` | gait and phase measures of E1 last-generation bodies and every E2 random checkpoint (50 / 200 / 800 / 1600 / 5000 evals) |
| Phase-shift replays | `__data__/ariel_paper/e3/phase_shift_{e2_random,e1}/` | 25 perturbed episodes per E2 final checkpoint / E1 champion |

Each E1 run: `<run>/__data__/evolve/forward/{run_config.json, run_data.jsonl, checkpoints/}`. Each E2 training: `<training>/{run_config.json, curve.jsonl, final.json, checkpoints/evalsNNNNN/}`.

## 2. Figures

Suggested output root for every command: `OUT=__data__/ariel_paper/paper`.

### Existing scripts

| plan figure | file | command |
| --- | --- | --- |
| 1a. Grid gap heatmap | `fig1a_grid_gap.png` | `analyse_e0.py --grid-root __data__/ariel_actuator_grid --budget-roots __data__/ariel_symmetry_pressure_sweep_dann __data__/ariel_dann_budget/sb40 __data__/ariel_dann_budget/sb80 --out $OUT/e0` |
| 1b. DANN gap vs budget | `fig1b_budget_gap.png` | same command |
| 2. E2 learning curves (tree vs sym only) | `fig5_learning_curves.png` | `analyse_e2.py --roots __data__/ariel_paper/e2/random --out $OUT/e2` |
| 3a. Body counts by genome | `fig2_bodies.png` | same command (module, hinge, limb counts) |
| 3c. Final score per body set (main figure) | `fig9_matched.png` | `analyse_matched.py --roots __data__/ariel_paper/e2/{random,tied,shape_matched} --bodies __data__/ariel_paper/e2/bodies_{random,tied,shape_matched}.jsonl --out $OUT/matched` |
| 4a. Gait class shares | `fig3_gait_classes.png` | `analyse_e3.py --roots __data__/ariel_paper/e3/e1 --training-roots __data__/ariel_paper/e3/e2_random --out $OUT/e3` |
| 5 (support). Phase measures along training | `fig6_phase_training_random.png` | same command |
| 5b. Phase-shift replay | `fig8_phase_shift.png` | `analyse_phase_shift.py --roots __data__/ariel_paper/e3/phase_shift_e2_random --out $OUT/phase_shift` (E1: `…/phase_shift_e1`, separate `--out`) |
| 6. E1 paired gaps | `fig7_ablation.png` | `analyse_e1.py --root __data__/ariel_paper/e1 --behaviour __data__/ariel_paper/e3/e1 --out $OUT/e1` |
| (tables) Body measures | `body_features.csv`, `bodies_locked.csv`, `bodies_score_corr.csv` | `analyse_bodies.py --bodies __data__/ariel_paper/e2/bodies_random.jsonl --replays __data__/ariel_paper/e3/e2_random --out $OUT/bodies` |

All scripts run in seconds to a few minutes locally, except `analyse_matched.py` and `analyse_bodies.py`, which simulate each body's 3 s settle (~10–30 s with all CPUs).

### Missing figures (no script yet)

| plan figure | what to plot | data |
| --- | --- | --- |
| 2. Learning curves, four body sets | Best score so far vs evaluations (log x), mean ± SEM over bodies (seed mean per body first), tree / sym / tied / shape-matched, one panel per brain (bang\_bang, DANN). Inset or second row: sym / tree ratio vs evals | `analyse_e2.load_trainings([random, tied, shape_matched roots])` returns per-training curves; body set = `genome_type`, except `body_source == "shape_matched"` |
| 3b. Shape measures by genome | Distributions (strip or box) of height, front–back contact spread (`contact_len`), leg hinges (`n_leg`), yaw-axis leg hinges (`n_leg_yaw`), lateral-axis hinges (`n_pitch`), for tree / sym / tied / shape-matched | `body_features.csv` (random tree and sym) and `matched_bodies.csv` (all four sets; shape measures only) |
| 4b. Gait vs actuator | Gait-class shares across kp × kv (per armature), sym vs tree, bang\_bang, with the score gap overlaid | `__data__/ariel_actuator_grid/rowing_grid/individuals.csv` (check its columns; numbers in `SYMMETRY_ADVANTAGE.md`) |
| 5a. Locked vs unlocked | Final score of E2 random bodies: tree, sym unlocked, sym locked, per brain (bang\_bang, drift, DANN) | see section 3, "Locked vs unlocked symmetric bodies" |
| 6 (absolute). E1 scores | Mean score per arm (brain × genome) beside the paired gaps, to show free frequency's collapse | `$OUT/e1/e1_runs.csv` (`score` by `brain`, `genome`) |

Figure style: use `runs.style(ax)` and the colours in `runs.GENOME_COLOR` (tree orange, symmetric blue, tied green; shape-matched tree as faded orange, as in `analyse_matched.py`), `runs.BRAIN_LABEL` for brain names.

## 3. Where each number comes from

**Scripted:** the number appears in the named script's printout or CSV with the command from section 2.

| plan number(s) | source |
| --- | --- |
| Grid +0.14 bang\_bang (17/18 cells, p < 0.001), DANN +0.02 (p = 0.31) | `analyse_e0` → `e0_grid_pooled.csv`, `e0_grid_cells.csv` |
| DANN budget −0.02 / +0.22 / +0.37, 5/5 wins at 1600 | `analyse_e0` → `e0_budget.csv` |
| E1 gaps (+0.08 p = 0.70; drift +0.15; tied − tree +0.13 / +0.14; sym ≈ tied) and arm means | `analyse_e1` → `e1_gaps.csv`, `e1_runs.csv` |
| E1 champions phase-locked 8/10 sym, 0/10 tree, 1/10 tied; free-frequency CV 0.10–0.19 | `analyse_e1` printout (champions table; `freq_cv`) |
| E2 random gaps +0.15 / +0.12 / +0.13 / +0.02 | `analyse_e2` → `e2_gaps.csv` (`metric == "final"`) |
| E2 sym / tree ratios (1.55 / 1.45 / 1.41 etc.) | `e2_bodies.csv`: mean of `score_at_{200,800,1600}` by `brain`, `genome_type`, sym / tree |
| E2 DANN gap at 200 evals +0.07 | `e2_gaps.csv`, `metric == "score_at_200"` |
| Gait shares (alternating, locked) by genome and brain; alternation from 50 evals | `analyse_e3` → `e3_gait_classes.csv`, `e3_champions.csv`, `e3_along_training.csv` |
| Phase-shift: side − random within ±0.05; detune 0.92 → 0.23 locked share | `analyse_phase_shift` → `phase_shift_summary.csv`; locked share after perturbation: mean `phase_locked_frac` per `condition` in the per-run `phase_shift.csv` files, for checkpoints whose baseline `phase_locked_frac ≥ 0.5` |
| Body shape by genome (height 0.14 vs 0.22, etc.); always vs never locked p > 0.1; ρ ≈ 0.6 for leg hinges and contact spread | `analyse_bodies` printout, `bodies_locked.csv`, `bodies_score_corr.csv` |
| E2 controls: 0.363 / 0.434 / 0.459 / 0.510 etc.; 40–50% closed; residual +0.075 / +0.082; adjusted gap | `analyse_matched` → `matched_gaps.csv` |

**Computed inline (not in any script).** Recompute with the recipe; each takes a few lines of pandas.

- **Locked vs unlocked symmetric bodies** (R5: 0.65 / 0.40 / 0.39 bang\_bang; drift 0.55 / 0.37 / 0.34; DANN 0.57 / 0.49 / 0.42). From `__data__/ariel_paper/e3/e2_random/*/episodes.csv`, keep each run's final checkpoint (`evals == max per run`). Locked = `phase_locked_frac ≥ 0.5`. Mean `score` by `brain`, `genome`, locked. Note: `score` here is the replayed normalised score without penalties, slightly different from E2's final score.
- **Locking as a body property** (most sym bodies lock in all or no seeds; 26/40 under bang\_bang; ρ = +0.26 bang\_bang, −0.02 DANN). Same final-checkpoint table; per `body_id` and brain, the share of the 3 seeds that are locked and the mean score; count bodies with share 0 or 1; Spearman of share vs mean score over the 40 sym bodies.
- **Shape-adjusted gap on random bodies** (+0.12 → +0.03). `body_features.csv`: OLS of `score_<brain>` on an is-symmetric indicator plus `n_leg, n_leg_yaw, contact_len, leg_reach, height, n_pitch`, over the 80 random bodies; bootstrap CI over bodies. The raw +0.12 differs from E2's +0.15 because these are replay scores without penalties. `analyse_matched.py`'s adjusted gap (sym vs shape-matched only) is the scripted counterpart and the one to prefer.
- **Tied − tree adjusted for leg hinges** (+0.096 → +0.061 bang\_bang; +0.111 → +0.070 DANN). `matched_bodies.csv`, sets `tree` and `tree_tied`: OLS of `final` on is-tied, then on is-tied + `n_leg`; bootstrap CI.
- **Diagonal hinges not predictive** (ρ = 0.03 in tree; sym 35% vs tree 53% diagonal). `body_features.csv`: `n_diag / n_hinges` by genome, Spearman with `score_bang_bang` within tree.
- **Hinge rotations by genome** (DEG\_0 56% sym vs 22% tree). Count the `rotation` of hinge nodes in `bodies_random.jsonl` genomes by `genome_type`.
- **Early stops** (5–11% under bang\_bang in every set). `final.json` `stopped_early` per training, by set and brain.
- **E1 zoo cell reproduces the grid** (seeds 42–44 identical to the grid's kp 1.36, kv 0.356, armature 0.081 cell). Compare `runs.load_runs` scores of the two roots.

## 4. Methods fact sheet

Values and where they are defined (all under `examples/symmetry_paper/` unless noted).

**Bodies**

- Modular robots built from a core, bricks and hinges ("zoo" modules), `sim.py` constants `ZOO_*`: core half-extents 0.075 m, mass 1.363 kg; brick half-extents 0.0375 m, mass 0.055 kg; hinge stator 0.065 kg, rotor 0.040 kg.
- Genome: tree genome of modules with face and rotation per node (`ariel.ec.genotypes.tree`), max 25 modules, depth limit 25, at least 4 hinges (`sim.MIN_HINGES`, `sim._morph_ok`).
- `tree_symmetric`: mirror-symmetric about the plane through the core containing the forward axis; mirrored subtrees with mirrored rotations; midline nodes forced to `DEG_0` (`src/ariel/ec/genotypes/tree/symmetry.py`).
- `tree_tied`: every placed module gets a randomly placed twin with probability 0.47 (`genomes.DEFAULT_TIE_PROB`), so about 64% of modules are tied, matching the mirrored share of symmetric bodies (`src/ariel/ec/genotypes/tree/tying.py`).
- Body operators: random tree, subtree crossover (probability 0.6), mutation, each in a symmetric / tied version (`sim._morph_ops`, `sim.make_offspring`); invalid offspring are re-mutated up to 50 times.

**Hinge actuator** (`sim.ZOO_HINGE`): position servo, kp 1.36, kv 0.356, armature 0.081, joint limit ±90°. The grid varies kp {1, 1.36, 2.5}, kv {0.356, 1, 2}, armature {0, 0.081}.

**Task and episode** (`sim.EvalProfile`, `sim.run_skill_episode`)

- MuJoCo flat floor, physics at 500 Hz, body spawns unrotated at 0.1 m and settles 3 s with zero control.
- 15 s episode, control every 25 physics steps (20 Hz), targets smoothed by α = 0.854 and clipped to ±90°.
- Score = clip(2 · d / (0.5 m/s · 15 s), 0, 2) = clip(d / 3.75 m, 0, 2), d = displacement along the forward axis (world −y).
- Fitness (minimised) = −(score − height penalty − jerk penalty): height penalty = spawn core height if above 0.5 m; jerk penalty 1.08 × mean |Δctrl| if that mean is at least 0.417.

**Controllers** (`brains.py`; docstring has the full parameterisation)

- `bang_bang`: open-loop square wave, each hinge switching between −90° and +90°. One shared frequency (4 genes: start, end, wobble amplitude and rate; 0.2–2 Hz), per hinge a duty cycle (4 genes) and a phase (1 gene): 4 + 5n parameters.
- `bang_bang_free`: a frequency per hinge, 9n parameters.
- `bang_bang_drift`: shared frequency plus per-hinge offsets scaled by 0.1, 4 + 9n parameters.
- `dann`: one MLP (42 inputs, 16 hidden ReLU, 1 tanh output, 705 parameters) shared by every hinge. Inputs per hinge: its own angle and velocity and module type, the same for its nearest hinge in each of 6 directions (zeros if none), a sine clock (period 25 control steps), the core orientation and the hinge's rest position.

**Learning and evolution**

- Inner loop (`sim.train_skill_for_body`): CMA-ES, σ 0.5, popsize 20, fixed starting mean (0.5 for the bang\_bang family, 0 for DANN), 10 generations = 200 evaluations per body in evolution. CMA-ES stops early on flat fitness (e.g. every candidate scoring 0); E2 carries the final score to the end of the budget.
- Outer loop (`evolve.py`): (μ + λ) = (10 + 20), 30 generations, tournament selection (size 4), `--repeat-evals` (surviving parents retrained every generation). Score of a run = best score of its last generation.
- Fixed bodies (`train_fixed_body.py`): the inner loop alone, bang\_bang family to 1,600 evaluations, DANN to 5,000, 3 training seeds (42–44); checkpoints at 50 / 200 / 800 / 1600 / 5000.
- Random bodies (`sample_bodies.py`): drawn from each genome's initial distribution, hinge-count-matched between genomes (pool of 400 per genome, seed 0). Tied: hinge counts of the symmetric bodies. Shape-matched: from 12,000 random tree bodies, Hungarian assignment per hinge count on standardised `n_leg, n_leg_yaw, contact_len, leg_reach, height, n_pitch`; median standardised distance 0.94.

**Behaviour measures** (`gait.py`, `replay.py`)

- Replay reproduces stored fitness exactly; poses recorded at 50 Hz; the first 2 s after settling are skipped.
- Sides: the vertical plane through the core within 45° of the heading that best mirrors the rest pose; modules within 0.02 m of it are midline. Left and right modules are paired by Hungarian matching of rest positions (mirror image within 0.03 m or 25% of distance to the core).
- Per pair: phase lag (cycles) from the cross-spectrum; rowing below 0.25, alternating at 0.40 or more.
- Episode gait class (`gait.label_of`), in order: static if no module moves (RMS stroke above 2% of body size and 3 mm); unpaired if paired modules carry under 25% of the stroke; aperiodic if over 50% is aperiodic; rowing or alternating if that share exceeds 0.25 and the other; otherwise mixed.
- Phase quality: each pair's trajectories cut into one-period windows, lag per window; quality = 1 − circular variance of the lags (needs 3 or more windows). A pair is locked at quality ≥ 0.9; `phase_locked_frac` = locked pairs' share of the stroke. A body or champion counts as phase-locked when `phase_locked_frac ≥ 0.5`.
- Phase-shift replay (`phase_shift.py`): left-side hinges (by the best mirror plane at spawn) shifted by 0.05 / 0.1 / 0.25 / 0.5 cycles or sped up by 2% / 15%; control: the same number of lateral hinges at random, never a whole side, 3 draws. DANN: the left hinges' clock input is shifted or sped up instead.

**Body measures** (`analyse_bodies.py`, after the settle): extent (length along forward, width, height) of module positions; floor contacts and their spread (std) along forward and lateral; per hinge the dominant world axis of its rotation axis (lateral = pitch, forward = roll, vertical = yaw; diagonal if no component reaches 0.9); leg hinge = a hinge whose distal part touches the floor.

**Statistics** (`runs.py`, scripts)

- Evolution: paired symmetric − tree by seed, Wilcoxon signed-rank, percentile bootstrap 95% CI (10,000 resamples).
- Fixed bodies: seed mean per body, then unpaired difference of body means, bootstrap CI over bodies, Mann-Whitney U; shape-matched pairs also by Wilcoxon.
- Correlations are Spearman. Body-measure analyses are exploratory (~25 measures on 40 bodies per genome); say so.

## 5. Still pending (do not report as done)

- **E4** (second random draw with tied and shape-matched sets), **E5** (tree bodies matched to tied on leg hinges), **E2b** (evolved bodies): planned, not run. R3's confidence ratings in the plan assume 40 bodies per set.
- **Detector hand-check** and the `static` gait videos: not done; gait-class results carry "pending hand-check".
- **DANN evolution puzzle** (no gap at 200 evals in evolution, gap on fixed bodies): open.
- Draft conclusion in the plan is marked for revision after E4 and E5; write the paper's conclusion from the current results with those caveats.
