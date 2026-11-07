# When does tree_symmetric beat tree on forward locomotion?

*Compiled 2026-10-30. The DANN budget series was still running when this was written. 6 of its 10 runs at
skill budget 80 and 2 of its 10 runs at budget 40 are missing. Update the budget section when they finish.*

## Summary

The old control-stride runs found that tree_symmetric beats tree on `forward`. The DANN symmetry sweep found no
difference. The obvious suspect was the hinge actuator: the old runs used a soft, slow servo, and the sweep used
the zoo hinge. The experiments below test that and the other differences between the two setups. The answer is
**how well trained the controller is, not the actuator or the simulator.**

- **Actuator: not the cause.**
  - **Bang_bang:** symmetric wins in 17 of 18 hinge cells (54 paired runs, Wilcoxon p < 0.001).
  - **DANN:** the gap is zero in every cell (pooled +0.02, p = 0.31).
  - **Hinge settings:** no setting creates or removes the effect. They only change its size.
- **Simulator: not the cause.** On the old simulator with the old budget, swapping only the brain flips the
  result.
  - **DANN:** tree wins, 1.75 m vs 0.55 m.
  - **Bang_bang:** symmetric wins, 3.11 m vs 2.49 m, as the old ANN did.
- **Inner budget: probably the cause.** Under the sweep settings, the DANN's symmetric gap grows with training
  per body.

  | inner training per body | 200 evals | 800 evals | 1600 evals |
  |---|---|---|---|
  | symmetric − tree (paired by rep) | −0.03 | +0.21 | +0.50 |
  | reps where symmetric is ahead | 3/5 | 2/4 | 2/2 |

  1600 evals is the old ANN runs' budget.
- **Explanation:** symmetric bodies have the higher ceiling, but only a controller trained enough to coordinate
  the mirrored limbs can use it.
  - **Bang_bang** shows the advantage at only 200 evals because it is a cheap search: one shared frequency plus
    a phase and duty cycle per hinge.
  - **The DANN** needs far more evaluations to reach that level of coordination.
- **Rowing is not needed.** In the ANN rowing-penalty experiment, symmetric still won (8.6 m vs ~6.9 m) after
  the penalty had removed all rowing.
- **Symmetry vs dimensionality:** the tied-genome control is inconclusive. tree_tied falls between the two, and
  differs significantly from neither.

## Units

Two scoring schemes are used.

| profile | used by | score |
|---|---|---|
| `sweep` (old simulator) | control-stride, rowing-penalty, tied-control runs; the local old-simulator runs | metres travelled in 30 s |
| `dsocial_zoo` | DANN sweep, actuator grid, budget series | normalised `clip(d / 3.75 m, 0, 2)` over 15 s, so 2 = 0.5 m/s |

- Each run's score is the best individual in the last generation, as minus the fitness, so penalties are
  included. Old-sim tables also give the best score ever reached.
- With `--repeat-evals`, every surviving body is retrained and rescored each generation. A best-ever score can
  therefore be one lucky training run, so the last-generation best is the steadier number.

## The experiments

### 1. Old ANN control-stride runs: the original result

These are in `__data__/ariel_control_stride_sweep/ctrlstride_forward_*_s20_*`.
- **Setup:** ANN brain, `sweep` profile, local modules, local hinge (kp 1, kv 1, no armature, no joint limit),
  20 Hz, 30 s episodes.
- **Training:** inner CMA-ES of 100 gens × 16 = 1600 evals per body, with the hinge-glitch guard. Outer EA
  μ10 λ20 over 30 generations.

Best ever, in metres:

| rep (seed) | tree | tree_symmetric |
|---|---|---|
| 0 (42) | 4.7 | 5.0 |
| 1 (43) | 1.6 | 4.7 |
| 2 (44) | 2.8 | 5.4 |

- Symmetric leads in all 3 reps. In reps 1 and 2 it leads from generation 3–5 onwards.
- Rowing appears in 3 of 3 symmetric reps and 1 of 3 tree reps (`ROWING.md`).
- Current code with `--hinge-glitch` reproduces these checkpoints' stored fitness exactly (24/24).

### 2. ANN with zoo modules: rowing penalty and tied control

These are in `__data__/ariel_rowing_penalty_ann`, `__data__/ariel_tied_control_ann`.
- **Setup:** ANN, `sweep` profile with `--zoo-modules` (zoo sizes, masses and hinge), 20 Hz, 1600 evals per
  body.
- **Rowing penalty** (3 reps per arm):

  | arm | tree_symmetric | tree |
  |---|---|---|
  | no penalty | 10.7 m | 6.9 m |
  | penalty | 8.6 m | ~6.9 m |

  - Without the penalty, little rowing appears at this stride: 7% of symmetric and 2% of tree individuals.
  - The penalty removes rowing entirely. Symmetric loses about 20% but still wins.
- **Tied control** (6 reps per genome; final best, median):

  | genome | median final best | Mann–Whitney vs tree_symmetric | vs tree |
  |---|---|---|---|
  | tree_symmetric | 9.2 m | – | p = 0.041 |
  | tree_tied | 7.9 m | p = 0.31 | p = 0.39 |
  | tree | 6.6 m | p = 0.041 | – |

  tree_tied gives each placed module a randomly placed twin and matches symmetric's number of free genes.
- **What it shows:** the zoo hinge does not remove the symmetric advantage once the brain is well trained. This
  was the first hint that the actuator is not the cause.

### 3. DANN symmetry sweep: where it disappeared

This is in `__data__/ariel_symmetry_pressure_sweep_dann`.
- **Setup:** DANN (`dann_pos_hinge_h16`), `dsocial_zoo`, zoo hinge, 20 Hz, 15 s episodes.
- **Training:** inner CMA-ES of 10 gens × 20 = 200 evals per body. 5 reps per task and genome.
- **Results:**

  | task | tree_symmetric | tree | |
  |---|---|---|---|
  | forward | 0.61 | 0.63 | paired difference −0.03, 3/5 reps positive |
  | multidirection | 0.61 | 0.42 | p = 0.008 |
  | turn_avg | | | no difference |

### 4. Hinge-actuator grid

This is in `__data__/ariel_actuator_grid`, with figures and tables in `analysis/` (`actuator_grid_delta.png`,
`actuator_grid_ranked.csv`).
- **Setup:** DANN sweep settings (`dsocial_zoo`, 200 evals per body, forward), with the hinge parameters
  varied.
- **Grid:** kp {1, 1.36 (zoo), 2.5} × kv {0.356 (zoo), 1, 2} × armature {0, 0.081 (zoo)}, joint limit always on.
  That is 18 cells, crossed with 2 brains, 2 genomes and 3 seeds (42–44, shared by every cell, so runs pair by
  seed): 216 runs, all complete.

| brain | mean tree | mean sym | paired difference | pairs positive | Wilcoxon p | cells with difference > 0 |
|---|---|---|---|---|---|---|
| dann | 0.457 | 0.474 | +0.016 | 31/54 | 0.31 | 12/18 |
| bang_bang | 0.694 | 0.832 | +0.138 | 39/54 | < 0.001 | 17/18 |

- **Correction:** this replaces the partial read of 2026-10-29, at 110 runs, which suggested no advantage for
  either brain. The bang_bang effect appeared only once all three seeds were in.
- **How each hinge parameter changes the difference** (cell means averaged over the other two parameters):

  | parameter | value | DANN | bang_bang |
  |---|---|---|---|
  | kp | 1 | +0.04 | +0.07 |
  | | 1.36 | +0.02 | +0.17 |
  | | 2.5 | −0.01 | +0.17 |
  | kv | 0.356 | +0.02 | +0.20 |
  | | 1 | +0.01 | +0.12 |
  | | 2 | +0.03 | +0.10 |
  | armature | 0 | +0.02 | +0.16 |
  | | 0.081 | +0.01 | +0.12 |

  - **Stiffer, less damped hinges** (high kp, low kv) let both genomes move further. For bang_bang, the mean
    score rises from 0.22 at (kp 1, kv 2) to 1.68 at (kp 2.5, kv 0.356).
  - **For bang_bang the gap grows in step with the overall score.** That is consistent with an effect that
    scales with how well the body can move, rather than one tied to a particular actuator.
  - **For the DANN** the gap stays at zero in every column.
- **Anchor cells:**

  | cell | DANN | bang_bang |
  |---|---|---|
  | zoo hinge | +0.10 (−0.43, +0.51, +0.23) | +0.17 (+0.30, +0.51, −0.32) |
  | old local hinge, but with the joint limit | +0.01 | +0.13 |

- **Ceiling:** 3 of 216 runs hit the score cap of 2, all bang_bang at kp 2.5, kv 0.356. The cap slightly
  compresses the gap in those cells.
- **Top cells by difference** are listed in `actuator_grid_ranked.csv`. With 3 seeds per cell, no single cell
  stands out from the others: only one has a confidence interval that excludes zero (bang_bang zoo kp/kv with
  armature 0: +0.45 [+0.09, +0.83]), and with 36 cells that is about what chance gives.

### 5. Old simulator with the brain swapped (local, seed 42)

These are in `__data__/ariel_dann_oldsim`, `__data__/ariel_bangbang_oldsim`.
- **Setup:** the command line of experiment 1 (`--eval-profile sweep`, local hinge, stride 20, 30 s,
  `--hinge-glitch`), except `--brain dann` or `--brain bang_bang` with `--skill-budget 10` (160 evals per body).
- **Results:** best ever (best per generation late in the run), in metres:

  | brain | tree | tree_symmetric |
  |---|---|---|
  | DANN | 1.75 (0.8–1.5) | 0.55 (~0.5, flat from generation 5) |
  | bang_bang | 2.49 (2.1–2.5) | 3.11 (2.2–3.1, symmetric leads from generation 3) |
  | old ANN, 1600 evals (from experiment 1) | 4.7 | 5.0 |

- **Incomplete runs:** in each pair, one run's main process died with no error. The DANN tree run reached
  generation 28 and the bang_bang symmetric run generation 25, and both are missing their last generations. No
  episode in any run triggered the glitch guard.
- **What it shows:** with the simulator held fixed, a cheap open-loop brain shows the old symmetric advantage.
  The DANN, at 1/10 of the budget, shows the reverse. Symmetric's DANN controllers stalled at about 0.5 m.

### 6. DANN inner-budget series (partial)

This is in `__data__/ariel_dann_budget/sb{40,80}`. The 200-eval point is the DANN sweep's forward runs.
- **Setup:** the DANN sweep's forward settings with `--skill-budget` 40 (800 evals) or 80 (1600 evals). Seeds
  42+rep.

| inner budget | tree, last-gen best per rep | tree_symmetric, last-gen best per rep | mean tree | mean sym | paired sym − tree |
|---|---|---|---|---|---|
| 200 evals (sweep) | 0.88 0.41 0.56 0.81 0.52 | 0.42 0.49 0.79 0.44 0.90 | 0.63 | 0.61 | −0.03 (3/5 positive) |
| 800 evals | 1.43 0.91 1.17 1.35 | 1.36 1.44 1.12 1.76 | 1.22 | 1.42 | +0.21 (2/4) |
| 1600 evals | 1.45 1.34 | 1.86 1.93 | 1.39 | 1.90 | +0.50 (2/2) |

- **Trend:** both genomes improve with budget, and symmetric improves faster. At 1600 evals, the old ANN's
  budget, symmetric wins both completed reps by about 0.4–0.6. That is the size of the old ANN gap.
- **Ceiling:** the symmetric runs at 1600 evals sit close to the cap of 2 (1.86, 1.93). If the remaining reps
  also approach it, the cap will understate the gap.
- **Still to come:** budget 40 rep 4, and budget 80 reps 2–4.

## Synthesis

| setup | brain | evals per body | symmetric advantage on forward? |
|---|---|---|---|
| old sim, local hinge | ANN | 1600 | yes, 3/3 reps |
| old sim, zoo hinge | ANN | 1600 | yes (rowing penalty: higher mean in both arms; tied control p = 0.04) |
| old sim, local hinge | bang_bang | 160 | yes (1 seed) |
| old sim, local hinge | DANN | 160 | no, reversed (1 seed) |
| `dsocial_zoo`, 18 hinge cells | bang_bang | 200 | yes, 17/18 cells, p < 0.001 |
| `dsocial_zoo`, 18 hinge cells | DANN | 200 | no, in any cell |
| `dsocial_zoo`, zoo hinge | DANN | 800 / 1600 | growing: +0.21, then +0.50 (partial) |

**What each row rules out or supports:**
- **The actuator:** it changes the size of the gap and how far bodies move, but never creates or removes the
  advantage.
- **The old simulator itself:** episode length, module masses and the glitch guard are not needed. Bang_bang
  shows the advantage under `dsocial_zoo`, and the DANN fails to show it under `sweep`.
- **Rowing:** symmetric still wins once rowing is penalised away.
- **Controller quality: what's left.**
  - In every setting where the controller is well optimised, symmetric wins: the ANN at 1600 evals, bang_bang
    (few parameters, cheap search), and the DANN at 1600 evals.
  - Where the controller is under-trained, it doesn't: the DANN at 160–200 evals.
  - In the zoo benchmark the DANN was at ~0.08 m/s after 200 evals and ~0.21 m/s after 1600. So the DANN sweep
    compared bodies early in learning, before their ceilings were reached.

**A plausible mechanism, not yet tested:** a mirrored body moves well only when its two sides are driven in a
coordinated pattern, whether in phase (rowing) or alternating. An asymmetric body has no such pattern to find,
so a weak controller loses less on it.
- **Bang_bang** has one shared frequency, so its hinges are phase-locked to each other by construction.
- **The DANN** must learn the coordination through its shared MLP and neighbour inputs, which takes many
  evaluations.

The multidirection result in the DANN sweep (symmetric wins at 200 evals) is a partial exception. There, the
benefit may come from the body's mirror plane making left/right steering easier, rather than from forward
gait quality.

**Symmetry or just fewer genes?** Still open. tree_tied falls between the two genomes with ANN controllers,
and differs significantly from neither. Six reps are too few to separate "mirroring helps" from "halving the
body search space helps".

## Caveats

- **Few seeds:** most comparisons rest on 2–6 seeds per arm, and the old-simulator brain swaps on a single seed
  (42). Seed 42 was also the rep where the old ANN runs showed the smallest gap (5.0 vs 4.7 m).
- **Budget series incomplete:** its strongest point (1600 evals) has 2 reps.
- **Noisy scores:** last-generation scores under `--repeat-evals` are noisy. A single retraining of a body can
  move its score by ±0.3 normalised.
- **Two scales:** `sweep` and `dsocial_zoo` scores are not on the same scale. Compare the direction of the gap
  across experiments, not its size.
- **Old-simulator runs incomplete:** in each local pair, one run's main process died with no error (cause
  unknown), so it is missing its last 2–5 generations.

## Next steps

1. **Finish the budget series** (6 runs left at 1600 evals, 2 at 800). If the gap stays near +0.5 at 1600 evals
   over 5 reps, this is the main result. Then rerun `plot_task_genome_summary.py` with
   `ARIEL_SWEEP_ROOT=__data__/ariel_dann_budget/sb80`.
2. **Learning curves:** at matched budgets, compare per-body inner learning curves
   (`per_skill.fwd.learning_curve` in `run_data.jsonl`). Do symmetric bodies start lower and climb faster? That
   would test the "needs a trained controller" explanation directly.
3. **Coordination check:** test the mechanism without more evolution. Take each genome's evolved best bodies
   and retrain their DANN at 200 vs 1600 evals, then compare the left–right phase relations. `rowing_detect.py`
   gives `mirror_sync` and `phase_lag` per pair.
4. **Skip further actuator-grid confirmation runs:** the grid already answers the actuator question.
5. **Tied control with bang_bang:** run it under `dsocial_zoo` to separate symmetry from dimensionality
   cheaply. Bang_bang shows the effect at 200 evals, about 10× cheaper per body than the ANN runs.
