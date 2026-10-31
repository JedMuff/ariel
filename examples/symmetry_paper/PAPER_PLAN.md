# Why symmetric robots win: module duplication, body shape and the symmetry advantage in body–brain evolution

Oct 31, 2026 · @Jed

## 1. Summary

Revised after E1, E2 (random bodies and controls), E3 and the phase-shift replay. The previous plan (git 8cc276d) argued that symmetry gives a coordinated gait that is cheap for the right controller to learn, and that removing the controller's coordination removes the advantage. The data no longer support that. Most of the advantage is already in the bodies before any learning, it is shared by bodies that duplicate modules without mirroring them, and the left–right phase lock that symmetric bodies show is a marker of good bodies, not the mechanism.

**Setup in one line.** Modular robots, outer EA over bodies, inner CMA-ES over each body's controller. Bodies from a mirror-symmetric genome (`tree_symmetric`) often walk further than bodies from an unconstrained genome (`tree`). A third genome, `tree_tied`, gives each placed module a randomly placed twin: as many duplicated modules as `tree_symmetric`, without mirroring.

**Question.** Where does the symmetry advantage come from: the body, the gait, or the controller's ability to learn it?

**Storyline.** Symmetric bodies start ahead: on fixed random bodies they score higher from the first evaluations, under every controller with a usable search space, and the lead does not grow with training. The lead is a property of the bodies. About half of it is shape (low, sprawled bodies whose floor contacts are spread front to back, with legs that sweep horizontally), and tied bodies, which have a different shape but more legs, get most of it too. Symmetric bodies do move differently: they alternate mirrored limbs and phase-lock, which tree bodies almost never do. But breaking the left–right timing of a trained gait costs no more than breaking any other timing, so the lock is something good symmetric bodies fall into, not why they are good.

**Thesis.** The symmetry advantage is mostly a body advantage, present before learning. Duplicating modules, of which mirroring is one kind, makes bodies with more usable legs and better resting shapes, and these start ahead under a fixed inner budget. Mirroring itself adds little beyond duplication.

**Task.** Forward locomotion (`dsocial_zoo` profile, zoo modules and hinge, 15 s episodes, 20 Hz).

## 2. Paper structure

| results section | question | finding so far | evidence | status |
| --- | --- | --- | --- | --- |
| R1. The advantage and its conditions | When do symmetric bodies win in evolution? | Bang\_bang: across hinge settings (pooled), but noisy per cell. DANN: no gap at 200 evals, grows with budget | Hinge grid, DANN budget series, E1 | done |
| R2. A head start, not faster learning | Do symmetric bodies learn faster or start better? | Ahead from the first 20 evals; sym/tree ratio shrinks with budget | E2 random learning curves | done |
| R3. The advantage is in the bodies | What about the bodies carries it? | Shape-matched tree bodies close ~half the gap; tied bodies close 65–85% with a different shape and more leg hinges | Body measures, E2 controls | done, underpowered (E4) |
| R4. Different behaviour | Do symmetric bodies move differently? | They alternate and phase-lock, tree bodies almost never; the actuator decides which gait | Grid gait labels, E3 on E1 and E2 | done, pending hand-check |
| R5. The gait is a marker, not the mechanism | Does the lock cause the gap? | Unlocked symmetric bodies score like tree; but breaking left–right timing costs no more than breaking any other | E3 along training, phase-shift replay | done |
| R6. Controllers | Does the advantage need a particular controller? | Present under bang\_bang, drift and DANN on fixed bodies; free frequency collapses every body | E1, E2 | done |

The discussion places the result in the morphology–control literature: the advantage of a regular encoding is first a property of the bodies it makes likely, and only second of what a controller can learn on them. Comparisons of encodings should control for the body statistics an encoding changes (here, module duplication and shape) before crediting the regularity itself.

**Questions each section answers on its own:**

- **Ceiling or learning speed?** R2: neither, a head start.
- **Mirroring or duplication?** R3: tied bodies close most of the gap; symmetric − tied is small and not significant.
- **Shape or size?** R3: hinge-matched by construction; shape matching closes about half.
- **Is the gait the mediator?** R5: no. The lock goes with good symmetric bodies, but the trained gaits do not depend on the left–right timing more than on any other.

## 3. Common setup

- **Simulator and task:** `dsocial_zoo`, zoo hinge (kp 1.36, kv 0.356, armature 0.081, joint limit), `forward`. Score is `clip(d / 3.75 m, 0, 2)` over 15 s.
- **Evolution (E0, E1):** outer (μ+λ) = (10+20) for 30 generations, `--repeat-evals`, max 25 modules, inner CMA-ES popsize 20. Seeds 42+rep; runs pair by seed. Score is the last-generation best.
- **Fixed bodies (E2):** hinge-matched random bodies from each genome's initial distribution, trained without the outer loop, 3 training seeds per body; bang\_bang variants to 1,600 evals, DANN to 5,000. Gaps are unpaired across bodies (bootstrap CI, Mann-Whitney), except shape-matched pairs.
- **Brains:** `bang_bang` (one shared frequency) and `dann` (`dann_pos_hinge_h16`). Two bang\_bang variants: `bang_bang_free` (a frequency per hinge) and `bang_bang_drift` (shared frequency plus small per-hinge offsets; the control for free's larger search).
- **Body measures** (`analyse_bodies.py`, after the 3 s settle): hinge and module counts; height, length, width; leg hinges (hinges whose distal part touches the floor); front–back and lateral spread of floor contacts; hinge axis classes (pitch, roll, yaw, diagonal).
- **Behaviour measures** (`replay.py`, `gait.py`): gait class, left–right phase relation, phase quality, phase-locked share.

## 4. Results so far

### R1. The advantage and its conditions

- **Hinge grid** (18 cells × 3 seeds): bang\_bang sym − tree +0.14 pooled (17/18 cells, p < 0.001); DANN +0.02 (p = 0.31).
- **E1 at the zoo hinge** (10 seeds): bang\_bang +0.08 (p = 0.70); seeds 42–44 reproduce the grid cell exactly, so the cell's +0.17 was three favourable seeds. Drift +0.15 (p = 0.08). Per-cell gaps need more than 10 seeds; the pooled grid carries R1.
- **DANN budget series** (n = 5): −0.02 / +0.22 / +0.37 at 200 / 800 / 1600 evals; symmetric wins 5/5 at 1600.
- **Tied in evolution (E1):** tied − tree +0.13 (bang\_bang, p = 0.06), +0.14 (drift, p = 0.01); symmetric − tied ≈ 0.

### R2. A head start, not faster learning

E2, 40 + 40 hinge-matched random bodies × 3 seeds:

| brain | sym − tree final | sym / tree at 200 / 800 / 1600 evals |
| --- | --- | --- |
| bang\_bang | +0.15 | 1.55 / 1.45 / 1.41 |
| drift | +0.12 | 1.54 / 1.43 / 1.40 |
| DANN | +0.13 (5,000 evals) | 1.84 / 1.50 / 1.39 |
| free | +0.02 | both collapse (0.14–0.16) |

Symmetric bodies lead from 20 evaluations and the ratio shrinks with budget. The evolution-level DANN puzzle (no gap at 200 evals) is therefore not a property of the bodies: on fixed bodies DANN shows the gap at 200 evals too (+0.07). It must come from the outer loop under a small, noisy inner budget (see section 7).

### R3. The advantage is in the bodies

Random symmetric bodies differ from tree bodies at the same hinge count: flatter (height 0.14 vs 0.22), floor contacts spread further front to back (0.16 vs 0.13), more leg hinges with a vertical axis (2.3 vs 1.5), fewer with a lateral axis (0.45 vs 1.6), fewer modules (18.6 vs 26; tree bodies always reach the cap). Leg hinges and contact spread predict score in both genomes (ρ ≈ 0.6).

E2 controls (40 bodies per set, bang\_bang and DANN × 3 seeds):

| brain | tree | tree, shape-matched | tied | symmetric |
| --- | --- | --- | --- | --- |
| bang\_bang | 0.363 | 0.434 | 0.459 | 0.510 |
| DANN | 0.396 | 0.447 | 0.507 | 0.529 |

- **Shape:** tree bodies matched to each symmetric body on hinge count and six shape measures close 40–50% of the gap; the residual (+0.075 / +0.082) is not significant, and adjusting for the remaining shape mismatch leaves it unchanged.
- **Duplication:** tied bodies close 65% (bang\_bang) to 85% (DANN) of the gap with a tree-like shape. They have more leg hinges (4.0 vs 3.45), which accounts for about a third of their lead over tree. Symmetric − tied: +0.05 / +0.02, not significant.
- **Not hinge orientation:** the symmetric genome forces midline hinges to DEG\_0, so it makes fewer diagonal hinges, but the diagonal share does not predict score among tree bodies (ρ = 0.03).
- **Power:** with 40 bodies per set even the original gap is borderline (bootstrap CI excludes 0 for bang\_bang; Mann-Whitney p = 0.12). E4 doubles the sets.

### R4. Different behaviour

- **Symmetric bodies alternate and phase-lock; tree bodies almost never do.** Grid: alternating gait in 40% of bang\_bang sym episodes vs 4% tree. E1 champions: 8/10 symmetric phase-locked vs 0/10 tree, 1/10 tied. On random bodies symmetric controllers alternate from the first 50 evals (35%).
- **Tied bodies reach symmetric scores with mixed or unpaired gaits**, so alternation is not the only route to a high score.
- **The actuator decides which gait appears:** rowing almost only at kv 1, a fast small-stroke shuffle at stiff, lightly damped hinges (pending videos).

### R5. The gait is a marker, not the mechanism

- **Correlation:** at the end of E2 training, unlocked symmetric bodies score like tree bodies (bang\_bang 0.40 vs 0.39; DANN 0.49 vs 0.42), and locked ones carry the gap (0.65, 0.57).
- **Intervention (phase-shift replay):** shifting or detuning the left side of a trained gait costs locked symmetric bodies no more than perturbing the same number of random lateral hinges, at every size (side − random within ±0.05, n.s., n = 53 locked bang\_bang bodies). Detuning by 15% unlocks the sides (locked share 0.92 → 0.23) yet keeps as much score as the random control. Every group loses the same share of its score to any timing change.
- **Locking is a body property:** most symmetric bodies lock in all three training seeds or in none, but no resting-shape measure predicts which (all p > 0.1), and a body's locked share barely tracks its score (ρ = +0.26 bang\_bang, −0.02 DANN).

### R6. Controllers

- The head start appears under bang\_bang, drift and DANN on fixed bodies.
- Free frequency removes the gap only by collapsing every body (scores 0.14–0.16 against 0.31–0.51); per-hinge frequencies stay spread out (CV 0.10–0.19). Drift, which keeps the search tractable, keeps the gap. Free frequency is reported as a negative result: it confounds coordination with search difficulty.

## 5. Remaining experiments

### E4. Power up the body controls (R3)

Doubles the random sets, so the shape and duplication results are not resting on 40 bodies.

- **Bodies:** a second random draw (`sample_bodies.py random --seed 1`), then tied and shape-matched sets against its symmetric bodies: 40 more per set, 160 bodies.
- **Training:** bang\_bang and DANN × 3 seeds, as the E2 controls: 960 trainings, **~355 CPU-h**.
- **Readout:** `analyse_matched.py` on both draws pooled: gap closed by shape matching, symmetric − tied, with CIs half as wide.

### E5. Which part of duplication? (R3)

Tied bodies have more leg hinges; does leg count explain their lead?

- **Bodies:** tree bodies matched to the tied bodies on hinge count and leg hinges (the `shape_matched` machinery with a different measure list), 40 bodies.
- **Training:** bang\_bang and DANN × 3 seeds: 240 trainings, **~90 CPU-h**.
- **Readout:** tied − leg-matched tree. Near zero: duplication works by adding usable legs. Still large: duplication helps in other ways (e.g. repeated limb geometry), which the body measures should then look for.

### E2b. Evolved bodies (R3, optional)

- **Bodies:** best final body of every E1 bang\_bang and drift run, tree, symmetric and tied (60).
- **Training:** bang\_bang and DANN × 3 seeds: 360 trainings, **~130 CPU-h**.
- **Readout:** is the gap among evolved bodies larger than among random ones (evolution finding something beyond the starting statistics), and do evolved tied and symmetric bodies converge on the same shapes?

### Analysis only

- **Detector hand-check** (R4): ~20 zoo-module bang\_bang and DANN episodes per gait label; watch the `static` videos.
- **R2 figure:** E2 curves for all four body sets (tree, symmetric, tied, shape-matched).
- **DANN evolution puzzle** (R1): compare outer-loop selection noise under `--repeat-evals` at 200 vs 1,600 evals per body, from the existing runs' repeated evaluations.

## 6. Cost and order

| experiment | trainings | compute | new code |
| --- | --- | --- | --- |
| E4 power-up | 960 | ~355 CPU-h | none |
| E5 leg-matched tree | 240 | ~90 CPU-h | `--measures` option for `sample_bodies.py shape_matched` (small) |
| E2b evolved bodies | 360 | ~130 CPU-h | none |
| **total** | **1,560** | **~575 CPU-h** | |

Costs from the E2 controls' wall times (bang\_bang 29 s, DANN 104 s per training at 20 CPUs).

**Order:** E4 and E5 together (both decide R3, the core of the paper); E2b if R3 holds; the detector hand-check alongside.

## 7. Planned figures

1. **R1 the advantage in evolution:** hinge-grid heatmap of sym − tree, bang\_bang vs DANN; DANN gap vs inner budget.
2. **R2 head start:** E2 learning curves, tree vs symmetric (and tied, shape-matched), one panel per brain; inset sym/tree ratio vs evals.
3. **R3 bodies:** shape measures by genome (height, contact spread, leg hinges, axis classes), with example bodies; final score per body set (`fig9_matched.png`), the main figure.
4. **R4 behaviour:** gait-class shares by genome × brain, with example phase-lag traces; gait vs actuator across the grid.
5. **R5 gait as marker:** score of locked vs unlocked symmetric bodies against tree; phase-shift replay, side vs random by perturbation size (`fig8_phase_shift.png`).
6. **R6 controllers:** E1 paired gaps under bang\_bang, drift and free frequency, with absolute scores to show free frequency's collapse.

## 8. Findings at a glance

| section | finding | evidence | confidence |
| --- | --- | --- | --- |
| R1 | Sym > tree under bang\_bang across hinge settings | grid pooled 17/18, p < 0.001; single cell n.s. at n = 10 | **moderate–high** |
| R1 | DANN gap absent at 200 evals in evolution, grows with budget | −0.02 / +0.22 / +0.37, n = 5 | **moderate** |
| R2 | Symmetric bodies start ahead; no faster learning | E2 ratio 1.55 → 1.41 | **high** |
| R3 | Shape accounts for about half the gap | shape matching closes 40–50% | **moderate** (n = 40) |
| R3 | Duplication without mirroring gets most of the advantage | tied closes 65–85%; sym − tied n.s.; E1 tied ≈ sym | **moderate** (n = 40) |
| R3 | Extra leg hinges explain part of duplication's effect | a third of tied − tree | **uncertain** (E5) |
| R4 | Symmetric bodies alternate and lock; tree bodies do not | grid, E1, E2 | **moderate–high** (pending hand-check) |
| R5 | Lock is a marker, not the mechanism | phase-shift replay: side ≈ random | **moderate–high** |
| R6 | Advantage needs a tractable controller search, not shared frequency | drift keeps it, free collapses all | **moderate** |

## 9. Risks, scope and draft conclusion

### Risks

- **E4 shrinks the controls' effects** (shape or tied no longer close a clear share of the gap): R3 becomes "symmetric bodies start ahead, partly through shape" with duplication as a suggestive result.
- **Symmetric − tied grows with more bodies:** mirroring adds something beyond duplication; then the gait results (R4) are the candidates for what, and the phase-shift result limits it to something other than left–right timing.
- **Detector does not hold on zoo modules:** report `mirror_sync` and phase-lag distributions directly.
- **The DANN evolution puzzle stays open:** report it as a limit; the fixed-body results do not depend on it.

### Out of scope (future work)

- **Mirror-tied DANN** (weights shared across mirrored pairs): no longer central, since the left–right timing is not the mechanism.
- **Symmetry dose–response** and other regularities (repetition, segmentation, radial symmetry): the natural follow-up, now framed as "which body statistics does an encoding change?".
- **Other tasks:** the DANN sweep found a symmetric advantage on multidirection at 200 evals; a separate question.

### Draft conclusion

*Written for the current results; revise after E4 and E5.*

> Symmetric robots win because of the bodies they make likely, not because of what their controllers learn. On fixed random bodies, symmetric ones score higher from the first evaluations, under every controller that can search effectively, and the lead does not grow with training. About half of it comes from shape: mirroring tends to make low, sprawled bodies with floor contacts spread front to back and legs that sweep horizontally. Most of it is shared by bodies that duplicate modules without mirroring them, which reach similar scores with a different shape and more legs.
>
> Symmetric bodies do move differently. They alternate mirrored limbs and lock the two sides in phase, which unconstrained bodies almost never do, and the hinge actuator decides which coordinated gait appears. But the lock is a symptom: perturbing the left–right timing of a trained gait costs no more than perturbing any other timing, so good symmetric bodies fall into it rather than depend on it.
>
> The symmetry advantage is therefore largely a duplication-and-shape advantage present before learning. Comparisons of morphological encodings should account for the body statistics an encoding changes, such as duplicated modules and resting shape, before crediting the regularity itself.
