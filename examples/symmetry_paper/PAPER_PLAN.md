# Why symmetric robots win: module duplication, body shape and the symmetry advantage in body–brain evolution

Oct 31, 2026 · @Jed · revised for the v2 protocol after the first draft review

## 1. Summary

Revised after E1, E2 (random bodies and controls), E3 and the phase-shift replay. The previous plan (git 8cc276d) argued that symmetry gives a coordinated gait that is cheap for the right controller to learn, and that removing the controller's coordination removes the advantage. The data no longer support that. Most of the advantage is already in the bodies before any learning, it is shared by bodies that duplicate modules without mirroring them, and the left–right phase lock that symmetric bodies show is a marker of good bodies, not the mechanism.

**Setup in one line.** Modular robots, outer EA over bodies, inner CMA-ES over each body's controller. Bodies from a mirror-symmetric genome (`tree_symmetric`) often walk further than bodies from an unconstrained genome (`tree`). A third genome, `tree_tied`, gives each placed module a randomly placed twin: as many duplicated modules as `tree_symmetric`, without mirroring.

**Question.** Where does the symmetry advantage come from: the body, the gait, or the controller's ability to learn it?

**Storyline.** Symmetric bodies start ahead: on fixed random bodies they score higher from the first evaluations, under every controller with a usable search space, and the lead does not grow with training. The lead is a property of the bodies. About half of it is shape (low, sprawled bodies whose floor contacts are spread front to back, with legs that sweep horizontally), and tied bodies, which have a different shape but more legs, get most of it too. Symmetric bodies do move differently: they alternate mirrored limbs and phase-lock, which tree bodies almost never do. But breaking the left–right timing of a trained gait costs no more than breaking any other timing, so the lock is something good symmetric bodies fall into, not why they are good.

**Thesis.** The symmetry advantage is mostly a body advantage, present before learning. Duplicating modules, of which mirroring is one kind, makes bodies with more usable legs and better resting shapes, and these start ahead under a fixed inner budget. Mirroring itself adds little beyond duplication.

**Task.** Forward locomotion (default modules and hinge, 15 s episodes, 20 Hz control). The paper never says "zoo": the hinge is "the default hinge", fitted to the physical servo.

**v2 protocol (after the draft review, 2026-10-31).** Fitness = forward displacement in metres, unclipped, no penalties. Outer EA (20+20), plus-selection, no retraining of parents. No retests (deterministic sim; decided 2026-10-31). E0 is DANN only, with the hinge grid at the largest budget. All first-round numbers below are in the old normalised units and will be replaced.

**Headline findings to carry:** symmetric bodies start ahead before learning; the relative lead shrinks with controller learning (ratio to tree 1.55 → 1.41) without closing; all gaits depend on timing, but not specially on the left–right lock.

## 2. Paper structure

| results section | question | finding so far | evidence | status |
| --- | --- | --- | --- | --- |
| R1. The advantage and its conditions | When do symmetric bodies win in evolution? | DANN: no gap at 200 evals, grows with budget; v2 grid at 1,600 evals tests the hinge dependence. E1 curves: genome and controller comparison | DANN budget series and grid, E1 curves | v2 pending |
| R2. A head start; the relative lead shrinks with learning | Do symmetric bodies learn faster or start better? | Ahead from the first 20 evals; sym/tree ratio shrinks with budget (absolute gap levels off) | E2 random learning curves | v2 pending |
| R3. The advantage is in the bodies | What about the bodies carries it? | Shape-matched tree bodies close ~half the gap; tied bodies close 65–85% with a different shape and more leg hinges | Body measures, E2 controls | done, underpowered (E4) |
| R4. Different behaviour | Do symmetric bodies move differently? | They alternate and phase-lock, tree bodies almost never; the actuator decides which gait | Grid gait labels, E3 on E1 and E2 | done, pending hand-check |
| R5. Gaits depend on timing, not specially on the lock | Does the lock cause the gap? | Unlocked symmetric bodies score like tree; all gaits lose 20–35% to a 0.05-cycle shift; left–right perturbation costs no more than random (body level); relative to baseline, locked symmetric lose the same or less than tree | E3 along training, phase-shift replay | v2 pending |
| R6. Controllers | Does the advantage need a particular controller? | Present under bang\_bang, drift and DANN on fixed bodies; free frequency collapses every body | E1, E2 | done |

The discussion places the result in the morphology–control literature: the advantage of a regular encoding is first a property of the bodies it makes likely, and only second of what a controller can learn on them. Comparisons of encodings should control for the body statistics an encoding changes (here, module duplication and shape) before crediting the regularity itself.

**Questions each section answers on its own:**

- **Ceiling or learning speed?** R2: neither, a head start.
- **Mirroring or duplication?** R3: tied bodies close most of the gap; symmetric − tied is small and not significant.
- **Shape or size?** R3: hinge-matched by construction; shape matching closes about half.
- **Is the gait the mediator?** R5: no. The lock goes with good symmetric bodies, but the trained gaits do not depend on the left–right timing more than on any other.

## 3. Common setup

- **Simulator and task:** default modules, default hinge (kp 1.36, kv 0.356, armature 0.081, joint limit; fitted to the physical servo), `forward`, 15 s, physics at 500 Hz (2 ms steps), control at 20 Hz.
- **Fitness (v2):** forward displacement in m, unclipped, no penalties (`--fitness displacement`). v1: `clip(d / 3.75 m, 0, 2)` minus height and jerk penalties.
- **Evolution (E0, E1, v2):** outer (μ+λ) = (20+20), plus-selection, tournament parents, 30 generations, no retraining; inner CMA-ES popsize 20; max 25 modules. Fitness = best CMA-ES fitness (no retests). Seeds 42 + rep; runs pair by seed. Score = last-generation best.
- **Fixed bodies (E2):** hinge-matched random bodies, 80 per genome (one draw, `--n 80 --pool 800`), trained without the outer loop, 3 seeds per body; bang\_bang family 1,600 evals, DANN 5,000. Gaps unpaired across bodies (bootstrap CI, Mann-Whitney), except matched pairs.
- **Brains:** `bang_bang` (one shared frequency) and `dann`; variants `bang_bang_free` and `bang_bang_drift`. In the paper: "bang-bang", "free-frequency bang-bang", "drift bang-bang", "DANN" (no underscores anywhere in text or figures).
- **Body measures** (`analyse_bodies.py`): as before.
- **Behaviour measures** (`replay.py`, `gait.py`): gait class, left–right phase relation, phase quality, locked share (locked = locked pairs carry ≥ 50% of the stroke).

## 4. Results so far (first round, v1 units)

### R1. The advantage and its conditions

- **E1 curves (v1):** over generations the bang\_bang sym − tree lead grows (+0.06 → +0.09), drift +0.04 → +0.18, tied − tree −0.10 → +0.20; DANN at 200 evals (n = 5) +0.13 → −0.15. So "the lead shrinks" is a learning (inner loop) result, not an evolution one; recheck with v2.

- **Hinge grid** (18 cells × 3 seeds): bang\_bang sym − tree +0.14 pooled (17/18 cells, p < 0.001); DANN +0.02 (p = 0.31).
- **E1 at the default hinge** (10 seeds): bang\_bang +0.08 (p = 0.70); seeds 42–44 reproduce the grid cell exactly, so the cell's +0.17 was three favourable seeds. Drift +0.15 (p = 0.08). Per-cell gaps need more than 10 seeds; the pooled grid carries R1.
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

- **Body-level and relative (added 2026-10-31):** side − random per body within −0.01…+0.06 for locked sym under bang\_bang (all p ≥ 0.43, 17 bodies). Every group loses ~20–35% to a 0.05-cycle shift and 70–90% to half a cycle. Locked sym lose more absolute score than tree but the same or less relative to baseline (bang\_bang 0.05 shift: 18–23% vs 30–31%).

- **Correlation:** at the end of E2 training, unlocked symmetric bodies score like tree bodies (bang\_bang 0.40 vs 0.39; DANN 0.49 vs 0.42), and locked ones carry the gap (0.65, 0.57).
- **Intervention (phase-shift replay):** shifting or detuning the left side of a trained gait costs locked symmetric bodies no more than perturbing the same number of random lateral hinges, at every size (side − random within ±0.05, n.s., n = 53 locked bang\_bang bodies). Detuning by 15% unlocks the sides (locked share 0.92 → 0.23) yet keeps as much score as the random control. Every group loses the same share of its score to any timing change.
- **Locking is a body property:** most symmetric bodies lock in all three training seeds or in none, but no resting-shape measure predicts which (all p > 0.1), and a body's locked share barely tracks its score (ρ = +0.26 bang\_bang, −0.02 DANN).

### R6. Controllers

- The head start appears under bang\_bang, drift and DANN on fixed bodies.
- Free frequency removes the gap only by collapsing every body (scores 0.14–0.16 against 0.31–0.51); per-hinge frequencies stay spread out (CV 0.10–0.19). Drift, which keeps the search tractable, keeps the gap. Free frequency is reported as a negative result: it confounds coordination with search difficulty.

## 5. v2 experiments (all to run)

Everything is rerun under the v2 protocol into `FINAL_ROOT=/scratch/$USER/ariel_paper_v2` (the `_common.sh` default). E4 (more bodies) is folded into E2.

| experiment | submit | design | runs | ≈ CPU-h |
| --- | --- | --- | --- | --- |
| E0 budget | `sbatch slurm/symmetry_paper/e0_dann_budget.sh` | DANN, tree / sym, 200 / 800 / 1,600 evals, 10 seeds | 60 | 3,800 |
| E0 grid | `sbatch slurm/symmetry_paper/e0_dann_grid.sh` | DANN at 1,600 evals, kp {1, 1.36, 2.5} × kv {0.356, 1, 2}, default armature, 5 seeds; default cell = budget sb80 seeds 42–46 | 80 | 9,300 |
| E1 | `sbatch slurm/symmetry_paper/e1_coordination_ablation.sh` | bang\_bang, drift, free × tree / sym / tied, 200 evals, 20 seeds; DANN arm = budget sb10 | 180 | 2,200 |
| E2 random | `sample_bodies.py random --n 80 --pool 800`, then `BODY_SET=random sbatch --array=0-639 …e2_fixed_body_curves.sh` | 80 + 80 bodies × 4 brains × 3 seeds | 640 tasks | 650 |
| E2 controls | `sample_bodies.py tied` / `shape_matched --pool 24000`, then `BODY_SET=… BRAINS="bang_bang dann" sbatch --array=0-159 …` | 80 tied, 80 shape-matched × 2 brains × 3 seeds | 320 tasks | 360 |
| E5 | `sample_bodies.py shape_matched --like bodies_tied.jsonl --target-genome tree_tied --measures n_leg --body-source leg_matched --id-prefix legmatch_tree`, then as above with `BODY_SET=leg_matched` | 80 tree bodies matched to tied on hinges + leg hinges | 160 tasks | 180 |
| E3 + phase shift | as before on the v2 roots | replays | — | small |

Total ≈ 16.5k CPU-h, mostly the DANN grid. Order: smoke test, E1 + E2 together, E0 budget, E0 grid. Rebuild everything with `V2=$FINAL_ROOT bash examples/symmetry_paper/paper/make_figures.sh` (after copying the roots locally).

**Retests dropped (2026-10-31):** the sim is deterministic on one machine, so same-node retests are identical. Optional: replay a sample of final controllers on a second machine and report the spread.

E2b (evolved bodies) stays optional.

## 6. Cost and order

See §5. Costs from the first round's wall times (bang\_bang 29 s, DANN 104 s per fixed-body training at 20 CPUs; ≈ 0.4 CPU-s per evaluation). A DANN evolution run at 1,600 evals is ≈ 620 bodies × 1,600 evals ≈ 6 h on 20 CPUs (scripts allow 24 h).

## 7. Planned figures

1. **R1, Figure 1:** (a) DANN kp × kv heatmap of sym − tree at 1,600 evals, default cell outlined, cell text gap / means / seeds ahead; (b) gap vs inner budget with paired seed lines (`analyse_e0.py` → `fig1_grid_budget.png`).
2. **R1, E1 curves:** Plot 1 genomes under bang-bang with paired gaps per generation; Plot 2 controllers, tree dashed / sym solid, with sym − tree per generation (`analyse_e1.py`).
3. **R2:** E2 learning curves for four body sets; rows score, gap to tree, ratio to tree (defined: mean ÷ mean of tree at equal evals, from 50 evals).
4. **R3:** shape measures by set; final score per body set.
5. **R4:** gait-class shares by genome × controller. (Gait vs hinge grid dropped; appendix only if the DANN grid shows something.)
6. **R5:** merged locking figure: (a) tree / sym unlocked / sym locked score on E2, (b) locked share of E1 champions; phase-shift replay as change ÷ baseline (absolute version in the appendix).

## 8. Findings at a glance

All from the first round (v1); every row is "v1 trend, v2 pending".

| section | finding | evidence | confidence |
| --- | --- | --- | --- |
| R1 | Sym > tree under bang\_bang across hinge settings | grid pooled 17/18, p < 0.001; single cell n.s. at n = 10 | **moderate–high** |
| R1 | DANN gap absent at 200 evals in evolution, grows with budget | −0.02 / +0.22 / +0.37, n = 5 | **moderate** |
| R2 | Symmetric bodies start ahead; the relative lead shrinks with learning | E2 ratio 1.55 → 1.41 | **high** (v1) |
| R3 | Shape accounts for about half the gap | shape matching closes 40–50% | **moderate** (n = 40) |
| R3 | Duplication without mirroring gets most of the advantage | tied closes 65–85%; sym − tied n.s.; E1 tied ≈ sym | **moderate** (n = 40) |
| R3 | Extra leg hinges explain part of duplication's effect | a third of tied − tree | **uncertain** (E5) |
| R4 | Symmetric bodies alternate and lock; tree bodies do not | grid, E1, E2 | **moderate–high** (pending hand-check) |
| R5 | Gaits depend on timing, not specially on the left–right lock | phase-shift replay: side ≈ random at body level; relative loss sym ≤ tree | **moderate–high** (v1) |
| R6 | Advantage needs a tractable controller search, not shared frequency | drift keeps it, free collapses all | **moderate** |

## 9. Risks, scope and draft conclusion

### Risks

- **E4 shrinks the controls' effects** (shape or tied no longer close a clear share of the gap): R3 becomes "symmetric bodies start ahead, partly through shape" with duplication as a suggestive result.
- **Symmetric − tied grows with more bodies:** mirroring adds something beyond duplication; then the gait results (R4) are the candidates for what, and the phase-shift result limits it to something other than left–right timing.
- **Detector does not hold on the current modules:** report `mirror_sync` and phase-lag distributions directly.
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
