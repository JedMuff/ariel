# What symmetric robots do differently: gaits, learning and the source of the symmetry advantage in body–brain evolution

Oct 30, 2026 · @Jed

## 1. Summary

The paper asks what symmetric robots do differently, and answers: they move differently, and those movements are cheap to learn. Revised from the 2026-10-30 plan; results so far are in `SYMMETRY_ADVANTAGE.md`. Costs come from `timing.jsonl` of finished runs; wall times assume about 7 runs of 20 CPUs at once.

**Setup in one line.** Modular robots, outer EA over bodies, inner CMA-ES over each body's controller. Bodies from a mirror-symmetric genome (`tree_symmetric`) often walk further than bodies from an unconstrained genome (`tree`), but not under every controller or budget:

- with an open-loop **bang\_bang** controller (shared frequency), the advantage appears at 200 evaluations per body;
- with a distributed ANN (**DANN**), it is absent at 200 evaluations and grows with the training budget.

**Question.** What do symmetric robots do differently, in body, behaviour and learning, and which of these differences produces the advantage?

**Storyline.** Symmetric bodies are larger yet stay controllable. They move differently: they alternate mirrored limbs, which tree bodies almost never do, while the hinge actuator decides which coordinated gait appears. They learn differently: a controller with built-in coordination finds these gaits quickly, while a learned controller finds them but holds them poorly until it has enough budget. Removing the controller's coordination removes the advantage.

**Thesis.** The symmetry advantage is a learnability advantage. Symmetry gives a body coordinated gaits that are cheap for the right controller to learn, and under a fixed inner budget evolution rewards exactly that.

**Task.** Forward locomotion (`dsocial_zoo` profile, zoo modules and hinge, 15 s episodes, 20 Hz).

## 2. Paper structure

The results run from the puzzle, through three kinds of difference, to the link between them and a causal test. Section 5 carries the argument; without it, the behaviour and learning sections sit side by side as a list.

| results section | question | expected finding | evidence | status |
| --- | --- | --- | --- | --- |
| R1. The advantage and its conditions | When do symmetric bodies win? | Bang\_bang gap across all hinge settings; DANN gap absent at 200 evals, growing with budget | Hinge grid, DANN budget series (E0) | done; budget series finishing |
| R2. Different bodies | Are symmetric bodies built differently? | Larger (more modules, hinges, limbs) but still controllable | Size descriptors of all champions and random bodies; hinge-matched gap (E2) | descriptors exist; matching new |
| R3. Different behaviour | Do they move differently? | They alternate mirrored limbs, tree bodies almost never do; the actuator sets which gait appears | Grid gait labels (E3, done); E1 and budget-series champions | grid done, pending hand-check |
| R4. Different learning | Do they learn faster, end higher, or both? | Faster climb under coordinated controllers; ceiling gap uncertain | E2 learning curves | new |
| R5. Behaviour meets learning | Does phase locking along training track the score gap? | DANN finds alternation early but holds it poorly; the gap opens as phase quality improves | E2 checkpoint replays (E3) | new |
| R6. Causal test | Does removing coordination remove the advantage? Do paired limbs without mirroring do the same? | Free frequency breaks phase locking and shrinks the gap; `tree_tied` uncertain | E1 | new |

The discussion places the result in the evolution-selects-for-learnability literature: under a fixed inner budget, evolution judges bodies by what the controller can extract, and symmetry is one way a body becomes easy to extract from.

**Questions each section answers on its own** (replacing the old H1–H4):

- **Ceiling or learning speed?** R4: gap at the final budget vs gap in area under the curve.
- **Mirroring or a smaller search space?** R6: `tree_tied` against tree and sym.
- **Is the gait the mediator?** R3 and R5: high-scoring symmetric controllers should be phase-locked, and symmetric bodies without phase locking should score like tree bodies.
- **Is it just size?** R2: the gap at matched hinge count. If it vanishes, the finding becomes "symmetry lets evolution build bigger bodies that stay controllable", which still fits the learnability story.

## 3. Common setup

Unchanged from the previous plan, except that behaviour metrics are now defined up front.

- **Simulator and task:** `dsocial_zoo`, zoo hinge (kp 1.36, kv 0.356, armature 0.081, joint limit), `forward`. Score is `clip(d / 3.75 m, 0, 2)` over 15 s.
- **Evolution:** outer (μ+λ) = (10+20) for 30 generations, `--repeat-evals`, max 25 modules. Inner CMA-ES with popsize 20.
- **Genomes:** `tree`, `tree_symmetric` (mirrored limbs), and `tree_tied` (each placed module gets a randomly placed twin, matching symmetric's gene count without mirroring; E1 only).
- **Seeds:** 10 per arm, seed 42+rep in every cell so runs pair by seed.
- **Statistics:** paired symmetric − tree differences with Wilcoxon signed-rank and bootstrap CIs. Score is the last-generation best, steadier than best-ever under `--repeat-evals`.

**Brains**

| brain | left–right coordination | status |
| --- | --- | --- |
| `bang_bang` | built in: one shared frequency, so relative phases stay fixed once found | exists |
| `bang_bang --free-frequency` | removed: each hinge has its own frequency, so phases drift unless frequencies match | new, small; a variant of bang\_bang, not a third brain |
| `dann` (`dann_pos_hinge_h16`) | learned: shared per-hinge MLP with neighbour inputs | exists |

**Body descriptors** (R2): module, hinge and limb count for every body. Symmetric bodies evaluate \~40% slower, consistent with being larger. Reported as a finding, and used as a covariate for hinge-matched gaps.

**Behaviour descriptors** (R3, R5):

- **gait class:** alternating, rowing, unpaired or static, from `rowing_detect.py`;
- **gait species:** clusters of gait cycles from `gait_species.py`, with cycle distance from `stride_gait_similarity.py`;
- **left–right phase relation:** `mirror_sync` and `phase_lag` per mirrored or tied hinge pair, classed as in-phase, anti-phase or unlocked;
- **phase quality (new, to define):** how steadily a pair holds its phase relation, for example the circular variance of `phase_lag` across gait cycles, or the share of cycles within a tolerance of the pair's mean lag. This is the metric behind the DANN execution claim and needs fixing before E2 runs.

## 4. Experiments

The experiments are unchanged in design; what changes is that E2 logs controller checkpoints and E3 becomes the main evidence, not a supporting analysis.

### E0. Existing results (R1, R3)

| result | serves | where |
| --- | --- | --- |
| Hinge grid: kp {1, 1.36, 2.5} × kv {0.356, 1, 2} × armature {0, 0.081}, 18 cells × 3 seeds. Bang\_bang sym wins 17/18 cells (+0.14, p < 0.001); DANN no gap in any cell (+0.02, p = 0.31) | R1 puzzle; R3 gait vs actuator | `__data__/ariel_actuator_grid` |
| DANN budget series: gap −0.03 / +0.21 / +0.50 at 200 / 800 / 1600 evals | R1 puzzle; R5 champions to replay | `__data__/ariel_dann_budget` (200-eval point: `__data__/ariel_symmetry_pressure_sweep_dann`) |
| Grid gait replay: last 5 generations of 216 runs, 23,709 episodes, exact fitness reproduction | R3 | `__data__/ariel_actuator_grid/rowing_grid/individuals.csv` |

- **Grid in the paper:** R1 reports parameter ranges and per-brain gaps; the bang\_bang gap grows with how well bodies can move (stiffer, less damped hinges) but never vanishes. R3 uses the same runs to show the actuator changes *which* gait evolves while the advantage persists.
- **Before writing up:** finish job 47059 (budget 40 rep 4, budget 80 reps 2–4), giving n = 5 per point.
- **Cap risk:** symmetric runs at 1600 evals reach 1.86–1.93, close to the cap of 2. If several hit it, rescore the champions without the clip.

### E1. Coordination ablation (R6)

The causal test: remove bang\_bang's shared frequency, keep everything else.

- **Design:** {`bang_bang`, `bang_bang --free-frequency`} × {tree, tree\_symmetric, tree\_tied} × 10 seeds, 200 evals per body. **60 runs, \~1,200 CPU-h, \~9 h wall.**
- **Readout:** sym − tree gap with vs without shared frequency; `tree_tied` against tree and sym; and, new in this framing, the gait class and phase quality of every champion, so the paper can show *how* the gap disappears, not only that it does.
- **Expected:**
  - *Gap with shared frequency:* high confidence (zoo-hinge grid cell +0.17; grid +0.14 pooled over 54 pairs).
  - *Gap shrinks with free frequency:* moderate. It can fail if per-hinge frequencies converge anyway (check: spread of evolved frequencies within each body), or if the extra parameters hurt CMA-ES on both genomes (check: absolute scores, not just the gap).
  - *Free-frequency sym champions lose phase locking:* moderate; this is the behavioural signature of the ablation.
  - *`tree_tied`:* uncertain, no data under `dsocial_zoo`. Report it whichever way it falls.

### E2. Fixed-body learning curves (R2, R4, R5)

Removes the outer EA, so controllability can be measured apart from what evolution found.

- **Bodies (120):** 40 random bodies per genome (tree, sym) from the initial distribution, stratified to matching hinge-count distributions; plus the best final body of each tree and sym run in E1 (40).
- **Training:** each body × {`bang_bang`, `bang_bang --free-frequency`, `dann`} × 3 training seeds: 1,080 trainings. Bang\_bang variants run to 1,600 evals (plateau), DANN to 5,000. The curve prefix gives the 200 / 800 / 1600-eval points.
- **New: controller checkpoints.** Save the CMA-ES mean (or best) controller at 50 / 200 / 800 / 1600 / 5000 evals. E3 replays them to get gait class and phase quality along training. Cost is a few kB per checkpoint.
- **Readout:**
  - **R4 ceiling:** gap at the final budget, per brain;
  - **R4 learning speed:** gap in area under the curve and in evals to reach 50% of the final score;
  - **R2 size:** gap at matched hinge count;
  - **R4 random vs evolved:** if the gap is much larger among evolved bodies, part of the advantage lies in what evolution finds, not in a typical symmetric body.
- **Cost:** \~400 CPU-h, **\~3 h wall.** Needs a small harness: sample a body, build it, run the existing inner loop, log `learning_curve` and checkpoints.
- **Expected:** faster climb for sym under bang\_bang and DANN (moderate–high); overlapping curves without shared frequency (moderate); ceiling gap at 5,000 evals, random vs evolved, and survival under hinge matching all genuinely uncertain.

### E3. Behaviour analysis (R3, R5, behaviour readouts of R6)

Now the main evidence of the paper. Rollouts only.

- **Inputs:** champions from the budget series and E1; E2 checkpoints along training.
- **Measures:** gait class, gait species, left–right phase relation and phase quality (section 3).
- **Readout:**
  - **R3:** gait class and species by genome × brain × budget; whether high-scoring symmetric controllers concentrate in a few phase-locked species, and whether symmetric bodies outside them score like tree bodies.
  - **R5:** phase quality vs evals for sym and tree, per brain. Prediction: bang\_bang sym locks within the first few hundred evals; DANN sym reaches alternation early but its phase quality rises slowly, and the score gap follows phase quality, not gait class.
  - **R6:** phase relations of E1 champions with and without shared frequency.
- **Grid results already in (2026-10-30):**
  - **Symmetric bodies alternate; tree bodies cannot use mirror coupling.** Alternating gait: bang\_bang sym 40% vs tree 4%; DANN sym 32% vs tree 3%. Tree bodies are 76–78% unpaired. Among last-generation champions, 24/54 bang\_bang sym runs alternate, against 0/54 tree runs.
  - **Rowing is rare and tied to damping.** Overall ≤ 5%. Under bang\_bang it appears almost only at kv 1: sym 9% at kv 1 vs 0.5% at kv 0.356, up to 60% in single runs (kp 1, kv 1, armature 0.081). It scores below alternation (bang\_bang sym mean 0.39 vs 0.63).
  - **Stiff, lightly damped hinges give a third behaviour:** bodies travel well (mean 1.1, up to the cap) with strokes too small for the detector, labelled `static`. 6% of bang\_bang episodes at kp 2.5, kv 0.356, against ≤ 0.3% at kp 1. Probably a fast vibrating shuffle; **check videos** before naming it.
  - **The DANN wrinkle, now the bridge to R5:** DANN sym champions alternate almost as often as bang\_bang ones (18/54) yet show no advantage; alternating sym mean score 0.31 DANN vs 0.63 bang\_bang. Finding a gait and executing it are different things, and execution is what learning budget buys.
- **Cost:** minutes to an hour per batch of rollouts (the grid took \~15 min locally).
- **Detector validation is on the critical path:** `rowing_detect.py` was validated on ANN gaits of the larger local modules. Hand-check \~20 zoo-module bang\_bang and DANN episodes per label before E1 finishes.

## 5. Cost and order

Total compute is unchanged at \~1,600 CPU-h, about half a day of cluster time; the order now puts the behaviour tools first because they carry the argument.

| experiment | jobs | compute | wall | new code |
| --- | --- | --- | --- | --- |
| E0 finish budget series | job 47059 | running | – | – |
| E1 coordination ablation | 60 runs | \~1,200 CPU-h | \~9 h | `--free-frequency` for bang\_bang (small) |
| E2 fixed-body curves | 1,080 trainings | \~400 CPU-h | \~3 h | body sampler, harness, checkpoint logging (small–moderate) |
| E3 behaviour analysis | rollouts | negligible | ≤ 1 h per batch | phase-quality metric (small) |
| **total** |  | **\~1,600 CPU-h** | **\~half a day** |  |

**Order:**

1. **Now, before compute:** define phase quality; hand-check the detector on zoo episodes and watch the `static` videos; add checkpoint logging to the E2 harness.
2. **E1** (cheapest decisive test). The random-body half of E2 can start in parallel.
3. **E2 evolved-body half** once E1 champions exist.
4. **E3** on each batch as it lands: E1 champions, then E2 checkpoints. Analyse R5 as early as possible, since it carries the paper.

## 6. Planned figures

Figures follow the storyline; the key new figure is phase quality along training (figure 6).

1. **R1 puzzle:** two panels. Hinge-grid heatmap of the sym − tree gap, bang\_bang vs DANN; and DANN gap vs inner budget (200 / 800 / 1600).
2. **R2 bodies:** module, hinge and limb count distributions, tree vs sym, random vs evolved.
3. **R3 behaviour:** gait-class shares by genome × brain, with example phase-lag traces of an alternating sym body and an unpaired tree body.
4. **R3 actuator:** gait-class shares across the hinge grid (kp × kv), sym vs tree, with the score gap overlaid.
5. **R4 learning:** E2 learning curves, tree vs sym, one panel per brain.
6. **R5 behaviour meets learning:** phase quality and score vs evals, sym vs tree, per brain, from E2 checkpoints.
7. **R6 causal test:** E1 paired gaps, {shared, free frequency} × {sym, tied} vs tree, with the share of phase-locked champions beside each bar.

Hinge-count-adjusted gaps and gait-species clustering details go in the appendix.

## 7. Findings at a glance

The behaviour findings are already moderately strong; the link to learning (R5) is the main untested piece.

| section | finding | evidence so far | confidence before new runs |
| --- | --- | --- | --- |
| R1 | Sym > tree under bang\_bang, across hinge settings | 17/18 cells, p < 0.001 | **high** |
| R1 | No gap for DANN at 200 evals | sweep + grid, p = 0.31 | **high** |
| R1 | Gap grows with DANN budget | 3-point series, n = 2 at the top | **moderate–high** |
| R2 | Symmetric bodies are larger | \~40% slower evaluation | **moderate** |
| R2 | Gap survives hinge-count matching | none | **uncertain** |
| R3 | Symmetric bodies alternate, tree bodies cannot | grid: 40% vs 4% (bang\_bang), 24/54 vs 0/54 champions | **moderate–high** (pending hand-check) |
| R3 | Gait type follows the actuator, the advantage does not | rowing at kv 1, shuffle at stiff, lightly damped hinges | **moderate–high** (pending hand-check) |
| R4 | Symmetric bodies learn faster | indirect | **moderate–high** |
| R4 | Symmetric bodies have a higher ceiling | near-cap scores only | **uncertain** |
| R5 | Phase quality, not gait class, tracks the score gap | DANN sym alternates (18/54) but scores 0.31 vs 0.63 | **moderate** |
| R6 | Removing shared frequency removes phase locking and the gap | none | **moderate** |
| R6 | Mirroring, not a smaller search space (tied ≈ tree) | none under `dsocial_zoo` | **uncertain** |

## 8. Risks, scope and draft conclusion

### Risks to the storyline

- **R5 fails** (phase quality along training does not track the score): behaviour and learning stay side by side without a link. Fallback: E1 still links them causally, and the paper becomes "symmetric robots move differently, and that difference needs a coordinated controller".
- **Detector does not hold on zoo modules:** R3 shares become unreliable. Fallback: report `mirror_sync` and `phase_lag` distributions directly, which need no labels.
- **Gap vanishes at matched hinge count:** reframe R2 as the headline difference (bigger bodies that stay learnable); the rest of the story holds.
- **Score cap at high budget:** rescore without the clip before reporting R1 and R4 ceilings.

### Out of scope (future work)

- **Mirror-tied DANN** (weights tied across mirrored pairs): the direct "give the learned controller the matching prior" test. Optional arm if R5 is ambiguous: {sym, tied} × 10 seeds at 200 evals, \~400 CPU-h.
- **Symmetry dose–response** (breaking 0–3 mirrored pairs): confounded by the morphology change; needs a matched "mutate both sides" control.
- **Other tasks:** the DANN sweep found a symmetric advantage on multidirection at 200 evals, possibly from easier steering rather than gait quality. A separate question.

### Draft conclusion

*Written for the expected outcome; revise once E1–E3 are in.*

> Symmetric robots do three things differently. They grow larger bodies, they move by alternating mirrored limbs, a gait unconstrained bodies almost never find, and they learn those movements quickly when the controller coordinates the two sides. The hinge actuator decides which coordinated gait appears, alternation, rowing or a fast shuffle, but not whether symmetric bodies win.
>
> The differences connect through learning. A controller with a shared frequency holds a phase relation from the start, so the gait comes almost for free. A distributed ANN finds alternation early but holds it poorly, and the symmetric advantage appears only as its phase quality improves with budget. Removing the shared frequency breaks phase locking and removes the advantage. *\[Ceiling result from E2: faster learning only / also a higher ceiling.\]* *\[Tied-genome result: mirroring itself / the smaller search space accounts for most of the effect.\]*
>
> The symmetry advantage is therefore a learnability advantage. Under a fixed inner budget, evolution judges a body by what the controller can extract from it, and a symmetric body offers gaits that are cheap to extract. Comparisons of morphological encodings are meaningful only relative to the controller and budget used to evaluate them, and we expect the same to hold for other regularities, such as repetition, segmentation or radial symmetry: they pay off when the controller shares them or has the budget to learn them.
