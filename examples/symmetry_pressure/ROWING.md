# Rowing-gait detection (control-stride sweep)

*Started 2026-10-07. Status (2026-10-08): validated against two hand-labelled batches of 96 (`--seed 0`,
`--seed 1`; 170 distinct individuals). Agreement is 0.93 / 0.94, which is above the labeller's own
repeatability (0.91). The over-generations plots are done: at stride 20 all 3 symmetric reps converge on
rowing, against 1 of 3 tree reps (see "Rowing over evolution").*

## Why

`__data__/ariel_control_stride_sweep/gait_similarity/FINDINGS.md` suggests that tree_symmetric populations
converge on a shared "rowing" gait: limbs on both sides lifted and swept together. Until now that came only
from watching videos. This pipeline labels each individual automatically and gives a way to check those labels
by hand.

## Method (`rowing_detect.py`)

**Input:** the 50Hz replays in `behavior_50hz/<run>/behavior.npz`. These come from `behavior_extract.py` run
with `RECORD_EVERY_N=10`. They cover the tree and tree_symmetric runs with the ANN brain. cppn has no 50Hz
traces. Per episode, after the settle phase plus a 2 s transient:

1. **Frame:** module positions are taken in the core's frame. The heading is the core's mean velocity in that
   frame, smoothed over 1 s.
2. **Split plane:** the best mirror plane within ±45° of the heading (`SNAP_DEG`).
   - `--snap-deg 0` splits strictly along the heading.
   - Snapping was added because many symmetric bodies travel 15–35° off their own mirror axis. A strict
     heading split then finds no pairs and labels them "unpaired".
3. **Pairing:** modules within 2 cm of the plane are midline. The rest are paired left↔right by Hungarian
   matching on their rest positions, with the right side reflected.
4. **Per pair:** the right module's trajectory is reflected first, which avoids hinge-sign ambiguity. Then:

   | metric | meaning |
   |---|---|
   | periodicity | autocorrelation peak |
   | frequency | 1 / period |
   | `mirror_sync` | zero-lag correlation: +1 = in sync, −1 = half a cycle apart |
   | `phase_lag` | phase offset between the two sides, cycles in [0, 0.5] |
   | `amp_ratio` | min / max stroke of the two sides |
   | `stroke_fore/lat/up` | share of the pair's stroke along each axis |

   A pair counts as **rowing** if all of these hold (each threshold is a module constant):

   | condition | constant |
   |---|---|
   | periodicity ≥ 0.3 on both sides | `PERIODIC_MIN` |
   | frequencies within 10% | `FREQ_TOL` |
   | `amp_ratio` ≥ 0.3 | `AMP_RATIO_MIN` |
   | lateral share of the stroke < 0.7 | `LAT_MAX` |
   | `phase_lag` < 0.25 cycles | `LAG_ROW` |
   | fore-aft share of the stroke ≥ 0.3 | `FORE_MIN` |

   A pair is **alternating** under the first four conditions with `phase_lag` ≥ 0.40 (`LAG_ALT`). Two pair
   kinds count as neither:
   - `lateral`: otherwise OK but at least 70% lateral (sway or clap).
   - `no_sweep`: in phase but less than 30% fore-aft (bobbing).
5. **Per individual:** shares of the stroke-weighted motion of moving, off-midline modules.
   - Outputs: `rowing_frac`, `alternating_frac`, `aperiodic_frac`, `paired_frac` and a continuous
     `rowing_score` (stroke-weighted `mirror_sync`; ROC AUC 0.95 against the hand labels).
   - The label is the first that applies: `static`, `unpaired` (paired < 25%), `aperiodic` (> 50%,
     `APERIODIC_MIN`), `rowing` (> 25%, `SHARE_MIN`, and more than alternating), `alternating` (the same
     the other way round), `mixed`.

## Validation

### Batch 1 (`--seed 0`, `rowing/rowing_labels.json`): re-tune

The first rule (`mirror_sync` ≥ 0.7, share > 50%) had precision 1.00 and recall 0.44 for rowing: it never
called a non-rower rowing, but missed over half the rowers. Lessons from the labels:

- **Phase lag separates the classes better than zero-lag sync.** Hand-labelled rowers lag a median of 0.08
  cycles (max 0.23), alternators 0.36–0.50, and mixed sit around 0.31. A lag of 0.15 cycles already drops
  `mirror_sync` to about 0.6, so the 0.7 cut missed slightly offset rowers.
- **Lateral strokes fool the mirror.** Mirroring flips the lateral axis, so a whole-body sideways sway reads
  as anti-phase (it was called "alternating") and an in/out clap as in-phase. No hand-labelled rower had
  more than 49% lateral stroke.
- **Rowing is often only part of the stroke.** A share of 0.25–0.3 catches it. The "more than alternating"
  condition stops a body that is evenly split from being called rowing.

After the re-tune, agreement went from 0.74 to 0.92.

### Batch 2 (`--seed 1`, `rowing/rowing_labels_2.json`): held-out check, then a small refinement

- **Held out:** on the 74 individuals not in batch 1, the batch-1 detector scored agreement 0.91, precision
  0.93 and recall 0.87. The tuning generalised.
- **Labeller repeatability:** 22 individuals appeared in both batches. They got the same label 19/22 times,
  and 20/22 (0.91) by the validator's measure (rowing vs not, alternating vs not). That is a rough ceiling:
  past it, a gain against one batch's labels is as likely to be fitting noise.
- **Threshold sweep across both batches:** the batch-1 thresholds were at or next to the optimum for every
  parameter on each batch independently.
- **Refinement:** two changes helped on *both* batches, and were adopted.
  - `FORE_MIN` 0.3 (`no_sweep`): rowers' rowing pairs are at least 34% fore-aft, and at most 40% vertical.
    The only two in-phase individuals that were mostly vertical (53% and 79%) were labelled "other".
    Caveat: both come from one run (`tree s100 rep2_26`).
  - `PERIODIC_MIN` 0.5 → 0.3 and `SHARE_MIN` 0.3 → 0.25: in-phase rowers with irregular cycles.

| | batch 1 | batch 2 | both (170) |
|---|---|---|---|
| agreement | 0.93 | 0.94 | 0.94 |
| rowing precision | 0.97 | 0.95 | 0.97 |
| rowing recall | 0.89 | 0.95 | 0.93 |

### The 11 remaining disagreements (both batches; repeats take the batch-2 label)

| group | cases | why |
|---|---|---|
| rowing → unpaired | `tree s20 rep2_24` g16 b10, g18 b21 | asymmetric bodies: the mirror split finds no pairs |
| rowing → aperiodic | `tree_symmetric s20 rep1_12` g23 b14, `s50 rep2_22` g13 b3 | in phase (lag ~0.14), but module periodicity 0.24–0.29; in g23 one side's period is estimated at double the other's |
| rowing → alternating | `tree_symmetric s100 rep2_23` g24 b23 | the data shows a fore-aft anti-phase stroke (lag 0.44); worth re-watching |
| false alternating | `tree s50 rep2_25` g29 b10, `tree s50 rep1_16` g22 b21, `tree s100 rep1_17` g18 b4, `tree_symmetric s100 rep1_14` g26 b4 | clean anti-phase in the data; g18 b4 was labelled alternating in batch 1 and other in batch 2 |
| false rowing | `tree s50 rep2_25` g29 b7, `tree s100 rep1_17` g8 b14 | g29 b7: four clean in-phase pairs (lag 0.06–0.18); g8 b14: 29% rowing, just over the share cut |

Most of these are either structural (pairing, period estimation) or are cases where a re-watch might change
the label. More thresholds would not fix them. Possible structural fixes, if ever needed:
- accept octave-related frequencies when pairing, for the aperiodic misses;
- a pairing that tolerates asymmetric bodies, for the unpaired ones.

## Results (detector as of 2026-10-08)

All 12,329 episodes, final generation, label shares (first-pass rowing share in brackets):

| genome | stride | rowing | alternating | mixed | aperiodic | unpaired | static |
|---|---|---|---|---|---|---|---|
| tree | 20 | 0.25 (0.07) | 0.02 | 0.14 | 0.05 | 0.29 | 0.25 |
| tree | 50 | 0.34 (0.05) | 0.02 | 0.10 | 0.11 | 0.40 | 0.03 |
| tree | 100 | 0.01 (0.00) | 0.00 | 0.04 | 0.10 | 0.71 | 0.14 |
| tree_symmetric | 20 | **0.60** (0.33) | 0.05 | 0.27 | 0.00 | 0.08 | 0.00 |
| tree_symmetric | 50 | 0.23 (0.09) | **0.26** | 0.22 | 0.08 | 0.22 | 0.00 |
| tree_symmetric | 100 | 0.18 (0.03) | 0.08 | 0.40 | 0.18 | 0.16 | 0.00 |

- **Sanity check:** the FINDINGS s20 symmetric medoid
  (`ctrlstride_forward_tree_symmetric_s20_rep0_43569_3`, gen 30, body 3) came out as `rowing` under the
  first-pass rule: 0.57 Hz, all 4 pairs in sync at 0.78, `heading_consistency` 1.0.
- **Rowing is not exclusive to symmetric bodies.** At stride 20, 60% of tree_symmetric individuals row in the
  final generation against 25% for tree. But tree at stride 50 rows at 34%. Many tree individuals are
  unpaired (29–71%), so their gait can't be judged by this method at all.
- **These are final-generation shares only.** The trajectories over generations are below.

## Rowing over evolution (`plot_rowing_stride_sweep.py`, 2026-10-08)

Plots in `rowing/plots/`: `rowing_share_over_gens.png` (mean ± SEM across reps, plus each rep) and
`label_composition.png` (stacked gait classes, reps pooled). Rowing share per rep, first vs last 5
generations, is in `rowing_early_late.csv`:

| genome | stride | early (gens 0–4) | late (gens 26–30), per rep | first gen ≥ 50% rowing, per rep |
|---|---|---|---|---|
| tree | 20 | 0.02 | 0.59 / 0.00 / 0.38 | 24 / – / – |
| tree | 50 | 0.01 | 0.57 / 0.00 / 0.28 | 25 / – / – |
| tree | 100 | 0.01 | 0.01 / 0.00 / 0.03 | – |
| tree_symmetric | 20 | 0.17 | **0.51 / 0.60 / 0.52** | **11 / 8 / 4** |
| tree_symmetric | 50 | 0.10 | 0.02 / 0.01 / 0.61 | – / – / 6 |
| tree_symmetric | 100 | 0.06 | 0.00 / 0.25 / 0.26 | – |

What it shows:
- **At stride 20 the hypothesis holds.** All 3 symmetric reps converge on rowing (51–60% of the population),
  and fast: past 50% by generation 4–11. Tree gets there in only 1 of 3 reps, by generation 24. Its other
  reps reach 38% and 0%.
- **Rowing is not exclusive to symmetry.** Tree evolves rowing too (2 of 3 reps at strides 20 and 50), but
  later and less reliably. The tree reps that never row (rep1 at s20 and s50) end up largely static
  (54% and 25%).
- **Symmetric bodies start ahead.** In generations 0–2, 9.5% of symmetric individuals already row, against
  0.6% of tree. Part of the gap is the genome's starting bias, before any selection.
- **At stride 50, symmetry converges on *a* mirror-coupled gait, not necessarily rowing.** Rep 2 rows (61%),
  rep 1 alternates (59%) and rep 0 is split (24% alternating, 47% unpaired). Both are bilateral
  coordination, but they are different gaits.
- **At stride 100 there is little of either.** Most tree individuals can't be judged (70–79% unpaired in two
  reps), and symmetric reaches only about 25% rowing in 2 of 3 reps.
- **Caveats:**
  - **Small sample.** With 3 reps per cell there is no meaningful significance test. The s20 result rests
    on all 3 symmetric reps agreeing, against 1 of 3 for tree.
  - **Many tree individuals can't be judged.** 44–85% of the final tree populations are unpaired or
    static, so their rowing share is a share of all individuals. Among judgeable individuals, late rowing is
    0.44 (tree) vs 0.60 (symmetric) at s20.

## Rowing-penalty experiment (set up 2026-10-13, not yet run)

The question: what evolves at 20Hz when rowing no longer pays, with symmetry enforced (`tree_symmetric`) and
without it (`tree`)?

- **Physics:** bodies are built with kgd's zoo modules, using `--zoo-modules` (`shared.zoo_modules_spec`):
  - hinge kp 1.36 (local 1.0), kv 0.36 (local 1.0), armature 0.081 (local 0), and a hard ±90° joint limit
    (local: none);
  - masses: core 1.363 kg (local 1.0) and stator 0.065 kg (local 0.020). The hinge body's own mass is
    0.105 kg. Brick and rotor masses already matched.
  - sizes (full edge): core 0.15 m (local 0.20), brick 0.075 m (local 0.10), and slightly different
    stator / rotor boxes.
  - The local module classes and their face layout are kept. kgd's core faces are rotated 90° and his bricks
    have no TOP/BOTTOM faces, which ~9% of evolved attachments use. Only the constants are swapped in while
    a body is built. Every attachment site scales with them, and the evolution process uses them for its
    collision pruning too (`use_zoo_modules_globally`).
  - Friction and contact settings already matched the zoo.
  - `test_zoo_modules.py` checks the compiled values against `zoo_benchmark/zoo_modules.py`.
  - Replays pick the flag up from `run_config.json`; older runs still replay with the local modules.
  - **Caveats:** two thresholds are in absolute units and were not rescaled. `HEIGHT_PENALTY_THRESHOLD`
    (0.21 m core height) is easier to stay under with a smaller core. The rowing detector's
    metre-based tolerances (`MIDLINE_TOL` 2 cm, `PAIR_TOL_M` 3 cm) are relatively looser on bodies ~25%
    smaller. The detector has not been re-validated on zoo-sized bodies.

- **Fitness:** the same forward task, but `gecko_skill_tasks.py --rowing-penalty` scales each episode's
  forward displacement by `1 - rowing_frac`, using `shared.rowing_scaled_disp`. A full rower scores about 0.
  A backward displacement is left unscaled, so the scaling never helps it. Height and jerk penalties are
  unchanged.
- **How:** `shared._skill_worker_eval` samples all robot body poses at 50Hz after settle, which is the same
  sampling as `behavior_50hz`. It then runs `rowing_detect.analyse_episode` on them with the validated
  thresholds above. This costs about 0.1 s per episode, on top of about 1 s of simulation. If the detector
  fails, the episode counts as no rowing.
- **Logged:** each skill in `run_data.jsonl` gets `raw_score` (the unpenalised displacement), `rowing_frac`
  and `rowing_label`. `behavior_common.replay_skill_episode` reads `rowing_penalty` from `run_config.json`
  and applies the same penalty, so `replayed_fitness` still matches `stored_fitness`.
- **Run:** `sbatch slurm/run_rowing_penalty.sh` starts 12 jobs: 2 genomes × {penalty, nopenalty} × 3 reps, at
  stride 25 (20Hz) with the ANN brain, seeds 42–44, all with the zoo modules.
  - The `nopenalty` arm is the baseline. The stride sweep's own runs used the local modules, so they can't
    serve as one.
  - Results go to `/scratch/jed/ariel_rowing_penalty_ann/<arm>`.
  - Run tags follow the stride sweep's format, so each arm directory works as `--stride-root` for
    `behavior_extract.py` → `rowing_detect.py` → `plot_rowing_stride_sweep.py` unchanged.
- **Sanity check:** the s20 symmetric medoid (gen 30, body 3) replays at 4.10 m raw, with `rowing_frac`
  1.00, and is penalised to 0.

## Outputs (`__data__/ariel_control_stride_sweep/rowing/`, not in git)

- `individuals.csv`: one row per episode, with every metric and the label.
- `runs/<run>.csv`: cached per run. It is recomputed when the npz is newer or `--snap-deg` changes.
- `pairs/<run>.csv`: one row per left/right pair, with its metrics, `pair_kind` and `weight`.
- `page/rowing_review.html` + `page/videos/`: the batch-1 review page (`--seed 0`).
- `page_seed1/rowing_review.html` + `page_seed1/videos/`: the batch-2 review page (`--seed 1`).
- `rowing_labels.json`, `rowing_labels_2.json`: the hand labels for batches 1 and 2.
- `validation/`: written by `rowing_validate.py` (overwritten on each run).
- `plots/`: written by `plot_rowing_stride_sweep.py`.

## How to pick this back up

1. **Validate:** run
   `python examples/symmetry_pressure/rowing_validate.py --labels <rowing>/rowing_labels.json <rowing>/rowing_labels_2.json`.
   - It accepts several label files. An individual in more than one takes the last file's label.
   - It prints the confusion matrix, precision/recall, the ROC AUC of `rowing_score`, a sweep over the
     thresholds (recomputed from the pair CSVs; `aperiodic_frac` is not recomputed), and the detector's
     reason for each disagreement.
2. **Re-check disagreements:** run
   `python examples/symmetry_pressure/rowing_review_page.py --only-disagreements <labels.json>`.
3. **Tune:** change the constants at the top of `rowing_detect.py`, then
   `python rowing_detect.py --force --workers 18` (about 3 min). Keep a change only if it helps on both
   batches, since one batch's labels are only about 0.91 repeatable.
4. **More labels:** `MUJOCO_GL=egl python rowing_review_page.py --seed N --out <rowing>/page_seedN/rowing_review.html`.
   - Keys: R rowing, A alternating, M mixed, N other, U unsure; J/K next/prev; B toggles blind mode.
   - Each seed saves to its own localStorage key (`rowing_review_labels_v1`, `..._seedN` for N > 0), so an
     export holds only that page's labels.
   - Seeds overlap: about a quarter of seed 1 was already in seed 0.
5. **Plots:** `python examples/symmetry_pressure/plot_rowing_stride_sweep.py` (seconds) reads
   `individuals.csv` and writes `rowing/plots/`. Re-run it after any detector change.

Run the scripts with `.venv/bin/python`; the system python lacks torch.

## Files (`examples/symmetry_pressure/`)

- `rowing_detect.py`: the detector library and CLI.
- `test_zoo_modules.py`: `--zoo-modules` bodies match kgd's zoo sizes, masses and hinge; the default build is unchanged.
- `test_rowing_penalty.py`: the fitness scaling and `shared.rowing_info` (rower, alternator, detector failure).
- `test_rowing_detect.py`: 14 synthetic tests (rowing under rotated heading/yaw, slightly lagged rowing,
  alternating, frequency mismatch, one-sided limb, noise, mixed, sideways sway, bobbing, off-axis snapping,
  morph plane, JSON payload).
- `rowing_review_page.py` + `rowing_review_page.html`: the review page and its template. Videos come from
  `stride_gait_page.render_one`, and existing renders are reused.
- `rowing_validate.py`: scoring, the threshold sweep and the disagreements list.
- `plot_rowing_stride_sweep.py`: rowing share and gait-class composition over generations, plus the
  early/late table.
- `stride_gait_similarity.py`: the shared loader `iter_episode_poses` was factored out of `run_cycles`; its
  behaviour is unchanged.
