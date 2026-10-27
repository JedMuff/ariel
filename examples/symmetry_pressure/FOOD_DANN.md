# Food collection with the distributed ANN, trained from scratch

*Started 2026-10-26. Status: the code, the Slurm script and the video renderer are in place but
uncommitted. Every result below is a single local run (seed 42, 100 generations unless noted); nothing has
been submitted to the cluster. The current setup is ordered progress plus camera coverage, with the vision
normalised to direction only (the last two rows of the table). With 300 generations it collects all 5
training foods and all 5 held-out foods, and, unchanged, 9.99 of 10 foods on the original random-chain
food task (see "Transfer to the original food task"). The policy is saved in `food_dann_policy/`.*

## Why

The food condition of `slurm/run_symmetry_pressure_sweep.sh` (`gecko_food_skills.py`) never trains a
controller on food directly. It trains loco, left and right skills, and a hand-coded, vision-gated
`SkillController` switches between them. This experiment asks how the distributed ANN does when **one
network** is trained end to end on food collection: no skills, no state machine, and a fixed body (no body
evolution).

## Setup

**Body:** kgd's canonical zoo gecko (`examples/zoo_benchmark/canonical_bodies.py`), with zoo modules and the
zoo hinge actuator.
- Physics matches the zoo benchmark exactly (diffed against `zoo_benchmark.build_world("gecko")`):
  0.002 s timestep, kp 1.36 / kv 0.36, masses, inertias, geom sizes, friction, joint ranges.
- Control follows the DSOCIAL_ZOO profile used by the DANN symmetry sweep, not the raw zoo benchmark:
  20 Hz, actions alpha-blended (α 0.854), and a 3 s zero-action settle after spawning at 0.1 m.
- The ariel prebuilt `gecko()` can't be used: its module names ("neck") break `brains.model_topology`.

**Brain:** `shared.DannBrain` (`dann_pos_hinge_h16`) with an `extra_inputs=3` option, which broadcasts the
camera's [left, centre, right] food fractions to every hinge, the same way core orientation is broadcast.
That gives 45 inputs and 753 parameters (705 without vision). With `extra_inputs=0` the brain is unchanged,
which `test_dann_brain.py::test_dann_extra_inputs` checks.

**Camera fix (important).** The zoo core camera looks out of the core's **right side** (world −Y), not the
front. The gecko's front is the core's free FRONT face (+X); the spine leaves the BACK face (−X) and the legs
leave the sides (±Y). `face_camera_forward()` turns the camera to look level out of the front face, in the
food world only (`zoo_modules.py` is untouched). **Every run before this fix used the sideways camera.** The
same issue would affect anything else that uses the zoo core camera.

**Brick colour.** Zoo bricks are green (0.1, 0.7, 0.1), the food's colour, so the food detector saw the
robot's own body as food. The food world recolours the bricks grey.

**Vision is ray-cast, not rendered (`RayVision`).** A 24×32 grid of rays is cast from the camera
(`mj_multiRay`), excluding the camera's own body. Each third of the image scores the share of its rays whose
first hit is the food. Along real gaits this is within 0.044 of the rendered 96×128 green-pixel fractions
(mean difference 0.003).
- Rendering in parallel serialised on the GPU: about 4.9 s per 200 episodes whatever the worker count, and 40
  workers were slower than 20.
- With rays it is 2.0 s on 20 workers and 1.3 s on 40, so it scales with cores.
- Pitfall if rendering is reintroduced: an EGL renderer created in the parent before the process pool forks
  kills the workers.

**Other speed-up:** the rotor-ground contact count (`c_hinge`) is a per-physics-step Python loop that cost
more than the physics itself. It is diagnostic only, so it now runs only for recorded (final-evaluation)
episodes.

## Task history and results

### Phase 1: random food chains (abandoned)

One episode with 10 foods collected in sequence, the first 1.0–1.5 m away, then 0.5–1 m between foods.

| Variant | Result |
|---|---|
| 60 s episodes, sweep fitness: food + d_norm (d_norm is flat until the gecko is within 1 m) | Plateau at 0 for the first ~5 generations; first food by generation 13 |
| Energy: 10 s at the start, +10 s per food; fitness = food + progress along the segment from the last food | No food in 60 generations. The first food was always at the same spot, so held-out sets had zero spread |
| + 10 repeat food sets per candidate, first food at a random angle | It learned to walk straight about 1.2 m along the camera direction whatever the food: held-out progress 0.23 ± 0.30 |

**Optimiser comparison** (sideways camera, 10 s energy; 3 seeds × 200 generations; 200 episodes per
generation for each setup; the controller was the best-validated search mean):

| Setup | Held-out food + progress per seed | Mean |
|---|---|---|
| CMA-ES, 20 candidates × 10 repeats | 0.87, 0.57, 0.41 | 0.62 |
| OpenAI-ES, 50 × 4 (σ 0.5, Adam lr 0.1) | 1.01, 0.49, 0.17 | 0.55 |
| sep-CMA-ES, 50 × 4 | 0.53, 0.38, 0.32 | 0.41 |

No winner: the seeds of one setup spread more than the setup means differ. CMA-ES 20×10 stays the default.
Figure: `food_dann_optimizers.png` in the repo root, made by `plot_food_dann_optimizers.py`. That script only
reads this phase's output format (`val_fitness`).

### Phase 2: deterministic subtasks (current)

5 separate episodes, each starting from the same reset pose with one food. The foods are evenly spaced
(every 72°) on a circle around the start. The first is straight ahead and the rest follow **clockwise**
(0°, −72°, −144°, +144°, +72°). After training, the best candidate is replayed on these angles and on
held-out angles offset by 36°.
- The task is deterministic: the same weights score identically in every worker, so there are no repeats and
  no validation set, and the best candidate ever seen is the controller.
- In 4 of the 5 subtasks the food starts out of view, so the inputs are identical until the food is seen.
  The gecko cannot tell the subtasks apart; it walks one blind search route until its food comes into view.
  Evolution bends that route towards the 5 training positions, which is why the training angles score better
  than the held-out ones.

| Food distance / time limit | Fitness per subtask | Training angles | Held-out angles | Notes |
|---|---|---|---|---|
| 1 m / 20 s (sideways camera) | collected + progress (max 2) | 8.0/10, 4 food | 5.3/10, 2 food | All subtasks follow one blind route until the food is seen; the +54° food lay on that route |
| 1 m / 20 s (front camera) | same | 8.0/10, 4 food | 5.7/10, 2 food | Collections faster: 9–17 s, against 14–19 s with the sideways camera |
| 2 m / 40 s | collected: 1 + (1 − t/T); missed: progress | 3.5/10, 1 food | 2.7/10, 0 food | Ignores vision. At 2 m the food reads only 0.11 (centre) |
| 2 m / 20 s | time left only (1 − t/T, else 0) | 0/5 | 0/5 | None of 2,000 candidates ever collected a food, so every score was 0 and nothing was learned |
| 2 m / 20 s | collected: 1 + (1 − t/T); progress of the **first missed food only**, in clockwise order | 2.9/10, 2 food (0° in 11.0 s, +72° in 11.3 s) | 0.9/10, 0 food | First food at generation 23. Tracks the forward food visually (in view 91%), but only curves left and never sees the foods on the right |
| 2 m / 20 s | as above + view term: (time the food is in view + time left after collection) / T, weight 1 | 4.1/15, 1 food (−72°) | 2.6/15, 1 food | First food at generation 6. Every food is in view 20–47% of the time: it learned to scan or circle. The view term is about 2.3 of the 4.1 points |
| 2 m / 20 s | **food's share of the camera image**, averaged over T (time left after collection = 1), max 1 per subtask | 0.06/5 | 0.05/5 | Stuck standing still in front of the forward food (0.04) from generation 0. Walking loses that view, and the coverage of distant food is too small to reward |
| 2 m / 20 s | ordered progress (as above) **+ coverage** (weight 1), max 3 per subtask | 3.3/15, 2 food (0° in 13.6 s, −72° in 16.7 s) | 1.3/15, 1 food (−36°) | First food at generation 11. Collects the first two foods in the clockwise order, so it curves right instead of left. The three foods behind and to the left are never seen. No clear gain over ordered progress alone |
| 2 m / 20 s | ordered progress + coverage, **vision normalised to direction** (`--vision-norm sum`) | **5.6/15, 3 food** (0° in 12.3 s, −72° in 11.3 s, −144° in 15.1 s) | **3.9/15, 2 food** (−36°, −108°) | Best so far. It turns clockwise, sees each food in turn (−72° at 1.7 s, −144° at 5.5 s, +144° at 13.1 s, +72° at 16.8 s) and homes in once one is in view (in view 53–94% for the collected ones). The left-side foods come into view too late to reach. Generalises better than before (2 held-out food). Still improving at generation 100 (new best 5.6 in the last generation) |
| 2 m / 20 s, **300 generations** | same | **10.3/15, 5 food** (7.8, 8.4, 8.8, 11.3, 14.9 s) | **9.8/15, 5 food** (8.3–16.2 s) | Solves the task, including every held-out angle. Best so far 3.1 at generation 50, 5.6 at 100, 7.9 at 150, 9.0 at 200, 10.3 at 297, still rising; population mean 8.7 at the end. The collection times grow in its clockwise turn order; the slowest held-out food is +36°, just left of straight ahead (16.2 s). 4.6 min on 40 workers |

What a gecko standing still scores under the coverage fitness: food 2 m ahead 0.04, 1 m 0.17, 0.5 m 0.80,
behind it 0. The coverage is nearly saturated by 0.5 m.

Vision signal (centre fraction, food straight ahead): 0.5 m → 0.75, 1 m → 0.44, 2 m → 0.11.

### Transfer to the original food task

`eval_food_dann_chain.py` replays the 300-generation controller, unchanged, on the original task (the
sweep's `gecko_food_skills` episode): one 120 s episode with a chain of 10 foods shown one at a time. The
first food is 1.5 m straight ahead and each next one is 0.5–1 m away at a random angle, appearing as soon as
the previous one is collected. That is shorter distances than training, foods appearing mid-gait, and food
directly behind it, none of which it trained on.

| 100 random chains (seeds 1000–1099) | Food (of 10) | Fitness (food + d_norm) |
|---|---|---|
| Vision | **9.99** (99 chains complete, the other 9) | 10.98 |
| Vision zeroed (blind control) | 0 | 0 |

- A full chain takes 57–105 s (median 81 s); the median food takes 7.8 s and 90% take under 13 s.
- The one miss (seed 1035) ran out of time at 117.7 s on its 10th food.
- Blind, it never gets within 1 m of a food: everything it does depends on the camera. Zeroed vision is
  what it sees when no food is in view, so blind it just searches.
- It reaches every food by the same behaviour as in training: it turns clockwise until the food enters
  the camera view, then homes in on it.

Videos (in `food_dann_videos/chain/`, each a full chain from start to finish; tracking view, the robot's
camera inset, foods collected, time and the vision inputs overlaid):

| Video | Food | Collection times (s) |
|---|---|---|
| `chain_seed1000.mp4` | 10/10 | 6.5, 15.4, 19.0, 21.8, 31.3, 44.0, 52.1, 55.2, 65.4, 77.1 |
| `chain_seed1001.mp4` | 10/10 | 6.5, 19.0, 22.3, 32.2, 34.9, 37.1, 47.6, 51.3, 54.6, 67.5 |
| `chain_seed1002.mp4` | 10/10 | 6.5, 19.8, 32.5, 46.2, 53.3, 63.7, 67.1, 75.3, 86.3, 88.6 |

This is a single training seed, and it is not directly comparable with the sweep's SkillController food
results: those used the sideways camera, green bricks and evolved bodies.

To reproduce (about 2 minutes on 40 workers):

```bash
cd examples/symmetry_pressure
MUJOCO_GL=egl ../../.venv/bin/python eval_food_dann_chain.py food_dann_policy \
  --episodes 100 --blind --workers 40 --video 3 --out ../../food_dann_videos/chain
```

### Tuning for training speed (set up, not run yet)

Goal: cut the training needed to collect every food, as a step towards running the food task inside body
evolution (about 3 days per sweep run at the current 300 generations per body).

- **"Collects every food"** is a fixed probe, the same whatever the training setup: 8 foods at 2 m, halfway
  between the eight compass directions (22.5°, 67.5°, ...), 30 s each. The 300-generation policy collects
  all 8 at 30 s, but only 7 at 20 s: its clockwise sweep reaches the +67.5° food after 21 s.
- **`gecko_food_dann.py`** now takes `--speed-weight`, `--progress-weight`, `--progress-mode first|all`,
  `--sigma` and `--hidden`; the defaults reproduce the setup above. Whenever the best score improves, the
  new best is probed (not counted as training compute). Stopping rules: `--stop-on-solve`,
  `--max-sim-seconds` (training compute budget), **`--hopeless-gen`** (no candidate has collected a
  training food by then) and `--stall-gens` (no new best for that long). `summary.json` records why it
  stopped and when the probe was solved (generation, simulated seconds, wall time).
- **`tune_food_dann.py`**: an Optuna TPE search over the number of training foods (2–8), the ring's
  rotation, their distance (1–2 m), the subtask time (15–30 s), the speed, progress and coverage weights, the progress mode, the
  hidden size (8/16/32), the optimiser (CMA-ES or sep-CMA-ES), the population (10–40) and sigma (0.1–1).
  The collection bonus stays 1: the optimisers rank candidates, so only weight ratios matter.
  - Objective: mean over seeds (default 2) of the simulated seconds of training until the probe is solved.
    Unsolved runs score budget × (2 − probe foods / 8), so they always rank below solved ones.
  - Default budget: 8e5 simulated seconds per run. The 300-generation run used at most ~6.9e5.
  - Defaults: `--hopeless-gen 60` (the reference run's first food came at generation 16) and
    `--stall-gens 150`. `--report` prints the best trials and parameter importance, writes `trials.csv`,
    and gives the latest first-food generation among solved runs, which bounds a safe hopeless cut-off for
    body evolution.
  - `--enqueue-baseline` queues the current defaults as the reference trial.
  - **Ring rotation** (`ring_offset_frac`, trainer `--ring-offset` in degrees): the first food sits
    ring_offset_frac × (360 / foods) degrees from straight ahead (positive = left), the rest clockwise.
    0 puts a food straight ahead (the setup so far); ±0.5 puts straight ahead halfway between two foods,
    with the first food in the progress order on the right (−0.5) or the left (+0.5).
  - **`--search a,b --grid`** varies only those parameters, keeping the rest at the defaults, and runs
    every combination. `--search subtasks,ring_offset_frac --grid` answers how many foods and where:
    2, 3, 4, 5, 6 or 8 foods × offsets −0.5, −0.25, 0, 0.25, 0.5 (30 trials). The report groups
    trials by the searched parameters.
- **`slurm/run_tune_food_dann.sh`**: array 0-19, 40 CPUs, 10 trials per task = 200 trials, 5 seeds per
  trial (seeds 42–46, the same for every trial) × 8 workers, all sharing one journal-file study in
  `/scratch/jed/ariel_food_dann_tune`. Measured throughput is ~64 simulated seconds per CPU-second (the same
  for hidden 16 and 32), so a trial takes ~15–26 min and a task ~2.5–4.5 h (limit 7 h). The whole search
  is about 2,000–3,500 CPU-hours.

## Lessons so far

- **Sparse fitness does not get started from scratch.** Time-left-only and coverage-only fitness both give
  random gaits no ranking to learn from at 2 m. Some progress signal is needed until the gecko can walk.
- **A progress term on its own makes evolution ignore vision.** A fixed search route fitted to the training
  positions scores well, and the deterministic subtasks make that easy to fit.
- **Once vision is normalised and it trains long enough, the deterministic subtasks generalise.** Five fixed
  food positions were enough to learn a general turn-until-seen, walk-at-it strategy that solves random
  chains it never trained on. Randomising the training foods was not needed.
- **The ordered-progress fitness (credit only the first missed food) worked best,** and adding a view term
  made the gecko look in every direction.
- **The weak vision signal at 2 m was a real limit:** raw, the food reads only 0.11 straight ahead.
  Normalising it to direction only gave the biggest single improvement.

## Next steps (proposed, not done)

1. ~~Return to ordered progress plus a camera term~~: done with coverage (weight 1), giving 2 foods, the same
   as without it.
2. ~~Normalise the vision to direction only~~: done (`--vision-norm sum`, now the default). It took the
   gecko from 2 to 3 training foods and from 1 to 2 held-out foods, and it now scans clockwise and homes in.
3. ~~Run 200+ generations~~: done (300). All 5 training and all 5 held-out foods, so the route fitting is
   no longer the limit.
4. If the seeds disagree, to stop the route fitting: randomly rotate the food ring each generation (shared by all candidates), or
   train on both rings (10 subtasks).
5. Then 10 seeds on the cluster (Slurm default is now 300 generations; needs a commit and push first).
6. Run `eval_food_dann_chain.py` on every seed's policy to check that the transfer holds across seeds.

## Files and how to run

- `gecko_food_dann.py`: the trainer. Options include `--optimizer cma|sepcma|oes`, `--pop`, `--subtasks`,
  `--food-radius`, `--subtask-time` and `--gens`. Output goes to `__data__/gecko_food_dann/` in the working
  directory: `run_config.json`, `learning_curve.jsonl`, `best_weights.npy`, `checkpoints/`, and
  `final_eval.json` (training and held-out subtasks, with trajectories and vision).
- `render_food_dann.py <run>/__data__/gecko_food_dann [--heldout] [--out dir]`: replays the controller
  through `run_subtask` (the scores match training exactly) and writes H.264 videos. Each video has a
  tracking view, the robot's camera view inset, and live vision values and scores. The latest videos are in
  `food_dann_videos/` in the repo root.
- `eval_food_dann_chain.py <run>/__data__/gecko_food_dann [--episodes 100] [--blind] [--video 3]`: the
  original random-chain food task, with the same metrics as the sweep (food, d_norm).
- `slurm/run_gecko_food_dann.sh`: array 0-9 (seeds 42–51). Settings via `OPTIMIZER=`, `POP=`, `SUBTASKS=`,
  `SUBTASK_TIME=` and `GENS=`. Results go to `/scratch/jed/ariel_gecko_food_dann`. About 1.5–3 s per
  generation on 20 CPUs.
- `food_dann_policy/`: the 300-generation controller (seed 42) behind the last results row and the transfer
  results: `best_weights.npy`, `run_config.json`, `learning_curve.jsonl`, `final_eval.json`. Pass the
  directory to `render_food_dann.py` or `eval_food_dann_chain.py`.
- `tune_food_dann.py --study-dir <dir> [--trials --jobs --seeds-per-trial --workers-per-run] [--report]`
  and `slurm/run_tune_food_dann.sh`: the speed tuning above.
- Brain changes: `brains.ZooDistributedAnnV2(extra_inputs=)` and `shared.DannBrain(extra_inputs=, topology=, hidden=)`.

```bash
cd <scratch dir>
MUJOCO_GL=egl python examples/symmetry_pressure/gecko_food_dann.py --gens 200 --workers 20 --seed 42
MUJOCO_GL=egl python examples/symmetry_pressure/render_food_dann.py __data__/gecko_food_dann --out videos/
```

The run directories for the results above were in the session scratchpad and are not kept, except the
300-generation run, saved in `food_dann_policy/`.
