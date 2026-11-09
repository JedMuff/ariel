# Social learning in morphological evolution: data and videos

Evolutionary records from a morphology-evolution experiment in two simulators:
ARIEL (MuJoCo, "zoo" module preset, tree-structured modular bodies) and EvoGym
(2-D soft voxel robots, 5×5 voxel bodies). In both, an outer (μ+λ) evolutionary
algorithm evolves robot bodies, and each new body learns a controller with CMA-ES
(the inner loop). The experimental factor is how the inner loop's starting
controller is chosen (the *inheritance scheme*). For example, it can be a random
draw, the parent's learned controller, or controllers taken from other members of
the population.

These are the runs behind the paper's figures (reps 0–9 of every condition).

## Contents

Each simulator has four zips, named `ariel_*` or `evogym_*`. Download only the zips
you need.

| Zip | Size | Contents |
|---|---|---|
| `ariel_1_core.zip` | 121 MB | every individual, its parent and donors, fitness, morphology, and the per-generation selection outcomes; plus `derived/`, the numbers the figures plot |
| `ariel_2_learning_curves.zip` | 1.2 GB | each individual's inner-loop (CMA-ES) learning curve |
| `ariel_3_raw_databases.zip` | 11 GB | the original SQLite databases of every run, including the learned and inherited controller weights |
| `ariel_4_videos.zip` | 331 MB | 360 videos of evolved robots |
| `evogym_1_core.zip` | 107 MB | every individual, its parent and donors, fitness, morphology; per-generation pool summaries; plus `derived/`, the numbers the figures plot |
| `evogym_2_learning_curves.zip` | 1.3 GB | each individual's inner-loop (CMA-ES) learning curve |
| `evogym_3_raw_databases.zip` | 19.5 GB | the original SQLite databases of every run, including the controller weights |
| `evogym_4_videos.zip` | 580 MB | 360 videos of evolved robots |

The zips have no enclosing folder and several contain a `runs.csv`, so unzip each
into its own folder. The data zips of a simulator share the key `run_id` +
`individual_id`, and each can be used on its own. `individual_id` is unique only
within a run. In all CSV files, empty cells mean "not applicable".

## Experimental design

| Factor | Levels |
|---|---|
| `scheme` | `Darwinian`, `Lamarckian`, `Random`, `Random-Many`, `Best`, `Best-Many`, `Similar`, `Similar-Many` |
| `x` (folder `x00` / `x05` / `x10`) | selection weight on fitness: 0 (novelty only), 0.5 (half fitness, half novelty) or 1.0 (fitness only) |
| `rep` | 0–9 |

Inheritance schemes (initial mean of the inner CMA-ES), named as in the paper:

- `Darwinian`: random initial weights (no inheritance). In ARIEL, uniform in [-1, 1].
- `Lamarckian`: the parent's learned controller.
- `Random`: the learned controller of one random population member. `Random-Many` averages k random members.
- `Best`: the controller of the population's fittest member. `Best-Many` averages the k fittest.
- `Similar`: the controller of the morphologically most similar member. `Similar-Many` averages the k most similar.

| | ARIEL | EvoGym |
|---|---|---|
| Runs | 400 = 8 schemes × 5 (budget, x) conditions × 10 reps | 240 = 8 schemes × 3 x values × 10 reps |
| Inner-loop budget | 10 or 20 CMA-ES generations of population 20 (folders `ig10`, `ig20`); x = 0 was run for `ig10` only | 10 CMA-ES generations of population 10 (100 episodes per individual) |
| μ, λ | 20, 20 | 100, 100 |
| k (`-Many` schemes) | 5 | 8 |
| Morphological distance (`Similar`, novelty) | exact top-down tree edit distance | aligned Hamming distance between voxel grids: the minimum number of differing cells over all relative placements of the two bodies |

Settings shared by all runs: elitist (μ+λ) survivor selection, 60 generations,
inner CMA-ES with σ = 0.45, distributed MLP controller with 16 hidden units, and
k-NN novelty within the selection pool.

**Selection.** Selection ranks on a pool-normalised blend,
`selection_score = x · norm(fitness) + (1 − x) · norm(novelty)`, where both terms
are min-max normalised within the generation's selection pool.

**ARIEL fitness.** After learning, the best controller is re-evaluated on 10
episodes from jittered start states. `fitness` is the mean of those 10 episodes.
Each episode scores the forward displacement of the core over 15 s, minus a jerk
penalty when the mean control jerk exceeds a threshold. The x = 0.5 and x = 1.0
runs executed in two parts: generations 0–30, then a resume for generations 31–60
from the saved population. The two parts are already merged in the core and
learning-curve zips. The x = 0 runs ran in one part.

**EvoGym task and fitness.** Walker on a long flat world (`walker_long`), 499
simulation steps per episode. Fitness (`fitness`) is the forward displacement of
the robot's centre of mass over the episode, in voxels, taken from CMA-ES's best
episode. EvoGym is deterministic for a fixed controller, so there are no repeated
final evaluations.

## Videos (`*_4_videos.zip`)

Each zip holds 360 videos, 15 per scheme and x. For each condition, the rep with
the highest final displacement is used, and the best, median and worst survivors
at generations 0, 15, 30, 45 and 60 are filmed, as
`<Scheme>_<xNN>/g<GG>_<best|median|worst>_id<individual_id>.mp4`. The ARIEL videos
use the budget of 10 inner generations.

An info panel in each video shows the scheme, x, rep, generation, the individual's
stored displacement, novelty, selection score and head start, and a bar along the
bottom shows the displacement so far.

- **ARIEL:** the panel's displacement is the mean of 10 evaluations. The video and
  its live bar show a single episode, re-simulated on a different machine from the
  original run, so the bar can end at a somewhat different value (within 0.34 m,
  and usually much closer).
- **EvoGym:** each video is a single 499-step episode with the individual's learned
  controller. EvoGym is deterministic, so the live bar ends at the displacement
  shown in the panel. Displacements are in voxels.

---

# ARIEL files

## `ariel_1_core.zip`

```
runs.csv                                       one row per run
derived/                                       figure inputs (see below)
data/<ig10|ig20>/<scheme>/<x00|x05|x10>/
    individuals.csv.gz                         one row per individual (10 runs per file)
    selection_history.csv.gz                   one row per (generation, pool member)
```

The key is `run_id` (`<ig>_<scheme>_<x>_rep<NN>`, e.g. `ig20_Best-Many_x10_rep03`).

### `runs.csv`

`run_id, scheme, x, inner_gens, rep, k_inherit, generations, mu, lambda, selection,
inner_pop, cma_sigma, mlp_hidden, final_evals, novelty_metric, sim_preset, seed,
seed_resume, started, started_resume, n_individuals, last_generation`

### `individuals.csv.gz`

Each run has 1220 individuals: 20 initial (generation 0) + 20 offspring × 60 generations.

| Column | Meaning |
|---|---|
| `run_id`, `individual_id` | identifiers |
| `birth_generation` | generation the individual was created (0 = initial population) |
| `death_generation` | generation whose selection removed it. Empty if it was alive at the end |
| `alive_at_end` | 1 if it is in the final population (20 per run) |
| `parent_id` | `individual_id` of the parent whose morphology was mutated. Empty for the initial population |
| `mutation_op`, `mutation_rotation`, `mutation_n_pruned` | tree mutation applied to the parent: `point`, `subtree`, `hoist` or `shrink`; whether a module rotation was changed; number of nodes pruned |
| `donor_ids` | `;`-separated `individual_id`s whose learned controllers seeded this individual's learning. Empty for `Darwinian` and the initial population |
| `fitness` | mean fitness over the 10 final evaluation episodes (the value selection uses) |
| `fitness_cma_best` | best single-episode fitness CMA-ES found during learning |
| `displacement` | mean displacement over the 10 final episodes, without the jerk penalty |
| `init_fitness` | fitness of the inherited (pre-learning) controller in one episode |
| `mean_jerk`, `hinge_contacts` | mean control jerk and count of hinge ground contacts in CMA-ES's best episode |
| `final_eval_fitness_1..10`, `final_eval_displacement_1..10` | per-episode values behind `fitness` / `displacement` |
| `novelty_last`, `selection_score_last` | novelty and blended selection score at the individual's last selection step. `selection_history` has the values for every generation |
| `B_branching` … `P_footprint` | 8-d morphological descriptor in [0, 1]: branching, limbs, limb length, coverage, joints, symmetry, module diversity, footprint proportion |
| `eval_seed` | seed of this individual's inner learning run |
| `morphology` | the body genome as JSON: `nodes` (`id → {type: CORE/BRICK/HINGE, rotation}`) and `edges` (`{parent, child, face}`) |

### `selection_history.csv.gz`

At generation g (1–60), the selection pool is the 20 survivors of generation
g−1 plus the 20 offspring born in generation g. There is one row per pool
member: 40 per generation, 2400 per run.

| Column | Meaning |
|---|---|
| `run_id`, `generation`, `individual_id` | |
| `novelty` | k-NN novelty (top-down tree edit distance) within that pool |
| `selection_score` | the blended, pool-normalised score that selection ranked on |
| `survived` | 1 if selected into the next population (20 per generation) |

### `derived/`

The per-run summaries the paper's figures and statistics were computed from
(`final_results_data.py` and `final_results_figures.py` in the code repository).
One file per budget (`ariel_ig10`, `ariel_ig20`).

- `curves_<budget>.csv`, one row per (scheme, x, rep, generation): `surv_disp` (mean
  displacement of the survivors), `surv_best_disp`, `best_so_far_disp`, `born_init_disp`
  (offspring displacement with the inherited controller, before learning: the "head
  start"), `born_learn_gain`, `pool_novelty`, `surv_diversity` (mean pairwise tree edit
  distance among survivors).
- `donor_dist_<budget>.csv`, one row per offspring that had donors: `donor_dist` (mean
  tree edit distance to its donors), `init_fitness`, `disp`.
- `stats.csv`: the statistics the text quotes (Vargha–Delaney A12, Holm-corrected
  p-values, etc.), one row per comparison; `table` names the figure or table it belongs to.

## `ariel_2_learning_curves.zip`

```
runs.csv                                       one row per run (same as in ariel_1_core.zip)
data/<ig10|ig20>/<scheme>/<x00|x05|x10>/
    learning_curves.csv.gz                     one row per individual (10 runs per file)
```

| Column | Meaning |
|---|---|
| `run_id`, `individual_id` | identifiers, as in `individuals.csv.gz` |
| `learning_curve` | JSON list with one entry per CMA-ES generation (10 for `ig10`, 20 for `ig20`). Each entry is the list of single-episode fitnesses of that generation's 20 candidates, in sampling order |

Reading a curve in Python:

```python
import json, pandas as pd
lc = pd.read_csv("data/ig10/Best/x10/learning_curves.csv.gz")
curve = json.loads(lc.learning_curve[0])            # [[f, f, ...], ...]
best_per_gen = [max(g) for g in curve]
```

## `ariel_3_raw_databases.zip`

The original SQLite databases written by `examples/d_social_learning/experiment.py`.
They hold everything the experiment recorded, including the learned and inherited
controller weights. The core and learning-curve zips were exported from these files.

```
runs.csv                                       one row per run (same as in ariel_1_core.zip)
data/<ig10|ig20>/<scheme>/<x00|x05|x10>/rep_<NN>/
    run_config.json                            seed, command-line arguments
    database.db                                generations 0–30 (all of a run for x00)
    database_part2.db                          the resume, generations 31–60 (x05 and x10 runs)
```

`rep_<NN>` maps to `run_id` `<ig>_<scheme>_<x>_rep<NN>` in the other zips.
For runs in two parts, an individual that appears in both files is taken from
`database_part2.db` (its later state).

**Table `individual`**: one row per individual.

| Column | Meaning |
|---|---|
| `id` | `individual_id` |
| `alive`, `time_of_birth`, `time_of_death` | state at the end of that part |
| `fitness_` | the blended, pool-normalised selection score at its last selection |
| `genotype_` | JSON: `morph` (the body tree: `nodes` and `edges`) and `brain` |
| `tags_` | JSON with everything else (below) |

Keys in `tags_`:

| Key | Meaning |
|---|---|
| `theta` | learned controller weights (the best CMA-ES solution) |
| `init_theta` | the starting mean of CMA-ES, i.e. the inherited controller |
| `learning_curve` | per CMA-ES generation, the episode fitness of every candidate |
| `distance`, `displacement`, `final_eval_fitness`, `final_eval_displacement` | final-evaluation results (`distance` is `fitness` in the core zip) |
| `distance_cma`, `init_fitness`, `mean_jerk`, `c_hinge` | as `fitness_cma_best`, `init_fitness`, `mean_jerk`, `hinge_contacts` in the core zip |
| `parent_id`, `mutation`, `donor_ids`, `descriptor`, `eval_seed`, `novelty`, `novelty_history` | lineage, tree mutation, donors, 8-d descriptor, inner-loop seed, novelty |

**Table `novelty_history`**: one row per (generation, pool member): `generation`,
`individual_id`, `novelty`, `fitness` (the selection score), `survivor`.
This is `selection_history.csv.gz` in the core zip.

Controllers are distributed MLPs with 16 hidden units. The code that rebuilds
them from `theta` is in the ARIEL repository (`examples/d_social_learning`).

---

# EvoGym files

## `evogym_1_core.zip`

```
runs.csv                                       one row per run
generation_summary.csv                         one row per (run, generation)
body_stats.csv                                 one row per (run, generation)
derived/                                       figure inputs (see below)
data/<scheme>/<x00|x05|x10>/
    individuals.csv.gz                         one row per individual (10 runs per file)
```

The key is `run_id` (`<scheme>_<x>_rep<NN>`, e.g. `Best-Many_x05_rep03`).

### `runs.csv`

`run_id, scheme, x, rep, inner_gens, inner_pop, k_inherit, generations, mu, lambda,
selection, cma_sigma, mlp_hidden, novelty_metric, env, episode_steps, n_individuals, last_generation`

### `individuals.csv.gz`

Each run has 6100 individuals: 100 initial (generation 0) + 100 offspring × 60 generations.

| Column | Meaning |
|---|---|
| `run_id`, `individual_id` | identifiers |
| `birth_generation` | generation the individual was created (0 = initial population) |
| `death_generation` | generation whose selection removed it. Empty if it was alive at the end |
| `alive_at_end` | 1 if it is in the final population (100 per run) |
| `parent_id` | `individual_id` of the parent whose body was mutated. Empty for the initial population |
| `donor_ids` | `;`-separated `individual_id`s whose learned controllers seeded this individual's learning. Empty for `Darwinian` and the initial population |
| `fitness` | displacement in CMA-ES's best episode |
| `init_fitness` | displacement of the inherited (pre-learning) controller in one episode |
| `novelty_last`, `selection_score_last` | novelty and blended selection score at the individual's last selection step |
| `relative_activity`, `size`, `compactness`, `elongation`, `symmetry` | 5-d morphological descriptor in [0, 1]: share of actuator voxels, share of the grid filled, compactness, elongation, symmetry |
| `morphology` | the 5×5 voxel grid as JSON (rows top to bottom). Materials: 0 empty, 1 rigid, 2 soft, 3 horizontal actuator, 4 vertical actuator |

### `generation_summary.csv`

One row per (scheme, x, rep, gen), computed over the generation's selection pool
(100 individuals at generation 0, then 200 = μ + λ): `n_individuals`,
`best_distance` and `mean_distance` (fitness), `mean_novelty` (novelty recomputed
within that pool; `novelty_last` above holds only each individual's last value),
`init_distance` (mean `init_fitness`).

### `body_stats.csv`

One row per (scheme, x, rep, gen) over the same pool: the mean of each of the
5 descriptors.

### `derived/`

The per-run summaries the paper's figures and statistics were computed from
(`final_results_data.py` and `final_results_figures.py` in the code repository).

- `curves_evogym.csv`, one row per (scheme, x, rep, generation): `surv_disp` (mean
  fitness of the survivors), `surv_best_disp`, `best_so_far_disp`, `born_init_disp`
  (offspring fitness with the inherited controller, before learning: the "head
  start"), `born_learn_gain`, `surv_diversity` (mean pairwise aligned Hamming distance
  among survivors), `pool_novelty`.
- `donor_dist_evogym.csv`, one row per offspring that had donors: `donor_dist` (mean
  aligned Hamming distance to its donors), `init_fitness`, `disp`.
- `stats.csv`: the statistics the text quotes (Vargha–Delaney A12, Holm-corrected
  p-values, etc.), one row per comparison; `table` names the figure or table it belongs to.

## `evogym_2_learning_curves.zip`

```
runs.csv                                       one row per run
data/<scheme>/<x00|x05|x10>/
    learning_curves.csv.gz                     one row per individual (10 runs per file)
```

| Column | Meaning |
|---|---|
| `run_id`, `individual_id` | identifiers, as in `individuals.csv.gz` |
| `learning_curve` | JSON list with one entry per CMA-ES generation (10). Each entry is the list of single-episode fitnesses (displacements) of that generation's 10 candidates, in sampling order |

Reading a curve in Python:

```python
import json, pandas as pd
lc = pd.read_csv("data/Best/x10/learning_curves.csv.gz")
curve = json.loads(lc.learning_curve[0])            # [[f, f, ...], ...]
best_per_gen = [max(g) for g in curve]
```

## `evogym_3_raw_databases.zip`

The original SQLite databases written by `examples/d_social_learning/experiment.py`
(`--platform evogym`). They hold everything the experiment recorded, including the
controller weights. The core and learning-curve zips were exported from these files.

```
runs.csv                                       one row per run
generation_summary.csv, body_stats.csv         per-generation pool summaries (as in evogym_1_core.zip)
data/<scheme>/<x00|x05|x10>/rep_<NN>/
    database.db
```

`rep_<NN>` maps to `run_id` `<scheme>_<x>_rep<NN>` in the other zips.

**Table `individual`**: one row per individual (6100 per run).

| Column | Meaning |
|---|---|
| `id` | `individual_id` |
| `alive`, `time_of_birth`, `time_of_death` | `time_of_death` is the generation whose selection removed it (60 for those alive at the end) |
| `fitness_` | the blended, pool-normalised selection score at its last selection |
| `genotype_` | JSON: `morph` (the 5×5 voxel grid) and `brain` |
| `tags_` | JSON with everything else (below) |

Keys in `tags_`:

| Key | Meaning |
|---|---|
| `theta` | learned controller weights (the best CMA-ES solution, 1201 values) |
| `learning_curve` | per CMA-ES generation, the episode fitness of every candidate |
| `distance` | `fitness` in the core zip |
| `init_fitness`, `parent_id`, `donor_ids`, `descriptor`, `novelty` | as in the core zip (`descriptor` is the 5 descriptor columns, in the same order) |

Controllers are distributed MLPs with 16 hidden units. The code that rebuilds
them from `theta` is in the ARIEL repository (`examples/d_social_learning/sim_evogym`).
