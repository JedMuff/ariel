# Symmetry paper: code and experiments

Code for the locomotion-only symmetry study (plan of Nov 5, 2026): five body genomes × four controllers × three learning budgets, evolved for forward locomotion (Experiment 1), the same genomes' random bodies trained without evolution (Experiment 2), and analysis in the morphological, behavioural and learning spaces. Experiment 3 reuses the zoo benchmark in `__data__/ariel_zoo_benchmark` and needs no code here.

The v1/v2 paper code (E0 actuator grid, coordination ablation, matched bodies, phase shift, head mass, LaTeX draft) is frozen in `legacy/` (tag `symmetry-paper-v2-final`), with its Slurm scripts in `slurm/symmetry_paper/legacy/`.

## Protocol

All settings live in `config.py` (`PROTOCOL`), and every run stores them in its `config.json`.

| | |
| --- | --- |
| Genomes | `tree` (Standard), `tree_symmetric` (Symmetric), `tree_tied` (Tied-matched, tie probability 0.47), `tree_tied_full` (Tied-full, 1.0), `grn` (Indirect: R2 GRN of Miras and Feenstra 2026) |
| Bodies | ≤ 25 modules besides the core, ≥ 4 hinges, depth ≤ 25, enforced for every genome by `genomes.is_valid`; random bodies and children are redrawn until valid |
| Controllers | `bang_bang`, `dann`, `ann`, `cpg` (`brains.py`) |
| Learning | CMA-ES, population 20, sigma 0.5, fixed x0; 100 / 200 / 800 evaluations per body |
| Evolution | (20 + 20), 50 generations, tournament of 4, crossover on 60% of children, mutation on all; every survivor is retrained with a fresh seed each generation |
| Simulation | zoo modules and hinge, flat floor, 3 s settle + 15 s episode, control at 20 Hz |
| Fitness | forward displacement of the core (m, world −Y), unclipped, no penalties |
| Main score | mean score of a run's final population (fixed before the runs) |

## Code

| file | what it does |
| --- | --- |
| `config.py` | the protocol, genome / brain / budget lists |
| `sim.py` | zoo bodies, the episode, CMA-ES training on a worker pool shared by concurrent trainings |
| `brains.py` | the four controllers |
| `genomes.py` | the five genomes behind one interface; validity (the module cap) and offspring |
| `evolve.py` | Experiment 1: one run (resumable) |
| `sample_bodies.py`, `train_fixed_body.py` | Experiment 2: random bodies; one brain on one body for every seed |
| `runs.py` | reading runs: main score, curves, episodes; paired statistics |
| `bodies.py` | body descriptors and rest-pose measures |
| `replay.py`, `gait.py`, `mechanics.py` | exact replay and gait metrics (gait class, phase quality, decomposition, contacts, effort) |
| `summarise.py` | per-run and per-condition tables, paired contrasts |
| `conditions.py`, `benchmark.py` | run lists for Slurm; cost per evaluation on a node type |
| `tools/grn_reference.py` | reference fixtures from the original GRN code (for `tests/test_grn.py`) |
| `tests/` | `pytest examples/symmetry_paper` (core only; `legacy/` is not collected) |

The GRN port itself is in `src/ariel/ec/genotypes/grn/` (development, mapping to ARIEL modules, operators).

## Output layout

An Experiment 1 run (`evolve.py --out DIR`):

```
config.json          protocol, arguments, git commit
evaluations.jsonl    one line per training: gen, id, parents, op, score, curve, body measures (genome on first training)
generations.jsonl    per generation: final population ids and scores, offspring fallbacks, wall time
weights/genGGG.npz   best weights of every training of generation GGG, keyed by body id
state.pkl, DONE      resume state; marker of a finished run
```

An Experiment 2 training (`train_fixed_body.py`) is `<body_id>__<brain>/` with `config.json`, `results.jsonl` (one line per seed) and `weights.npz`.

## Running on Snellius

From a checkout of this branch with `uv sync` done, on the login node:

```bash
export OUT_ROOT=/projects/0/<project>/symmetry_paper    # default /scratch-shared/$USER/symmetry_paper (purged after 14 days)
S=slurm/symmetry_paper/submit.sh

bash $S bench rome genoa      # 1. cost per evaluation on each node type (benchmark.py)
bash $S pilot                 # 2. one seed per condition (60 runs)
bash $S e1                    # 3. all 1,200 runs (or: bash $S e1 800)
bash $S bodies                # 4. Experiment 2 bodies (80 per genome), then
bash $S e2                    #    1,600 body x brain jobs, 20 seeds each
bash $S replay $OUT_ROOT/e1   # 5. gait measures of finished runs
```

`DRY_RUN=1` prints the `sbatch` commands instead. Rerunning `e1` submits only runs without `DONE`; runs stopped by the time limit resume from their last generation. Set `PARTITION`, `CPUS`, `ACCOUNT`, `THROTTLE`, `MAX_ARRAY` and `TIME_E100` / `TIME_E200` / `TIME_E800` to match the cluster's limits (see `submit.sh`).

Analysis is cheap and runs anywhere:

```bash
python examples/symmetry_paper/summarise.py --e1 $OUT_ROOT/e1 --e2 $OUT_ROOT/e2 --bodies $OUT_ROOT/e2/bodies.jsonl -o $OUT_ROOT/analysis
```

## Notes

- **Determinism.** A run is deterministic on one machine type: resumed runs reproduce uninterrupted ones, and replays reproduce stored scores exactly. Replay on the node type that trained a run.
- **Early CMA-ES stops.** CMA-ES stops when its own criteria say so (e.g. every candidate scores the same); the training is flagged `stopped_early`.
- **GRN bodies.** Only ~15% of random R2 genomes grow ≥ 4 hinges (39% grow only a head), so valid random GRN bodies are much larger (~20 modules) than unfiltered ones (~7).
- **Do not run plain `ruff check`** here. The repo's `ruff.toml` has `fix = true` and `unsafe-fixes = true`, so it rewrites files and deletes every `print`. Use `ruff check --no-fix`.
