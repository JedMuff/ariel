# Symmetry paper: code and experiments

Code for the experiments in `PAPER_PLAN.md`. It is the part of the `symmetry-pressure-investigation` branch that the paper needs, pruned to forward locomotion under the `dsocial_zoo` profile. Seeded runs reproduce that branch's `run_data.jsonl` exactly, and the analysis scripts read its data, so the E0 results in `__data__/ariel_actuator_grid`, `__data__/ariel_dann_budget` and `__data__/ariel_symmetry_pressure_sweep_dann` remain valid. `SYMMETRY_ADVANTAGE.md` records the results so far; its paths and script names refer to the investigation branch.

## Code

| file | what it does |
| --- | --- |
| `evolve.py` | body–brain evolution: (μ+λ) over bodies, inner CMA-ES over each body's controller (E0, E1) |
| `sim.py` | world, zoo modules and hinge actuator, episode and fitness, inner CMA-ES, body operators |
| `brains.py` | `bang_bang`, `bang_bang_free` (one frequency per hinge), `dann` (`dann_pos_hinge_h16`) |
| `genomes.py` | `tree`, `tree_symmetric`, `tree_tied` |
| `sample_bodies.py`, `train_fixed_body.py` | E2: hinge-matched random or evolved bodies; one brain trained on one fixed body, with per-generation curve and checkpoints |
| `replay.py`, `gait.py`, `gait_species.py` | E3: exact replay, gait class (the validated rowing detector), left–right phase relation and phase quality, gait cycles and species |
| `runs.py` | run discovery, last-generation score, paired statistics, figure style |
| `phase_shift.py`, `analyse_phase_shift.py` | E3 causal check: replay trained controllers with the left side's timing shifted or detuned, against the same number of random lateral hinges |
| `analyse_e0.py` … `analyse_e3.py` | tables and figures per experiment |
| `analyse_bodies.py` | R2: rest-pose body measures (size, height, contact spread, hinge axes, leg hinges) of the E2 bodies against training score and phase locking |
| `tests/` | `pytest examples/symmetry_paper/tests` |

Score is the best forward score of a run's last generation, in [0, 2] minus penalties. Runs pair by seed (42 + rep in every cell).

**Phase quality** (new, `gait.phase_quality`): each left–right pair's trajectories are cut into one-period windows, and the cross-spectrum lag is taken per window. Phase quality is 1 − the circular variance of those lags: 1 means the same lag every cycle, near 0 means drifting. Per episode it is the stroke-weighted mean over pairs; `phase_locked_frac` is the stroke share of pairs with quality ≥ 0.9.

## Running on the cluster

Use a separate checkout of this branch, so jobs still running from the investigation branch's checkout are not disturbed:

```bash
git clone -b symmetry-paper <remote> /home/jed/workspaces/ariel-symmetry/ariel_paper
cd /home/jed/workspaces/ariel-symmetry/ariel_paper && uv sync
```

The scripts in `slurm/symmetry_paper/` take `REPO` (default: that checkout) and `FINAL_ROOT` (default `/scratch/$USER/ariel_paper`) from the environment. Submit them from `REPO`.

| experiment | submit | jobs | then |
| --- | --- | --- | --- |
| **E0** hinge grid (already done; reproduction only) | `sbatch slurm/symmetry_paper/e0_actuator_grid.sh` | 216 | `analyse_e0.py --grid-root $FINAL_ROOT/e0_grid` |
| **E0** DANN budget series (200 / 800 / 1600 evals; done up to job 47059) | `sbatch slurm/symmetry_paper/e0_dann_budget.sh` | 30 | `analyse_e0.py --budget-roots $FINAL_ROOT/e0_budget/sb{10,40,80}` |
| **E1** coordination ablation | `sbatch slurm/symmetry_paper/e1_coordination_ablation.sh` | 60 | E3 on `e1`, then `analyse_e1.py --root $FINAL_ROOT/e1 --behaviour $FINAL_ROOT/e3/e1` |
| **E2** random bodies (can start with E1) | `sample_bodies.py random --n 40 -o $FINAL_ROOT/e2/bodies_random.jsonl`, then `BODY_SET=random sbatch --array=0-239 slurm/symmetry_paper/e2_fixed_body_curves.sh` | 240 | E3 on `e2/random`, then `analyse_e2.py` |
| **E2** evolved bodies (after E1) | `sample_bodies.py evolved --runs-root $FINAL_ROOT/e1 -o $FINAL_ROOT/e2/bodies_evolved.jsonl`, then `BODY_SET=evolved sbatch --array=0-119 slurm/symmetry_paper/e2_fixed_body_curves.sh` | 120 | as above |
| **E3** behaviour | `E3_ROOTS=<run root(s)> E3_NAME=<name> sbatch --array=0-<n-1> slurm/symmetry_paper/e3_behaviour.sh` (n = `replay.py --list-runs <roots> \| wc -l`) | one per run | `gait_species.py --root $FINAL_ROOT/e3/<name>`; `analyse_e3.py --roots … --training-roots …` |
| **E3** phase shift (after E3) | `PS_ROOTS=$FINAL_ROOT/e2/random PS_NAME=e2_random sbatch --array=0-23 slurm/symmetry_paper/e3_phase_shift.sh` (40 runs per task) | ceil(n / 40) | `analyse_phase_shift.py --roots $FINAL_ROOT/e3/phase_shift_e2_random …` |

All `analyse_*.py` and `gait_species.py` calls are `python examples/symmetry_paper/<script>`, cheap, and fine on a login node or locally after copying the results. Each writes CSVs and figures to `<root>/analysis/` (or `--out`).

Typical E3 batches:

```bash
E3_ROOTS=$FINAL_ROOT/e1 E3_NAME=e1 sbatch --array=0-59 slurm/symmetry_paper/e3_behaviour.sh
E3_ROOTS="$FINAL_ROOT/e0_budget/sb10 $FINAL_ROOT/e0_budget/sb40 $FINAL_ROOT/e0_budget/sb80" E3_NAME=budget \
    sbatch --array=0-29 slurm/symmetry_paper/e3_behaviour.sh
E3_ROOTS=$FINAL_ROOT/e2/random E3_NAME=e2_random sbatch --array=0-719%100 slurm/symmetry_paper/e3_behaviour.sh
```

The existing E0 data can be analysed directly, e.g. `analyse_e0.py --grid-root __data__/ariel_actuator_grid --budget-roots __data__/ariel_symmetry_pressure_sweep_dann __data__/ariel_dann_budget/sb40 __data__/ariel_dann_budget/sb80`, and replayed with `replay.py` (old DANN sweep runs go through the neighbour-tie search automatically).

## Notes

- **Early CMA-ES stops.** The inner CMA-ES stops when a generation's fitness is flat, e.g. when every candidate scores exactly 0 (backward motion is clipped to 0). This is also true of the investigation's runs. E2 keeps the final score and checkpoint up to the full budget and flags these trainings (`stopped_early`).
- **Do not run plain `ruff check`** here. The repo's `ruff.toml` has `fix = true` and `unsafe-fixes = true`, so it rewrites files and deletes every `print`. Use `ruff check --no-fix`.
