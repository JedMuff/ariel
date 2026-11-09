# Social learning in morphological evolution

Code for the paper's experiments. An outer (μ+λ) evolutionary algorithm evolves
robot bodies, and each new body learns a controller with CMA-ES (the inner loop).
The experimental factor is how the inner loop's starting controller is chosen
(the *inheritance scheme*). Experiments run in two simulators: ARIEL (MuJoCo, 3D
modular robots) and EvoGym (2D voxel-based soft robots).

The data and videos are published separately on Zenodo.
[`publication/README.md`](publication/README.md) is that record's README, and
describes the schemes, conditions and every published file.

## Setup

ARIEL uses the repository's main environment (Python 3.12):

```bash
uv sync
```

EvoGym needs Python 3.10, in a separate venv that takes `ariel` from `src/`:

```bash
uv venv evogym-venv --python 3.10
uv pip install --python evogym-venv/bin/python evogym gymnasium "numpy<2" \
    nevergrad rich sqlmodel sqlalchemy pydantic pydantic-settings "setuptools<81" pillow
uv pip install --python evogym-venv/bin/python torch \
    --index-url https://download.pytorch.org/whl/cpu
```

Run everything from the repository root. Outputs go to `__data__/` (gitignored).

## Reproducing the paper

The run scripts are Slurm arrays; edit the `#SBATCH` lines for your cluster.

| Step | Command | Output |
|---|---|---|
| ARIEL runs, generations 0–30, x ∈ {0.5, 1} | `sbatch examples/d_social_learning/run_social_zoo_budget.sh` | `__data__/social/ariel_zoo_fe10/ig{10,20}/` |
| ARIEL runs, generations 31–60 (resumes the above) | `sbatch examples/d_social_learning/run_social_zoo_budget_extend.sh` | same run folders, `database_part2.db` |
| ARIEL runs, x = 0, 60 generations | `sbatch examples/d_social_learning/run_social_zoo_budget_x0.sh` | `__data__/social/ariel_zoo_fe10/ig10/` |
| EvoGym runs | `sbatch examples/d_social_learning/run_social_evogym_budget.sh` | `__data__/social/evogym_budget/ib{50,100}/` |
| Flat dataset for analysis | `uv run examples/d_social_learning/analysis/export_final_results.py --budgets 10 20 --x x00 x05 x10` (ARIEL) | `__data__/final_results/` |
| Figures, tables and statistics | `bash examples/d_social_learning/make_final_results_figures.sh` | `__data__/social/figures/` |
| Zenodo data zips | `bash examples/d_social_learning/make_publication_data.sh` | `__data__/publication/{ariel,evogym}/` (the Zenodo record holds these zips in one flat folder) |
| Videos | `bash examples/d_social_learning/make_stage_videos.sh` (needs `xvfb`, `tmux`) | `__data__/videos/` |

The published EvoGym data is the budget of 100 episodes (`ib100`), and uses reps 0–9.
The figure step and the data export both expect the runs under
`__data__/final_results/` (`ariel/` from `export_final_results.py`; EvoGym run
folders as `evogym/social/evogym/<scheme>/<xNN>/k8/rep_<R>_novSTRUCT/`). To work from the published data instead, unzip the core
and raw-database zips into that layout.

## Files

| Path | Contents |
|---|---|
| `experiment.py` | the outer evolutionary loop, for `--platform ariel` or `evogym` |
| `evaluate.py`, `simulator_dependent_functions.py` | evaluates one individual: inheritance, CMA-ES learning, final evaluation |
| `core/` | inheritance schemes, selection fitness, novelty and its per-generation history |
| `sim_ariel/` | ARIEL evaluator, tree genome operators, descriptors, tree edit distance |
| `sim_evogym/` | EvoGym evaluator, environment, voxel genome operators, descriptors |
| `run_social_*.sh` | the Slurm arrays behind the published runs |
| `analysis/` | `final_results_*` make the figures, tables and statistics; `export_*` make the published datasets; the other modules are shared helpers |
| `make_videos/`, `evosight_adapter.py` | selecting and rendering the published videos |
| `publication/README.md` | the Zenodo record's README |
