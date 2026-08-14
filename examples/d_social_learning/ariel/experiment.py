"""ARIEL social-learning experiment: (mu+lambda) morphology EA + CMA-ES brain learning.

Usage:
    uv run examples/d_social_learning/ariel/experiment.py \
        --scheme lamarckian --x 0.5 --rep 0 [--gens 100] [--pop 20] [--lam 100] \
        [--inner-gens 20] [--inner-pop 16] [--sigma 0.5] [--hidden 32] [--workers N] \
        [--comma-selection] [--selection elitist|tournament] [--tournament-size 4]

Each invocation without --resume-dir creates a fresh, timestamped output
directory (__data__/social/ariel/{scheme}/x{x}/rep_{rep}_{timestamp}) so
re-running the same scheme/x/rep never clobbers a previous run, and prints
that directory as `RUN_DIR=<path>` on its own stdout line. To continue a run,
pass that exact directory back in:
    uv run examples/d_social_learning/ariel/experiment.py \
        --scheme lamarckian --x 0.5 --rep 0 --gens 20 \
        --resume-dir __data__/social/ariel/lamarckian/x05/rep_0_20260813_143022
"""

import argparse
import datetime
import multiprocessing
import os
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Add local module dirs to sys.path WITHOUT adding d_social_learning/ itself,
# because that directory contains an ariel/ subdirectory that would shadow the
# installed ariel package.
_THIS_DIR = Path(__file__).parent          # d_social_learning/ariel/
_SOCIAL_DIR = _THIS_DIR.parent             # d_social_learning/
_CORE_DIR = _SOCIAL_DIR / "core"
for _p in [str(_THIS_DIR), str(_CORE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from rich.console import Console

from ariel.ec import EA, EAOperation, Individual, Population

# Local social-learning modules (resolved via _THIS_DIR on sys.path)
from descriptor import tree_descriptor       # d_social_learning/ariel/descriptor.py
from morphology_ops import mutate, random_individual
from evaluator import evaluate_individual

from fitness import combined_fitness
from inheritance import SCHEMES
from novelty import compute_novelty

console = Console()

N_NEIGHBORS = 6  # must match evaluator.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pop_state(individuals: list[Individual]) -> list[dict]:
    """Build lightweight state list for inheritance scheme functions."""
    states = []
    for ind in individuals:
        theta = ind.tags_.get("theta") if ind.tags_ else None
        fitness = ind.fitness_ if not ind.requires_eval else None
        descriptor = ind.tags_.get("descriptor") if ind.tags_ else None
        if descriptor is None:
            descriptor = [0.0] * 8
        states.append({
            "descriptor": np.array(descriptor, dtype=np.float64),
            "theta": np.array(theta, dtype=np.float64) if theta else None,
            "fitness": fitness,
            "db_id": ind.id,
        })
    return states


def _compute_descriptors(individuals: list[Individual]) -> list[np.ndarray]:
    descs = []
    for ind in individuals:
        stored = ind.tags_.get("descriptor") if ind.tags_ else None
        if stored:
            descs.append(np.array(stored, dtype=np.float64))
        else:
            try:
                d = tree_descriptor(ind.genotype_["morph"])
            except Exception:  # noqa: BLE001
                d = np.zeros(8, dtype=np.float64)
            descs.append(d)
    return descs


def _timeout_result(donor_ids: list[int]) -> dict:
    """Placeholder for an individual whose evaluation timed out.

    ``distance`` is NaN (not e.g. 0.0) so it's visibly distinguishable from a
    legitimate low-fitness individual in stored tags — see
    analysis/curve_utils.py's ``np.isfinite`` filtering, which already treats
    non-finite fitness as "simulator failure" and drops it from plots.
    """
    return {
        "distance": float("nan"),
        "best_theta": [],
        "init_fitness": float("nan"),
        "learning_curve": [],
        "donor_ids": donor_ids,
        "mean_jerk": float("nan"),
        "c_hinge": 0,
    }


def _evaluate_with_timeout(
    worker_args: list[tuple], num_workers: int, timeout_s: float,
) -> list[dict]:
    """Like ``pool.map(evaluate_individual, worker_args)``, but a task that
    doesn't return within ``timeout_s`` is replaced with a NaN result instead
    of blocking every remaining task forever.

    A plain ``pool.map`` waits on every task unconditionally, so one worker
    wedged in a hang (observed in practice: a whole generation frozen for
    16+ hours with zero CPU usage across every worker, most likely a
    native-level deadlock inside MuJoCo for some pathological morphology —
    root cause unconfirmed, but reproduced both with and without
    ``forkserver``) blocks the entire generation indefinitely. Submitting
    each task individually and polling each with its own ``.get(timeout=)``
    bounds the damage to one bad individual per generation.

    A worker that's actually hung never returns to the pool for more work,
    so the pool is unconditionally terminated at the end of this batch
    (matching ``with Pool(...) as pool:``'s terminate-on-exit semantics) —
    safe since a fresh Pool is created for every generation anyway.
    """
    pool = Pool(processes=num_workers)
    try:
        async_results = [pool.apply_async(evaluate_individual, (a,)) for a in worker_args]
        results = []
        for i, ar in enumerate(async_results):
            try:
                results.append(ar.get(timeout=timeout_s))
            except multiprocessing.TimeoutError:
                donor_ids = worker_args[i][2]
                console.log(
                    f"[red]individual {i} timed out after {timeout_s:.0f}s "
                    f"-- marking NaN and moving on[/red]"
                )
                results.append(_timeout_result(donor_ids))
        return results
    finally:
        pool.terminate()
        pool.join()


def _tournament_select(
    individuals: list[Individual], n: int, tournament_size: int,
) -> list[Individual]:
    """n survivor tournaments without replacement across tournaments (a
    winner can't compete again), each picking the fittest of
    tournament_size random contestants from what's left of `individuals`.

    Without-replacement is required here (unlike typical parent-selection
    tournaments): survivors are the literal Individual objects going on to
    become alive, so the same individual can't "win" twice.
    """
    pool = list(individuals)
    survivors = []
    for _ in range(n):
        contestants = random.sample(pool, k=min(tournament_size, len(pool)))
        winner = max(contestants, key=lambda ind: ind.fitness_)
        survivors.append(winner)
        pool.remove(winner)
    return survivors


# ---------------------------------------------------------------------------
# (mu+lambda) EA operations as EAOperation functions
# ---------------------------------------------------------------------------

def build_ops(
    scheme_name: str,
    x_val: float,
    mu: int,
    lam: int,
    inner_gens: int,
    inner_pop: int,
    num_workers: int,
    eval_timeout: float,
    sigma: float,
    hidden: int,
    comma_selection: bool = False,
    selection_method: str = "elitist",
    tournament_size: int = 4,
) -> list[EAOperation]:
    """Return the ordered list of EAOperation steps for the outer EA."""
    @EAOperation
    def generate_offspring(population: Population) -> Population:
        parents = [ind for ind in population if ind.alive and not ind.requires_eval]
        if not parents:
            parents = [ind for ind in population if ind.alive]

        offspring_list = []
        # sample parents without replacement where possible, cycling if lam > mu
        import random as _random
        parent_pool = list(parents)
        for i in range(lam):
            if i % len(parent_pool) == 0:
                _random.shuffle(parent_pool)
            parent = parent_pool[i % len(parent_pool)]
            child_morph = mutate(parent.genotype_["morph"])
            child = Individual()
            parent_brain = parent.genotype_.get("brain") or []
            child.genotype = {"morph": child_morph, "brain": parent_brain}
            child.tags = {"parent_id": parent.id}
            offspring_list.append(child)

        pop_out = Population(list(population))
        pop_out.extend(offspring_list)
        return pop_out

    @EAOperation
    def evaluate_and_select(population: Population) -> Population:
        parents = [ind for ind in population if ind.alive and not ind.requires_eval]
        offspring = [ind for ind in population if ind.alive and ind.requires_eval]
        all_alive = parents + offspring

        descs = _compute_descriptors(all_alive)
        novelties = compute_novelty(descs)

        if x_val == 0.0:
            # Pure novelty — no simulation needed
            for i, ind in enumerate(all_alive):
                novelty = float(novelties[i])
                desc = descs[i]
                prior = ind.tags_ or {}
                ind.fitness = novelty
                ind.tags = {
                    "parent_id": prior.get("parent_id"),
                    "distance": 0.0,
                    "novelty": novelty,
                    "descriptor": desc.tolist(),
                    "theta": prior.get("theta", []),
                    "init_fitness": 0.0,
                    "learning_curve": [],
                    "donor_ids": [],
                    "mean_jerk": 0.0,
                    "c_hinge": 0,
                }
        else:
            all_state = _pop_state(all_alive)
            scheme_fn = SCHEMES[scheme_name]

            from ariel.simulation.controllers.distributed_mlp import DistributedMLP
            n_params = DistributedMLP(n_neighbors=N_NEIGHBORS, hidden=hidden).n_params

            worker_args = []
            for i, ind in enumerate(all_alive):
                init_mean_arr, donor_ids = scheme_fn(all_state, i, n_params)
                worker_args.append((
                    ind.genotype_["morph"],
                    init_mean_arr.tolist(),
                    donor_ids,
                    inner_gens,
                    inner_pop,
                    sigma,
                    hidden,
                ))

            if num_workers > 1:
                results = _evaluate_with_timeout(worker_args, num_workers, eval_timeout)
            else:
                results = [evaluate_individual(a) for a in worker_args]

            for i, ind in enumerate(all_alive):
                r = results[i]
                distance = r["distance"]
                theta_list = r["best_theta"]
                novelty = float(novelties[i])
                desc = descs[i]
                prior = ind.tags_ or {}
                computed_fitness = combined_fitness(distance, novelty, x_val)
                # NaN (e.g. a timed-out individual's distance, see
                # _evaluate_with_timeout) doesn't reliably sink to the bottom
                # under Population.best()'s comparison-based sort the way
                # -inf does (see _safe_attr) -- substitute -inf explicitly so
                # a failed individual can never be selected as a survivor.
                ind.fitness = computed_fitness if np.isfinite(computed_fitness) else float("-inf")
                ind.tags = {
                    "parent_id": prior.get("parent_id"),
                    "distance": distance,
                    "novelty": novelty,
                    "descriptor": desc.tolist(),
                    "theta": theta_list,
                    "init_fitness": r["init_fitness"],
                    "learning_curve": r["learning_curve"],
                    "donor_ids": r["donor_ids"],
                    "mean_jerk": r.get("mean_jerk", 0.0),
                    "c_hinge": r.get("c_hinge", 0),
                }
                ind.genotype_ = {"morph": ind.genotype_["morph"], "brain": theta_list}

        if comma_selection:
            # (mu,lambda): survivors drawn only from offspring, parents always die.
            selection_pool = Population(offspring)
        else:
            # (mu+lambda): survivors drawn from parents + offspring.
            selection_pool = Population(all_alive)

        if selection_method == "tournament":
            survivors = _tournament_select(
                selection_pool.to_list(), n=mu, tournament_size=tournament_size,
            )
        else:
            survivors = selection_pool.best(n=mu).to_list()
        survivor_ids = {id(s) for s in survivors}
        for ind in all_alive:
            ind.alive = id(ind) in survivor_ids

        return Population(all_alive)

    return [generate_offspring, evaluate_and_select]


# ---------------------------------------------------------------------------
# Initial population
# ---------------------------------------------------------------------------

def make_initial_population(mu: int) -> Population:
    inds = []
    for _ in range(mu):
        ind = Individual()
        ind.genotype = {"morph": random_individual(), "brain": []}
        inds.append(ind)
    return Population(inds)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _numbered_db_parts(out_dir: Path) -> list[Path]:
    """database_part{N}.db files in out_dir, sorted by N (database.db is
    implicitly part 1 and isn't included here)."""
    return sorted(
        out_dir.glob("database_part*.db"),
        key=lambda p: int(re.search(r"database_part(\d+)\.db", p.name).group(1)),
    )


def _latest_db_path(out_dir: Path) -> Path:
    """The db file to restart from: the highest-numbered database_part{N}.db
    if this run has already been resumed before, else database.db."""
    parts = _numbered_db_parts(out_dir)
    return parts[-1] if parts else out_dir / "database.db"


def _next_db_path(out_dir: Path) -> Path:
    """First free database_part{N}.db in out_dir (N starts at 2)."""
    parts = _numbered_db_parts(out_dir)
    next_n = int(re.search(r"database_part(\d+)\.db", parts[-1].name).group(1)) + 1 if parts else 2
    return out_dir / f"database_part{next_n}.db"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", required=True, choices=list(SCHEMES.keys()))
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--pop", type=int, default=20, help="mu")
    parser.add_argument("--lam", type=int, default=100)
    parser.add_argument("--comma-selection", action="store_true",
                        help="Use (mu,lambda) selection (survivors from offspring only) "
                             "instead of the default (mu+lambda) (survivors from parents+offspring). "
                             "Requires --lam >= --pop.")
    parser.add_argument("--selection", choices=["elitist", "tournament"], default="elitist",
                        help="Survivor selection algorithm applied to the pool chosen by "
                             "--comma-selection: elitist (top-mu by fitness, default) or tournament.")
    parser.add_argument("--tournament-size", type=int, default=4,
                        help="Tournament size when --selection=tournament (ignored otherwise).")
    parser.add_argument("--inner-gens", type=int, default=20)
    parser.add_argument("--inner-pop", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.5, help="CMA-ES initial step size")
    parser.add_argument("--hidden", type=int, default=32, help="DistributedMLP hidden width")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--eval-timeout", type=float, default=1800.0,
        help="Per-individual wall-clock timeout in seconds (default 1800 = 30min, "
             "several times a typical individual's eval time). A worker that hangs "
             "past this (e.g. a native-level MuJoCo deadlock on some pathological "
             "morphology) is killed and that individual gets a NaN result instead "
             "of blocking the whole generation forever.",
    )
    parser.add_argument(
        "--resume-dir", type=str, default=None,
        help="Continue an existing run from this exact directory (must contain "
             "database.db). Writes the continuation to the next "
             "database_part{N}.db in that same directory. If omitted, a fresh "
             "timestamped directory is created instead.",
    )
    args = parser.parse_args()

    if args.comma_selection and args.lam < args.pop:
        parser.error("--comma-selection requires --lam >= --pop (not enough offspring to fill mu)")

    if args.selection == "tournament" and args.tournament_size < 2:
        parser.error("--tournament-size must be >= 2 when --selection=tournament")

    if args.resume_dir:
        out_dir = Path(args.resume_dir)
        if not out_dir.is_dir():
            parser.error(f"--resume-dir does not exist: {out_dir}")
        if not (out_dir / "database.db").exists():
            parser.error(f"--resume-dir has no database.db to resume from: {out_dir}")
    else:
        x_str = str(args.x).replace(".", "")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Non-default selection schemes get a label in the rep-dir name so
        # runs of different schemes for the same (scheme, x, rep) don't get
        # silently mixed together by analysis/curve_utils.py's discover_reps
        # (which globs "rep_*/database.db"). The true default (mu+lambda,
        # elitist) keeps the original unlabeled name so every other run
        # script's output layout is unaffected.
        sel_bits = []
        if args.comma_selection:
            sel_bits.append("comma")
        if args.selection == "tournament":
            sel_bits.append(f"tourn{args.tournament_size}")
        sel_suffix = ("_" + "_".join(sel_bits)) if sel_bits else ""
        out_dir = Path(
            f"__data__/social/ariel/{args.scheme}/x{x_str}/rep_{args.rep}{sel_suffix}_{timestamp}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

    # Printed on its own line (not via `console`, whose rich markup/box-drawing
    # would make this harder to grep) so calling scripts can pick up exactly
    # which directory this run landed in — see run_social_parta.sh.
    print(f"RUN_DIR={out_dir}")

    console.rule(f"[bold cyan]ARIEL | scheme={args.scheme} x={args.x} rep={args.rep} dir={out_dir}")

    ops = build_ops(
        scheme_name=args.scheme,
        x_val=args.x,
        mu=args.pop,
        lam=args.lam,
        inner_gens=args.inner_gens,
        inner_pop=args.inner_pop,
        num_workers=args.workers,
        eval_timeout=args.eval_timeout,
        sigma=args.sigma,
        hidden=args.hidden,
        comma_selection=args.comma_selection,
        selection_method=args.selection,
        tournament_size=args.tournament_size,
    )

    if args.resume_dir:
        ea = EA(
            restart=_latest_db_path(out_dir),
            operations=ops,
            num_steps=args.gens,
            db_file_path=_next_db_path(out_dir),
            db_handling="delete",
        )
    else:
        ea = EA(
            population=make_initial_population(args.pop),
            operations=ops,
            num_steps=args.gens,
            db_file_path=out_dir / "database.db",
            db_handling="delete",
        )
    ea.run()

    console.rule("[bold green]Done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
