"""Learn controllers from scratch on the final population of a finished run.

Each morphology alive at the end of a run gets a fresh controller, with nothing
inherited, trained the way the run trained its own. Physics preset, final evals,
CMA-ES settings and platform come from the run's run_config.json, so a zoo fe10
run stays zoo fe10. Command-line flags override individual settings; runs without
a run_config.json need --platform and use the defaults noted in the flag help.

Resumed runs keep their final population in the highest-numbered
database_part{N}.db, which is the file read here.

Usage:
    uv run examples/d_social_learning/run_scratch_learning.py \
        --data-dir __data__/social/ariel_zoo_fe10/ig10 \
        --out-dir __data__/scratch_learning/ariel_zoo_fe10/ig10 \
        --scheme best --x 05 --rep 0 --k-inherit 5 --workers 24

Writes scratch_results.json and scratch_config.json to the matching directory
under --out-dir. A run that already has results is skipped unless --force is set.
"""

import argparse
import datetime
import hashlib
import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

# First: experiment.py caps BLAS threads before numpy is imported.
from experiment import (
    RUN_CONFIG,
    _evaluate_with_timeout,
    _latest_db_path,
    evaluate_individual,
)

from rich.console import Console
from sqlalchemy import func
from sqlmodel import Session, create_engine, select

import simulator_dependent_functions
from ariel.ec import Individual
from evaluate import init_worker

console = Console()

# Used for a setting that neither the command line nor run_config.json gives.
FALLBACKS = {
    "sim_preset": "legacy",
    "final_evals": 0,
    "inner_gens": 20,
    "inner_pop": 16,
    "sigma": 0.5,
    "hidden": 16,
    "eval_timeout": 1800.0,
}


def load_final_population(db_path: Path) -> list[Individual]:
    """Individuals alive at the end of this database, else the last to die."""
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        alive = session.exec(select(Individual).where(Individual.alive == True)).all()  # noqa: E712
        if alive:
            return list(alive)
        last = session.exec(select(func.max(Individual.time_of_death))).one()
        if last is None:
            return []
        return list(session.exec(select(Individual).where(Individual.time_of_death == last)).all())


def individual_seed(run_seed: int, ind_id: int) -> int:
    """Stable 32-bit seed, so a rerun repeats each individual's learning."""
    key = f"scratch|{run_seed}|{ind_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def find_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [Path(args.run_dir)]
    data_root = Path(args.data_dir)
    if args.scheme and args.x is not None and args.rep is not None:
        root = data_root / args.platform if args.platform else data_root
        base = root / args.scheme / f"x{args.x}" / f"k{args.k_inherit}"
        # The trailing underscore stops rep_1 matching rep_10. As in the
        # analysis, the newest run wins if a rep was run more than once.
        runs = sorted(d for d in base.glob(f"rep_{args.rep}_*") if (d / "database.db").exists())
        return runs[-1:]
    return sorted(db.parent for db in data_root.rglob("database.db"))


def resolve_settings(args: argparse.Namespace, config: dict) -> dict:
    run_args = config.get("args") or {}
    settings = {}
    for name, fallback in FALLBACKS.items():
        given = getattr(args, name)
        settings[name] = given if given is not None else run_args.get(name, fallback)
    settings["platform"] = run_args.get("platform") or args.platform
    if settings["platform"] is None:
        raise SystemExit("Run has no run_config.json: pass --platform")
    return settings


def process_run(run_dir: Path, args: argparse.Namespace, out_root: Path) -> None:
    config_path = run_dir / RUN_CONFIG
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    settings = resolve_settings(args, config)

    if args.platform and args.platform != settings["platform"]:
        console.print(f"Skipping {run_dir}: platform is {settings['platform']}")
        return

    data_root = Path(args.data_dir)
    rel = run_dir.resolve().relative_to(data_root.resolve()) if run_dir.resolve().is_relative_to(data_root.resolve()) else Path(run_dir.name)
    out_dir = out_root / rel
    result_file = out_dir / "scratch_results.json"
    if result_file.exists() and not args.force:
        console.print(f"Results exist, skipping (use --force to redo): {result_file}")
        return

    db_path = _latest_db_path(run_dir)
    console.rule(f"[bold cyan]{db_path}")
    console.print(f"Settings: {settings}")

    inds = load_final_population(db_path)
    if not inds:
        raise SystemExit(f"No individuals found in {db_path}")
    generation = max(ind.time_of_death for ind in inds)
    console.print(f"Loaded {len(inds)} individuals from generation {generation}.")

    platform = settings["platform"]
    simulator_dependent_functions.simulator = platform
    simulator_dependent_functions.set_sim_preset(settings["sim_preset"])

    run_seed = config.get("seed", 0)
    seeds = [individual_seed(run_seed, ind.id) for ind in inds]
    worker_args = [
        (
            ind.genotype_["morph"],
            None,  # nothing inherited
            [],
            settings["inner_gens"],
            settings["inner_pop"],
            settings["sigma"],
            settings["hidden"],
            seed,
            settings["final_evals"],
        )
        for ind, seed in zip(inds, seeds)
    ]

    if args.workers > 1:
        results = _evaluate_with_timeout(worker_args, args.workers, settings["eval_timeout"], platform)
    else:
        init_worker(platform, settings["sim_preset"])
        results = [evaluate_individual(a) for a in worker_args]

    skip = {"distance", "best_theta", "learning_curve", "donor_ids", "init_fitness"}
    records = [
        {
            "ind_id": ind.id,
            "eval_seed": seed,
            "morphology": ind.genotype_["morph"],
            "evolved_distance": ind.tags.get("distance"),
            "distance": res["distance"],
            "init_fitness": res["init_fitness"],
            "best_theta": res["best_theta"],
            "learning_curve": res["learning_curve"],
            "extra": {k: v for k, v in res.items() if k not in skip},
        }
        for ind, seed, res in zip(inds, seeds, results)
    ]

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
    ).stdout.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scratch_config.json").write_text(json.dumps({
        "source_run": str(run_dir),
        "source_db": db_path.name,
        "generation": generation,
        "run_seed": run_seed,
        "settings": settings,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
    }, indent=2))
    # Written last and renamed in, so its presence means a complete run.
    tmp = result_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, indent=2))
    tmp.replace(result_file)
    console.print(f"Saved results to {result_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="__data__/social", help="Directory the runs are under.")
    parser.add_argument("--out-dir", default="__data__/scratch_learning", help="Directory to save results.")
    parser.add_argument("--run-dir", default=None, help="Process this one run directory.")
    parser.add_argument("--platform", choices=["ariel", "evogym"], default=None,
                        help="With --scheme/--x/--rep: data is under DATA_DIR/PLATFORM. Otherwise only process this platform.")
    parser.add_argument("--scheme", default=None)
    parser.add_argument("--x", default=None, help="X folder value as a string (00, 05, 10)")
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--k-inherit", type=int, default=8, help="k value in the run's path")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Redo runs that already have results.")
    parser.add_argument("--sim-preset", dest="sim_preset", choices=["legacy", "zoo"], default=None,
                        help="Default: from run_config.json, else legacy.")
    parser.add_argument("--final-evals", dest="final_evals", type=int, default=None,
                        help="Default: from run_config.json, else 0.")
    parser.add_argument("--inner-gens", dest="inner_gens", type=int, default=None,
                        help="Default: from run_config.json, else 20.")
    parser.add_argument("--inner-pop", dest="inner_pop", type=int, default=None,
                        help="Default: from run_config.json, else 16.")
    parser.add_argument("--sigma", type=float, default=None, help="Default: from run_config.json, else 0.5.")
    parser.add_argument("--hidden", type=int, default=None, help="Default: from run_config.json, else 16.")
    parser.add_argument("--eval-timeout", dest="eval_timeout", type=float, default=None,
                        help="Seconds per individual. Default: from run_config.json, else 1800.")
    args = parser.parse_args()

    run_dirs = find_run_dirs(args)
    if not run_dirs:
        console.print(f"[red]No matching runs under {args.data_dir}[/red]")
        sys.exit(1)
    console.print(f"Found {len(run_dirs)} run(s).")

    out_root = Path(args.out_dir)
    for run_dir in run_dirs:
        process_run(run_dir, args, out_root)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
