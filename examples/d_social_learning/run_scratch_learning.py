import argparse
import datetime
import json
import multiprocessing
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from rich.console import Console
from sqlmodel import Session, create_engine, select

# Add this dir to sys.path so local modules resolve.
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import simulator_dependent_functions
from ariel.ec import Individual
from evaluate import evaluate_individual, init_worker

console = Console()

def get_latest_generation(db_path: Path):
    """Load individuals from the latest generation in the database."""
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        # Find the maximum time_of_death among all individuals
        statement = select(Individual).where(Individual.alive == True)
        results = session.exec(statement).all()
        
        if not results:
            # If no 'alive' individuals, maybe the experiment finished.
            # Try to find the latest generation based on time_of_death.
            max_death = session.exec(select(np.max(Individual.time_of_death))).one()
            if max_death is None:
                return []
            statement = select(Individual).where(Individual.time_of_death == max_death)
            results = session.exec(statement).all()
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Run controller learning from scratch on last generation morphologies.")
    parser.add_argument("--data-dir", type=str, default="__data__/social", help="Root directory for experiment data.")
    parser.add_argument("--out-dir", type=str, default="__data__/scratch_learning", help="Directory to save results.")
    parser.add_argument("--inner-gens", type=int, default=20)
    parser.add_argument("--inner-pop", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--platform", type=str, choices=["ariel", "evogym"], default="ariel", help="Filter experiments by platform.")
    parser.add_argument("--db-path", type=str, default=None, help="Path to a specific database.db file to process.")
    parser.add_argument("--scheme", type=str, default=None, choices=["darwinian", "lamarckian", "random", "random_many", "best", "best_many", "similar_STRUCT", "similar_many_STRUCT"])
    parser.add_argument("--x", type=str, default=None, help="X value as string (e.g. 00, 05, 10)")
    parser.add_argument("--rep", type=int, default=None, help="Repetition number")
    parser.add_argument("--k-inherit", type=int, default=8, help="k value in path")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Discover experiments
    if args.db_path:
        db_paths = [Path(args.db_path)]
    elif args.scheme and args.x is not None and args.rep is not None:
        # Calculate path based on parameters
        # Pattern: __data__/social/$PLATFORM/$SCHEME/x$X/k$K/rep_${REP}_novSTRUCT/database.db
        platform = args.platform
        base_path = data_root / platform / args.scheme / f"x{args.x}" / f"k{args.k_inherit}"
        
        # Search for database.db within the specific rep directory
        # The rep directory might have a suffix like _novSTRUCT or _novDESCR
        search_pattern = f"rep_{args.rep}*"
        rep_dirs = list(base_path.glob(search_pattern))
        
        db_paths = []
        for rd in rep_dirs:
            db_file = rd / "database.db"
            if db_file.exists():
                db_paths.append(db_file)
        
        if not db_paths:
            console.print(f"[red]Could not find database.db for scheme={args.scheme}, x={args.x}, rep={args.rep} in {base_path}[/red]")
            return
    else:
        db_paths = list(data_root.rglob("database.db"))
    
    if not db_paths:
        console.print(f"[red]No database.db files found in {data_root}[/red]")
        return

    console.print(f"Found {len(db_paths)} experiments.")

    for db_path in db_paths:
        # Extract platform from path. 
        # If data_root is __data__/social, platform is the next part.
        # If data_root is deeper, we might need to find where 'social' is.
        parts = db_path.parts
        if "social" in parts:
            social_idx = parts.index("social")
            platform = parts[social_idx + 1]
        else:
            # Fallback to first part relative to data_root if 'social' not found
            platform = db_path.relative_to(data_root).parts[0]
        
        if platform != args.platform:
            continue

        console.rule(f"[bold cyan]Processing {db_path} (Platform: {platform})")
        
        # Determine output directory for this experiment
        # e.g. __data__/scratch_learning/evogym/best/x00/k8/rep_0_novSTRUCT
        rel_path = db_path.parent.relative_to(data_root)
        exp_out_dir = out_root / rel_path
        exp_out_dir.mkdir(parents=True, exist_ok=True)

        # Set simulator
        simulator_dependent_functions.simulator = platform
        
        # Load individuals
        inds = get_latest_generation(db_path)
        if not inds:
            console.print(f"[yellow]No individuals found in {db_path}[/yellow]")
            continue
        
        console.print(f"Loaded {len(inds)} individuals from the last generation.")

        # Prepare learning tasks
        from ariel.simulation.controllers.distributed_mlp import DistributedMLP
        n_params = DistributedMLP(n_neighbors=simulator_dependent_functions.n_neighbours(), hidden=args.hidden).n_params

        console.print(f"Running scratch learning")
        
        worker_args = []
        for ind in inds:
            # Learning from scratch means no inherited brain
            init_mean_list = None 
            donor_ids = []
            
            worker_args.append((
                ind.genotype_["morph"],
                init_mean_list,
                donor_ids,
                args.inner_gens,
                args.inner_pop,
                args.sigma,
                args.hidden,
            ))

        # Run parallel evaluation
        if args.workers > 1:
            with Pool(processes=args.workers, initializer=init_worker, initargs=(platform,)) as pool:
                results = pool.map(evaluate_individual, worker_args)
        else:
            init_worker(platform)
            results = [evaluate_individual(a) for a in worker_args]

        # Save results for this repetition
        rep_results = []
        for i, res in enumerate(results):
            rep_results.append({
                "ind_id": inds[i].id,
                "morphology": inds[i].genotype_["morph"],
                "distance": res["distance"],
                "best_theta": res["best_theta"],
                "learning_curve": res["learning_curve"],
                "extra": {k: v for k, v in res.items() if k not in ["distance", "best_theta", "learning_curve", "donor_ids", "init_fitness"]}
            })

        result_file = exp_out_dir / f"scratch_results.json"
        with open(result_file, "w") as f:
            json.dump(rep_results, f, indent=2)
        
        console.print(f"Saved results to {result_file}")

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
