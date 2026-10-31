"""
Train one brain on one fixed body and log its learning curve (E2).

The inner loop of evolve.py (sim.train_skill_for_body: CMA-ES popsize 20,
sigma 0.5, fixed x0, zoo hinge) run for --evals evaluations, with no outer
loop, once per training seed. Every generation is logged, and the controller
is checkpointed at the first generation reaching each of CHECKPOINT_EVALS, in
the layout replay.py reads, so E3 can measure gait and phase quality along
training.

Writes, per seed, <out-root>/<body_id>__<brain>__s<seed>/:
  run_config.json   body, brain, seed, budget, hinge actuator
  curve.jsonl       per generation: gen, evals, gen_best, gen_median, best_so_far, sigma
  checkpoints/evalsNNNNN/  best_genome.json, fwd_weights.npy (best so far),
                    fwd_mean.npy (CMA-ES mean), meta.json (fitness = best so far,
                    plus retest_scores / retest_mean / retest_sd with --retests)
  final.json        final best fitness and score (the retest mean with --retests,
                    else the training best), wall time; a seed with it is skipped

--fitness displacement (paper v2, default) scores forward displacement in m,
unclipped, without penalties; --fitness normalised reproduces the v1 runs.

Usage:
    python train_fixed_body.py --bodies bodies_random.jsonl --line 0 --brain dann \\
        --evals 5000 --seeds 42 43 44 --workers 20 -o /scratch/$USER/ariel_paper_v2/e2
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from brains import BRAIN_KINDS
from sim import FITNESS_MODES, PROFILE, ZOO_HINGE, retest_controller, retest_summary, train_skill_for_body, zoo_spec_fn

CHECKPOINT_EVALS = (50, 200, 800, 1600, 5000)


def train(body: dict, brain: str, seed: int, evals: int, workers: int, out: Path,
          fitness: str = "displacement", retests: int = 0) -> None:
    if (out / "final.json").exists():
        print(f"{out.name}: done, skipped", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    gens = math.ceil(evals / PROFILE.cma_popsize)
    profile = replace(PROFILE, skill_budget=gens, fitness=fitness)
    to_spec = zoo_spec_fn(ZOO_HINGE)
    (out / "run_config.json").write_text(json.dumps({
        "body_id": body["body_id"], "body_source": body["body_source"], "genome_type": body["genome_type"],
        "n_modules": body["n_modules"], "n_hinges": body["n_hinges"], "n_limbs": body["n_limbs"],
        "brain": brain, "seed": seed, "evals": evals, "skill_budget": gens, "eval_profile": "dsocial_zoo",
        "fitness": fitness, "retests": retests,
        "eval_profile_settings": asdict(profile), "hinge_actuator": asdict(ZOO_HINGE),
    }, indent=2))
    curve_path = out / "curve.jsonl"
    curve_path.unlink(missing_ok=True)
    pending = [e for e in CHECKPOINT_EVALS if e <= evals]
    last: dict = {}
    t0 = time.time()

    def save_checkpoint(threshold: int, stopped: bool) -> None:
        ckpt = out / "checkpoints" / f"evals{threshold:05d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        np.save(ckpt / "fwd_weights.npy", last["best_w"])
        np.save(ckpt / "fwd_mean.npy", last["mean"])
        (ckpt / "best_genome.json").write_text(json.dumps(body["genome"], indent=2))
        (ckpt / "meta.json").write_text(json.dumps({"fitness": last["best_fit"], "evals": last["evals"],
                                                    "gen": last["gen"], "stopped_early": stopped}))

    def on_generation(gen, n_evals, es, fits, best_w, best_fit) -> None:  # noqa: ANN001
        last.update(gen=gen, evals=n_evals, best_w=best_w.copy(), best_fit=float(best_fit),
                    mean=np.asarray(es.mean, dtype=np.float32))
        with curve_path.open("a") as fh:
            fh.write(json.dumps({"gen": gen, "evals": n_evals, "gen_best": float(min(fits)),
                                 "gen_median": float(np.median(fits)), "best_so_far": float(best_fit),
                                 "sigma": float(es.sigma), "wall_s": round(time.time() - t0, 2)}) + "\n")
        while pending and n_evals >= pending[0]:
            save_checkpoint(pending.pop(0), stopped=False)

    _, curve, eval_time, info = train_skill_for_body(
        body["genome"], gens, workers, seed, to_spec_fn=to_spec, brain_kind=brain,
        profile=profile, on_generation=on_generation,
    )
    # CMA-ES stops early on a flat fitness (e.g. every candidate scores 0, the
    # clip for moving backwards), as in evolve.py's inner loop: the controller
    # no longer changes, so later checkpoints repeat the final one.
    stopped_early = last["evals"] < gens * PROFILE.cma_popsize
    while pending:
        save_checkpoint(pending.pop(0), stopped=True)
    best = float(min(curve)) if curve else math.inf
    # Retest every checkpoint's best-so-far controller in fresh processes
    # (after training, so the retest pool does not compete with CMA-ES).
    final_retest: dict = {}
    if retests > 0:
        for ckpt in sorted((out / "checkpoints").glob("evals*")):
            meta = json.loads((ckpt / "meta.json").read_text())
            summ = retest_summary(retest_controller(body["genome"], np.load(ckpt / "fwd_weights.npy"), brain,
                                                    retests, workers, seed, to_spec_fn=to_spec, profile=profile))
            (ckpt / "meta.json").write_text(json.dumps({**meta, **summ}))
            final_retest = summ  # the last checkpoint holds the final controller
    score = final_retest["retest_mean"] if final_retest else -best
    (out / "final.json").write_text(json.dumps({"fitness": -score, "score": score, "train_best_score": -best,
                                                "evals": last["evals"], "stopped_early": stopped_early,
                                                "wall_s": round(eval_time, 1), **info, **final_retest}))
    print(f"{out.name}: train best {-best:.3f}, score {score:.3f} in {eval_time / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bodies", type=Path, required=True, help="sample_bodies.py output")
    parser.add_argument("--line", type=int, required=True, help="0-based body index in --bodies")
    parser.add_argument("--brain", choices=BRAIN_KINDS, required=True)
    parser.add_argument("--evals", type=int, required=True, help="evaluations per training (rounded up to whole generations)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("-o", "--out-root", type=Path, required=True)
    parser.add_argument("--fitness", choices=FITNESS_MODES, default="displacement",
                        help="displacement (m, unclipped, no penalties) or normalised (v1)")
    parser.add_argument("--retests", type=int, default=0, help="fresh-process retests per checkpoint (default 0: none)")
    args = parser.parse_args()

    lines = args.bodies.read_text().splitlines()
    if args.line >= len(lines):
        print(f"line {args.line} is past the {len(lines)} bodies in {args.bodies}; nothing to do")
        return
    body = json.loads(lines[args.line])
    for seed in args.seeds:
        train(body, args.brain, seed, args.evals, args.workers, args.out_root / f"{body['body_id']}__{args.brain}__s{seed}",
              fitness=args.fitness, retests=args.retests)


if __name__ == "__main__":
    main()
