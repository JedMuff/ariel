"""
Check that runs saved what they should: every invariant of the output layout
of evolve.py (Experiment 1) and train_fixed_body.py (Experiment 2).

E1 run:
  config       protocol equals config.PROTOCOL (generations may differ)
  generations  contiguous from 0; mu ids and scores, best first; DONE iff the
               last generation is the configured one
  trainings    gen 0: mu; later: mu + lam, with every survivor of the previous
               generation retrained (re-evaluation); ids unique per generation
  records      score = max of the curve; curve length = CMA-ES generations
               (unless stopped early); evals = popsize x curve length;
               train_seed = seed + 1000 gen + index; population scores = their
               trainings' scores; children's parents are in the previous population
  genomes      logged once, on the first training; valid (module cap, hinges,
               builds); genome_hash and body measures match; GRN genomes regrow
               their body
  weights      one npz per generation, a vector per training, sized for the
               body and brain
  --replay N   the best N of the final population replay to their stored score
               (exact: run this on the machine type that trained the run)
E2 directory:  seeds unique and as configured, score = max of curve, a weight
               vector per seed of the right size; --replay replays the best seed

Usage:
    python tools/check_runs.py ROOT [ROOT ...] [--replay 1]
Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import genomes  # noqa: E402
import runs  # noqa: E402
from brains import make_brain_for_model  # noqa: E402
from config import PROTOCOL, cma_generations  # noqa: E402
from replay import record_episode  # noqa: E402
from sim import build_world, genome_hash  # noqa: E402

_param_cache: dict[tuple[str, str], int] = {}


def n_params(genome: dict, brain: str) -> int:
    key = (genome_hash(genome), brain)
    if key not in _param_cache:
        model, data = build_world(genome)
        _param_cache[key] = make_brain_for_model(brain, model, data).num_params
    return _param_cache[key]


class Checker:
    def __init__(self, name: str) -> None:
        self.name, self.errors, self.n = name, [], 0

    def check(self, ok: bool, msg: str) -> None:
        self.n += 1
        if not ok and len(self.errors) < 20:
            self.errors.append(msg)

    def report(self) -> bool:
        status = "ok" if not self.errors else f"{len(self.errors)} FAILED"
        print(f"{self.name}: {self.n} checks, {status}")
        for e in self.errors:
            print(f"    {e}")
        return not self.errors


def check_e1(run_dir: Path, replay: int) -> bool:
    c = Checker(str(run_dir))
    cfg = runs.config(run_dir)
    proto = {k: v for k, v in cfg["protocol"].items() if k != "generations"}
    c.check(proto == {k: v for k, v in asdict(PROTOCOL).items() if k != "generations"}, "protocol differs from config.PROTOCOL")
    mu, lam, seed, brain = cfg["protocol"]["mu"], cfg["protocol"]["lam"], cfg["seed"], cfg["brain"]
    n_gen_cma = cma_generations(cfg["evals"])
    gens, evals = runs.generations(run_dir), runs.evaluations(run_dir)

    c.check([g["gen"] for g in gens] == list(range(len(gens))), "generations not contiguous from 0")
    done = (run_dir / "DONE").exists()
    c.check(done == (gens[-1]["gen"] == cfg["protocol"]["generations"]), f"DONE={done} but last gen {gens[-1]['gen']}")

    by_gen: dict[int, list[dict]] = {}
    for e in evals:
        by_gen.setdefault(e["gen"], []).append(e)
    c.check(sorted(by_gen) == [g["gen"] for g in gens], "trainings and generations cover different generations")

    genome_of: dict[int, dict] = {}
    prev_ids: list[int] = []
    for g in gens:
        recs = by_gen.get(g["gen"], [])
        ids = [r["id"] for r in recs]
        c.check(len(ids) == (mu if g["gen"] == 0 else mu + lam), f"gen {g['gen']}: {len(ids)} trainings")
        c.check(len(set(ids)) == len(ids), f"gen {g['gen']}: a body trained twice")
        c.check(set(prev_ids) <= set(ids), f"gen {g['gen']}: survivors not all retrained")
        c.check(len(g["ids"]) == mu and g["scores"] == sorted(g["scores"], reverse=True),
                f"gen {g['gen']}: population not {mu} sorted scores")
        score_of = {r["id"]: r["score"] for r in recs}
        c.check(all(score_of.get(i) == s for i, s in zip(g["ids"], g["scores"])),
                f"gen {g['gen']}: population scores differ from their trainings")
        c.check(sorted(g["scores"], reverse=True)[:mu] == sorted(score_of.values(), reverse=True)[:mu],
                f"gen {g['gen']}: population is not the best {mu} trainings")
        w = np.load(run_dir / "weights" / f"gen{g['gen']:03d}.npz")
        c.check(sorted(w.files) == sorted(str(i) for i in ids), f"gen {g['gen']}: weights keys differ from trainings")
        for k, r in enumerate(recs):
            tag = f"gen {g['gen']} id {r['id']}"
            if r["born"] == r["gen"]:
                c.check("genome" in r and r["id"] not in genome_of, f"{tag}: genome not logged exactly once")
                genome_of[r["id"]] = r.get("genome", {})
                if r["gen"] > 0:
                    c.check(set(r["parents"]) <= set(prev_ids) and r["parents"], f"{tag}: parents not in the population")
            else:
                c.check("genome" not in r and r["id"] in genome_of, f"{tag}: retraining of an unknown body")
            genome = genome_of.get(r["id"], {})
            curve = r["curve"]
            c.check(len(curve["best"]) == len(curve["median"]) == len(curve["sigma"]), f"{tag}: curve lengths differ")
            c.check(bool(curve["best"]) and abs(r["score"] - max(curve["best"])) < 1e-5,  # curves are rounded
                    f"{tag}: score is not the curve's best")
            c.check(len(curve["best"]) == n_gen_cma or r["stopped_early"], f"{tag}: {len(curve['best'])} CMA-ES generations")
            c.check(r["evals"] == PROTOCOL.cma_popsize * len(curve["best"]), f"{tag}: evals {r['evals']}")
            c.check(r["train_seed"] == seed + 1000 * g["gen"] + k, f"{tag}: train seed {r['train_seed']}")
            c.check(r["genome_hash"] == genome_hash(genome), f"{tag}: genome hash differs")
            if r["born"] == r["gen"]:
                c.check(genomes.is_valid(genome), f"{tag}: invalid body ({genomes.module_count(genome)} modules, "
                        f"{genomes.hinge_count(genome)} hinges)")
                desc = genomes.describe(genome)
                c.check(all(r[k] == v for k, v in desc.items()), f"{tag}: body measures differ")
                c.check(w[str(r["id"])].shape == (n_params(genome, brain),), f"{tag}: weights shape")
                if cfg["genome"] == "grn":
                    from ariel.ec.genotypes.grn import develop_body  # noqa: PLC0415
                    from ariel.ec.genotypes.tree.collision import prune_colliding_subtrees  # noqa: PLC0415
                    from ariel.ec.genotypes.tree.tree_genome import TreeGenome  # noqa: PLC0415
                    t = TreeGenome.from_dict(develop_body(genome["grn"], max_modules=PROTOCOL.max_modules + 1))
                    prune_colliding_subtrees(t)
                    c.check(genome_hash({**t.to_dict(), "grn": genome["grn"]}) == r["genome_hash"],
                            f"{tag}: GRN values do not regrow the body")
        prev_ids = g["ids"]

    for ep in list(runs.episodes(run_dir, "final"))[:replay]:
        score = record_episode(ep.genome, ep.brain, ep.weights)[0]
        c.check(score == ep.score, f"replay {ep.key}: {score} vs stored {ep.score}")
    return c.report()


def check_e2(d: Path, replay: int) -> bool:
    c = Checker(str(d))
    cfg = runs.config(d)
    res = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]
    seeds = [r["seed"] for r in res]
    c.check(len(set(seeds)) == len(seeds) and set(seeds) <= set(cfg["seeds"]), "seeds duplicated or unexpected")
    w = np.load(d / "weights.npz")
    c.check(sorted(w.files) == sorted(f"s{s}" for s in seeds), "weights keys differ from seeds")
    c.check(genomes.is_valid(cfg["genome"]), "invalid body")
    for r in res:
        c.check(abs(r["score"] - max(r["curve"]["best"])) < 1e-5, f"seed {r['seed']}: score is not the curve's best")
        c.check(len(r["curve"]["best"]) == cma_generations(cfg["evals"]) or r["stopped_early"], f"seed {r['seed']}: curve length")
        c.check(w[f"s{r['seed']}"].shape == (n_params(cfg["genome"], cfg["brain"]),), f"seed {r['seed']}: weights shape")
    for ep in list(runs.episodes(d, "champion"))[:replay]:
        score = record_episode(ep.genome, ep.brain, ep.weights)[0]
        c.check(score == ep.score, f"replay {ep.key}: {score} vs stored {ep.score}")
    return c.report()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--replay", type=int, default=1, help="final-population bodies to replay per run (0: none)")
    args = ap.parse_args()
    e1, e2 = runs.find(args.roots, "e1"), runs.find(args.roots, "e2")
    ok = [check_e1(d, args.replay) for d in e1] + [check_e2(d, args.replay) for d in e2]
    print(f"\n{sum(ok)}/{len(ok)} runs pass ({len(e1)} E1, {len(e2)} E2)")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
