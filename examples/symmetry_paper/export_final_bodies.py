"""
Experiment 4: export the final-generation bodies of Experiment 1 for long retraining.

Every run's last generation (normally 50) holds 20 survivors. Each distinct body of a run
(by genome hash; Symmetric runs often hold copies) becomes one line of the output, with the
controller kind it evolved with, so train_long.py can retrain it from scratch for a much
larger budget (default 10,000 evaluations, as Experiment 3).

Lines are ordered by rank within the run (every run's best body first, then every run's
second best, ...), so a partial set still covers every genome x controller x budget cell.

Output: one JSON line per body:
  body_id      <genome>__<brain>__e<evals>__s<seed>__i<id> (unique)
  genome_type, brain, e1_evals, e1_seed, e1_run (path relative to --e1), e1_gen
  e1_id        the survivor's id (the best-scoring one when the run holds copies)
  e1_ids       every survivor id with this body; copies = len(e1_ids)
  rank         0 = the run's best distinct body
  e1_score     the survivor's score in Experiment 1 (best of its copies)
  n_modules, n_hinges, symmetry, tied_frac, genome_hash, genome

Usage (from this directory):
    python export_final_bodies.py --e1 ../../__data__/symmetry_paper/e1 -o e4_bodies.jsonl
    python export_final_bodies.py --e1 ... --per-run 5 -o e4_top5.jsonl   # a cheaper subset
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

KEEP = ("n_modules", "n_hinges", "symmetry", "tied_frac", "genome_hash", "genome")


def run_bodies(run: Path, root: Path) -> list[dict]:
    """The distinct bodies of a run's last generation, best first."""
    gens = run / "generations.jsonl"
    last = json.loads(gens.read_text().splitlines()[-1])
    score = dict(zip(last["ids"], last["scores"]))
    recs, genome_of = {}, {}
    with (run / "evaluations.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("genome") is not None:  # a retraining's record leaves the genome out
                genome_of.setdefault(r["genome_hash"], r["genome"])
            if r["id"] in score:
                recs[r["id"]] = r
    missing = set(score) - set(recs)
    if missing:
        raise SystemExit(f"{run}: survivors {sorted(missing)} not in evaluations.jsonl")
    genome, brain, e, s = run.relative_to(root).parts
    by_hash: dict[str, list[int]] = {}
    for i in sorted(score, key=lambda i: -score[i]):
        by_hash.setdefault(recs[i]["genome_hash"], []).append(i)
    out = []
    for rank, ids in enumerate(by_hash.values()):
        r = {**recs[ids[0]], "genome": genome_of.get(recs[ids[0]]["genome_hash"])}
        if r["genome"] is None:
            raise SystemExit(f"{run}: no genome stored for survivor {ids[0]}")
        out.append({"body_id": f"{genome}__{brain}__{e}__{s}__i{ids[0]}", "genome_type": genome, "brain": brain,
                    "e1_evals": int(e[1:]), "e1_seed": int(s[1:]), "e1_run": str(run.relative_to(root)),
                    "e1_gen": last["gen"], "e1_id": ids[0], "e1_ids": ids, "copies": len(ids), "rank": rank,
                    "e1_score": score[ids[0]], **{k: r.get(k) for k in KEEP}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1", type=Path, required=True, help="Experiment 1 root: <genome>/<brain>/e<evals>/s<seed>/")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--per-run", type=int, default=None, help="keep only each run's best N distinct bodies")
    ap.add_argument("--allow-unfinished", action="store_true", help="include runs without DONE")
    a = ap.parse_args()
    runs = sorted(p.parent for p in a.e1.glob("*/*/e*/s*/generations.jsonl"))
    unfinished = [r for r in runs if not (r / "DONE").exists()]
    if unfinished and not a.allow_unfinished:
        raise SystemExit(f"{len(unfinished)} runs without DONE (e.g. {unfinished[0]}); --allow-unfinished to include")
    per_run = [run_bodies(r, a.e1)[: a.per_run] for r in runs]
    rows = [b for k in range(max(map(len, per_run))) for bodies in per_run if k < len(bodies) for b in bodies[k:k + 1]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for b in rows:
            fh.write(json.dumps(b) + "\n")
    cells = Counter((b["genome_type"], b["brain"], b["e1_evals"]) for b in rows)
    genomes = Counter(b["genome_type"] for b in rows)
    print(f"{len(runs)} runs -> {len(rows)} bodies ({sum(b['copies'] for b in rows)} survivors) in {a.out}")
    print("per genome:", dict(genomes))
    print(f"per cell: {min(cells.values())}-{max(cells.values())} bodies over {len(cells)} cells")


if __name__ == "__main__":
    main()
