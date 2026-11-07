"""evolve.Run on a tiny protocol: output layout, re-evaluation and resuming."""

import json
from dataclasses import replace

import numpy as np

from config import PROTOCOL
from evolve import Run

TINY = replace(PROTOCOL, mu=4, lam=4, generations=2)


def run(out, generations=2):
    Run("tree", "bang_bang", 20, 42, 4, out, replace(TINY, generations=generations)).run()


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_layout_and_reevaluation(tmp_path):
    run(tmp_path)
    evals, gens = lines(tmp_path / "evaluations.jsonl"), lines(tmp_path / "generations.jsonl")
    assert [g["gen"] for g in gens] == [0, 1, 2]
    assert len(evals) == 4 + 2 * 8               # initial + (survivors retrained + offspring) per generation
    for g in gens[1:]:
        trained = [e for e in evals if e["gen"] == g["gen"]]
        prev = next(p for p in gens if p["gen"] == g["gen"] - 1)
        # every survivor of the previous generation was retrained
        assert set(prev["ids"]) <= {e["id"] for e in trained}
        # and the population is the best mu of this generation's trainings
        assert g["scores"] == sorted((e["score"] for e in trained), reverse=True)[:4]
    assert all("genome" in e for e in evals if e["born"] == e["gen"])
    assert not any("genome" in e for e in evals if e["born"] != e["gen"])
    w = np.load(tmp_path / "weights" / "gen002.npz")
    assert sorted(w.files) == sorted(str(e["id"]) for e in evals if e["gen"] == 2)
    assert (tmp_path / "DONE").exists()


def test_resume_matches_an_uninterrupted_run(tmp_path):
    run(tmp_path / "a")
    run(tmp_path / "b", generations=1)
    with (tmp_path / "b" / "evaluations.jsonl").open("a") as fh:  # a crash mid-generation
        fh.write(json.dumps({"gen": 2, "partial": True}) + "\n")
    run(tmp_path / "b")
    for name in ("evaluations.jsonl", "generations.jsonl"):
        a, b = lines(tmp_path / "a" / name), lines(tmp_path / "b" / name)
        strip = [{k: v for k, v in r.items() if k != "wall_s"} for r in a]
        assert strip == [{k: v for k, v in r.items() if k != "wall_s"} for r in b]
