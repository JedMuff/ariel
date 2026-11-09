"""Reading a run back: main score, episodes, and an exact replay of a stored score."""

from dataclasses import replace

import numpy as np
import pytest

import runs
from config import PROTOCOL
from evolve import Run
from replay import record_episode
from summarise import e1_run, evals_to_frac


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("e1") / "tree_symmetric" / "cpg" / "s7"
    Run("tree_symmetric", "cpg", 40, 7, 4, out, replace(PROTOCOL, mu=4, lam=4, generations=1)).run()
    return out


def test_main_score_is_the_final_population_mean(run_dir):
    df = runs.load_e1(run_dir.parents[1])
    assert len(df) == 1
    final = runs.generations(run_dir)[-1]
    assert df["score"].iloc[0] == pytest.approx(np.mean(final["scores"]))
    assert df["champion"].iloc[0] == max(final["scores"])
    assert bool(df["done"].iloc[0])


def test_final_episodes_replay_their_stored_scores(run_dir):
    eps = list(runs.episodes(run_dir, "final"))
    assert len(eps) == 4
    assert [e.score for e in eps] == sorted(runs.generations(run_dir)[-1]["scores"], reverse=True)
    champion = next(runs.episodes(run_dir, "champion"))
    score, P, Q, dt = record_episode(champion.genome, champion.brain, champion.weights)
    assert score == champion.score
    assert dt == pytest.approx(0.02, abs=1e-6)
    assert P.shape[0] == pytest.approx(PROTOCOL.episode_s / dt, abs=2)


def test_survivor_episodes_of_chosen_generations(run_dir):
    gens = runs.generations(run_dir)
    eps = list(runs.episodes(run_dir, "s:0,1"))
    assert [e.meta["gen"] for e in eps] == [0] * 4 + [1] * 4
    assert [e.score for e in eps] == sorted(gens[0]["scores"], reverse=True) + sorted(gens[1]["scores"], reverse=True)
    assert {e.meta["id"] for e in eps if e.meta["gen"] == 1} == set(gens[1]["ids"])
    assert len(list(runs.episodes(run_dir, "g:1"))) == 8  # survivors and offspring


def test_summary_row(run_dir):
    row = e1_run(run_dir)
    assert row["final_n_modules"] > 0 and np.isfinite(row["curve_jitter"])


def test_evals_to_frac():
    assert evals_to_frac([0.1, 0.5, 0.95, 1.0]) == 60
    assert np.isnan(evals_to_frac([-0.2, -0.1]))
