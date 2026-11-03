"""Which checkpoints and score stand for an evolution run (runs.py)."""

import json
from pathlib import Path

import pytest
import replay
import runs


def make_run(root: Path, repeat_evals: bool | None, pop: int = 2) -> Path:
    """Gen 0: fitness -1, -5. Gen 1: -2, -3 (so the champion, -5, is not in the last generation)."""
    run_dir = root / "run"
    data = run_dir / "__data__" / "evolve" / "forward"
    data.mkdir(parents=True)
    cfg = {"genome_type": "tree", "seed": 42, "skill_budget": 10, "pop": pop, "fitness": "displacement"}
    if repeat_evals is not None:
        cfg["repeat_evals"] = repeat_evals
    (data / "run_config.json").write_text(json.dumps(cfg))
    with (data / "run_data.jsonl").open("w") as f:
        for gen, body, fitness in [(0, 0, -1.0), (0, 1, -5.0), (1, 0, -2.0), (1, 1, -3.0)]:
            f.write(json.dumps({"gen": gen, "fitness": fitness}) + "\n")
            ckpt = data / "checkpoints" / f"gen{gen:03d}_body{body:02d}"
            ckpt.mkdir(parents=True)
            (ckpt / "meta.json").write_text(json.dumps({"fitness": fitness}))
    return run_dir


def names(ckpts: list[Path]) -> list[str]:
    return sorted(c.name for c in ckpts)


def test_without_repeat_evals_the_final_population_spans_all_generations(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, repeat_evals=False)
    assert names(runs.final_checkpoints(run_dir)) == ["gen000_body01", "gen001_body01"]
    assert runs.best_checkpoint(run_dir).name == "gen000_body01"
    assert runs.run_row(run_dir)["score"] == 5.0
    assert names(replay.select_checkpoints(run_dir, "last")) == ["gen000_body01", "gen001_body01"]


@pytest.mark.parametrize("repeat_evals", [True, None])
def test_with_repeat_evals_the_final_population_is_the_last_generation(tmp_path: Path, repeat_evals) -> None:  # noqa: ANN001
    run_dir = make_run(tmp_path, repeat_evals=repeat_evals)
    assert names(runs.final_checkpoints(run_dir)) == ["gen001_body00", "gen001_body01"]
    assert runs.best_checkpoint(run_dir).name == "gen001_body01"
    assert runs.run_row(run_dir)["score"] == 3.0
    assert names(replay.select_checkpoints(run_dir, "last")) == ["gen001_body00", "gen001_body01"]


def test_with_retrain_survivors_only_the_latest_scores_count(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, repeat_evals=False, pop=1)
    cfg_path = run_dir / "__data__" / "evolve" / "forward" / "run_config.json"
    cfg_path.write_text(json.dumps({**json.loads(cfg_path.read_text()), "retrain_survivors": True}))
    assert names(runs.final_checkpoints(run_dir)) == ["gen001_body01"]
    assert runs.run_row(run_dir)["score"] == 3.0
    assert runs.curve_by_gen(run_dir)["best_so_far"].tolist() == [5.0, 3.0]  # may fall


def test_an_explicit_generation_is_respected(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, repeat_evals=False)
    assert runs.best_checkpoint(run_dir, gen=1).name == "gen001_body01"
    assert names(replay.select_checkpoints(run_dir, "1")) == ["gen001_body00", "gen001_body01"]
    assert len(replay.select_checkpoints(run_dir, "all")) == 4
