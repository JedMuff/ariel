"""
Tests for the distributed ANN brain kind ("dann") and the d_social evaluation
profile in shared.py.

Covers:
  - brains.model_topology reads the genome graph's modules and (parent, child,
    face) edges back from a compiled tree / tree_symmetric / cppn body, with
    local and zoo modules
  - no child ever attaches to a hinge's or brick's BACK face (the DANN's
    parent slot)
  - the dann brain's parameter count, output range and clock reset
  - translate_score / turn_score normalisation, and the early end of a
    normalised turn episode at a full turn
  - SWEEP_PROFILE is the module's original constants
"""

import contextlib
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

import genome_adapter
import shared
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from shared import (
    DSOCIAL_ZOO,
    SWEEP_PROFILE,
    TURN_TARGET,
    EvalProfile,
    SkillReward,
    build_loco_world_for_body,
    get_robot_state,
    make_brain_for_model,
    translate_score,
    turn_score,
    with_zoo_modules,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "zoo_benchmark"))
import brains  # noqa: E402

MAX_MODULES = 25
GENOMES = ["tree", "tree_symmetric", "cppn"]


def _bodies(genome_type: str, n: int = 12, seed: int = 0):
    adapter = genome_adapter.genome_adapter_from_cli(genome_type, MAX_MODULES)
    rng = np.random.default_rng(seed)
    return adapter, [adapter.create_individual(rng, MAX_MODULES, MAX_MODULES).genotype[adapter.genotype_key]
                     for _ in range(n)]


def _graph(genome_type: str, genotype: dict):
    if genome_type == "cppn":
        return genome_adapter._cppn_decode(genotype, MAX_MODULES)  # noqa: SLF001
    return TreeGenome.from_dict(genotype).to_networkx()


def _first_body():
    adapter, genotypes = _bodies("tree", n=1)
    return genotypes[0], adapter.to_spec


@pytest.mark.parametrize("zoo", [False, True])
@pytest.mark.parametrize("genome_type", GENOMES)
def test_model_topology_matches_genome_graph(genome_type, zoo):
    adapter, genotypes = _bodies(genome_type)
    to_spec = with_zoo_modules(adapter.to_spec) if zoo else adapter.to_spec
    for genotype in genotypes:
        model, _ = build_loco_world_for_body(genotype, to_spec)
        topo = brains.model_topology(model)
        # the cppn decoder's collision checks depend on module sizes
        with shared.zoo_module_constants() if zoo else contextlib.nullcontext():
            graph = _graph(genome_type, genotype)
        root = next(n for n, d in graph.nodes(data=True) if d["type"] == ModuleType.CORE.name)

        def node(name: str) -> int:  # module name "...-<parent>-<child>-<face>"
            return root if name == "core" else int(name.split("-")[-2])

        assert {(node(p), node(c), f) for p, c, f in topo["edges"]} == {
            (p, c, ModuleFaces[f]) for p, c, f in graph.edges(data="face")}
        assert {node(n): t for n, t in topo["type"].items()} == {
            n: brains._DIST_TYPE_ONEHOT[ModuleType[d["type"]]] for n, d in graph.nodes(data=True)}  # noqa: SLF001
        actuated = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[a, 0])
                    .removeprefix("robot1_").removesuffix("-servo") for a in range(model.nu)}
        assert actuated <= set(topo["type"])  # the DANN's joint -> module lookup


@pytest.mark.parametrize("genome_type", GENOMES)
def test_no_child_on_hinge_or_brick_back_face(genome_type):
    adapter, genotypes = _bodies(genome_type, n=20, seed=1)
    for genotype in genotypes:
        graph = _graph(genome_type, genotype)
        for p, _, face in graph.edges(data="face"):
            if graph.nodes[p]["type"] != ModuleType.CORE.name:
                assert face != ModuleFaces.BACK.name
        model, _ = build_loco_world_for_body(genotype, adapter.to_spec)
        brains.model_topology(model, parent_slot="back")  # raises on a parent-slot collision


@pytest.mark.parametrize("genome_type", GENOMES)
def test_dann_builds_on_many_bodies(genome_type):
    adapter, genotypes = _bodies(genome_type, n=40, seed=2)
    nested = 0
    for genotype in genotypes:
        model, data = build_loco_world_for_body(genotype, adapter.to_spec)
        brain = make_brain_for_model("dann", model, data)
        assert brain.num_params == 705
        nested += any(n.count("-") > 3 for n in brains.model_topology(model)["type"])
    if genome_type == "tree_symmetric":  # nested module names do turn up here
        assert nested > 0


def test_dann_brain_shape_range_and_reset():
    model, data = build_loco_world_for_body(*_first_body())
    brain = make_brain_for_model("dann", model, data)
    assert brain.num_params == 705
    brain.set_params(np.random.default_rng(0).normal(0, 2, brain.num_params))

    def rollout() -> np.ndarray:
        mujoco.mj_resetData(model, data)
        outs = []
        for _ in range(10):
            outs.append(brain.act(data, get_robot_state(data)))
            mujoco.mj_step(model, data)
        return np.array(outs)

    first = rollout()
    assert first.shape == (10, model.nu)
    assert np.all(np.abs(first) <= math.pi / 2)
    brain.reset()
    np.testing.assert_array_equal(rollout(), first)


def test_dann_rejects_wrong_length():
    model, data = build_loco_world_for_body(*_first_body())
    with pytest.raises(IndexError):
        make_brain_for_model("dann", model, data).set_params(np.zeros(10))


def test_translate_score_clips_to_unit_range():
    assert translate_score(-1.0, 15.0) == 0.0
    assert translate_score(0.25 * 15.0, 15.0) == pytest.approx(0.5)
    assert translate_score(100.0, 15.0) == 1.0


def test_turn_score_clips_to_unit_range():
    assert turn_score(-1.0) == 0.0
    assert turn_score(math.pi) == pytest.approx(0.5)
    assert turn_score(3 * TURN_TARGET) == 1.0


def _turn_episode(monkeypatch, profile: EvalProfile, yaw_per_step: float):
    """One turn episode with signed_vertical_yaw_delta stubbed to a constant,
    returning (fitness, info, sim time at the end)."""
    monkeypatch.setattr(shared, "signed_vertical_yaw_delta", lambda *_: yaw_per_step)
    genotype, to_spec = _first_body()
    reward = SkillReward(kind="rotate", turn_sign=+1, duration=5.0)
    shared._skill_worker_init(0, reward, genotype, to_spec, 25, "dann", profile=profile)  # noqa: SLF001
    n = shared._skill_worker_ctx["brain"].num_params  # noqa: SLF001
    fitness, info = shared._skill_worker_eval([0.0] * n)  # noqa: SLF001
    return fitness, info, float(shared._skill_worker_ctx["data"].time)  # noqa: SLF001


def test_normalised_turn_ends_at_full_turn(monkeypatch):
    profile = EvalProfile(**{**DSOCIAL_ZOO.__dict__, "turn_duration": 5.0})
    _, info, t_end = _turn_episode(monkeypatch, profile, 0.01)  # 2 pi after ~629 steps (1.26 s)
    assert info["score"] == 1.0
    assert t_end < shared.SETTLE_DURATION + 2.0


def test_normalised_turn_counts_net_yaw(monkeypatch):
    _, info, t_end = _turn_episode(monkeypatch, DSOCIAL_ZOO, -0.001)  # turning the wrong way
    assert info["raw_score"] < 0
    assert info["score"] == 0.0
    assert t_end >= shared.SETTLE_DURATION + 5.0


def test_sweep_turn_runs_full_episode(monkeypatch):
    _, info, t_end = _turn_episode(monkeypatch, SWEEP_PROFILE, 0.01)
    assert "score" not in info
    assert info["raw_score"] > TURN_TARGET
    assert t_end >= shared.SETTLE_DURATION + 5.0


def test_sweep_profile_is_the_original_constants():
    assert SWEEP_PROFILE.loco_duration == shared.LOCO_DURATION
    assert SWEEP_PROFILE.turn_duration == shared.TURN_DURATION
    assert SWEEP_PROFILE.control_step_freq == shared.CONTROL_STEP_FREQ
    assert SWEEP_PROFILE.ctrl_alpha == shared.CTRL_ALPHA
    assert SWEEP_PROFILE.jerk_weight == shared.JERK_PENALTY_WEIGHT
    assert SWEEP_PROFILE.jerk_threshold == shared.JERK_THRESHOLD
    assert SWEEP_PROFILE.height_threshold == shared.HEIGHT_PENALTY_THRESHOLD
    assert not SWEEP_PROFILE.height_at_spawn
    assert (SWEEP_PROFILE.cma_sigma, SWEEP_PROFILE.cma_popsize, SWEEP_PROFILE.cma_init_scale) == (
        shared.CMA_SIGMA, shared.CMA_POPSIZE, shared.CMA_INIT_SCALE)
    assert not SWEEP_PROFILE.normalised
