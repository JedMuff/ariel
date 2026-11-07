"""Gait measures that need no left-right pairing, and the hinge knockout (mechanics.py)."""

import random

import numpy as np
import pytest
from ariel.ec.genotypes.tree.tying import random_tree_tied
from brains import BangBangBrain
from mechanics import (
    FrozenHinges,
    actuator_nodes,
    contact_stats,
    decompose,
    hinge_relations,
    inverse_participation,
    knockout_episode,
    mean_angles,
    measure_episode,
    nested_pairs,
    pair_rows,
    participation_ratio,
)
import genomes
from sim import build_world

DT = 0.02
T = np.arange(round(13.0 / DT)) * DT


def wave(freq=1.0, phase=0.0):
    return np.sin(2 * np.pi * freq * T + phase)


def test_decomposition_multiplies_back_to_displacement() -> None:
    t = np.arange(751) * DT
    xy = np.c_[0.1 * t + 0.01 * np.sin(2 * np.pi * t), 0.02 * t]
    displacement = 1.2
    d = decompose(xy, displacement, 1.0, DT, 15.0)
    assert d["cycles"] * d["path_per_cycle"] * d["straightness"] == pytest.approx(displacement)
    assert d["cycles"] == 15.0
    assert d["wobble"] >= 1.0


def test_decomposition_of_a_straight_line() -> None:
    t = np.arange(751) * DT
    xy = np.c_[0.1 * t, 0 * t]
    d = decompose(xy, 1.5, 1.0, DT, 15.0)
    assert d["straightness"] == pytest.approx(1.0, abs=1e-6)
    assert d["path_per_cycle"] == pytest.approx(0.1, abs=1e-6)


def test_decomposition_without_a_frequency_is_undefined() -> None:
    assert np.isnan(decompose(np.zeros((10, 2)), 0.0, float("nan"), DT, 15.0)["cycles"])


def test_participation_ratio_extremes() -> None:
    same = np.c_[wave(), wave(), wave()]
    assert participation_ratio(same) == pytest.approx(1.0, abs=1e-6)
    rng = np.random.default_rng(0)
    assert participation_ratio(rng.normal(size=(5000, 4))) == pytest.approx(4.0, abs=0.1)
    # Quarter-cycle sine and cosine are independent over whole cycles.
    assert participation_ratio(np.c_[wave(), wave(phase=np.pi / 2)]) == pytest.approx(2.0, abs=0.01)


def test_inverse_participation() -> None:
    assert inverse_participation(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(4.0)
    assert inverse_participation(np.array([5.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert np.isnan(inverse_participation(np.zeros(3)))


def _pairs(b, freq=1.0):
    a = wave()
    x = np.c_[a, b]
    return pair_rows(x, x, [1, 2], {1: 2, 2: 1}, {}, np.array([True, True]), np.ones(2), freq, DT)


def test_pair_quality_at_known_lags() -> None:
    row = _pairs(wave(phase=-np.pi / 2))[0]
    assert row["kind"] == "twin"
    assert row["limb_quality"] == pytest.approx(1.0, abs=1e-3)
    assert min(row["joint_lag"], 1 - row["joint_lag"]) == pytest.approx(0.25, abs=0.01)


def test_detuned_pair_is_not_locked() -> None:
    assert _pairs(wave(1.2), 1.1)[0]["limb_quality"] < 0.5


def test_pair_kinds() -> None:
    x = np.c_[wave(), wave(), wave()]
    rows = pair_rows(x, x, [1, 2, 3], {1: 2, 2: 1}, {1: 3, 3: 1}, np.ones(3, bool), np.ones(3), 1.0, DT)
    assert sorted(r["kind"] for r in rows) == ["mirror", "other", "twin"]


def test_inactive_hinges_make_no_pairs() -> None:
    x = np.c_[wave(), wave(), wave()]
    assert len(pair_rows(x, x, [1, 2, 3], {}, {}, np.array([True, False, False]), np.ones(3), 1.0, DT)) == 0


def test_nested_pairs_follow_the_tree() -> None:
    # core -> 1 -> 2 (brick) -> 3, core -> 4: 1 carries 3; 4 is on another branch.
    nodes = {str(i): {"type": t, "rotation": "DEG_0"} for i, t in
             enumerate(["CORE", "HINGE", "BRICK", "HINGE", "HINGE", "BRICK", "BRICK"])}
    edges = [{"parent": 0, "child": 1, "face": "FRONT"}, {"parent": 1, "child": 2, "face": "FRONT"},
             {"parent": 2, "child": 3, "face": "FRONT"}, {"parent": 0, "child": 4, "face": "BACK"},
             {"parent": 3, "child": 5, "face": "FRONT"}, {"parent": 4, "child": 6, "face": "FRONT"}]
    assert nested_pairs({"nodes": nodes, "edges": edges}, [1, 3, 4]) == {frozenset((1, 3))}


def test_planted_foot_and_dragged_foot() -> None:
    n = 200
    t = np.arange(n) * DT
    core = np.c_[0.1 * t, 0 * t, 0 * t]
    planted = np.tile([0.5, 0.0, 0.0], (n, 1))
    dragged = core + [0.1, 0.0, 0.0]
    contact = np.zeros((n, 2), dtype=bool)
    contact[:, 1] = True
    s = contact_stats(contact, np.stack([core, planted], axis=1), DT)
    assert s["foot_fwd_frac"] == pytest.approx(0.0, abs=1e-6)
    assert s["n_always"] == 1 and s["n_stepping"] == 0
    s = contact_stats(contact, np.stack([core, dragged], axis=1), DT)
    assert s["foot_fwd_frac"] == pytest.approx(1.0, abs=1e-6)


def test_stepping_foot_and_its_changes() -> None:
    n = 200
    pos = np.zeros((n, 2, 3))
    contact = np.zeros((n, 2), dtype=bool)
    contact[:100, 1] = True
    s = contact_stats(contact, pos, DT)
    assert s["n_stepping"] == 1
    assert s["stepping_duty"] == pytest.approx(0.5)
    assert s["contact_changes_per_s"] == pytest.approx(1 / (n * DT))


@pytest.fixture(scope="module")
def tied_body():
    random.seed(3)
    np.random.seed(3)
    return random_tree_tied(25, 0.47).to_dict()


def test_actuator_nodes_are_the_genome_hinges(tied_body) -> None:
    model, _ = build_world(tied_body)
    hinges = {int(n) for n, v in tied_body["nodes"].items() if v["type"] == "HINGE"}
    assert set(actuator_nodes(model)) == hinges


def test_twins_are_symmetric_and_only_hinges(tied_body) -> None:
    model, _ = build_world(tied_body)
    nodes = actuator_nodes(model)
    twin, _ = hinge_relations(tied_body, nodes)
    assert all(twin[twin[n]] == n for n in twin)
    assert set(twin) <= set(nodes)


def test_mirror_partners_of_a_symmetric_body() -> None:
    from ariel.ec.genotypes.tree.symmetry import MirrorAxis
    random.seed(1)
    genome = None
    for _ in range(50):
        g = genomes.random_body(genomes.genome_ops("tree_symmetric"))[0]
        model, _ = build_world(g)
        if model.nu >= 4:
            genome = g
            break
    assert genome is not None
    nodes = actuator_nodes(model)
    _, mirror = hinge_relations(genome, nodes)
    assert mirror, "a symmetric body with four hinges has an off-midline pair"
    assert all(mirror[mirror[n]] == n for n in mirror)


def _frozen_outputs(frozen):
    brain = BangBangBrain(3)
    brain.set_params(np.full(brain.num_params, 0.5))
    plain = BangBangBrain(3)
    plain.set_params(np.full(plain.num_params, 0.5))

    class D:
        time = 3.2

    return plain.act(D, None), FrozenHinges(brain, np.array(frozen, dtype=int)).act(D, None)


def test_frozen_hinge_only_changes_that_hinge() -> None:
    plain, frozen = _frozen_outputs([1])
    assert frozen[1] == 0.0
    assert frozen[0] == plain[0] and frozen[2] == plain[2]


def test_frozen_hinge_holds_its_target() -> None:
    brain = BangBangBrain(3)
    brain.set_params(np.full(brain.num_params, 0.5))

    class D:
        time = 3.2

    assert FrozenHinges(brain, np.array([2]), 0.7).act(D, None)[2] == 0.7


def test_mean_angle_of_an_idle_body_is_near_rest(tied_body) -> None:
    # Bang-bang at x0 drives hinges; the mean angle stays inside the joint limits.
    model, _ = build_world(tied_body)
    held = mean_angles(tied_body, np.full(4 + 5 * model.nu, 0.5), "bang_bang")
    assert held.shape == (model.nu,)
    assert np.all(np.abs(held) <= np.pi / 2 + 0.05)


def test_no_frozen_hinges_change_nothing() -> None:
    plain, frozen = _frozen_outputs([])
    assert np.array_equal(plain, frozen)


def test_recording_does_not_change_the_episode(tied_body) -> None:
    """The measuring replay and the bare episode reach the same displacement."""
    model, _ = build_world(tied_body)
    weights = np.full(4 + 5 * model.nu, 0.5)
    row, hinges, _, _ = measure_episode(tied_body, weights, "bang_bang")
    assert row["displacement"] == pytest.approx(knockout_episode(tied_body, weights, "bang_bang", None), abs=1e-9)
    assert len(hinges) == model.nu
    assert row["cycles"] * row["path_per_cycle"] * row["straightness"] == pytest.approx(row["displacement"], abs=1e-6)


def test_all_frozen_barely_moves(tied_body) -> None:
    model, _ = build_world(tied_body)
    weights = np.full(4 + 5 * model.nu, 0.5)
    assert abs(knockout_episode(tied_body, weights, "bang_bang", list(range(model.nu)))) < 0.05
