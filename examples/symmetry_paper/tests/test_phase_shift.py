import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brains import BangBangBrain, BangBangDriftBrain  # noqa: E402
from phase_shift import perturb_brain, random_subset  # noqa: E402


def _outputs(brain, ts):
    brain.reset()
    return np.array([brain.output_at(t) for t in ts])


def test_random_subset_is_never_a_whole_side():
    sides = np.array([1, 1, -1, -1, 0])
    for d in range(50):
        c = set(random_subset(sides, np.random.default_rng(d)))
        assert len(c) == 2 and c not in ({0, 1}, {2, 3})


def test_random_subset_none_when_only_whole_sides():
    assert random_subset(np.array([1, -1, 0]), np.random.default_rng(0)) is None


def test_no_mask_leaves_bang_bang_unchanged():
    rng = np.random.default_rng(1)
    ts = np.arange(0, 15, 0.04)
    for cls in (BangBangBrain, BangBangDriftBrain):
        a, b = cls(4), cls(4)
        params = rng.normal(0.5, 0.5, a.num_params)
        a.set_params(params)
        b.set_params(params)
        perturb_brain(b, np.zeros(4, dtype=bool), 0.5, 1.15)
        assert np.array_equal(_outputs(a, ts), _outputs(b, ts))


def test_side_shift_moves_only_masked_hinges_by_the_shift():
    brain = BangBangBrain(2)
    brain.set_params(np.full(brain.num_params, 0.5))
    ts = np.arange(0, 15, 0.01)
    before = _outputs(brain, ts)
    perturb_brain(brain, np.array([True, False]), 0.25, 1.0)
    after = _outputs(brain, ts)
    assert np.array_equal(before[:, 1], after[:, 1])
    assert not np.array_equal(before[:, 0], after[:, 0])
    # A quarter cycle later in the phase accumulator = hinge 0's square wave a quarter period earlier.
    f = float(np.asarray(brain.freq.value(0.0, brain.duration))[0])
    lag = round(0.25 / f / 0.01)
    assert np.mean(after[: -lag - 1, 0] == before[lag : -1, 0]) > 0.95
