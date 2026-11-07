"""brains.CpgBrain against the zoo benchmark's revolve_cpg (shared.RevolveCpgBrain)."""

import itertools
import math

import numpy as np
import pytest
from brains import SETTLE_DURATION, CpgBrain, make_brain_for_model
import genomes
from sim import build_world


def reference_targets(w: np.ndarray, n: int, times: np.ndarray) -> np.ndarray:
    """shared.RevolveCpgBrain (one RK4 step per call, no time scale), copied verbatim."""
    m = np.zeros((2 * n, 2 * n))
    for i in range(n):
        m[i, n + i] = +w[i]
        m[n + i, i] = -w[i]
    for (i, j), wij in zip(itertools.combinations(range(n), 2), w[n:], strict=True):
        m[i, j] = +wij
        m[j, i] = -wij
    s = np.hstack([np.full(n, 1.0), np.full(n, -1.0)]) * 0.5 * np.sqrt(2)
    t_prev, out = 0.0, []
    for t in times:
        dt = t - t_prev
        a1 = m @ s
        a2 = m @ (s + dt / 2 * a1)
        a3 = m @ (s + dt / 2 * a2)
        a4 = m @ (s + dt * a3)
        s = np.clip(s + dt / 6 * (a1 + 2 * (a2 + a3) + a4), -1, 1)
        t_prev = t
        out.append(s[:n] * (math.pi / 2))
    return np.array(out)


TIMES = np.arange(301) * 0.05


@pytest.mark.parametrize("n", [1, 2, 8, 14])
def test_matches_zoo_revolve_cpg(n):
    brain = CpgBrain(n)
    assert brain.num_params == n + n * (n - 1) // 2
    w = np.random.default_rng(n).normal(0.5, 1.0, brain.num_params)
    brain.set_params(w)
    got = np.array([brain.output_at(t) for t in TIMES])
    np.testing.assert_allclose(got, reference_targets(w, n, TIMES), atol=1e-12)


def test_reset_and_make_brain_for_model():
    model, data = build_world(genomes.random_body(genomes.genome_ops("tree"))[0])
    brain = make_brain_for_model("cpg", model, data)
    assert isinstance(brain, CpgBrain) and brain.n == model.nu
    brain.set_params(np.full(brain.num_params, 0.5))
    data.time = SETTLE_DURATION + 1.0
    first = brain.act(data, None)
    brain.reset()
    np.testing.assert_allclose(brain.act(data, None), first)
