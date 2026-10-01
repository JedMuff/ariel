"""Tests for the square-wave engine (run: pytest test_square_wave.py)."""

import math

import numpy as np
import pytest

from square_wave import (
    F_MAX,
    ModulatedValue,
    PatternState,
    ServoLimits,
    ServoPattern,
    Timeline,
    TimelineController,
    eased_level,
    from_vector,
    load_timeline,
    rhythm,
    step,
    to_vector,
    validate,
)


def pattern(f=2.0, duty=0.25, transition=0.0, offset=0.0, mode="freq_duty", **kw) -> ServoPattern:
    a = f if isinstance(f, ModulatedValue) else ModulatedValue.constant(f)
    b = duty if isinstance(duty, ModulatedValue) else ModulatedValue.constant(duty)
    return ServoPattern(servo_id=0, mode=mode, a=a, b=b, low_angle=kw.get("low", -30.0),
                        high_angle=kw.get("high", 30.0), phase_offset=offset, transition_time=transition)


def run(pat, duration, rate, t_end=None, end_behavior="hold"):
    """Step a pattern from t = 0 to t_end; return times, angles, phases and cycle counts (after each step)."""
    dt = 1.0 / rate
    n = round((t_end if t_end is not None else duration) * rate)
    state = PatternState.initial(pat)
    ts, angles, phases, cycles = [], [], [], []
    for k in range(n):
        t = k * dt
        state, angle = step(pat, state, t, dt, duration=duration, end_behavior=end_behavior)
        ts.append(t)
        angles.append(float(angle))
        phases.append(float(state.phase))
        cycles.append(int(state.cycles))
    return np.array(ts), np.array(angles), np.array(phases), np.array(cycles)


def unwrapped(phases, cycles):
    return phases + cycles


# 1. Constant rhythm
def test_constant_rhythm():
    pat = pattern(f=2.0, duty=0.25, low=0.0, high=10.0)
    _, angles, _, _ = run(pat, duration=10.0, rate=1000)
    high = angles > 5.0
    rising_edges = np.count_nonzero(high[1:] & ~high[:-1]) + int(high[0])
    assert abs(rising_edges - 20) <= 1
    assert abs(high.mean() - 0.25) < 0.01


# 2. Phase continuity
def test_phase_increment_bounded_and_non_negative():
    rate = 1000
    pat = pattern(f=ModulatedValue(0.5, 5.0), duty=0.5)
    _, _, phases, cycles = run(pat, duration=10.0, rate=rate)
    inc = np.diff(unwrapped(phases, cycles))
    assert np.all(inc >= 0)
    assert np.all(inc <= F_MAX / rate + 1e-12)


# 3. Naive vs integrated
def test_integrated_cycles_match_integral_of_f():
    # f ramps 0.5 -> 5 Hz over 10 s, so the integral of f is (0.5 + 5) / 2 * 10 = 27.5 cycles.
    # The naive frac(f(t) * t) would count f(10) * 10 = 50 cycles: its effective
    # frequency is d(f t)/dt = f + t f'(t), not f.
    duration, rate = 10.0, 1000
    pat = pattern(f=ModulatedValue(0.5, 5.0), duty=0.5)
    _, _, phases, cycles = run(pat, duration=duration, rate=rate)
    integral = (0.5 + 5.0) / 2 * duration
    assert abs(unwrapped(phases, cycles)[-1] - integral) < 1.0
    naive_cycles = 5.0 * duration
    assert abs(naive_cycles - integral) > 1.0


# 4. Mode conversion
def test_high_low_equals_freq_duty():
    hl = pattern(f=0.3, duty=0.7, mode="high_low", transition=0.05)
    fd = pattern(f=1.0, duty=0.3, transition=0.05)
    r = rhythm(hl, 0.0, 10.0)
    assert math.isclose(float(r.f), 1.0) and math.isclose(float(r.duty), 0.3)
    np.testing.assert_allclose(run(hl, 10.0, 200)[1], run(fd, 10.0, 200)[1], atol=1e-9)


# 5. Easing
def test_easing_bounds_continuity_and_plateaus():
    p = np.linspace(0, 1, 100001, endpoint=False)
    duty, w = 0.4, 0.1
    level = eased_level(p, duty, w)
    assert level.min() >= 0 and level.max() <= 1
    # Max slope of the cosine ease is pi / (2 w) per unit phase.
    assert np.abs(np.diff(level)).max() <= math.pi / (2 * w) * (p[1] - p[0]) + 1e-9
    assert np.all(level[(p >= w) & (p < duty)] == 1.0)
    assert np.all(level[p >= duty + w] == 0.0)
    # Hard square wave when w == 0.
    hard = eased_level(p, duty, 0.0)
    assert set(np.unique(hard)) == {0.0, 1.0}
    assert abs(hard.mean() - duty) < 1e-4


def test_angle_continuity_in_time():
    rate, transition = 1000, 0.1
    pat = pattern(f=1.0, duty=0.5, transition=transition, low=0.0, high=60.0)
    _, angles, _, _ = run(pat, duration=5.0, rate=rate)
    max_rate = 60.0 * math.pi / (2 * transition)  # deg/s at the steepest point of the ease
    assert np.abs(np.diff(angles)).max() <= max_rate / rate + 1e-9


# 6. Clamping
def test_clamps_and_warnings():
    pat = pattern(f=ModulatedValue(0.001, 20.0), duty=ModulatedValue(0.0, 1.0), transition=1.0,
                  low=-120.0, high=30.0)
    r = rhythm(pat, 0.0, 10.0)
    assert float(r.f) == 0.01 and float(r.duty) == 0.02
    r = rhythm(pat, 10.0, 10.0)
    assert float(r.f) == 10.0 and float(r.duty) == 0.98

    _, angles, _, _ = run(pat, duration=10.0, rate=100)
    assert angles.min() >= -90.0

    tl = Timeline(duration=10.0, patterns=[pat])
    errors, _ = validate(tl, ServoLimits())
    assert any("low_angle" in e for e in errors)

    tl = Timeline(duration=10.0, patterns=[pattern(f=ModulatedValue(0.001, 20.0),
                                                    duty=ModulatedValue(0.0, 1.0), transition=1.0)])
    errors, warnings = validate(tl, ServoLimits())
    assert not errors
    text = "\n".join(warnings)
    assert "frequency clamped" in text and "duty cycle clamped" in text and "transition_time shortened" in text

    hl = pattern(f=0.001, duty=0.5, mode="high_low", transition=0.0)
    _, warnings = validate(Timeline(duration=1.0, patterns=[hl]))
    assert any("high or low time" in w for w in warnings)
    assert any("hard square wave" in w for w in warnings)


def test_validate_errors():
    errors, _ = validate(Timeline(duration=0.0, patterns=[pattern()]))
    assert any("duration" in e for e in errors)
    errors, _ = validate(Timeline(duration=1.0, patterns=[pattern()]), known_servo_ids={1, 2})
    assert any("unknown servo_id" in e for e in errors)


# 9. Phase offset
def test_phase_offset_antiphase():
    # Eased edges keep the levels continuous, so float round-off at the switch points can't flip a sample.
    a = pattern(f=1.0, duty=0.5, transition=0.1, offset=0.0, low=0.0, high=1.0)
    b = pattern(f=1.0, duty=0.5, transition=0.1, offset=0.5, low=0.0, high=1.0)
    _, la, _, _ = run(a, 5.0, 1000)
    _, lb, _, _ = run(b, 5.0, 1000)
    np.testing.assert_allclose(la + lb, 1.0, atol=1e-6)


# 10. End behaviours
def test_end_hold():
    pat = pattern(f=ModulatedValue(1.0, 2.0, wobble_amp=0.5, wobble_rate=0.3), duty=0.5)
    r_end = rhythm(pat, 4.0, 4.0)
    _, _, phases, cycles = run(pat, duration=4.0, rate=1000, t_end=6.0)
    # After the timeline, f is held at f(duration).
    inc = np.diff(unwrapped(phases, cycles))[5000:]
    np.testing.assert_allclose(inc, float(r_end.f) / 1000)


def test_end_loop_phase_continuous():
    pat = pattern(f=ModulatedValue(1.0, 3.0), duty=0.5)
    _, _, phases, cycles = run(pat, duration=2.0, rate=1000, t_end=4.0, end_behavior="loop")
    inc = np.diff(unwrapped(phases, cycles))
    assert np.all(inc > 0)
    # Just before the loop point f ~ 3 Hz, just after it restarts at 1 Hz, with no phase jump.
    assert inc[1998] == pytest.approx(3.0 / 1000, rel=1e-2)
    assert inc[2000] == pytest.approx(1.0 / 1000, rel=1e-2)
    # The second pass repeats the first pass's frequency profile.
    np.testing.assert_allclose(inc[2000:3998], inc[0:1998], rtol=1e-6)


def test_end_stop():
    pat = pattern(f=1.0, duty=0.5, low=-20.0, high=40.0)
    ts, angles, _, _ = run(pat, duration=2.0, rate=100, t_end=3.0, end_behavior="stop")
    assert np.all(angles[ts >= 2.0] == -20.0)
    assert angles[ts < 2.0].max() == 40.0


# 11. Vector round trip
def test_vector_round_trip():
    p = ServoPattern(servo_id="hip", mode="high_low", a=ModulatedValue(0.3, 0.4, 0.05, 0.2),
                     b=ModulatedValue(0.7, 0.6, 0.01, 0.1), low_angle=-10.0, high_angle=35.0,
                     phase_offset=0.25, transition_time=0.08)
    v = to_vector(p)
    assert v.shape == (12,)
    assert from_vector(v, p) == p


# Vectorised use (as in the zoo brains): array fields give the same result as scalar patterns.
def test_vectorised_matches_scalar():
    vec = ServoPattern(servo_id=(0, 1), mode="freq_duty", a=ModulatedValue(np.array([1.0, 2.0]), np.array([2.0, 0.5])),
                       b=ModulatedValue.constant(np.array([0.3, 0.6])), low_angle=np.array([-10.0, 20.0]),
                       high_angle=np.array([50.0, -40.0]), phase_offset=np.array([0.0, 0.7]),
                       transition_time=np.array([0.05, 0.2]))
    state = PatternState.initial(vec)
    out = []
    for k in range(400):
        state, angle = step(vec, state, k / 20, 1 / 20 if k else 0.0, duration=15.0)
        out.append(angle)
    out = np.array(out)
    for i in range(2):
        single = ServoPattern(servo_id=i, mode="freq_duty",
                              a=ModulatedValue(vec.a.start[i], vec.a.end[i]),
                              b=ModulatedValue.constant(vec.b.start[i]), low_angle=vec.low_angle[i],
                              high_angle=vec.high_angle[i], phase_offset=vec.phase_offset[i],
                              transition_time=vec.transition_time[i])
        s = PatternState.initial(single)
        for k in range(400):
            s, angle = step(single, s, k / 20, 1 / 20 if k else 0.0, duration=15.0)
            assert out[k, i] == pytest.approx(float(angle))


def test_example_timeline_runs_on_gecko():
    from pathlib import Path

    from zoo_benchmark import build_world, run_episode

    timeline = load_timeline(Path(__file__).parent / "square_wave_example.json")
    model, data = build_world("gecko")
    controller = TimelineController(timeline, model)
    assert not any("clamped" in w for w in controller.warnings)
    ctrl = np.array([controller.act(k / 20, None) for k in range(40)])
    assert np.count_nonzero(np.abs(ctrl).max(axis=0)) == 2  # only the two patterned hinges move
    np.testing.assert_allclose(ctrl[:, 0], -ctrl[:, 1], atol=1e-9)  # half a cycle apart, symmetric angles
    np.testing.assert_allclose(controller.safe_ctrl()[:2], math.radians(-45.0))
    assert np.isfinite(run_episode(model, data, controller, timeline.duration, 20))
