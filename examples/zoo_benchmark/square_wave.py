"""Time-varying square-wave motion patterns for hinge servos.

Each servo swings between a low and a high angle. Its rhythm is a frequency f
and a duty cycle D (the fraction of each cycle spent high). Both can change
smoothly over a timeline, and every switch between angles is cosine-eased
rather than instantaneous.

Each servo keeps a phase accumulator, measured in cycles:

    phase += f(t) * dt          (integrated, never frac(f(t) * t))
    p = frac(phase)             position within the current cycle
    high while p < D(t)

The naive frac(f(t) * t) is only right for a constant f. When f changes, it
jumps whenever f(t) * t does, which skips or repeats cycles, and its actual
frequency is d(f t)/dt = f + t f'(t) rather than f. Integrating the phase keeps
the motion continuous and the cycle count equal to the integral of f.

The engine is pure (no I/O): step() maps (pattern, state, t, dt) to
(new_state, angle). The numeric fields of a pattern may be numpy arrays, so
one pattern can drive many servos at once (this is how the zoo brains in
brains.py use it). Angles are in degrees. TimelineController is the adapter
that writes them to a MuJoCo model's ctrl (radians).

Usage (validate a timeline file and run it on a zoo body):
  python square_wave.py square_wave_example.json --body gecko
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

F_MIN, F_MAX = 0.01, 10.0     # Hz
D_MIN, D_MAX = 0.02, 0.98     # duty cycle
SEGMENT_MIN = 0.01            # s, minimum high/low time in high_low mode

Mode = Literal["freq_duty", "high_low"]
EndBehavior = Literal["hold", "loop", "stop"]
MODES = ("freq_duty", "high_low")
END_BEHAVIORS = ("hold", "loop", "stop")


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class ModulatedValue:
    """start -> end linear ramp over the timeline, plus an optional sinusoidal wobble."""

    start: Any
    end: Any
    wobble_amp: Any = 0.0
    wobble_rate: Any = 0.0  # Hz

    @classmethod
    def constant(cls, value: float) -> ModulatedValue:
        return cls(value, value)

    def value(self, t: float, duration: float) -> Any:
        ramp = min(max(t / duration, 0.0), 1.0)
        return (self.start + (np.asarray(self.end) - self.start) * ramp
                + self.wobble_amp * np.sin(2 * math.pi * np.asarray(self.wobble_rate) * t))


@dataclass
class ServoPattern:
    """One servo's pattern.

    mode "freq_duty": a = frequency (Hz), b = duty cycle (0-1).
    mode "high_low":  a = high time (s),  b = low time (s).
    """

    servo_id: Any
    mode: Mode
    a: ModulatedValue
    b: ModulatedValue
    low_angle: Any         # degrees
    high_angle: Any        # degrees
    phase_offset: Any = 0.0      # cycles, 0-1
    transition_time: Any = 0.0   # s per eased move


@dataclass
class Timeline:
    duration: float
    patterns: list[ServoPattern]
    end_behavior: EndBehavior = "hold"
    update_rate_hz: float = 50.0


@dataclass
class ServoLimits:
    min_angle: float = -90.0  # kgd hinge: ctrlrange +-pi/2
    max_angle: float = 90.0


@dataclass
class PatternState:
    phase: Any          # in [0, 1)
    cycles: Any = 0     # completed cycles (integer part removed from phase)
    halted: bool = False

    @classmethod
    def initial(cls, pattern: ServoPattern) -> PatternState:
        phase = np.mod(np.asarray(pattern.phase_offset, dtype=np.float64), 1.0)
        return cls(phase=phase, cycles=np.zeros_like(phase, dtype=np.int64))


@dataclass
class Rhythm:
    f: Any
    duty: Any
    f_clamped: Any = field(default=False)
    duty_clamped: Any = field(default=False)
    segment_clamped: Any = field(default=False)


# ── Engine ────────────────────────────────────────────────────────────────────


def param_time(t: float, duration: float, end_behavior: EndBehavior) -> float:
    """Time at which to evaluate the pattern parameters (end-of-timeline handling)."""
    if t < duration:
        return t
    if end_behavior == "loop":
        return math.fmod(t, duration)
    return duration  # hold (and stop, which never gets this far)


def rhythm(pattern: ServoPattern, t: float, duration: float) -> Rhythm:
    """Evaluate and sanitise (f, D) at parameter time t."""
    a = np.asarray(pattern.a.value(t, duration), dtype=np.float64)
    b = np.asarray(pattern.b.value(t, duration), dtype=np.float64)
    segment_clamped = np.zeros(np.broadcast(a, b).shape, dtype=bool)
    if pattern.mode == "freq_duty":
        f_raw, d_raw = a, b
    elif pattern.mode == "high_low":
        segment_clamped = (a < SEGMENT_MIN) | (b < SEGMENT_MIN)
        th, tl = np.maximum(a, SEGMENT_MIN), np.maximum(b, SEGMENT_MIN)
        f_raw, d_raw = 1.0 / (th + tl), th / (th + tl)
    else:
        raise ValueError(f"Unknown mode {pattern.mode!r} (expected one of {MODES})")
    f, duty = np.clip(f_raw, F_MIN, F_MAX), np.clip(d_raw, D_MIN, D_MAX)
    return Rhythm(f, duty, f != f_raw, duty != d_raw, segment_clamped)


def transition_width(transition_time: Any, f: Any, duty: Any) -> tuple[Any, Any]:
    """Transition width in phase units, clamped to min(D, 1 - D); also returns whether it was clamped."""
    w_raw = np.asarray(transition_time, dtype=np.float64) * f
    w_max = np.minimum(duty, 1.0 - duty)
    return np.minimum(w_raw, w_max), w_raw > w_max + 1e-12


def ease(x: Any) -> Any:
    return 0.5 - 0.5 * np.cos(math.pi * x)


def eased_level(p: Any, duty: Any, w: Any) -> Any:
    """Level in [0, 1] at cycle position p.

    The rise is eased over [0, w) at the start of the high segment and the fall
    over [D, D + w), so the servo sits fully at high_angle for (D - w) / f
    seconds per cycle, not D / f. This is intentional: the duty cycle measures
    when the move towards high starts, not how long the servo holds there.
    With w == 0 this is a hard square wave.
    """
    p, duty, w = np.broadcast_arrays(*(np.asarray(x, dtype=np.float64) for x in (p, duty, w)))
    w_safe = np.where(w > 0, w, 1.0)
    rising = ease(p / w_safe)
    falling = 1.0 - ease((p - duty) / w_safe)
    return np.select(
        [p < w, p < duty, p < duty + w],
        [rising, 1.0, falling],
        default=0.0,
    )


def step(
    pattern: ServoPattern,
    state: PatternState,
    t: float,
    dt: float,
    *,
    duration: float,
    end_behavior: EndBehavior = "hold",
    limits: ServoLimits = ServoLimits(),  # noqa: B008
) -> tuple[PatternState, Any]:
    """Advance one update: evaluate params at t, integrate phase by f * dt, return the angle (deg)."""
    if state.halted or (end_behavior == "stop" and t >= duration):
        low = np.clip(pattern.low_angle, limits.min_angle, limits.max_angle)
        return replace(state, halted=True), low

    r = rhythm(pattern, param_time(t, duration, end_behavior), duration)
    phase = state.phase + r.f * dt
    wraps = np.floor(phase)
    phase = phase - wraps  # keep the float64 accumulator in [0, 1) for long runs
    new_state = PatternState(phase=phase, cycles=state.cycles + wraps.astype(np.int64))

    w, _ = transition_width(pattern.transition_time, r.f, r.duty)
    level = eased_level(phase, r.duty, w)
    target = pattern.low_angle + level * (np.asarray(pattern.high_angle) - pattern.low_angle)
    return new_state, np.clip(target, limits.min_angle, limits.max_angle)


# ── Validation ────────────────────────────────────────────────────────────────


def validate(
    timeline: Timeline,
    limits: ServoLimits | dict[Any, ServoLimits] = ServoLimits(),  # noqa: B008
    known_servo_ids: set[Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Check a timeline at load time; return (errors, warnings).

    limits is either one ServoLimits for every servo or a dict keyed by servo_id.
    Warnings are found by sampling the timeline at update_rate_hz and report the
    first time each problem occurs.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if timeline.duration <= 0:
        errors.append(f"duration must be > 0 (got {timeline.duration})")
    if timeline.end_behavior not in END_BEHAVIORS:
        errors.append(f"end_behavior {timeline.end_behavior!r} not in {END_BEHAVIORS}")
    if timeline.update_rate_hz <= 0:
        errors.append(f"update_rate_hz must be > 0 (got {timeline.update_rate_hz})")

    for pat in timeline.patterns:
        sid = pat.servo_id
        if known_servo_ids is not None and sid not in known_servo_ids:
            errors.append(f"servo {sid!r}: unknown servo_id")
            continue
        if pat.mode not in MODES:
            errors.append(f"servo {sid!r}: unknown mode {pat.mode!r} (expected one of {MODES})")
            continue
        lim = limits.get(sid, ServoLimits()) if isinstance(limits, dict) else limits
        for name in ("low_angle", "high_angle"):
            angle = getattr(pat, name)
            if np.any(np.asarray(angle) < lim.min_angle) or np.any(np.asarray(angle) > lim.max_angle):
                errors.append(f"servo {sid!r}: {name} {angle} outside [{lim.min_angle}, {lim.max_angle}] deg")
        if np.all(np.asarray(pat.transition_time) == 0):
            warnings.append(f"servo {sid!r}: transition_time is 0, so the angle switches as a hard square wave")

    if errors:
        return errors, warnings

    times = np.arange(0.0, timeline.duration + 0.5 / timeline.update_rate_hz, 1.0 / timeline.update_rate_hz)
    for pat in timeline.patterns:
        sid = pat.servo_id
        first: dict[str, float] = {}
        for t in times:
            r = rhythm(pat, float(t), timeline.duration)
            _, w_clamped = transition_width(pat.transition_time, r.f, r.duty)
            for key, hit in (("f", r.f_clamped), ("duty", r.duty_clamped),
                             ("segment", r.segment_clamped), ("transition", w_clamped)):
                if key not in first and np.any(hit):
                    first[key] = float(t)
        messages = {
            "f": f"frequency clamped to [{F_MIN}, {F_MAX}] Hz",
            "duty": f"duty cycle clamped to [{D_MIN}, {D_MAX}]",
            "segment": f"high or low time clamped to >= {SEGMENT_MIN} s",
            "transition": "transition_time shortened to fit min(D, 1 - D) of the cycle",
        }
        for key, t in first.items():
            warnings.append(f"servo {sid!r}: {messages[key]} (first at t = {t:.3f} s)")
    return errors, warnings


# ── Flat parameter vector ─────────────────────────────────────────────────────

VECTOR_FIELDS = (
    "a.start", "a.end", "a.wobble_amp", "a.wobble_rate",
    "b.start", "b.end", "b.wobble_amp", "b.wobble_rate",
    "low_angle", "high_angle", "phase_offset", "transition_time",
)


def to_vector(pattern: ServoPattern) -> np.ndarray:
    """The 12 numeric fields of a pattern, in VECTOR_FIELDS order (mode and servo_id stay in the template)."""
    return np.array([
        pattern.a.start, pattern.a.end, pattern.a.wobble_amp, pattern.a.wobble_rate,
        pattern.b.start, pattern.b.end, pattern.b.wobble_amp, pattern.b.wobble_rate,
        pattern.low_angle, pattern.high_angle, pattern.phase_offset, pattern.transition_time,
    ], dtype=np.float64)


def from_vector(vector: np.ndarray, template: ServoPattern) -> ServoPattern:
    v = [float(x) for x in np.asarray(vector, dtype=np.float64)]
    if len(v) != len(VECTOR_FIELDS):
        raise IndexError(f"Expected {len(VECTOR_FIELDS)} values, got {len(v)}")
    return replace(
        template,
        a=ModulatedValue(*v[0:4]), b=ModulatedValue(*v[4:8]),
        low_angle=v[8], high_angle=v[9], phase_offset=v[10], transition_time=v[11],
    )


# ── Config ────────────────────────────────────────────────────────────────────


def _modulated_from_json(obj: Any) -> ModulatedValue:
    if isinstance(obj, (int, float)):
        return ModulatedValue.constant(float(obj))
    return ModulatedValue(**{f.name: float(obj[f.name]) for f in fields(ModulatedValue) if f.name in obj})


def timeline_from_dict(cfg: dict) -> Timeline:
    """Build a Timeline from a JSON-style dict; a and b may be a number (constant) or a ModulatedValue dict."""
    patterns = [
        ServoPattern(
            servo_id=p["servo_id"], mode=p["mode"],
            a=_modulated_from_json(p["a"]), b=_modulated_from_json(p["b"]),
            low_angle=float(p["low_angle"]), high_angle=float(p["high_angle"]),
            phase_offset=float(p.get("phase_offset", 0.0)),
            transition_time=float(p.get("transition_time", 0.0)),
        )
        for p in cfg["patterns"]
    ]
    return Timeline(
        duration=float(cfg["duration"]), patterns=patterns,
        end_behavior=cfg.get("end_behavior", "hold"),
        update_rate_hz=float(cfg.get("update_rate_hz", 50.0)),
    )


def load_timeline(path: Path) -> Timeline:
    return timeline_from_dict(json.loads(Path(path).read_text()))


# ── MuJoCo adapter ────────────────────────────────────────────────────────────


class TimelineController:
    """Drives a MuJoCo model's actuators from a Timeline, with the zoo brain interface.

    servo_id is an actuator name or index. Unpatterned actuators are held at 0.
    act(t, state) steps every pattern by the time since the previous call (the
    control period) and returns ctrl in radians. Raises if validate() reports errors.
    """

    def __init__(self, timeline: Timeline, model: Any, limits: ServoLimits = ServoLimits()) -> None:  # noqa: B008
        import mujoco

        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
        known = set(names) | set(range(model.nu))
        errors, self.warnings = validate(timeline, limits, known)
        if errors:
            raise ValueError("Invalid timeline:\n  " + "\n  ".join(errors))
        self.timeline, self.limits, self.nu = timeline, limits, model.nu
        self._index = [names.get(p.servo_id, p.servo_id) for p in timeline.patterns]
        self.reset()

    def reset(self) -> None:
        self._states = [PatternState.initial(p) for p in self.timeline.patterns]
        self._t = 0.0

    def safe_ctrl(self) -> np.ndarray:
        """Every patterned actuator at its low_angle (the shutdown / stop pose)."""
        out = np.zeros(self.nu)
        for idx, pat in zip(self._index, self.timeline.patterns, strict=True):
            out[idx] = math.radians(float(np.clip(pat.low_angle, self.limits.min_angle, self.limits.max_angle)))
        return out

    def act(self, t: float, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        dt, self._t = t - self._t, t
        out = np.zeros(self.nu)
        tl = self.timeline
        for k, (idx, pat) in enumerate(zip(self._index, tl.patterns, strict=True)):
            self._states[k], angle = step(pat, self._states[k], t, dt, duration=tl.duration,
                                          end_behavior=tl.end_behavior, limits=self.limits)
            out[idx] = math.radians(float(angle))
        return out


def main() -> None:
    import argparse

    import canonical_bodies
    from zoo_benchmark import build_world, run_episode

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--body", default="gecko", choices=sorted(canonical_bodies.get_all()))
    parser.add_argument("--control-freq", type=float, default=20.0)
    args = parser.parse_args()

    timeline = load_timeline(args.timeline)
    model, data = build_world(args.body)
    controller = TimelineController(timeline, model)
    for w in controller.warnings:
        print(f"WARNING: {w}")
    speed = run_episode(model, data, controller, timeline.duration, args.control_freq)
    print(f"{args.body}: x-speed {speed:+.4f} m/s over {timeline.duration:g} s")


if __name__ == "__main__":
    main()
