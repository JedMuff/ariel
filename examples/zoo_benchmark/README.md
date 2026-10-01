# Zoo brain benchmark

CMA-ES optimises one brain on one of kgd's 23 canonical bodies, following the
apets-ariel zoo protocol:

- 15 s episodes
- 20 Hz control
- fitness = core x-speed

`zoo_benchmark.py` has the details. `slurm/run_zoo_benchmark.sh` runs ann, sine,
revolve_cpg and matsuoka. `slurm/run_zoo_square_benchmark.sh` runs one
square-wave brain (`bang_bang` by default, or `BRAIN=square_sync`) into the
same output directory. It leaves out `square`, which trained far slower in a
gecko pilot. Aggregate and plot all of them together:

```bash
python aggregate_zoo_benchmark.py __data__/ariel_zoo_benchmark
python plot_zoo_fitness_curves.py __data__/ariel_zoo_benchmark
python plot_zoo_wall_time.py __data__/ariel_zoo_benchmark
```

## Square-wave motion patterns (`square_wave.py`)

Each hinge is driven back and forth between `low_angle` and `high_angle`. Two
things can change smoothly over the timeline:

- the rhythm (frequency `f`)
- the duty cycle `D`, the fraction of each cycle spent high

Each hinge keeps a phase accumulator, `phase += f(t) * dt`, and is high while
`frac(phase) < D(t)`. Integrating, rather than using `frac(f(t) * t)`, keeps the
motion continuous when `f` changes and makes the cycle count equal the integral
of `f`.

### Parameters

Every time-varying parameter is a `ModulatedValue`:

```
value(t) = start + (end - start) * clamp(t / duration, 0, 1) + wobble_amp * sin(2 pi wobble_rate t)
```

A `ServoPattern` has the following fields:

- `servo_id`: an actuator name or index
- `mode`: `freq_duty` or `high_low` (see below)
- `a` and `b`: two ModulatedValues, whose meaning depends on the mode
- `low_angle` and `high_angle`: in degrees
- `phase_offset`: in cycles, used to stagger hinges
- `transition_time`: in seconds, the length of each cosine-eased move

The two modes:

- **`freq_duty`**: `a` is the frequency in Hz and `b` is the duty cycle (0–1).
- **`high_low`**: `a` is the high time and `b` is the low time, both in seconds.
  These are converted to `f = 1 / (Th + Tl)` and `D = Th / (Th + Tl)`.

The values are sanitised as follows:

- `f` is clamped to [0.01, 10] Hz.
- `D` is clamped to [0.02, 0.98].
- High and low times are clamped to at least 0.01 s.

Easing takes `w = transition_time * f` cycles, limited to `min(D, 1 - D)`. The
rise happens at the start of the high segment. That means the hinge sits fully
at `high_angle` for `(D - w) / f` seconds per cycle, not `D / f`.

A `Timeline` holds the shared `duration`, the patterns, and an `end_behavior`:

- `hold` keeps the parameters at their values at `t = duration`.
- `loop` restarts the parameter timeline, with the phase staying continuous.
- `stop` sends every hinge to its `low_angle`.

`to_vector` / `from_vector` flatten a pattern to 12 floats, in this order:

```
a.start, a.end, a.wobble_amp, a.wobble_rate, b.start, b.end, b.wobble_amp, b.wobble_rate,
low_angle, high_angle, phase_offset, transition_time
```

### Validation

`validate(timeline, limits, known_servo_ids)` runs when a `TimelineController`
is built and returns errors and warnings.

**Errors** (the timeline is rejected):

- an angle outside the hinge limits (±90° for kgd's hinge)
- `duration <= 0`
- an unknown `servo_id` or mode

**Warnings** (each reports the first time it happens on the timeline,
sampled at `update_rate_hz`):

- the frequency, duty cycle or high/low time hit its clamp
- `transition_time` had to be shortened to fit the cycle
- `transition_time` is 0, which gives a hard square wave

There is no slew-rate limit. The hinge's own position servo
(kp = 1.36, kv = 0.36) limits how fast it actually moves.

### Running a timeline

```bash
python square_wave.py square_wave_example.json --body gecko
```

The example drives gecko's two front hips (`robot1_C-LH-servo`,
`robot1_C-RH-servo`) half a cycle apart:

- the frequency ramps from 1 to 2 Hz
- duty cycle 0.5
- ±45°
- 0.1 s eased transitions

### Benchmark brains

All three brains use `freq_duty` mode, with the timeline set to the episode
(15 s, `hold`). The phase advances by the control period at each call.

- **`square`**: the full 12 genes per hinge, so each hinge has its own frequency.
- **`square_sync`**: one frequency ModulatedValue (4 genes) shared by all
  hinges, plus 8 genes per hinge. Sharing the frequency keeps the phase
  relationships fixed.
- **`bang_bang`**: `square_sync` with the angles fixed at ±90° and
  `transition_time` 0, so every hinge switches instantly between its limits,
  as the ANN champions do. That leaves 5 genes per hinge: the duty-cycle
  ModulatedValue and the phase offset.

CMA-ES genes are unbounded and squashed into ranges by the `SQUARE_*` constants
in `brains.py`. The ranges centre x0 = 0.5 near the ANN champions'
saturated 0.7–1 Hz bang-bang gaits. Phase is scaled by 0.1 because the gait is
far more sensitive to it than to any other gene.

Tests: `pytest test_square_wave.py test_zoo_benchmark.py`.
