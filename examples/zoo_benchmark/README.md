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

## Gait similarity (`gait_similarity.py`)

Compares champions' gaits across bodies. Each champion is replayed and every
module's (geom-owning body's) position is recorded at 20 Hz. Two gaits are
compared by matching modules between the robots with the Hungarian algorithm
(cost = trajectory MSE) and taking the RMSE over the matched modules (metres).
Leftover modules in the larger robot are charged their best match. Every
distance is also written with the leftovers ignored (`_ignore` file suffix):
the matching gives each module of the smaller robot a partner, and only those
pairs count. `--leftovers charge` skips that variant. There are four
comparisons:

- **world**: start-aligned world positions, so travel counts
- **core**: positions in the core's moving frame, each module centred on its
  mean, so only motion relative to the body counts
- **cycle**: the core-frame motion averaged over one gait cycle. The period
  comes from the autocorrelation after a 2 s transient. Cycles are compared at
  the best phase shift, yaw rotation and left/right mirror. This makes the
  comparison independent of gait speed, start phase and the body's orientation
  on its core. The relative timing of the modules still counts (trot ≠ pace).
- **cycle_shape**: as `cycle`, but each mean cycle is first scaled to unit RMS,
  so only the shape and timing of the motion count, not its amplitude. The
  distance is dimensionless (0 = same shape, about 1.4 = unrelated).

`world` and `core` compare the episodes sample by sample, so they depend on
speed, phase and orientation. Use `cycle_shape` for the gait map. In `cycle`
the amplitude difference dominates: on the 23 ANN champions the distance
correlates 0.83 with it, and amplitude grows with body size (correlation
0.84), so `cycle` mostly sorts the bodies by size. `cycle_shape` drops that to
0.37. The amplitude is kept as a separate number instead.
`gait_cycles_<tag>.csv` holds the period, amplitude (`cycle_rms_m`), body size
(`body_size_m`, RMS module distance from the core) and amplitude relative to
size per gait. `gait_alignment_<tag>[_cycle_shape].csv` holds the best phase,
yaw and mirror per pair.

```bash
python gait_similarity.py __data__/ariel_zoo_benchmark --brain ann   # best ANN seed per body
python plot_gait_similarity.py __data__/ariel_zoo_benchmark --tag ann   # one figure per comparison
```

`gait_similarity_page.py` writes an interactive version of the map,
`gait_similarity/gait_similarity_<tag>.html`, with a switch between the
`cycle_shape` and `cycle` maps (`--frames` picks others), a switch between
charged and ignored leftover modules, and dot colour by x-speed or relative
amplitude. Hover a body to see its
champion video, its behavioural descriptors
(`symmetry_pressure/behavior_common.compute_behavior_descriptors` on a replay
of the zoo episode, cached in `gait_descriptors_<tag>.csv`) and the other gaits
ranked by distance. Click a body to pin it; `#<body>` in the URL pins it on
load. Videos come from `render_zoo_best.py` and are transcoded to H.264 into
`gait_similarity/videos/`, so keep that folder next to the page.

```bash
python gait_similarity_page.py __data__/ariel_zoo_benchmark --tag ann
```

`--body gecko` compares the best gait of each brain on one body instead
(tag `gecko_brains`). `--max-lag N` also tries time shifts of up to N control
steps. Tests: `pytest test_gait_similarity.py`.
