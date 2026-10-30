"""
Gait analysis of one episode's body poses: gait class, left-right phase
relations, phase quality, and the mean gait cycle (for gait species).

Input: post-settle world poses P (T, B, 3) and quaternions Q (T, B, 4) of the
robot bodies that own geoms, core first, sampled every dt s (replay.py: 50 Hz,
float16 like the investigation's behavior_extract.py traces, on which the
detector was validated).

Gait class (the investigation's rowing_detect.py, unchanged). After a
TRANSIENT_S transient:

  1. Module positions in the core's frame. The heading is the core's mean
     velocity in that frame (1 s smoothed). The split plane is the vertical
     plane within SNAP_DEG of the heading that best mirrors the rest pose onto
     itself. It defines sagittal coordinates (fore, lat, up); lat > 0 is left.
  2. Modules within MIDLINE_TOL of the sagittal plane are midline. The rest are
     paired left <-> right by Hungarian matching of their rest positions with
     the right side reflected; a pair is kept if the distance is at most
     max(PAIR_TOL_M, PAIR_TOL_FRAC * distance from the core).
  3. Per pair, the right module's trajectory is reflected too, so a mirror-image
     motion counts as identical:
       periodicity_l/r  autocorrelation at the period (estimate_period on that module)
       freq_l/r         1 / period
       freq_match       |f_l - f_r| <= FREQ_TOL * mean frequency
       mirror_sync      zero-lag normalised correlation of left and reflected right
       phase_lag        cross-spectrum phase at the shared frequency, cycles in [0, 0.5]
       amp_ratio        min / max RMS stroke of the two sides
     A pair rows if both sides are cyclic, freq_match, amp_ratio >= AMP_RATIO_MIN
     and phase_lag < LAG_ROW; it alternates if the same holds with phase_lag >=
     LAG_ALT. Mostly lateral strokes are "lateral", in-phase strokes with little
     fore-aft sweep "no_sweep" (neither counts as rowing).
  4. Per episode, as shares of the RMS stroke of the moving off-midline modules:
     paired_frac, rowing_frac, alternating_frac, aperiodic_frac, and
     rowing_label (the gait class), the first that applies: static (nothing
     moves), unpaired, aperiodic, rowing, alternating, mixed.

Phase quality (new): how steadily a pair holds its phase relation. The pair's
trajectories are cut into windows of one period (at the pair's mean
frequency), and the cross-spectrum phase is taken per window (lag_k, cycles in
[0, 1), not folded). Per pair:
  phase_quality   |mean exp(2 pi i lag_k)|, 1 - circular variance: 1 = the same
                  lag every cycle, ~0 = drifting (e.g. detuned frequencies)
  lag_within_frac share of cycles within LOCK_TOL cycles of the circular mean lag
  phase_relation  unlocked (phase_quality < LOCK_MIN), else in_phase (mean lag,
                  folded, < LAG_ROW), anti_phase (>= LAG_ALT) or offset
Per episode, over pairs with at least MIN_CYCLES windows: phase_quality and
lag_within_frac are their stroke-weighted means, phase_locked_frac the stroke
share (of all moving off-midline stroke) of locked pairs, in_phase_frac and
anti_phase_frac of locked in- and anti-phase pairs.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── Constants ─────────────────────────────────────────────────────────────────

TRANSIENT_S = 2.0      # skipped before estimating the gait cycle
MIN_PERIOD_S = 0.3
PEAK_FRACTION = 0.9    # first autocorrelation peak this close to the highest is the period
N_PHASE = 64           # samples per averaged cycle
N_YAW = 72             # yaw grid for cycle_distance (5 degrees)

MOVE_FRAC = 0.02        # a module moves if its RMS stroke exceeds this fraction of body size ...
MOVE_MIN_M = 0.003      # ... and this many metres (float16 poses are ~1 mm noisy)
MIDLINE_TOL = 0.02      # m, |lat| below this at rest is midline
PAIR_TOL_M = 0.03       # m, max distance between a module and its partner's mirror image ...
PAIR_TOL_FRAC = 0.25    # ... or this fraction of its distance from the core, if larger
PERIODIC_MIN = 0.3      # autocorrelation at the period for a module to count as cyclic
FREQ_TOL = 0.10         # relative frequency difference for freq_match
LAG_ROW = 0.25          # phase_lag (cycles) below this: a rowing pair
LAG_ALT = 0.40          # phase_lag at or above this: an alternating pair
LAT_MAX = 0.7           # a pair whose stroke is at least this share lateral is "lateral" (sway / clap)
FORE_MIN = 0.3          # a rowing pair's stroke is at least this share fore-aft; less is "no_sweep"
AMP_RATIO_MIN = 0.3     # both sides of a pair must really move
UNPAIRED_MAX = 0.25     # paired_frac below this -> unpaired
APERIODIC_MIN = 0.5     # aperiodic_frac above this -> aperiodic
SHARE_MIN = 0.25        # stroke share above which an episode is rowing / alternating
HEADING_SMOOTH_S = 1.0
MORPH_GRID_DEG = 5
SNAP_DEG = 45           # split plane: the best mirror plane within this of the heading

LOCK_MIN = 0.9          # phase_quality at or above this: a phase-locked pair
LOCK_TOL = 0.1          # cycles, for lag_within_frac
MIN_CYCLES = 3          # windows needed for a pair's phase quality

LABELS = ("rowing", "alternating", "mixed", "aperiodic", "unpaired", "static")
SIDE_CODE = {"core": 0, "left": 1, "right": -1, "midline": 2}


# ── Frames ────────────────────────────────────────────────────────────────────


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(T, 4) MuJoCo quaternions (w, x, y, z) -> (T, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def core_frame(pos: np.ndarray, core_pos: np.ndarray, core_R: np.ndarray) -> np.ndarray:
    """Positions (T, M, 3) in the core's moving frame, (M, T, 3), each module centred on its mean."""
    local = np.einsum("tji,tmj->mti", core_R, pos - core_pos[:, None, :])  # R^T (p - p_core)
    return local - local.mean(axis=1, keepdims=True)


# ── Gait cycle ────────────────────────────────────────────────────────────────


def estimate_period(traj: np.ndarray, dt: float, transient_s: float = TRANSIENT_S) -> tuple[float, float]:
    """(period in s, autocorrelation at it) of a (M, T, 3) trajectory after the transient.

    Uses the normalised autocorrelation of all module coordinates together and
    takes the first peak within PEAK_FRACTION of the highest one, refined by
    parabolic interpolation; (nan, nan) if there is no positive peak.
    """
    x = traj[:, round(transient_s / dt):]
    x = (x - x.mean(axis=1, keepdims=True)).transpose(1, 0, 2).reshape(x.shape[1], -1)
    T = len(x)
    lags = np.arange(T // 2)
    ac = np.array([
        np.sum(x[:T - k] * x[k:]) / max(np.sqrt(np.sum(x[:T - k] ** 2) * np.sum(x[k:] ** 2)), 1e-12)
        for k in lags
    ])
    min_lag = max(2, round(MIN_PERIOD_S / dt))
    peaks = [k for k in range(min_lag, len(ac) - 1) if ac[k] >= ac[k - 1] and ac[k] > ac[k + 1]]
    top = max((ac[k] for k in peaks), default=0.0)
    if top <= 0:  # no peak, or only anti-correlated ones: not periodic
        return float("nan"), float("nan")
    k = next(k for k in peaks if ac[k] >= PEAK_FRACTION * top)
    denom = ac[k - 1] - 2 * ac[k] + ac[k + 1]
    shift = 0.5 * (ac[k - 1] - ac[k + 1]) / denom if denom < 0 else 0.0
    return float((k + shift) * dt), float(ac[k])


def average_cycle(
    traj: np.ndarray, dt: float, period: float, transient_s: float = TRANSIENT_S, n_phase: int = N_PHASE,
) -> np.ndarray:
    """Mean cycle (M, n_phase, 3) over the complete cycles after the transient,
    each sampled at the same n_phase phases by linear interpolation; centred
    on its mean per module."""
    t = np.arange(traj.shape[1]) * dt
    n_cycles = int((t[-1] - transient_s) // period)
    if n_cycles < 1:
        raise ValueError(f"No complete cycle of {period:.2f} s after the {transient_s} s transient")
    phase = np.arange(n_phase) / n_phase
    ts = transient_s + (np.arange(n_cycles)[:, None] + phase[None]) * period  # (cycles, n_phase)
    flat = traj.transpose(0, 2, 1).reshape(-1, traj.shape[1])                 # (M*3, T)
    samples = np.stack([np.interp(ts.ravel(), t, row) for row in flat])        # (M*3, cycles*n_phase)
    cycle = samples.reshape(traj.shape[0], 3, n_cycles, n_phase).mean(axis=2).transpose(0, 2, 1)
    return cycle - cycle.mean(axis=1, keepdims=True)


def cycle_rms(cycle: np.ndarray) -> float:
    """RMS module displacement (m) of a (M, P, 3) mean cycle."""
    return float(np.sqrt((cycle ** 2).sum(axis=2).mean()))


def shape_cycle(cycle: np.ndarray) -> np.ndarray:
    """cycle scaled to unit RMS, for the amplitude-independent cycle_shape comparison."""
    return cycle / max(cycle_rms(cycle), 1e-12)


def episode_cycle(P: np.ndarray, Q: np.ndarray, dt: float) -> tuple[np.ndarray, dict]:
    """Mean cycle (M, N_PHASE, 3) of an episode's core-frame motion, with its
    period stats; the whole post-transient window counts as one cycle when no
    period is found."""
    traj = core_frame(P, P[:, 0], quat_to_mat(Q[:, 0]))
    period, autocorr = estimate_period(traj, dt)
    aperiodic = not np.isfinite(period)
    if aperiodic:
        period = (traj.shape[1] - 1) * dt - TRANSIENT_S
    cycle = average_cycle(traj, dt, period)
    size = float(np.sqrt(((P[0] - P[0, 0]) ** 2).sum(axis=1).mean()))
    rms = cycle_rms(cycle)
    return cycle, {"period_s": float(period), "autocorr": float(autocorr), "aperiodic": aperiodic,
                   "cycle_rms_m": rms, "rel_amplitude": rms / size if size > 0 else float("nan")}


# ── Cycle distance (gait species) ─────────────────────────────────────────────


def _charged_mean(cost: np.ndarray) -> float:
    """Mean cost of the Hungarian module matching, each module of the larger
    robot left unmatched charged its best match."""
    rows, cols = linear_sum_assignment(cost)
    charged = list(cost[rows, cols])
    na, nb = cost.shape
    if na > nb:
        charged += list(cost[np.setdiff1d(np.arange(na), rows)].min(axis=1))
    elif nb > na:
        charged += list(cost[:, np.setdiff1d(np.arange(nb), cols)].min(axis=0))
    return float(np.mean(charged))


def _charged_mean_lower_bound(cost: np.ndarray) -> np.ndarray:
    """Lower bound on _charged_mean for a stack (..., Ma, Mb) of cost matrices."""
    na, nb = cost.shape[-2:]
    rows = cost.min(axis=-1).mean(axis=-1)
    cols = cost.min(axis=-2).mean(axis=-1)
    if na == nb:
        return np.maximum(rows, cols)
    return rows if na > nb else cols


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Circular cross-correlation r[i, j, s] = sum_p u[i, p] v[j, p + s]."""
    U, V = np.fft.rfft(u, axis=-1), np.fft.rfft(v, axis=-1)
    return np.fft.irfft(np.conj(U)[:, None] * V[None], n=u.shape[-1], axis=-1)


def cycle_distance(a: np.ndarray, b: np.ndarray, n_yaw: int = N_YAW) -> float:
    """Module-matched RMSE between mean cycles a (Ma, P, 3) and b (Mb, P, 3),
    minimised over circular phase shift, yaw rotation of b and mirroring b
    across a vertical plane. On unit-RMS cycles (shape_cycle): 0 = same gait,
    ~1.0-1.4 = unrelated. The Hungarian matching runs in order of a lower
    bound until no candidate can beat the best, so the minimum is exact."""
    P = a.shape[1]
    ax, ay, az = (a[..., k] for k in range(3))
    bx, by, bz = (b[..., k] for k in range(3))
    zz = _cross(az, bz)
    xx, yy, xy, yx = _cross(ax, bx), _cross(ay, by), _cross(ax, by), _cross(ay, bx)
    norms = (a ** 2).sum(axis=(1, 2))[:, None] + (b ** 2).sum(axis=(1, 2))[None, :]
    yaw = np.arange(n_yaw) * 2 * np.pi / n_yaw
    c, s = np.cos(yaw)[:, None, None, None], np.sin(yaw)[:, None, None, None]

    costs = []  # a . (R_yaw M b) summed over coords; M = identity or y -> -y
    for p_term, q_term in ((xx + yy, yx - xy), (xx - yy, yx + xy)):
        dot = zz[None] + c * p_term[None] + s * q_term[None]               # (yaw, Ma, Mb, shift)
        costs.append((norms[None, :, :, None] - 2 * dot).transpose(0, 3, 1, 2) / P)
    cost = np.maximum(np.stack(costs), 0.0)                                   # (mirror, yaw, shift, Ma, Mb)

    bound = _charged_mean_lower_bound(cost)
    best = np.inf
    for idx in np.argsort(bound, axis=None):
        idx = np.unravel_index(idx, bound.shape)
        if bound[idx] >= best:
            break
        best = min(best, _charged_mean(cost[idx]))
    return float(np.sqrt(best))


# ── Gait class: geometry ──────────────────────────────────────────────────────


def core_local(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World poses (T, B, 3/4), core first -> (positions R^T (p - p_core) of every
    body (T, B, 3), core rotations (T, 3, 3))."""
    R = quat_to_mat(Q[:, 0])
    return np.einsum("tji,tbj->tbi", R, P - P[:, :1]), R


def heading(core_pos: np.ndarray, R: np.ndarray, dt: float) -> tuple[float, float, float]:
    """(heading angle in the core's xy plane (rad), consistency in [0, 1], mean
    speed m/s) from the core's world velocity rotated into its own frame."""
    v = np.einsum("tji,tj->ti", R[:-1], np.diff(core_pos, axis=0) / dt)[:, :2]
    k = max(1, round(HEADING_SMOOTH_S / dt))
    if len(v) > k:
        c = np.cumsum(np.vstack([np.zeros((1, 2)), v]), axis=0)
        v = (c[k:] - c[:-k]) / k  # 1 s moving average: the stroke's own back-and-forth cancels
    mean = v.mean(axis=0)
    speed = float(np.linalg.norm(v, axis=1).mean())
    consistency = float(np.linalg.norm(mean) / speed) if speed > 1e-12 else 0.0
    return float(np.arctan2(mean[1], mean[0])), consistency, speed


def sagittal(local: np.ndarray, theta: float) -> np.ndarray:
    """Core-frame positions (..., 3) -> (fore, lat, up) with fore along heading theta."""
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ rot.T


def mirror(x: np.ndarray) -> np.ndarray:
    """Reflect sagittal coordinates through the sagittal plane (lat -> -lat)."""
    return x * np.array([1.0, -1.0, 1.0])


def morph_plane(rest_xy: np.ndarray, center_deg: float = 0.0, half_width_deg: float = 90.0) -> tuple[float, float]:
    """(angle (rad) of the vertical plane through the core that best mirrors the
    modules onto each other, mean mismatch (m)), searched on a MORPH_GRID_DEG
    grid within half_width_deg of center_deg. Rest positions (B, 2) without the core."""
    if len(rest_xy) == 0:
        return float("nan"), float("nan")
    best = (float("nan"), np.inf)
    for off in np.arange(-half_width_deg, half_width_deg + 1e-9, MORPH_GRID_DEG):
        if half_width_deg >= 90 and off >= 90:
            break  # planes are undirected: a full search covers [-90, 90)
        a = np.deg2rad(center_deg + off)
        u = np.array([np.cos(a), np.sin(a)])
        refl = 2 * (rest_xy @ u)[:, None] * u - rest_xy
        cost = np.linalg.norm(refl[:, None] - rest_xy[None], axis=-1).min(axis=1).mean()
        if cost < best[1]:
            best = (a, float(cost))
    return best


def axial_diff_deg(a: float, b: float) -> float:
    """Angle (deg, in [0, 90]) between two undirected lines at angles a, b (rad)."""
    d = np.rad2deg(abs(a - b)) % 180
    return float(min(d, 180 - d))


# ── Gait class: signals ───────────────────────────────────────────────────────


def periodicity(x: np.ndarray, dt: float) -> tuple[float, float]:
    """(period s, autocorrelation at it) of one module's centred (T, 3) trajectory;
    (nan, 0) if there is no autocorrelation peak."""
    period, ac = estimate_period(x[None], dt, transient_s=0.0)
    return (period, ac) if np.isfinite(period) else (float("nan"), 0.0)


def mirror_sync(xl: np.ndarray, xr_m: np.ndarray) -> float:
    den = np.sqrt((xl ** 2).sum() * (xr_m ** 2).sum())
    return float((xl * xr_m).sum() / den) if den > 1e-12 else float("nan")


def _cross_phase(xl: np.ndarray, xr_m: np.ndarray, freq: float, dt: float) -> float:
    """Phase of left relative to reflected right (cycles, in [0, 1)) at freq, summed over coordinates."""
    e = np.exp(-2j * np.pi * freq * dt * np.arange(len(xl)))
    cross = np.sum((e @ xl) * np.conj(e @ xr_m))
    return float((np.angle(cross) / (2 * np.pi)) % 1.0)


def phase_lag(xl: np.ndarray, xr_m: np.ndarray, freq: float, dt: float) -> float:
    """Phase difference (cycles, folded to [0, 0.5]) at freq, summed over coordinates."""
    if not np.isfinite(freq) or freq <= 0:
        return float("nan")
    lag = _cross_phase(xl, xr_m, freq, dt)
    return float(min(lag, 1.0 - lag))


def phase_quality(xl: np.ndarray, xr_m: np.ndarray, freq: float, dt: float) -> dict:
    """Phase quality of one pair (see module docstring): the cross-spectrum
    phase per one-period window, and how steady it is."""
    nan = {"phase_quality": float("nan"), "lag_within_frac": float("nan"), "mean_lag": float("nan"),
           "n_cycles": 0, "phase_relation": "unlocked"}
    if not np.isfinite(freq) or freq <= 0:
        return nan
    n = round(1.0 / (freq * dt))
    n_cycles = len(xl) // n if n >= 2 else 0
    if n_cycles < MIN_CYCLES:
        return {**nan, "n_cycles": n_cycles}
    lags = np.array([_cross_phase(xl[k * n:(k + 1) * n], xr_m[k * n:(k + 1) * n], freq, dt)
                     for k in range(n_cycles)])
    z = np.exp(2j * np.pi * lags)
    mean = z.mean()
    mean_lag = float((np.angle(mean) / (2 * np.pi)) % 1.0)
    dev = np.abs((lags - mean_lag + 0.5) % 1.0 - 0.5)
    quality = float(np.abs(mean))
    folded = min(mean_lag, 1.0 - mean_lag)
    relation = ("unlocked" if quality < LOCK_MIN else "in_phase" if folded < LAG_ROW
                else "anti_phase" if folded >= LAG_ALT else "offset")
    return {"phase_quality": quality, "lag_within_frac": float((dev <= LOCK_TOL).mean()),
            "mean_lag": mean_lag, "n_cycles": int(n_cycles), "phase_relation": relation}


# ── Episode analysis ──────────────────────────────────────────────────────────


def analyse_episode(P: np.ndarray, Q: np.ndarray, dt: float, snap_deg: float = SNAP_DEG) -> tuple[dict, list[dict]]:
    """(episode metrics, per-pair rows) of one episode's post-settle poses (T, B, 3/4), core first."""
    i0 = round(TRANSIENT_S / dt)
    P, Q = P[i0:], Q[i0:]
    local, R = core_local(P, Q)
    theta, consistency, speed = heading(P[:, 0], R, dt)
    # Split plane: the plane within snap_deg of the heading that best mirrors
    # the rest pose, so a symmetric body that travels a little off its own axis
    # is still split along that axis.
    split = theta
    if snap_deg > 0 and local.shape[1] > 1:
        a, _ = morph_plane(local[:, 1:, :2].mean(axis=0), np.rad2deg(theta), snap_deg)
        if np.isfinite(a):
            split = a
    sag = sagittal(local, split)                       # (T, B, 3) fore, lat, up
    rest = sag.mean(axis=0)
    x = sag - rest                                     # centred trajectories
    amp = np.sqrt((x ** 2).sum(axis=2).mean(axis=0))   # RMS stroke per body
    size = float(np.sqrt((rest[1:] ** 2).sum(axis=1).mean())) if len(rest) > 1 else 0.0
    B = len(rest)
    moving = amp > max(MOVE_FRAC * size, MOVE_MIN_M)
    moving[0] = False

    side = np.full(B, SIDE_CODE["midline"])
    side[0] = SIDE_CODE["core"]
    side[1:][rest[1:, 1] > MIDLINE_TOL] = SIDE_CODE["left"]
    side[1:][rest[1:, 1] < -MIDLINE_TOL] = SIDE_CODE["right"]
    left, right = np.flatnonzero(side == 1), np.flatnonzero(side == -1)

    pairs: list[tuple[int, int]] = []
    if len(left) and len(right):
        cost = np.linalg.norm(rest[left][:, None] - mirror(rest[right])[None], axis=-1)
        for a, b in zip(*linear_sum_assignment(cost)):
            reach = np.linalg.norm(rest[left[a], :2])
            if cost[a, b] <= max(PAIR_TOL_M, PAIR_TOL_FRAC * reach):
                pairs.append((int(left[a]), int(right[b])))

    per = {b: periodicity(x[:, b], dt) for b in np.flatnonzero(moving)}

    pair_rows = []
    for l, r in pairs:
        if not (moving[l] or moving[r]):
            continue
        xl, xr = x[:, l], mirror(x[:, r])
        (pl, acl), (pr, acr) = per.get(l, (np.nan, 0.0)), per.get(r, (np.nan, 0.0))
        fl, fr = 1 / pl if np.isfinite(pl) else np.nan, 1 / pr if np.isfinite(pr) else np.nan
        fmean = np.nanmean([fl, fr]) if np.isfinite([fl, fr]).any() else np.nan
        freq_match = bool(np.isfinite(fl) and np.isfinite(fr) and abs(fl - fr) <= FREQ_TOL * fmean)
        sync = mirror_sync(xl, xr)
        ratio = float(min(amp[l], amp[r]) / max(amp[l], amp[r]))
        var = (xl ** 2).mean(axis=0) + (xr ** 2).mean(axis=0)
        share = var / max(var.sum(), 1e-12)
        cyclic = min(acl, acr) >= PERIODIC_MIN
        ok = cyclic and freq_match and ratio >= AMP_RATIO_MIN
        lag = phase_lag(xl, xr, fmean, dt)
        # Mirroring the right side flips lat, so a whole-body sideways sway
        # reads as anti-phase and an in/out clap as in-phase: judge lateral
        # strokes separately.
        lateral = share[1] >= LAT_MAX
        pair_rows.append({
            "left": l, "right": r, "amp_l": amp[l], "amp_r": amp[r], "amp_ratio": ratio,
            "periodicity_l": acl, "periodicity_r": acr, "freq_l": fl, "freq_r": fr,
            "freq_match": freq_match, "mirror_sync": sync, "phase_lag": lag,
            "stroke_fore": share[0], "stroke_lat": share[1], "stroke_up": share[2],
            "rest_dist": float(np.linalg.norm(rest[l] - mirror(rest[r]))),
            "pair_kind": ("lateral" if ok and lateral else
                          "no_sweep" if ok and lag < LAG_ROW and share[0] < FORE_MIN else
                          "rowing" if ok and lag < LAG_ROW else
                          "alternating" if ok and lag >= LAG_ALT else
                          "amp_mismatch" if ratio < AMP_RATIO_MIN else
                          "aperiodic" if not cyclic else
                          "freq_mismatch" if not freq_match else "out_of_sync"),
            **phase_quality(xl, xr, fmean, dt),
        })

    off = moving & (side != SIDE_CODE["midline"])
    W = float(amp[off].sum())
    for p in pair_rows:  # the pair's share of the stroke: its moving modules' RMS strokes
        p["weight"] = float(amp[p["left"]] * off[p["left"]] + amp[p["right"]] * off[p["right"]])

    def weight(rows: list[dict]) -> float:
        return sum(p["weight"] for p in rows)

    ind = {
        "heading_deg": float(np.rad2deg(theta)), "heading_consistency": consistency, "speed": speed,
        "split_offset_deg": float(np.rad2deg(split - theta)),
        "body_size_m": size, "n_bodies": B, "n_moving": int(moving.sum()),
        "n_left": int(len(left)), "n_right": int(len(right)), "n_midline": int((side == 2).sum()),
        "n_pairs": len(pair_rows), "n_rowing_pairs": sum(p["pair_kind"] == "rowing" for p in pair_rows),
        "n_alternating_pairs": sum(p["pair_kind"] == "alternating" for p in pair_rows),
        "midline_amp_frac": float(amp[moving & (side == 2)].sum() / max(amp[moving].sum(), 1e-12)),
        "stroke_total_m": W,
    }
    if W > 0:
        cyc = [p for p in pair_rows if p["freq_match"]
               and min(p["periodicity_l"], p["periodicity_r"]) >= PERIODIC_MIN]
        ind.update({
            "paired_frac": float(weight(pair_rows) / W),
            "rowing_frac": float(weight([p for p in pair_rows if p["pair_kind"] == "rowing"]) / W),
            "alternating_frac": float(weight([p for p in pair_rows if p["pair_kind"] == "alternating"]) / W),
            "aperiodic_frac": float(sum(amp[b] for b in np.flatnonzero(off) if per[b][1] < PERIODIC_MIN) / W),
            "rowing_score": float(sum(p["weight"] * p["mirror_sync"] for p in cyc) / W),
        })
    else:
        ind.update(dict.fromkeys(("paired_frac", "rowing_frac", "alternating_frac", "aperiodic_frac"), 0.0))
        ind["rowing_score"] = 0.0
    ind["rowing_label"] = label_of(ind, W)

    measured = [p for p in pair_rows if p["n_cycles"] >= MIN_CYCLES and p["weight"] > 0]
    wq = weight(measured)
    locked = [p for p in measured if p["phase_relation"] != "unlocked"]
    ind.update({
        "phase_quality": float(sum(p["weight"] * p["phase_quality"] for p in measured) / wq) if wq > 0 else float("nan"),
        "lag_within_frac": float(sum(p["weight"] * p["lag_within_frac"] for p in measured) / wq) if wq > 0 else float("nan"),
        "phase_locked_frac": float(weight(locked) / W) if W > 0 else 0.0,
        "in_phase_frac": float(weight([p for p in locked if p["phase_relation"] == "in_phase"]) / W) if W > 0 else 0.0,
        "anti_phase_frac": float(weight([p for p in locked if p["phase_relation"] == "anti_phase"]) / W) if W > 0 else 0.0,
    })

    a, mismatch = morph_plane(rest[1:, :2], np.rad2deg(theta - split))  # sagittal coords: heading at theta - split
    ind["morph_plane_offset_deg"] = axial_diff_deg(a, theta - split) if np.isfinite(a) else float("nan")
    ind["morph_plane_mismatch_m"] = mismatch
    return ind, pair_rows


def label_of(ind: dict, total_stroke: float) -> str:
    if total_stroke <= 0:
        return "static"
    if ind["paired_frac"] < UNPAIRED_MAX:
        return "unpaired"
    if ind["aperiodic_frac"] > APERIODIC_MIN:
        return "aperiodic"
    row, alt = ind["rowing_frac"], ind["alternating_frac"]
    if row > SHARE_MIN and row > alt:
        return "rowing"
    if alt > SHARE_MIN and alt > row:
        return "alternating"
    return "mixed"
