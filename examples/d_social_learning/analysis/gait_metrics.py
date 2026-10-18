"""Per-episode behaviour descriptors and the rowing / alternating gait label.

Ported from ~/workspaces/ariel_symmetry/ariel/examples (not a git repo):
    behavior_descriptors   symmetry_pressure/behavior_common.py::compute_behavior_descriptors
                           (locomotion, posture, contact, gait, left/right joint
                           symmetry, effort); the task axis is world +x here
    rowing                 symmetry_pressure/rowing_detect.py::analyse_episode
                           (constants unchanged; validated against hand labels
                           there, 2026-10-08), with estimate_period from
                           zoo_benchmark/gait_similarity.py

rowing() labels a body that moves only midline modules "midline" (the
original calls it "static", like a body that moves nothing).

Both take post-settle traces sampled every dt seconds (behavior_replay.py
records 50 Hz). Bodies are the robot's geom-owning MuJoCo bodies, core first.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── Behaviour descriptors (behavior_common.py) ────────────────────────────────

MIRROR_TOLERANCE = 0.03  # m, max distance between a hinge and its partner's mirror image
ON_PLANE_TOLERANCE = 0.01  # m, hinges this close to the mirror plane have no partner


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) MuJoCo quaternions (w, x, y, z) -> (..., 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def yaw_deltas(R: np.ndarray) -> np.ndarray:
    """Signed rotation about world z between consecutive core rotations (T, 3, 3):
    the geodesic incremental rotation's vertical component, so tumbling about
    a horizontal axis contributes ~0 (shared.py::signed_vertical_yaw_delta)."""
    rel = R[1:] @ np.swapaxes(R[:-1], 1, 2)
    angle = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
    s = np.sin(angle)
    axis_z = np.where(np.abs(s) > 1e-9, (rel[:, 1, 0] - rel[:, 0, 1]) / (2 * np.where(s == 0, 1, s)), 0.0)
    return angle * axis_z


def _mirror_pairs(anchors: np.ndarray, axis: int) -> list[tuple[int, int]]:
    """Mutual-nearest hinge pairs under reflection of core-frame coordinate
    ``axis``. Hinges within ON_PLANE_TOLERANCE of the plane are unpaired."""
    n = len(anchors)
    if n < 2:
        return []
    mirrored = anchors.copy()
    mirrored[:, axis] *= -1
    dist = np.linalg.norm(anchors[:, None, :] - mirrored[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    off_plane = np.abs(anchors[:, axis]) > ON_PLANE_TOLERANCE
    nearest = np.argmin(dist, axis=1)
    return [(i, int(nearest[i])) for i in range(n)
            if i < nearest[i] and nearest[nearest[i]] == i and dist[i, nearest[i]] <= MIRROR_TOLERANCE
            and off_plane[i] and off_plane[nearest[i]]]


def _dominant_frequency(x: np.ndarray, dt: float) -> tuple[float, np.ndarray]:
    """(dominant non-DC frequency in Hz, rfft) of a mean-removed signal."""
    spec = np.fft.rfft(x - x.mean())
    freqs = np.fft.rfftfreq(len(x), d=dt)
    if len(spec) < 2:
        return float("nan"), spec
    k = 1 + int(np.argmax(np.abs(spec[1:])))
    return float(freqs[k]), spec


def behavior_descriptors(P: np.ndarray, Q: np.ndarray, joint_angles: np.ndarray, contact: np.ndarray,
                         ctrl: np.ndarray, joint_anchors: np.ndarray, dt: float) -> dict[str, float]:
    """Descriptors of one episode: body poses P (T, B, 3) and quaternions Q
    (T, B, 4), core first; hinge angles (T, J); floor contact per body (T, B);
    applied control (T, nu); hinge anchors in the core frame at rest (J, 3)."""
    nan = float("nan")
    d: dict[str, float] = {}
    T, duration = len(P), len(P) * dt
    core_xy, core_z = P[:, 0, :2], P[:, 0, 2]
    R = quat_to_mat(Q[:, 0])

    disp = core_xy[-1] - core_xy[0]
    path = float(np.linalg.norm(np.diff(core_xy, axis=0), axis=1).sum())
    d["net_displacement"] = float(np.linalg.norm(disp))
    d["path_length"] = path
    d["mean_speed"] = path / duration
    d["straightness"] = d["net_displacement"] / path if path > 1e-9 else nan
    d["axis_displacement"] = float(disp[0])
    d["lateral_drift"] = float(abs(disp[1]))
    d["heading_offset_deg"] = float(np.degrees(np.arctan2(disp[1], disp[0]))) if path > 1e-9 else nan
    yaw = yaw_deltas(R)
    d["yaw_total"] = float(yaw.sum())
    d["mean_abs_yaw_rate"] = float(np.abs(yaw).sum() / duration)

    tilt = np.arccos(np.clip(R[:, 2, 2], -1, 1))
    d["core_height_mean"], d["core_height_std"] = float(core_z.mean()), float(core_z.std())
    d["core_tilt_mean"], d["core_tilt_std"] = float(tilt.mean()), float(tilt.std())

    d["duty_factor_mean"] = float(contact.mean())
    d["n_contact_bodies_mean"] = float(contact.sum(axis=1).mean())
    d["frac_bodies_ever_contact"] = float(contact.any(axis=0).mean())
    _, counts = np.unique(np.packbits(contact.astype(bool), axis=1), axis=0, return_counts=True)
    p = counts / counts.sum()
    d["contact_entropy_bits"] = max(0.0, float(-(p * np.log2(p)).sum()))

    for k in ("gait_frequency_hz", "joint_amplitude_mean", "joint_corr_abs_mean", "mirror_pair_frac",
              "mirror_pair_corr_mean", "mirror_pair_phase_lag", "mirror_pair_amp_asym", "mirror_plane"):
        d[k] = nan
    J = joint_angles.shape[1]
    if J > 0:
        amps = joint_angles.std(axis=0)
        d["joint_amplitude_mean"] = float(amps.mean())
        freqs_specs = [_dominant_frequency(joint_angles[:, j], dt) for j in range(J)]
        if amps.sum() > 1e-9:
            d["gait_frequency_hz"] = float(np.average([f for f, _ in freqs_specs], weights=amps))
        moving = amps > 1e-6
        if moving.sum() >= 2:
            corr = np.corrcoef(joint_angles[:, moving].T)
            d["joint_corr_abs_mean"] = float(np.nanmean(np.abs(corr[np.triu_indices(len(corr), k=1)])))
        best_axis, best_pairs, best_off = None, [], 0
        for ax in (0, 1):
            pairs = _mirror_pairs(joint_anchors, ax)
            if len(pairs) > len(best_pairs):
                best_axis, best_pairs = ax, pairs
                best_off = int((np.abs(joint_anchors[:, ax]) > ON_PLANE_TOLERANCE).sum())
        d["mirror_pair_frac"] = 2 * len(best_pairs) / best_off if best_pairs else 0.0
        if best_pairs:
            d["mirror_plane"] = float(best_axis)
            corrs, lags, asyms = [], [], []
            for i, j in best_pairs:
                if amps[i] > 1e-6 and amps[j] > 1e-6:
                    corrs.append(float(np.corrcoef(joint_angles[:, i], joint_angles[:, j])[0, 1]))
                    si, sj = freqs_specs[i][1], freqs_specs[j][1]
                    k = 1 + int(np.argmax(np.abs(si[1:]) + np.abs(sj[1:])))
                    lags.append(abs(float(np.angle(si[k] * np.conj(sj[k])))))
                if amps[i] + amps[j] > 1e-9:
                    asyms.append(float(abs(amps[i] - amps[j]) / (amps[i] + amps[j])))
            if corrs:
                d["mirror_pair_corr_mean"] = float(np.mean(corrs))
                d["mirror_pair_phase_lag"] = float(np.mean(lags))
            if asyms:
                d["mirror_pair_amp_asym"] = float(np.mean(asyms))

    d["ctrl_abs_mean"] = float(np.abs(ctrl).mean()) if ctrl.size else nan
    d["ctrl_jerk_mean"] = float(np.abs(np.diff(ctrl, axis=0)).mean()) if len(ctrl) > 1 and ctrl.size else nan
    return d


# ── Rowing detector (rowing_detect.py) ────────────────────────────────────────

TRANSIENT_S = 2.0       # skipped before estimating the gait cycle
MIN_PERIOD_S = 0.3
PEAK_FRACTION = 0.9     # first autocorrelation peak this close to the highest is the period
MOVE_FRAC = 0.02        # a module moves if its RMS stroke exceeds this fraction of body size ...
MOVE_MIN_M = 0.003      # ... and this many metres
MIDLINE_TOL = 0.02      # m, |lat| below this at rest is midline
PAIR_TOL_M = 0.03       # m, max distance between a module and its partner's mirror image ...
PAIR_TOL_FRAC = 0.25    # ... or this fraction of its distance from the core, if larger
PERIODIC_MIN = 0.3      # autocorrelation at the period for a module to count as cyclic
FREQ_TOL = 0.10         # relative frequency difference for freq_match
LAG_ROW = 0.25          # phase_lag (cycles) below this: a rowing pair
LAG_ALT = 0.40          # phase_lag at or above this: an alternating pair
LAT_MAX = 0.7           # a pair whose stroke is at least this share lateral is "lateral" (sway / clap)
FORE_MIN = 0.3          # a rowing pair's stroke is at least this share fore-aft, else "no_sweep" (bobbing)
AMP_RATIO_MIN = 0.3     # both sides of a pair must really move
UNPAIRED_MAX = 0.25     # paired_frac below this -> unpaired
APERIODIC_MIN = 0.5     # aperiodic_frac above this -> aperiodic
SHARE_MIN = 0.25        # stroke share above which an individual is rowing / alternating
HEADING_SMOOTH_S = 1.0
MORPH_GRID_DEG = 5
SNAP_DEG = 45           # split plane: the best mirror plane within this of the heading

LABELS = ("rowing", "alternating", "mixed", "aperiodic", "unpaired", "midline", "static")
SIDE_CODE = {"core": 0, "left": 1, "right": -1, "midline": 2}


def estimate_period(traj: np.ndarray, dt: float) -> tuple[float, float]:
    """(period in s, autocorrelation at it) of a (M, T, 3) trajectory: the first
    autocorrelation peak within PEAK_FRACTION of the highest, refined by
    parabolic interpolation; (nan, nan) if there is no positive peak."""
    x = (traj - traj.mean(axis=1, keepdims=True)).transpose(1, 0, 2).reshape(traj.shape[1], -1)
    T = len(x)
    ac = np.array([
        np.sum(x[:T - k] * x[k:]) / max(np.sqrt(np.sum(x[:T - k] ** 2) * np.sum(x[k:] ** 2)), 1e-12)
        for k in range(T // 2)
    ])
    min_lag = max(2, round(MIN_PERIOD_S / dt))
    peaks = [k for k in range(min_lag, len(ac) - 1) if ac[k] >= ac[k - 1] and ac[k] > ac[k + 1]]
    top = max((ac[k] for k in peaks), default=0.0)
    if top <= 0:
        return float("nan"), float("nan")
    k = next(k for k in peaks if ac[k] >= PEAK_FRACTION * top)
    denom = ac[k - 1] - 2 * ac[k] + ac[k + 1]
    shift = 0.5 * (ac[k - 1] - ac[k + 1]) / denom if denom < 0 else 0.0
    return float((k + shift) * dt), float(ac[k])


def _periodicity(x: np.ndarray, dt: float) -> tuple[float, float]:
    period, ac = estimate_period(x[None], dt)
    return (period, ac) if np.isfinite(period) else (float("nan"), 0.0)


def _heading(core_pos: np.ndarray, R: np.ndarray, dt: float) -> tuple[float, float, float]:
    """(heading angle in the core's xy plane, consistency in [0, 1], mean speed)."""
    v = np.einsum("tji,tj->ti", R[:-1], np.diff(core_pos, axis=0) / dt)[:, :2]
    k = max(1, round(HEADING_SMOOTH_S / dt))
    if len(v) > k:
        c = np.cumsum(np.vstack([np.zeros((1, 2)), v]), axis=0)
        v = (c[k:] - c[:-k]) / k
    mean = v.mean(axis=0)
    speed = float(np.linalg.norm(v, axis=1).mean())
    consistency = float(np.linalg.norm(mean) / speed) if speed > 1e-12 else 0.0
    return float(np.arctan2(mean[1], mean[0])), consistency, speed


def _sagittal(local: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return local @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]).T


def _mirror(x: np.ndarray) -> np.ndarray:
    return x * np.array([1.0, -1.0, 1.0])


def _morph_plane(rest_xy: np.ndarray, center_deg: float = 0.0, half_width_deg: float = 90.0) -> tuple[float, float]:
    """(angle of the vertical plane through the core that best mirrors the
    modules (B, 2) onto each other, mean mismatch in m)."""
    if len(rest_xy) == 0:
        return float("nan"), float("nan")
    best = (float("nan"), np.inf)
    for off in np.arange(-half_width_deg, half_width_deg + 1e-9, MORPH_GRID_DEG):
        if half_width_deg >= 90 and off >= 90:
            break
        a = np.deg2rad(center_deg + off)
        u = np.array([np.cos(a), np.sin(a)])
        refl = 2 * (rest_xy @ u)[:, None] * u - rest_xy
        cost = np.linalg.norm(refl[:, None] - rest_xy[None], axis=-1).min(axis=1).mean()
        if cost < best[1]:
            best = (a, float(cost))
    return best


def _axial_diff_deg(a: float, b: float) -> float:
    d = np.rad2deg(abs(a - b)) % 180
    return float(min(d, 180 - d))


def _phase_lag(xl: np.ndarray, xr_m: np.ndarray, freq: float, dt: float) -> float:
    if not np.isfinite(freq) or freq <= 0:
        return float("nan")
    e = np.exp(-2j * np.pi * freq * dt * np.arange(len(xl)))
    lag = (np.angle(np.sum((e @ xl) * np.conj(e @ xr_m))) / (2 * np.pi)) % 1.0
    return float(min(lag, 1.0 - lag))


def _label(ind: dict, total_stroke: float) -> str:
    # rowing_detect.py calls both of these "static"; here "midline" is a body
    # that moves only modules on its split plane (a chain flexing in one
    # vertical plane), and "static" one that moves nothing.
    if total_stroke <= 0:
        return "midline" if ind["n_moving"] else "static"
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


def rowing(P: np.ndarray, Q: np.ndarray, dt: float) -> dict:
    """Rowing metrics and label of one episode's post-settle poses (T, B, 3/4),
    core first. See rowing_detect.py's docstring in ariel_symmetry."""
    i0 = round(TRANSIENT_S / dt)
    P, Q = P[i0:], Q[i0:]
    R = quat_to_mat(Q[:, 0])
    local = np.einsum("tji,tbj->tbi", R, P - P[:, :1])
    theta, consistency, speed = _heading(P[:, 0], R, dt)
    split = theta
    if local.shape[1] > 1:
        a, _ = _morph_plane(local[:, 1:, :2].mean(axis=0), np.rad2deg(theta), SNAP_DEG)
        if np.isfinite(a):
            split = a
    sag = _sagittal(local, split)
    rest = sag.mean(axis=0)
    x = sag - rest
    amp = np.sqrt((x ** 2).sum(axis=2).mean(axis=0))
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
        cost = np.linalg.norm(rest[left][:, None] - _mirror(rest[right])[None], axis=-1)
        for a, b in zip(*linear_sum_assignment(cost)):
            if cost[a, b] <= max(PAIR_TOL_M, PAIR_TOL_FRAC * np.linalg.norm(rest[left[a], :2])):
                pairs.append((int(left[a]), int(right[b])))

    per = {b: _periodicity(x[:, b], dt) for b in np.flatnonzero(moving)}
    pair_rows = []
    for l, r in pairs:
        if not (moving[l] or moving[r]):
            continue
        xl, xr = x[:, l], _mirror(x[:, r])
        (pl, acl), (pr, acr) = per.get(l, (np.nan, 0.0)), per.get(r, (np.nan, 0.0))
        fl, fr = 1 / pl if np.isfinite(pl) else np.nan, 1 / pr if np.isfinite(pr) else np.nan
        fmean = np.nanmean([fl, fr]) if np.isfinite([fl, fr]).any() else np.nan
        freq_match = bool(np.isfinite(fl) and np.isfinite(fr) and abs(fl - fr) <= FREQ_TOL * fmean)
        den = np.sqrt((xl ** 2).sum() * (xr ** 2).sum())
        sync = float((xl * xr).sum() / den) if den > 1e-12 else float("nan")
        ratio = float(min(amp[l], amp[r]) / max(amp[l], amp[r]))
        var = (xl ** 2).mean(axis=0) + (xr ** 2).mean(axis=0)
        share = var / max(var.sum(), 1e-12)
        cyclic = min(acl, acr) >= PERIODIC_MIN
        ok = cyclic and freq_match and ratio >= AMP_RATIO_MIN
        lag = _phase_lag(xl, xr, fmean, dt)
        pair_rows.append({
            "left": l, "right": r, "freq_match": freq_match, "cyclic": cyclic, "mirror_sync": sync,
            "pair_kind": ("lateral" if ok and share[1] >= LAT_MAX else
                          "no_sweep" if ok and lag < LAG_ROW and share[0] < FORE_MIN else
                          "rowing" if ok and lag < LAG_ROW else
                          "alternating" if ok and lag >= LAG_ALT else
                          "amp_mismatch" if ratio < AMP_RATIO_MIN else
                          "aperiodic" if not cyclic else
                          "freq_mismatch" if not freq_match else "out_of_sync"),
        })

    off = moving & (side != SIDE_CODE["midline"])
    W = float(amp[off].sum())
    for p in pair_rows:
        p["weight"] = float(amp[p["left"]] * off[p["left"]] + amp[p["right"]] * off[p["right"]])
    weight = lambda rows: sum(p["weight"] for p in rows)  # noqa: E731
    ind = {
        "heading_deg": float(np.rad2deg(theta)), "heading_consistency": consistency, "speed": speed,
        "body_size_m": size, "n_moving": int(moving.sum()), "n_pairs": len(pair_rows),
        "midline_amp_frac": float(amp[moving & (side == 2)].sum() / max(amp[moving].sum(), 1e-12)),
        "stroke_total_m": W,
    }
    if W > 0:
        ind.update({
            "paired_frac": float(weight(pair_rows) / W),
            "rowing_frac": float(weight([p for p in pair_rows if p["pair_kind"] == "rowing"]) / W),
            "alternating_frac": float(weight([p for p in pair_rows if p["pair_kind"] == "alternating"]) / W),
            "aperiodic_frac": float(sum(amp[b] for b in np.flatnonzero(off) if per[b][1] < PERIODIC_MIN) / W),
            "rowing_score": float(sum(p["weight"] * p["mirror_sync"] for p in pair_rows
                                      if p["freq_match"] and p["cyclic"]) / W),
        })
    else:
        ind.update(dict.fromkeys(("paired_frac", "rowing_frac", "alternating_frac", "aperiodic_frac", "rowing_score"), 0.0))
    ind["rowing_label"] = _label(ind, W)
    a, mismatch = _morph_plane(rest[1:, :2], np.rad2deg(theta - split))
    ind["morph_plane_offset_deg"] = _axial_diff_deg(a, theta - split) if np.isfinite(a) else float("nan")
    ind["morph_plane_mismatch_m"] = mismatch
    return ind
