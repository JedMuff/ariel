"""
Detect a "rowing" gait (limbs on both sides swept together) in the
control-stride sweep (slurm/run_control_stride_sweep.sh, forward task).

Input: 50Hz traces from behavior_extract.py, one behavior.npz per run (see
stride_gait_similarity.py). Per episode, after the settle phase and a
TRANSIENT_S transient:

  1. Module positions in the core's frame. The heading is the core's mean
     velocity in that frame (1 s smoothed), projected onto the core's xy
     plane. The split plane is the vertical plane within SNAP_DEG of the
     heading that best mirrors the rest pose onto itself (--snap-deg 0: the
     heading itself). It defines sagittal coordinates (fore, lat, up) with
     lat = up x fore; lat > 0 is the left side.
  2. Modules within MIDLINE_TOL of the sagittal plane are midline. The rest are
     paired left <-> right by Hungarian matching of their rest positions (mean
     over the window) with the right side reflected (lat -> -lat). A pair is kept
     if the distance is at most max(PAIR_TOL_M, PAIR_TOL_FRAC * distance from
     the core).
  3. Per pair, the right module's trajectory is reflected too, so a mirror-image
     motion counts as identical, whatever the hinge axes' signs:
       periodicity_l/r  autocorrelation at the period (estimate_period on that module)
       freq_l/r         1 / period
       freq_match       |f_l - f_r| <= FREQ_TOL * mean frequency
       mirror_sync      zero-lag normalised correlation of left and reflected right,
                        in [-1, 1]: +1 = in sync (rowing), -1 = half a cycle apart
       phase_lag        cross-spectrum phase at the shared frequency, cycles in [0, 0.5]
       amp_ratio        min / max RMS stroke of the two sides
     A pair rows if both sides are cyclic (periodicity >= PERIODIC_MIN),
     freq_match, amp_ratio >= AMP_RATIO_MIN and phase_lag < LAG_ROW. It
     alternates if the same holds with phase_lag >= LAG_ALT. A pair whose
     stroke is at least LAT_MAX lateral is "lateral" instead: after mirroring,
     a sideways sway looks anti-phase and an in/out clap in-phase, and hand
     labelling counts neither. An in-phase pair whose stroke is less than
     FORE_MIN fore-aft (mostly bobbing up and down) is "no_sweep": hand
     labelled rowers sweep fore-aft (2026-10-08, seed 1 labels). Phase lag rather than mirror_sync, because hand
     labelled rowers often lag 0.1-0.2 cycles, which drops mirror_sync well
     below 1 (rowing_validate.py, 2026-10-08 labels).
  4. Per individual, as shares of the RMS stroke of the moving
     (> MOVE_FRAC of body size), off-midline modules:
       paired_frac, rowing_frac, alternating_frac, aperiodic_frac
       rowing_score  stroke-weighted mirror_sync of the cyclic, frequency-matched
                     pairs, over all moving off-midline stroke (continuous, in [-1, 1])
     rowing_label is the first that applies: static (nothing moves), unpaired
     (paired_frac < UNPAIRED_MAX), aperiodic (aperiodic_frac > APERIODIC_MIN),
     rowing (rowing_frac > SHARE_MIN and > alternating_frac), alternating
     (the same the other way round), mixed.

The morphology's own best mirror plane (5-degree grid) is also found and its
angle to the heading reported (morph_plane_offset_deg), to flag bodies that
are symmetric but travel off-axis.

Writes to <stride-root>/rowing/:
  runs/<run>.csv    one row per episode (cached per run, recomputed when the npz is newer)
  pairs/<run>.csv   one row per left/right pair
  individuals.csv   all runs
Check the labels by hand with rowing_review_page.py, and score the detector
against your labels with rowing_validate.py.

Usage:
    python rowing_detect.py [--workers 18]
    python rowing_detect.py --runs ctrlstride_forward_tree_symmetric_s20_rep0_43569_3
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "examples" / "zoo_benchmark"))

from gait_similarity import TRANSIENT_S, estimate_period  # noqa: E402
from stride_gait_similarity import (  # noqa: E402
    GENOMES,
    META_COLS,
    discover,
    episode_cycle,
    iter_episode_poses,
    quat_to_mat,
)

MOVE_FRAC = 0.02        # a module moves if its RMS stroke exceeds this fraction of body size ...
MOVE_MIN_M = 0.003      # ... and this many metres (float16 poses are ~1 mm noisy)
MIDLINE_TOL = 0.02      # m, |lat| below this at rest is midline
PAIR_TOL_M = 0.03       # m, max distance between a module and its partner's mirror image ...
PAIR_TOL_FRAC = 0.25    # ... or this fraction of its distance from the core, if larger
PERIODIC_MIN = 0.3      # autocorrelation at the period for a module to count as cyclic
FREQ_TOL = 0.10         # relative frequency difference for freq_match
LAG_ROW = 0.25          # phase_lag (cycles) below this: a rowing pair
LAG_ALT = 0.40          # phase_lag at or above this: an alternating pair
LAT_MAX = 0.7           # a pair whose stroke is at least this share lateral is "lateral" (sway / clap), neither
FORE_MIN = 0.3          # a rowing pair's stroke is at least this share fore-aft; in phase but less is "no_sweep" (bobbing)
AMP_RATIO_MIN = 0.3     # both sides of a pair must really move
UNPAIRED_MAX = 0.25     # paired_frac below this -> unpaired
APERIODIC_MIN = 0.5     # aperiodic_frac above this -> aperiodic
SHARE_MIN = 0.25        # stroke share above which an individual is rowing / alternating
HEADING_SMOOTH_S = 1.0
MORPH_GRID_DEG = 5
SNAP_DEG = 45           # split plane: the best mirror plane within this of the heading (0 = heading only)

LABELS = ("rowing", "alternating", "mixed", "aperiodic", "unpaired", "static")
SIDE_CODE = {"core": 0, "left": 1, "right": -1, "midline": 2}


# ── Geometry ──────────────────────────────────────────────────────────────────


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


# ── Signals ───────────────────────────────────────────────────────────────────


def periodicity(x: np.ndarray, dt: float) -> tuple[float, float]:
    """(period s, autocorrelation at it) of one module's centred (T, 3) trajectory;
    (nan, 0) if there is no autocorrelation peak."""
    period, ac = estimate_period(x[None], dt, transient_s=0.0)
    return (period, ac) if np.isfinite(period) else (float("nan"), 0.0)


def mirror_sync(xl: np.ndarray, xr_m: np.ndarray) -> float:
    den = np.sqrt((xl ** 2).sum() * (xr_m ** 2).sum())
    return float((xl * xr_m).sum() / den) if den > 1e-12 else float("nan")


def phase_lag(xl: np.ndarray, xr_m: np.ndarray, freq: float, dt: float) -> float:
    """Phase difference (cycles, folded to [0, 0.5]) at freq, summed over coordinates."""
    if not np.isfinite(freq) or freq <= 0:
        return float("nan")
    e = np.exp(-2j * np.pi * freq * dt * np.arange(len(xl)))
    cross = np.sum((e @ xl) * np.conj(e @ xr_m))
    lag = (np.angle(cross) / (2 * np.pi)) % 1.0
    return float(min(lag, 1.0 - lag))


# ── Episode analysis ──────────────────────────────────────────────────────────


def analyse_episode(P: np.ndarray, Q: np.ndarray, dt: float, detail: bool = False, snap_deg: float = SNAP_DEG):
    """Rowing metrics of one episode's post-settle poses (T, B, 3/4), core first.

    Returns (individual metrics, list of per-pair dicts[, detail dict for the
    review page: sagittal trajectories, rest positions, sides, pairs])."""
    i0 = round(TRANSIENT_S / dt)
    P, Q = P[i0:], Q[i0:]
    local, R = core_local(P, Q)
    theta, consistency, speed = heading(P[:, 0], R, dt)
    # Split plane: the heading, or (snap_deg > 0) the plane within snap_deg of
    # it that best mirrors the rest pose, so a symmetric body that travels a
    # little off its own axis is still split along that axis.
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

    per = {}
    for b in np.flatnonzero(moving):
        per[b] = periodicity(x[:, b], dt)

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
        })

    off = moving & (side != SIDE_CODE["midline"])
    W = float(amp[off].sum())
    for p in pair_rows:  # the pair's share of the stroke: its moving modules' RMS strokes
        p["weight"] = float(amp[p["left"]] * off[p["left"]] + amp[p["right"]] * off[p["right"]])
    weight = lambda rows: sum(p["weight"] for p in rows)
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

    a, mismatch = morph_plane(rest[1:, :2], np.rad2deg(theta - split))  # sagittal coords: heading at theta - split
    ind["morph_plane_offset_deg"] = axial_diff_deg(a, theta - split) if np.isfinite(a) else float("nan")
    ind["morph_plane_mismatch_m"] = mismatch

    if not detail:
        return ind, pair_rows
    return ind, pair_rows, {"heading_rel": float(theta - split), "sag": sag, "rest": rest, "side": side, "moving": moving, "amp": amp,
                            "pairs": [(p["left"], p["right"]) for p in pair_rows], "dt": dt}


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


def review_detail(d: dict, trace_s: float = 6.0) -> dict:
    """Small JSON-able summary of analyse_episode's detail for the review page:
    the mean gait cycle in sagittal coordinates (rest + cycle) and trace_s of
    each pair's left / reflected-right fore and up coordinates."""
    sag, rest, dt = d["sag"], d["rest"], d["dt"]
    traj = (sag - rest).transpose(1, 0, 2)
    cycle, stats = episode_cycle(traj, dt)
    cyc = cycle + rest[:, None]
    n = min(len(sag), round(trace_s / dt))
    step = max(1, round(0.04 / dt))  # 25Hz is plenty to draw
    traces = []
    for l, r in d["pairs"]:
        tl = sag[:n:step, l] - rest[l]
        tr = mirror(sag[:n:step, r] - rest[r])
        traces.append({"fore_l": tl[:, 0], "fore_r": tr[:, 0], "up_l": tl[:, 2], "up_r": tr[:, 2],
                       "lat_l": tl[:, 1], "lat_r": tr[:, 1]})
    r4 = lambda a: np.round(np.asarray(a, dtype=float), 4).tolist()
    return {
        "cycle": r4(cyc), "rest": r4(rest), "side": d["side"].tolist(), "moving": d["moving"].tolist(),
        "amp": r4(d["amp"]), "pairs": [list(p) for p in d["pairs"]], "period_s": stats["period_s"],
        "aperiodic": bool(stats["aperiodic"]), "trace_dt": dt * step,
        "traces": [{k: r4(v) for k, v in t.items()} for t in traces],
    }


# ── Runs ──────────────────────────────────────────────────────────────────────


def analyse_run(npz: Path, snap_deg: float = SNAP_DEG) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta, episodes = iter_episode_poses(npz)
    rows, pair_rows = [], []
    for e, P, Q, dt in episodes:
        ind, prs = analyse_episode(P, Q, dt, snap_deg=snap_deg)
        rows.append({**meta.loc[e].to_dict(), **ind, "snap_deg": snap_deg})
        pair_rows += [{"gen": meta.loc[e, "gen"], "body_idx": meta.loc[e, "body_idx"], **p} for p in prs]
    ind = pd.DataFrame(rows)
    ints = ["gen", "body_idx", "ind_id", "n_joints", "n_bodies"]
    ind[ints] = ind[ints].astype(int)  # .loc[e].to_dict() of the mixed-type meta row made them floats
    prs = pd.DataFrame(pair_rows)
    if not prs.empty:
        prs[["gen", "body_idx"]] = prs[["gen", "body_idx"]].astype(int)
    return ind, prs


def _run_job(job: tuple[dict, Path, float]) -> str:
    r, out, snap_deg = job
    ind, prs = analyse_run(r["npz"], snap_deg)
    for i, k in enumerate(("run", "genome", "stride", "rep")):
        ind.insert(i, k, r[k])
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    ind.to_csv(out / "runs" / f"{r['run']}.csv", index=False)
    prs.to_csv(out / "pairs" / f"{r['run']}.csv", index=False)
    return f"{r['run']}: {len(ind)} episodes, " + ", ".join(
        f"{k}={v}" for k, v in ind["rowing_label"].value_counts().items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
    parser.add_argument("--behavior-subdir", default="behavior_50hz")
    parser.add_argument("--genomes", nargs="+", default=list(GENOMES))
    parser.add_argument("--runs", nargs="+", default=None, help="only these run names")
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--snap-deg", type=float, default=SNAP_DEG,
                        help="split along the best mirror plane within this of the heading; 0 = heading only")
    parser.add_argument("--force", action="store_true", help="recompute cached runs")
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="default: <stride-root>/rowing")
    args = parser.parse_args()
    out = args.out_dir or args.stride_root / "rowing"

    runs = discover(args.stride_root, args.behavior_subdir, args.genomes, args.runs)
    if not runs:
        raise SystemExit(f"No runs with behavior.npz under {args.stride_root / args.behavior_subdir}")
    def stale(r: dict) -> bool:
        p = out / "runs" / f"{r['run']}.csv"
        return (args.force or not p.exists() or p.stat().st_mtime < r["npz"].stat().st_mtime
                or pd.read_csv(p, usecols=["snap_deg"])["snap_deg"].iloc[0] != args.snap_deg)

    todo = [(r, out, args.snap_deg) for r in runs if stale(r)]
    print(f"{len(runs)} runs, {len(todo)} to analyse")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for msg in pool.map(_run_job, todo):
            print(msg)

    # individuals.csv covers every run analysed so far, not only --runs.
    df = pd.concat([pd.read_csv(p) for p in sorted((out / "runs").glob("*.csv"))], ignore_index=True)
    df.to_csv(out / "individuals.csv", index=False)
    last = df[df["gen"] == df.groupby("run")["gen"].transform("max")]
    for name, d in (("all generations", df), ("final generation", last)):
        tab = pd.crosstab([d["genome"], d["stride"]], d["rowing_label"], normalize="index")
        print(f"\nLabel shares, {name}:\n{tab.reindex(columns=[c for c in LABELS if c in tab]).round(2)}")
    print(f"-> {out / 'individuals.csv'} ({len(df)} episodes)")


if __name__ == "__main__":
    main()
