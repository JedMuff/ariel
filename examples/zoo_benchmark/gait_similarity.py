"""Gait similarity between zoo champions from their module trajectories.

Each champion is replayed with the training episode loop and the world
position of every robot body that owns a geom (core, stators, rotors, bricks;
the "modules") is recorded at every control step. Two gaits are compared in
four ways:
  world  positions minus the core's start xy: how far and which way each
         module travels counts
  core   positions in the core's moving frame, each module centred on its own
         mean: only how the modules move relative to the core counts, not
         where they sit on the body
  cycle  the core-frame motion averaged over one gait cycle on a 0-1 phase
         axis (period from the autocorrelation, after a transient), compared
         at the best circular phase shift, yaw rotation and left/right mirror:
         independent of gait speed, start phase and the body's orientation on
         its core, while the relative timing of the modules still counts
  cycle_shape  cycle with each mean cycle scaled to unit RMS first: only the
         shape and coordination of the motion counts, not its amplitude
         (which otherwise dominates, since it grows with body size); the
         amplitude is reported separately in gait_cycles_<tag>.csv
The distance between robots A and B is a module-matched RMSE in metres:
C[i, j] is the MSE between module i of A and module j of B, the Hungarian
algorithm pairs min(Ma, Mb) modules, each module left over in the larger
robot is charged its best match, and the result is the root of the mean
charged cost. It is symmetric, zero for identical gaits and independent of
module order. Every distance is also computed with the leftovers ignored: only
the smaller robot's modules, each with its matched partner, count (files with
an _ignore suffix; --leftovers picks which variants are written). world and core compare the whole episode sample by sample
(trajectory_distance), so they are speed, phase and orientation dependent;
cycle and cycle_shape use cycle_distance; cycle_shape distances are
dimensionless (0 = same shape, ~1.4 = uncorrelated).

By default one gait per body: the best-seed champion of --brain on each body.
With --body, one gait per brain on that body instead.

Usage:
  python gait_similarity.py __data__/ariel_zoo_benchmark [--brain ann] [--max-lag 0] [-o OUTDIR]
  python gait_similarity.py __data__/local_gecko --body gecko
Writes <out>/gait_trajectories_<tag>.npz (replay cache, reused if present),
<out>/gait_distance_<tag>_{world,core,cycle,cycle_shape}.csv, gait_cycles_<tag>.csv
(period, amplitude and body size per gait) and gait_alignment_<tag>[_cycle_shape].csv
(best phase/yaw/mirror per pair), each also with an _ignore suffix. Plot with
plot_gait_similarity.py.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from zoo_benchmark import CORE_BODY, build_world, get_state_from_data, make_brain_for

FRAMES = ("world", "core", "cycle", "cycle_shape")
LEFTOVERS = ("charge", "ignore")              # how unmatched modules of the larger robot count
LEFTOVER_SUFFIX = {"charge": "", "ignore": "_ignore"}  # on output file names
TRANSIENT_S = 2.0      # skipped before estimating the gait cycle
MIN_PERIOD_S = 0.3
PEAK_FRACTION = 0.9    # first autocorrelation peak this close to the highest is the period
N_PHASE = 64           # samples per averaged cycle
N_YAW = 72             # yaw grid for cycle_distance (5 degrees)


# ── Recording ─────────────────────────────────────────────────────────────────


def _robot_bodies(model: mujoco.MjModel) -> list[int]:
    """Robot bodies that own at least one geom (as behavior_common._robot_bodies)."""
    has_geom = set(int(b) for b in model.geom_bodyid)
    return [b for b in range(model.nbody) if model.body(b).name.startswith("robot1_") and b in has_geom]


def record_module_trajectories(run_dir: Path) -> dict:
    """Replay run_dir's champion; module positions at t = 0 and after every control step."""
    torch.set_num_threads(1)
    cfg = json.loads((run_dir / "config.json").read_text())
    model, data = build_world(cfg["body"])
    brain = make_brain_for(cfg["brain"], model, data, cfg["duration"])
    brain.set_params(np.load(run_dir / "champion.npy"))
    steps_per_ctrl = round(1.0 / (cfg["control_freq"] * model.opt.timestep))
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CORE_BODY)
    bodies = _robot_bodies(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    brain.reset()
    t, pos, core_R = [], [], []

    def record() -> None:
        t.append(float(data.time))
        pos.append(data.xpos[bodies].copy())
        core_R.append(data.xmat[core_id].reshape(3, 3).copy())

    record()
    for _ in range(round(cfg["duration"] * cfg["control_freq"])):
        data.ctrl[:] = brain.act(float(data.time), get_state_from_data(data))
        mujoco.mj_step(model, data, nstep=steps_per_ctrl)
        record()

    pos = np.array(pos)
    core_col = bodies.index(core_id)
    return {
        "body": cfg["body"], "brain": cfg["brain"], "seed": cfg["seed"], "n_hinges": model.nu,
        "t": np.array(t), "pos": pos, "core_pos": pos[:, core_col], "core_R": np.array(core_R),
        "body_names": [model.body(b).name for b in bodies],
        "xspeed": float(pos[-1, core_col, 0] - pos[0, core_col, 0]) / t[-1],
    }


# ── Frames: (T, M, 3) recording -> (M, T, 3) trajectories ─────────────────────


def world_frame(pos: np.ndarray, core_pos: np.ndarray) -> np.ndarray:
    """Positions minus the core's start xy (z kept)."""
    offset = np.array([core_pos[0, 0], core_pos[0, 1], 0.0])
    return (pos - offset).transpose(1, 0, 2)


def core_frame(pos: np.ndarray, core_pos: np.ndarray, core_R: np.ndarray) -> np.ndarray:
    """Positions in the core's moving frame, each module centred on its mean."""
    local = np.einsum("tji,tmj->mti", core_R, pos - core_pos[:, None, :])  # R^T (p - p_core)
    return local - local.mean(axis=1, keepdims=True)


def to_frame(rec: dict, frame: str) -> np.ndarray:
    if frame == "world":
        return world_frame(rec["pos"], rec["core_pos"])
    if frame == "core":
        return core_frame(rec["pos"], rec["core_pos"], rec["core_R"])
    raise ValueError(f"Unknown frame {frame!r}")


# ── Distance ──────────────────────────────────────────────────────────────────


def _charged_mean(cost: np.ndarray, leftover: str = "charge") -> float:
    """Mean charged cost of the Hungarian module matching (see module docstring).

    The matching gives every module of the smaller robot a partner; with
    leftover="ignore" the larger robot's unmatched modules are not charged.
    """
    rows, cols = linear_sum_assignment(cost)
    charged = list(cost[rows, cols])
    if leftover == "ignore":
        return float(np.mean(charged))
    na, nb = cost.shape
    if na > nb:
        charged += list(cost[np.setdiff1d(np.arange(na), rows)].min(axis=1))
    elif nb > na:
        charged += list(cost[:, np.setdiff1d(np.arange(nb), cols)].min(axis=0))
    return float(np.mean(charged))


def _charged_mean_lower_bound(cost: np.ndarray, leftover: str = "charge") -> np.ndarray:
    """Lower bound on _charged_mean for a stack (..., Ma, Mb) of cost matrices:
    every module of the larger robot (of the smaller one with leftover="ignore")
    is charged once, at least its row/column min."""
    na, nb = cost.shape[-2:]
    rows = cost.min(axis=-1).mean(axis=-1)
    cols = cost.min(axis=-2).mean(axis=-1)
    if na == nb:
        return np.maximum(rows, cols)
    return rows if (na > nb) == (leftover == "charge") else cols


def _matched_mse(a: np.ndarray, b: np.ndarray, leftover: str = "charge") -> float:
    T = a.shape[1]
    return _charged_mean(cdist(a.reshape(len(a), -1), b.reshape(len(b), -1), "sqeuclidean") / T, leftover)


def trajectory_distance(a: np.ndarray, b: np.ndarray, max_lag: int = 0, leftover: str = "charge") -> float:
    """Module-matched RMSE (m) between trajectories a (Ma, T, 3) and b (Mb, T, 3).

    With max_lag > 0, b is also shifted by up to +-max_lag samples against a
    (both cropped to the overlap) and the smallest distance is kept.
    """
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Trajectory lengths differ: {a.shape[1]} vs {b.shape[1]}")
    T = a.shape[1]
    best = np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            best = min(best, _matched_mse(a[:, lag:], b[:, :T - lag], leftover))
        else:
            best = min(best, _matched_mse(a[:, :T + lag], b[:, -lag:], leftover))
    return float(np.sqrt(best))


def distance_matrix(trajs: list[np.ndarray], max_lag: int = 0, leftover: str = "charge") -> np.ndarray:
    n = len(trajs)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = trajectory_distance(trajs[i], trajs[j], max_lag, leftover)
    return dist


# ── Gait cycle: speed, phase and orientation independent ─────────────────────


def estimate_period(traj: np.ndarray, dt: float, transient_s: float = TRANSIENT_S) -> tuple[float, float]:
    """(period in s, autocorrelation at it) of a (M, T, 3) trajectory after the transient.

    Uses the normalised autocorrelation of all module coordinates together and
    takes the first peak within PEAK_FRACTION of the highest one, so a multiple
    of the period is not picked when every cycle repeats almost exactly. The
    peak is refined by parabolic interpolation.
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


def body_size(rec: dict) -> float:
    """RMS distance (m) of the modules from the core at t = 0."""
    return float(np.sqrt(((rec["pos"][0] - rec["core_pos"][0]) ** 2).sum(axis=1).mean()))


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Circular cross-correlation r[i, j, s] = sum_p u[i, p] v[j, p + s]."""
    U, V = np.fft.rfft(u, axis=-1), np.fft.rfft(v, axis=-1)
    return np.fft.irfft(np.conj(U)[:, None] * V[None], n=u.shape[-1], axis=-1)


def cycle_distance(
    a: np.ndarray, b: np.ndarray, n_yaw: int = N_YAW, leftover: str = "charge",
) -> tuple[float, dict]:
    """Module-matched RMSE (m) between mean cycles a (Ma, P, 3) and b (Mb, P, 3),
    minimised over circular phase shift, yaw rotation of b (n_yaw steps) and
    mirroring b across a vertical plane. Returns (distance, best alignment).

    All shift x yaw cost matrices come from FFT cross-correlations; the
    Hungarian matching is run in order of a lower bound until no remaining
    candidate can beat the best, so the minimum is exact on the grid.
    """
    P = a.shape[1]
    ax, ay, az = (a[..., k] for k in range(3))
    bx, by, bz = (b[..., k] for k in range(3))
    zz = _cross(az, bz)
    xx, yy, xy, yx = _cross(ax, bx), _cross(ay, by), _cross(ax, by), _cross(ay, bx)
    norms = (a ** 2).sum(axis=(1, 2))[:, None] + (b ** 2).sum(axis=(1, 2))[None, :]
    yaw = np.arange(n_yaw) * 2 * np.pi / n_yaw
    c, s = np.cos(yaw)[:, None, None, None], np.sin(yaw)[:, None, None, None]

    # a . (R_yaw M b) summed over coords; M = identity or y -> -y.
    costs = []
    for p_term, q_term in ((xx + yy, yx - xy), (xx - yy, yx + xy)):
        dot = zz[None] + c * p_term[None] + s * q_term[None]               # (yaw, Ma, Mb, shift)
        costs.append((norms[None, :, :, None] - 2 * dot).transpose(0, 3, 1, 2) / P)
    cost = np.maximum(np.stack(costs), 0.0)                                   # (mirror, yaw, shift, Ma, Mb)

    bound = _charged_mean_lower_bound(cost, leftover)
    best, best_idx = np.inf, None
    for idx in np.argsort(bound, axis=None):
        idx = np.unravel_index(idx, bound.shape)
        if bound[idx] >= best:
            break
        value = _charged_mean(cost[idx], leftover)
        if value < best:
            best, best_idx = value, idx
    mirror, k_yaw, shift = best_idx
    return float(np.sqrt(best)), {"mirrored": bool(mirror), "yaw_deg": float(np.rad2deg(yaw[k_yaw])),
                                  "phase_shift": float(shift / P)}


def cycle_distance_matrix(
    cycles: list[np.ndarray], labels: list[str], leftover: str = "charge",
) -> tuple[np.ndarray, pd.DataFrame]:
    n = len(cycles)
    dist, rows = np.zeros((n, n)), []
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j], align = cycle_distance(cycles[i], cycles[j], leftover=leftover)
            dist[j, i] = dist[i, j]
            rows.append({"a": labels[i], "b": labels[j], "distance": dist[i, j], **align})
    return dist, pd.DataFrame(rows)


# ── Cache ─────────────────────────────────────────────────────────────────────


_ARRAYS = ("t", "pos", "core_pos", "core_R", "body_names")
_SCALARS = ("body", "brain", "seed", "n_hinges", "xspeed")


def save_recordings(path: Path, labels: list[str], recs: list[dict]) -> None:
    arrays = {"labels": np.array(labels)}
    for label, rec in zip(labels, recs):
        for k in _ARRAYS + _SCALARS:
            arrays[f"{label}__{k}"] = np.asarray(rec[k])
    np.savez_compressed(path, **arrays)


def load_recordings(path: Path) -> tuple[list[str], list[dict]]:
    with np.load(path) as z:
        labels = [str(s) for s in z["labels"]]
        recs = [{k: z[f"{label}__{k}"] for k in _ARRAYS + _SCALARS} for label in labels]
    for rec in recs:
        rec["body_names"] = [str(s) for s in rec["body_names"]]
        for k in _SCALARS:
            rec[k] = rec[k].item()
    return labels, recs


# ── Main ──────────────────────────────────────────────────────────────────────


def select_runs(root: Path, brain: str, body: str | None) -> pd.DataFrame:
    """Best-seed run per body for `brain`, or per brain on `body`; indexed by label."""
    df = pd.read_csv(root / "summary.csv")
    key = "brain" if body else "body"
    df = df[df["body"] == body] if body else df[df["brain"] == brain]
    if df.empty:
        raise SystemExit(f"No runs for {'body ' + body if body else 'brain ' + brain} in {root / 'summary.csv'}")
    return df.loc[df.groupby(key)["best_xspeed"].idxmax()].set_index(key).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="zoo benchmark output dir with summary.csv")
    parser.add_argument("--brain", default="ann", help="brain whose best gait per body is compared")
    parser.add_argument("--body", default=None, help="compare the best gait of every brain on this body instead")
    parser.add_argument("--max-lag", type=int, default=0, help="max time shift tried, in control steps")
    parser.add_argument("--leftovers", nargs="+", default=list(LEFTOVERS), choices=LEFTOVERS,
                        help="how the larger robot's unmatched modules count: charged their best match, or ignored")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="default: <root>")
    args = parser.parse_args()
    out_dir = args.out_dir or args.root
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.body}_brains" if args.body else args.brain

    cache = out_dir / f"gait_trajectories_{tag}.npz"
    runs = select_runs(args.root, args.brain, args.body)
    labels = list(runs.index)
    if cache.exists() and load_recordings(cache)[0] == labels:
        _, recs = load_recordings(cache)
        print(f"Loaded {len(recs)} recordings from {cache}")
    else:
        run_dirs = [args.root / runs.loc[label, "run_dir"] for label in labels]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            recs = list(pool.map(record_module_trajectories, run_dirs))
        save_recordings(cache, labels, recs)
        print(f"Recorded {len(recs)} champions -> {cache}")

    for label, rec in zip(labels, recs):
        if not np.isclose(rec["xspeed"], runs.loc[label, "best_xspeed"], atol=1e-6):
            print(f"WARNING: {label} replay x-speed {rec['xspeed']:+.4f} vs stored {runs.loc[label, 'best_xspeed']:+.4f}")

    cycles, cycle_rows = [], []
    for label, rec in zip(labels, recs):
        core, dt = to_frame(rec, "core"), float(rec["t"][1] - rec["t"][0])
        period, autocorr = estimate_period(core, dt)
        cycles.append(average_cycle(core, dt, period))
        rms, size = cycle_rms(cycles[-1]), body_size(rec)
        cycle_rows.append({"label": label, "period_s": period, "autocorr": autocorr,
                           "cycle_rms_m": rms, "body_size_m": size, "rel_amplitude": rms / size})
    pd.DataFrame(cycle_rows).to_csv(out_dir / f"gait_cycles_{tag}.csv", index=False)

    for leftover in args.leftovers:
        lsuffix = LEFTOVER_SUFFIX[leftover]
        for frame in FRAMES:
            if frame in ("cycle", "cycle_shape"):
                shapes = cycles if frame == "cycle" else [shape_cycle(c) for c in cycles]
                dist, align = cycle_distance_matrix(shapes, labels, leftover)
                suffix = "" if frame == "cycle" else f"_{frame}"
                align.to_csv(out_dir / f"gait_alignment_{tag}{suffix}{lsuffix}.csv", index=False)
            else:
                dist = distance_matrix([to_frame(r, frame) for r in recs], args.max_lag, leftover)
            out = out_dir / f"gait_distance_{tag}_{frame}{lsuffix}.csv"
            pd.DataFrame(dist, index=labels, columns=labels).to_csv(out)
            print(f"{frame + lsuffix:18s}: median pairwise distance "
                  f"{np.median(dist[np.triu_indices(len(dist), 1)]):.4f}{"" if frame == "cycle_shape" else " m"} -> {out}")


if __name__ == "__main__":
    main()
