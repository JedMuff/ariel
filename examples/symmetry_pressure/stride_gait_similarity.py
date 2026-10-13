"""
Gait similarity within populations of the control-stride sweep
(slurm/run_control_stride_sweep.sh, ANN brain, forward task), to test whether
symmetric genomes converge on a shared gait.

Input: 50Hz traces from behavior_extract.py, one behavior.npz per run, e.g.
    RECORD_EVERY_N=10 BEHAVIOR_SUBDIR=behavior_50hz SWEEP_ROOTS=... \\
        sbatch --array=9-26 slurm/run_behavior_extract_array.sh
Each episode's post-settle body poses are turned into a mean gait cycle in the
core's moving frame, using the zoo benchmark's gait_similarity.py:
estimate_period + average_cycle. Gaits are then compared with its
cycle_distance, a module-matched RMSE minimised over phase shift, yaw and
mirror, in two variants:
  cycle_shape  cycles scaled to unit RMS: only the shape and coordination of
               the motion count (dimensionless, 0 = same, ~1.4 = unrelated)
  cycle        cycles in metres: amplitude counts too
An episode with no autocorrelation peak gets one "cycle" spanning the whole
post-transient window and is flagged aperiodic rather than dropped.

Writes to <stride-root>/gait_similarity/:
  cycles/<run>.npz       per-episode cycles + period/amplitude (reused if present)
  pairwise/<run>.npz     per generation: condensed pairwise distances, both variants
  summary.csv            per run x gen x variant: mean/median/quartiles of the
                         pairwise distances (behavioural diversity)
  pool_<genome>_s<stride>_<variant>.csv   final generation of every rep pooled:
                         labelled distance matrix
  pool_<genome>_s<stride>_<variant>_alignment.csv   best mirror/yaw/phase per pair
  pool_<genome>_s<stride>_meta.csv        one row per pooled individual
  medoid_summary.csv     per genome x stride (scope=pool) and per run (scope=run):
                         the medoid (smallest median distance to the rest) and
                         the median / IQR of its distances to everyone else

Plot with plot_gait_similarity_stride_sweep.py; browse with stride_gait_page.py.

Usage:
    python stride_gait_similarity.py [--workers 48]
    python stride_gait_similarity.py --runs ctrlstride_forward_tree_symmetric_s20_rep0_43569_3 --gens 0 30
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "zoo_benchmark"))

from gait_similarity import (  # noqa: E402
    TRANSIENT_S,
    average_cycle,
    core_frame,
    cycle_distance,
    cycle_rms,
    estimate_period,
    shape_cycle,
)

SETTLE_S = 3.0  # shared.SETTLE_DURATION: poses before this are the settle phase
VARIANTS = ("cycle_shape", "cycle")
GENOMES = ("tree", "tree_symmetric")
STRIDE_TAG_RE = re.compile(
    r"^ctrlstride_forward_(?P<genome>tree_symmetric|tree|cppn)_s(?P<stride>\d+)_rep(?P<rep>\d+)_\d+_\d+$"
)
META_COLS = ("gen", "body_idx", "ind_id", "stored_fitness", "replayed_fitness", "n_joints", "n_bodies")
CYCLE_COLS = ("period_s", "autocorr", "aperiodic", "cycle_rms_m", "body_size_m", "rel_amplitude")


# ── Poses -> cycles ───────────────────────────────────────────────────────────


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(T, 4) MuJoCo quaternions (w, x, y, z) -> (T, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def episode_cycle(traj: np.ndarray, dt: float) -> tuple[np.ndarray, dict]:
    """Mean cycle (M, N_PHASE, 3) of a core-frame trajectory (M, T, 3), with
    its period stats; falls back to the whole post-transient window as one
    cycle when no period is found."""
    period, autocorr = estimate_period(traj, dt)
    aperiodic = not np.isfinite(period)
    if aperiodic:
        period = (traj.shape[1] - 1) * dt - TRANSIENT_S
    cycle = average_cycle(traj, dt, period)
    return cycle, {"period_s": float(period), "autocorr": float(autocorr), "aperiodic": aperiodic}


def body_size(pos0: np.ndarray) -> float:
    """RMS distance (m) of the modules from the core (column 0) in one (B, 3) pose."""
    return float(np.sqrt(((pos0 - pos0[0]) ** 2).sum(axis=1).mean()))


def iter_episode_poses(behavior_npz: Path, meta_cols=META_COLS):
    """(meta, generator of (episode, P (T, B, 3), Q (T, B, 4), dt)) for the
    post-settle body poses of every episode of a behavior_extract.py output;
    body 0 is the core."""
    with np.load(behavior_npz) as z:
        meta = pd.DataFrame({c: z[f"episode_meta__{c}"] for c in meta_cols})
        ep, t = z["body_poses__episode_idx"], z["body_poses__time"]
        pos, quat = z["body_poses__pos"], z["body_poses__quat"]
    bounds = np.searchsorted(ep, np.arange(len(meta) + 1))  # rows are written episode by episode

    def episodes():
        for e in range(len(meta)):
            sl = slice(bounds[e], bounds[e + 1])
            n_b = int(meta.loc[e, "n_bodies"])
            keep = t[sl] >= SETTLE_S
            P = pos[sl][keep].astype(np.float64).reshape(-1, n_b, 3)
            Q = quat[sl][keep].astype(np.float64).reshape(-1, n_b, 4)
            dt = float(np.median(np.diff(t[sl][keep][::n_b])))
            yield e, P, Q, dt

    return meta, episodes()


def run_cycles(behavior_npz: Path) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """One row + mean cycle per episode of a behavior_extract.py output."""
    meta, episodes = iter_episode_poses(behavior_npz)
    rows, cycles = [], []
    for _, P, Q, dt in episodes:
        traj = core_frame(P, P[:, 0], quat_to_mat(Q[:, 0]))
        cycle, stats = episode_cycle(traj, dt)
        rms, size = cycle_rms(cycle), body_size(P[0])
        rows.append({**stats, "cycle_rms_m": rms, "body_size_m": size, "rel_amplitude": rms / size})
        cycles.append(cycle.astype(np.float32))
    return pd.concat([meta, pd.DataFrame(rows)], axis=1), cycles


def save_cycles(path: Path, df: pd.DataFrame, cycles: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cycles=np.concatenate(cycles), n_modules=np.array([len(c) for c in cycles]),
                        **{c: df[c].to_numpy() for c in df.columns})


def load_cycles(path: Path) -> tuple[pd.DataFrame, list[np.ndarray]]:
    with np.load(path) as z:
        df = pd.DataFrame({k: z[k] for k in z.files if k not in ("cycles", "n_modules")})
        cycles = np.split(z["cycles"].astype(np.float64), np.cumsum(z["n_modules"])[:-1])
    return df, cycles


# ── Distances ─────────────────────────────────────────────────────────────────


def _prep(cycles: list[np.ndarray], variant: str) -> list[np.ndarray]:
    return [shape_cycle(c) for c in cycles] if variant == "cycle_shape" else cycles


def condensed_distances(cycles: list[np.ndarray], variant: str) -> np.ndarray:
    """Pairwise cycle_distance in scipy's condensed order."""
    c = _prep(cycles, variant)
    return np.array([cycle_distance(c[i], c[j])[0] for i in range(len(c)) for j in range(i + 1, len(c))])


def distance_row(cycles: list[np.ndarray], variant: str, i: int) -> list[dict]:
    """Distance and best alignment of cycle i to every later one."""
    c = _prep(cycles, variant)
    out = []
    for j in range(i + 1, len(c)):
        d, align = cycle_distance(c[i], c[j])
        out.append({"i": i, "j": j, "distance": d, **align})
    return out


def medoid_stats(dist: np.ndarray) -> dict:
    """Medoid of a square distance matrix (smallest median distance to the
    others) and the spread of its distances, plus all-pairs stats for scale."""
    n = len(dist)
    others = [np.delete(dist[i], i) for i in range(n)]
    med = np.array([np.median(o) for o in others])
    i = int(np.argmin(med))
    q25, q50, q75 = np.percentile(others[i], [25, 50, 75])
    pairs = squareform(dist, checks=False)
    p25, p50, p75 = np.percentile(pairs, [25, 50, 75])
    return {"medoid": i, "n": n, "medoid_median": q50, "medoid_q25": q25, "medoid_q75": q75,
            "medoid_iqr": q75 - q25, "pairs_mean": pairs.mean(), "pairs_median": p50,
            "pairs_q25": p25, "pairs_q75": p75, "pairs_iqr": p75 - p25}


def label_of(rep: int, gen: int, body_idx: int) -> str:
    return f"rep{rep}_g{gen:03d}_b{body_idx:02d}"


# ── Main ──────────────────────────────────────────────────────────────────────


def discover(root: Path, subdir: str, genomes: list[str], only: list[str] | None) -> list[dict]:
    runs = []
    for d in sorted((root / subdir).glob("ctrlstride_*")):
        m = STRIDE_TAG_RE.match(d.name)
        if not m or m["genome"] not in genomes or (only and d.name not in only):
            continue
        if not (d / "behavior.npz").exists():
            print(f"[warn] no behavior.npz in {d}, skipping")
            continue
        runs.append({"run": d.name, "genome": m["genome"], "stride": int(m["stride"]),
                     "rep": int(m["rep"]), "npz": d / "behavior.npz"})
    return runs


def _cycles_job(job: tuple[Path, Path]) -> str:
    npz, out = job
    df, cycles = run_cycles(npz)
    save_cycles(out, df, cycles)
    bad = (df["stored_fitness"] - df["replayed_fitness"]).abs().max()
    return f"{out.stem}: {len(df)} episodes, {int(df['aperiodic'].sum())} aperiodic, max|fitness delta|={bad:.1e}"


def _gen_job(job: tuple) -> tuple:
    run, gen, variant, cycles = job
    return run, gen, variant, condensed_distances(cycles, variant)


def _row_job(job: tuple) -> tuple:
    key, variant, cycles, i = job
    return key, variant, distance_row(cycles, variant, i)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
    parser.add_argument("--behavior-subdir", default="behavior_50hz")
    parser.add_argument("--genomes", nargs="+", default=list(GENOMES))
    parser.add_argument("--runs", nargs="+", default=None, help="only these run names")
    parser.add_argument("--gens", nargs="+", type=int, default=None, help="only these gens in the per-gen pass")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--force", action="store_true", help="recompute cached cycles and distances")
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="default: <stride-root>/gait_similarity")
    args = parser.parse_args()
    out = args.out_dir or args.stride_root / "gait_similarity"

    runs = discover(args.stride_root, args.behavior_subdir, args.genomes, args.runs)
    if not runs:
        raise SystemExit(f"No runs with behavior.npz under {args.stride_root / args.behavior_subdir}")
    print(f"{len(runs)} runs")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # 1. Cycles per episode.
        todo = [(r["npz"], out / "cycles" / f"{r['run']}.npz") for r in runs]
        todo = [j for j in todo if args.force or not j[1].exists()
                or j[1].stat().st_mtime < j[0].stat().st_mtime]
        for msg in pool.map(_cycles_job, todo):
            print(msg)
        data = {r["run"]: load_cycles(out / "cycles" / f"{r['run']}.npz") for r in runs}

        # 2. Pairwise distances within each generation; cached gens are kept.
        jobs, results = [], {}
        for r in runs:
            path = out / "pairwise" / f"{r['run']}.npz"
            if path.exists() and not args.force:
                with np.load(path) as z:
                    results[r["run"]] = {k: z[k] for k in z.files}
            done = results.get(r["run"], {})
            df, cycles = data[r["run"]]
            for gen in sorted(df["gen"].unique()):
                if args.gens is not None and gen not in args.gens:
                    continue
                idx = np.flatnonzero(df["gen"].to_numpy() == gen)
                jobs += [(r["run"], int(gen), v, [cycles[i] for i in idx])
                         for v in VARIANTS if f"g{gen:03d}__{v}" not in done]
        for run, gen, variant, d in pool.map(_gen_job, jobs, chunksize=1):
            results.setdefault(run, {})[f"g{gen:03d}__{variant}"] = d
        for run in {j[0] for j in jobs}:
            (out / "pairwise").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "pairwise" / f"{run}.npz", **results[run])
        print(f"Per-generation distances: {len(jobs)} gen x variant blocks")

        # 3. Final generation of every rep pooled per genome x stride.
        groups: dict[tuple, list[dict]] = {}
        for r in runs:
            groups.setdefault((r["genome"], r["stride"]), []).append(r)
        pool_data = {}
        for (genome, stride), rs in sorted(groups.items()):
            metas, cycles = [], []
            for r in sorted(rs, key=lambda r: r["rep"]):
                df, cyc = data[r["run"]]
                last = int(df["gen"].max())
                for i in np.flatnonzero(df["gen"].to_numpy() == last):
                    metas.append({"label": label_of(r["rep"], last, int(df.loc[i, "body_idx"])), "run": r["run"],
                                  "rep": r["rep"], **df.loc[i].to_dict()})
                    cycles.append(cyc[i])
            meta = pd.DataFrame(metas).set_index("label")
            key = f"{genome}_s{stride}"
            meta.to_csv(out / f"pool_{key}_meta.csv")
            pool_data[key] = (meta, cycles)
        jobs = []
        for key, (meta, cycles) in pool_data.items():
            if not args.force and all((out / f"pool_{key}_{v}.csv").exists() for v in VARIANTS):
                continue
            jobs += [(key, v, cycles, i) for v in VARIANTS for i in range(len(cycles) - 1)]
        rows: dict[tuple, list] = {}
        for key, variant, part in pool.map(_row_job, jobs, chunksize=1):
            rows.setdefault((key, variant), []).extend(part)
        for (key, variant), part in rows.items():
            labels = list(pool_data[key][0].index)
            align = pd.DataFrame(part)
            dist = np.zeros((len(labels), len(labels)))
            dist[align["i"], align["j"]] = dist[align["j"], align["i"]] = align["distance"]
            pd.DataFrame(dist, index=labels, columns=labels).to_csv(out / f"pool_{key}_{variant}.csv")
            align.insert(0, "a", [labels[i] for i in align["i"]])
            align.insert(1, "b", [labels[j] for j in align["j"]])
            align.drop(columns=["i", "j"]).to_csv(out / f"pool_{key}_{variant}_alignment.csv", index=False)
        print(f"Pooled final generations: {len(pool_data)} experiments")

    # 4. Summaries.
    summary, medoids = [], []
    for r in runs:
        path = out / "pairwise" / f"{r['run']}.npz"
        if not path.exists():
            continue
        df = data[r["run"]][0]
        last = int(df["gen"].max())
        with np.load(path) as z:
            for k in sorted(z.files):
                gen, variant = int(k[1:4]), k.split("__")[1]
                d = z[k]
                if len(d) == 0:
                    continue
                gdf = df[df["gen"] == gen]
                q25, q50, q75 = np.percentile(d, [25, 50, 75])
                summary.append({"run": r["run"], "genome": r["genome"], "stride": r["stride"], "rep": r["rep"],
                                "gen": gen, "variant": variant, "n": len(gdf),
                                "n_aperiodic": int(gdf["aperiodic"].sum()), "mean": d.mean(),
                                "median": q50, "q25": q25, "q75": q75})
                if gen == last:
                    st = medoid_stats(squareform(d))
                    body = int(gdf["body_idx"].iloc[st.pop("medoid")])
                    medoids.append({"scope": "run", "genome": r["genome"], "stride": r["stride"],
                                    "run": r["run"], "variant": variant,
                                    "medoid_label": label_of(r["rep"], gen, body), **st})
    for key, (meta, _) in pool_data.items():
        genome, stride = key.rsplit("_s", 1)
        for variant in VARIANTS:
            path = out / f"pool_{key}_{variant}.csv"
            if not path.exists():
                continue
            dist = pd.read_csv(path, index_col=0)
            st = medoid_stats(dist.to_numpy())
            label = dist.index[st.pop("medoid")]
            medoids.append({"scope": "pool", "genome": genome, "stride": int(stride), "run": meta.loc[label, "run"],
                            "variant": variant, "medoid_label": label, **st})
    if summary:
        pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    med = pd.DataFrame(medoids).sort_values(["scope", "variant", "stride", "genome"])
    med.to_csv(out / "medoid_summary.csv", index=False)
    show = med[(med["scope"] == "pool") & (med["variant"] == "cycle_shape")]
    print(show[["genome", "stride", "n", "medoid_label", "medoid_median", "medoid_iqr", "pairs_median"]]
          .to_string(index=False, float_format="%.3f"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
