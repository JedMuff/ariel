"""
Gait species (E3): clusters of similar gait cycles among replayed episodes.

Distance: gait.cycle_distance on unit-RMS mean cycles (gait.shape_cycle):
module-matched RMSE minimised over phase shift, yaw and mirror; 0 = same
gait, ~1.0-1.4 = unrelated. Species: average-linkage clustering of each
group's episodes (by default genome x brain x skill budget), cut at
--threshold, numbered by size (1 = largest). A species' medoid is its member
with the smallest mean distance to the rest.

Input: replay.py outputs (<root>/<run>/episodes.csv + cycles.npz).
Writes to <root>/gait_species/:
  pairwise/<group>.npz      condensed distances (reused when newer than the inputs)
  species_t<thr>.csv        one row per episode: group, run, ckpt, score, species,
                            species size and share, medoid flag
  summary_t<thr>.csv        per group: n episodes, n species, largest species share,
                            mean score of the largest species vs the rest

Usage:
    python gait_species.py --root E3_OUT [--groups genome brain skill_budget] [--threshold 0.5]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from gait import cycle_distance, shape_cycle

DEFAULT_THRESHOLD = 0.5
CHUNK = 64  # cycle pairs per job


def load_episodes(root: Path) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Every replayed episode under root with its shape cycle, in matching order."""
    rows, cycles = [], []
    for ep in sorted(root.glob("*/episodes.csv")):
        npz = ep.parent / "cycles.npz"
        if not npz.exists():
            continue
        df = pd.read_csv(ep)
        with np.load(npz) as z:
            by_ckpt = dict(zip(z["ckpt"], np.split(z["cycles"].astype(np.float64), np.cumsum(z["n_modules"])[:-1])))
        df = df[df["ckpt"].isin(by_ckpt)]
        rows.append(df)
        cycles += [shape_cycle(by_ckpt[c]) for c in df["ckpt"]]
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), cycles


def _pairs_job(pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
    return [cycle_distance(a, b) for a, b in pairs]


def condensed_distances(cycles: list[np.ndarray], workers: int) -> np.ndarray:
    pairs = list(itertools.combinations(range(len(cycles)), 2))
    chunks = [[(cycles[i], cycles[j]) for i, j in pairs[k:k + CHUNK]] for k in range(0, len(pairs), CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return np.array([d for part in pool.map(_pairs_job, chunks) for d in part])


def species_labels(condensed: np.ndarray, n: int, threshold: float) -> np.ndarray:
    """Average-linkage clusters cut at threshold, numbered by size (1 = largest)."""
    if n < 2:
        return np.ones(n, dtype=int)
    raw = fcluster(linkage(condensed, method="average"), t=threshold, criterion="distance")
    order = {c: k + 1 for k, (c, _) in enumerate(sorted(
        ((c, (raw == c).sum()) for c in np.unique(raw)), key=lambda x: (-x[1], x[0])))}
    return np.array([order[c] for c in raw])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="replay.py output root")
    parser.add_argument("--groups", nargs="+", default=["genome", "brain", "skill_budget"])
    parser.add_argument("--threshold", type=float, nargs="+", default=[DEFAULT_THRESHOLD])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    out = args.root / "gait_species"

    df, cycles = load_episodes(args.root)
    if df.empty:
        raise SystemExit(f"no replayed episodes under {args.root}")
    newest = max(p.stat().st_mtime for p in args.root.glob("*/cycles.npz"))
    groups = list(df.groupby(args.groups).groups.items())
    dist = {}
    for key, idx in groups:
        key = key if isinstance(key, tuple) else (key,)
        name = "__".join(str(k) for k in key)
        path = out / "pairwise" / f"{name}.npz"
        if path.exists() and path.stat().st_mtime >= newest:
            with np.load(path) as z:
                d = z["d"]
        else:
            print(f"{name}: {len(idx)} episodes, {len(idx) * (len(idx) - 1) // 2} distances", flush=True)
            d = condensed_distances([cycles[i] for i in idx], args.workers)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, d=d)
        dist[key] = (np.asarray(idx), d)

    for thr in args.threshold:
        rows, summ = [], []
        for key, (idx, d) in dist.items():
            lab = species_labels(d, len(idx), thr)
            sq = squareform(d) if len(idx) > 1 else np.zeros((1, 1))
            g = df.loc[idx].copy()
            g["species"] = lab
            g["species_size"] = [int((lab == s).sum()) for s in lab]
            g["species_share"] = g["species_size"] / len(lab)
            medoids = set()
            for s in np.unique(lab):
                members = np.flatnonzero(lab == s)
                within = sq[np.ix_(members, members)].sum(axis=1) / max(len(members) - 1, 1)
                medoids.add(int(members[int(np.argmin(within))]))
            g["medoid"] = [i in medoids for i in range(len(lab))]
            rows.append(g[[*args.groups, "run", "ckpt", "score", "rowing_label", "species", "species_size",
                           "species_share", "medoid"]])
            top = g["species"] == 1
            summ.append({**dict(zip(args.groups, key)), "n": len(lab), "n_species": int(lab.max()),
                         "largest_share": float(top.mean()), "score_largest": float(g.loc[top, "score"].mean()),
                         "score_rest": float(g.loc[~top, "score"].mean()) if (~top).any() else float("nan"),
                         "mean_distance": float(d.mean()) if len(d) else float("nan")})
        tag = f"t{thr:g}"
        pd.concat(rows).to_csv(out / f"species_{tag}.csv", index=False)
        summ = pd.DataFrame(summ)
        summ.to_csv(out / f"summary_{tag}.csv", index=False)
        print(f"\nthreshold {thr:g}:\n{summ.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
