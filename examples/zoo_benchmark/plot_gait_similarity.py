"""Plot gait-similarity maps from gait_similarity.py output.

For each comparison (world, core, cycle, cycle_shape) writes gait_similarity_<tag>_<frame>.png with:
  left   distance matrix, rows/columns ordered by average-linkage clustering,
         dendrogram on top
  right  2D metric-MDS embedding of the same distances, one labelled point per
         gait, coloured by champion x-speed and sized by hinge count
The MDS stress-1 is printed in the panel title: below ~0.1 the 2D distances
are a fair picture of the real ones, above ~0.2 trust the matrix instead.

Usage:
  python plot_gait_similarity.py __data__/ariel_zoo_benchmark [--tag ann] [-o OUTDIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from sklearn.manifold import MDS  # noqa: E402

from gait_similarity import FRAMES, load_recordings  # noqa: E402
from plot_zoo_fitness_curves import INK, MUTED, style_axis  # noqa: E402

# Single-hue blue ramp, light = small.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIST_CMAP = LinearSegmentedColormap.from_list("dist", SEQ_BLUE)
SPEED_CMAP = LinearSegmentedColormap.from_list("speed", SEQ_BLUE[2:])  # markers stay visible on white
FRAME_TITLE = {"world": "world frame (travel counts)", "core": "core frame (motion relative to the core)",
               "cycle": "mean gait cycle (speed, phase and orientation independent)",
               "cycle_shape": "mean gait cycle shape (also amplitude independent)"}
FRAME_UNIT = {"world": "m", "core": "m", "cycle": "m", "cycle_shape": None}  # cycle_shape is dimensionless


def mds_embedding(dist: np.ndarray) -> tuple[np.ndarray, float]:
    """2D metric MDS; returns (coords, Kruskal stress-1)."""
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=8, normalized_stress="auto")
    xy = mds.fit_transform(dist)
    iu = np.triu_indices(len(dist), 1)
    emb = np.linalg.norm(xy[:, None] - xy[None], axis=-1)[iu]
    stress1 = float(np.sqrt(((dist[iu] - emb) ** 2).sum() / (dist[iu] ** 2).sum()))
    return xy, stress1


def plot_frame(dist: pd.DataFrame, meta: pd.DataFrame, title: str, out: Path, dpi: int, unit: str | None = "m") -> None:
    labels = list(dist.index)
    d = dist.to_numpy()
    link = linkage(squareform(d, checks=False), method="average")

    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 0.04, 0.22, 1.05], height_ratios=[0.18, 1],
                          wspace=0.08, hspace=0.02)  # column 2 is a spacer
    ax_dend, ax_mat = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_cbar, ax_map = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[:, 3])

    dn = dendrogram(link, ax=ax_dend, no_labels=True, color_threshold=0, above_threshold_color=MUTED,
                    link_color_func=lambda _: MUTED)
    ax_dend.axis("off")
    order = dn["leaves"]
    im = ax_mat.imshow(d[np.ix_(order, order)], cmap=DIST_CMAP, vmin=0, aspect="auto",
                       interpolation="nearest")
    ticks = np.arange(len(order))
    ax_mat.set_xticks(ticks, [labels[i] for i in order], rotation=90, fontsize=8, color=INK)
    ax_mat.set_yticks(ticks, [labels[i] for i in order], fontsize=8, color=INK)
    ax_mat.tick_params(length=0)
    for s in ax_mat.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, cax=ax_cbar)
    cb.set_label("module-matched RMSE" + (f" ({unit})" if unit else ", unit-RMS cycles"), fontsize=9, color=INK)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)

    xy, stress = mds_embedding(d)
    speed = meta.loc[labels, "xspeed"].to_numpy()
    hinges = meta.loc[labels, "n_hinges"].to_numpy()
    sc = ax_map.scatter(xy[:, 0], xy[:, 1], c=speed, cmap=SPEED_CMAP, s=30 + 12 * hinges,
                        edgecolors="white", linewidths=2, zorder=3)
    for (x, y), label in zip(xy, labels):
        ax_map.annotate(label, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8, color=INK)
    style_axis(ax_map)
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("MDS 1" + (f" ({unit})" if unit else ""), fontsize=9, color=INK)
    ax_map.set_ylabel("MDS 2" + (f" ({unit})" if unit else ""), fontsize=9, color=INK)
    ax_map.set_title(f"2D map, stress-1 = {stress:.2f}  (marker size = hinges)", fontsize=10, color=INK)
    cb2 = fig.colorbar(sc, ax=ax_map, fraction=0.04, pad=0.08)
    cb2.set_label("champion x-speed (m/s)", fontsize=9, color=INK)
    cb2.ax.tick_params(colors=MUTED, labelsize=8)
    cb2.outline.set_visible(False)

    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  (MDS stress-1 {stress:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="gait_similarity.py output dir")
    parser.add_argument("--tag", default="ann", help="output tag: the brain, or <body>_brains")
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="default: data_dir")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    out_dir = args.out_dir or args.data_dir

    labels, recs = load_recordings(args.data_dir / f"gait_trajectories_{args.tag}.npz")
    meta = pd.DataFrame(recs, index=labels)
    for frame in FRAMES:
        dist = pd.read_csv(args.data_dir / f"gait_distance_{args.tag}_{frame}.csv", index_col=0)
        plot_frame(dist, meta, f"Gait similarity ({args.tag}), {FRAME_TITLE[frame]}",
                   out_dir / f"gait_similarity_{args.tag}_{frame}.png", args.dpi, FRAME_UNIT[frame])


if __name__ == "__main__":
    main()
