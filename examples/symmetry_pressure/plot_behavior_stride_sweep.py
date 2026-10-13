"""
Behavioural-descriptor plots for the control-stride sweep
(slurm/run_control_stride_sweep.sh): forward task, 3 genome types x strides
{20, 50, 100}, 3 reps, plus the stride-9 baseline (forward runs of the
symmetry-pressure sweep, reps 0-2), as in plot_control_stride_sweep.py.

Reads <root>/behavior/<run>/descriptors.csv written by behavior_extract.py
(one row per checkpointed body; forward has a single skill). Every
generation is represented by its whole evaluated population.

Produces, under <stride-root>/analysis/behavior/:
  gens_<family>.png
    rows = the family's descriptors, cols = genome type; one line per stride,
    per-generation population mean, then mean ± 1 SEM across reps.
  symmetry_by_stride.png
    cols = stride, one line per genome type: share of bodies with any mirror
    joint pair, mirror_pair_frac, and the pair corr / phase lag / amplitude
    asymmetry (pair stats only exist for bodies with pairs).
  symmetry_distributions.png
    histograms of mirror_pair_corr_mean and mirror_pair_phase_lag over the
    last generations, pooled over reps, per genome type x stride.
  descriptor_fitness_corr.png
    Spearman correlation of each descriptor with performance (-fitness, so
    > 0 = goes with better fitness) per genome type x stride.
  descriptors_vs_fitness.png
    scatter of key descriptors against fitness, coloured by stride.
Glitch episodes (c_hinge > HINGE_CONTACT_LIMIT, fitness pinned to 1.0) are
kept for the behaviour curves and dropped from the fitness analysis.

Run behavior_extract.py on the baseline runs first (default out dir), e.g.
    python behavior_extract.py --run-dir <sympress_forward_run> --no-traces

Usage:
    python plot_behavior_stride_sweep.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sweep_common import GENOME_TYPES, parse_run_tag

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK = "forward"
HINGE_CONTACT_LIMIT = 200  # shared.HINGE_CONTACT_LIMIT
LAST_GENS = 5              # generations pooled for the distribution plot
MIN_CORR_POINTS = 20

parser = argparse.ArgumentParser()
parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
parser.add_argument("--baseline-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_symmetry_pressure_sweep")
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()

# Styling / baseline selection copied from plot_control_stride_sweep.py (which
# parses its CLI at import time, so it can't be imported).
BASELINE_STRIDE = 9
BASELINE_REPS = {0, 1, 2}
STRIDE_COLOR = {9: "#86b6ef", 20: "#3987e5", 50: "#1c5cab", 100: "#0d366b"}
STRIDE_MARKER = {9: "o", 20: "s", 50: "^", 100: "D"}
GENOME_LABEL = {"tree": "tree", "tree_symmetric": "tree (symmetric)", "cppn": "cppn"}
GENOME_COLOR = {"tree": "#2a78d6", "tree_symmetric": "#eb6834", "cppn": "#1baf7a"}
CI_ALPHA = 0.15

_STRIDE_TAG_RE = re.compile(
    r"^ctrlstride_forward_(?P<genome>tree_symmetric|tree|cppn)_s(?P<stride>\d+)_rep(?P<rep>\d+)_\d+_\d+$"
)

# behavior_common.DESCRIPTOR_NAMES, grouped (not imported: that module pulls in
# mujoco / torch via shared.py). mirror_plane is categorical and left out.
FAMILIES = {
    "locomotion": ["net_displacement", "path_length", "mean_speed", "straightness",
                   "axis_displacement", "lateral_drift", "yaw_total", "mean_abs_yaw_rate"],
    "posture": ["core_height_mean", "core_height_std", "core_tilt_mean", "core_tilt_std"],
    "contact": ["duty_factor_mean", "n_contact_bodies_mean", "frac_bodies_ever_contact",
                "contact_entropy_bits", "c_hinge"],
    "gait": ["gait_frequency_hz", "joint_amplitude_mean", "joint_corr_abs_mean"],
    "symmetry": ["mirror_pair_frac", "mirror_pair_corr_mean", "mirror_pair_phase_lag",
                 "mirror_pair_amp_asym"],
    "effort": ["ctrl_abs_mean", "ctrl_jerk_mean"],
}
ALL_DESCRIPTORS = [d for ds in FAMILIES.values() for d in ds]
SCATTER_DESCRIPTORS = ["straightness", "gait_frequency_hz", "mirror_pair_corr_mean",
                       "core_tilt_mean", "ctrl_jerk_mean", "contact_entropy_bits"]


def stride_label(stride: int) -> str:
    base = " (baseline)" if stride == BASELINE_STRIDE else ""
    return f"stride {stride} (~{500 / stride:.0f} Hz){base}"


# ── Loading ───────────────────────────────────────────────────────────────


def load() -> pd.DataFrame:
    frames = []
    for d in sorted((args.stride_root / "behavior").glob("ctrlstride_*")):
        m = _STRIDE_TAG_RE.match(d.name)
        if m:
            frames.append(_load_run(d, m["genome"], int(m["stride"]), int(m["rep"])))
    for d in sorted((args.baseline_root / "behavior").glob("sympress_forward_*")):
        info = parse_run_tag(d.name)
        if info and info["rep"] in BASELINE_REPS:
            frames.append(_load_run(d, info["genome"], BASELINE_STRIDE, info["rep"]))
    df = pd.concat([f for f in frames if f is not None], ignore_index=True)
    df["glitch"] = df["d_c_hinge"] > HINGE_CONTACT_LIMIT
    df["has_pair"] = (df["d_mirror_pair_frac"] > 0).astype(float)
    return df


def _load_run(run_dir: Path, genome: str, stride: int, rep: int) -> pd.DataFrame | None:
    path = run_dir / "descriptors.csv"
    if not path.exists():
        print(f"[warn] no descriptors.csv in {run_dir}, skipping")
        return None
    df = pd.read_csv(path)
    bad = df["control_step_freq"] != stride
    assert not bad.any(), f"{run_dir.name}: control_step_freq {df.loc[bad, 'control_step_freq'].iloc[0]} != {stride}"
    df = df[df["skill"] == "fwd"].copy()
    df["genome"], df["stride"], df["rep"] = genome, stride, rep
    return df


# ── Aggregation ───────────────────────────────────────────────────────────


def gen_curve(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-rep population mean per generation, then mean ± 1 SEM across reps."""
    per_rep = df.groupby(["rep", "gen"])[col].mean().unstack("rep")
    m = per_rep.mean(axis=1)
    sem = per_rep.apply(lambda r: stats.sem(r.dropna()) if r.notna().sum() > 1 else 0.0, axis=1)
    return per_rep.index.to_numpy(), m.to_numpy(), (m - sem).to_numpy(), (m + sem).to_numpy()


def _plot_curve(ax, df: pd.DataFrame, col: str, color: str, label: str, marker: str | None = None) -> None:
    if df.empty or df[col].notna().sum() == 0:
        return
    gens, m, lo, hi = gen_curve(df, col)
    ax.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA, linewidth=0)
    ax.plot(gens, m, color=color, linewidth=2, label=label, marker=marker, markevery=5, markersize=5)


# ── Plots ─────────────────────────────────────────────────────────────────


def plot_family(df: pd.DataFrame, family: str, strides: list[int], out_path: Path) -> None:
    cols = FAMILIES[family]
    fig, axes = plt.subplots(len(cols), len(GENOME_TYPES), figsize=(15, 2.8 * len(cols) + 1),
                             sharex=True, sharey="row", squeeze=False)
    for c, genome in enumerate(GENOME_TYPES):
        for r, desc in enumerate(cols):
            ax = axes[r, c]
            for stride in strides:
                sub = df[(df.genome == genome) & (df.stride == stride)]
                _plot_curve(ax, sub, f"d_{desc}", STRIDE_COLOR[stride], stride_label(stride), STRIDE_MARKER[stride])
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(GENOME_LABEL[genome])
            if r == len(cols) - 1:
                ax.set_xlabel("Generation")
            if c == 0:
                ax.set_ylabel(desc)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(f"Behavioural descriptors ({family}) over generations — population mean, "
                 f"mean ± 1 SEM across reps")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_symmetry_by_stride(df: pd.DataFrame, strides: list[int], out_path: Path) -> None:
    rows = [
        ("has_pair", "share of bodies with\nany mirror joint pair"),
        ("d_mirror_pair_frac", "mirror_pair_frac"),
        ("d_mirror_pair_corr_mean", "mirror_pair_corr_mean\n(+1 in phase, −1 anti-phase)"),
        ("d_mirror_pair_phase_lag", "mirror_pair_phase_lag (rad)\n(0 in phase, π anti-phase)"),
        ("d_mirror_pair_amp_asym", "mirror_pair_amp_asym\n(0 = equal swing)"),
    ]
    fig, axes = plt.subplots(len(rows), len(strides), figsize=(4.2 * len(strides), 14),
                             sharex=True, sharey="row", squeeze=False)
    for c, stride in enumerate(strides):
        for r, (col, ylabel) in enumerate(rows):
            ax = axes[r, c]
            for genome in GENOME_TYPES:
                sub = df[(df.genome == genome) & (df.stride == stride)]
                _plot_curve(ax, sub, col, GENOME_COLOR[genome], GENOME_LABEL[genome])
            if col == "d_mirror_pair_phase_lag":
                ax.axhline(np.pi / 2, color="#888888", linewidth=1, linestyle=":", zorder=1)
            if col == "d_mirror_pair_corr_mean":
                ax.axhline(0, color="#888888", linewidth=1, zorder=1)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(stride_label(stride))
            if r == len(rows) - 1:
                ax.set_xlabel("Generation")
            if c == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Left/right mirror-joint behaviour per stride — population mean, mean ± 1 SEM across reps\n"
                 "(pair statistics average only over bodies that have a mirror pair; see top row)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_symmetry_distributions(df: pd.DataFrame, strides: list[int], out_path: Path) -> None:
    specs = [("d_mirror_pair_corr_mean", np.linspace(-1, 1, 21), "mirror_pair_corr_mean"),
             ("d_mirror_pair_phase_lag", np.linspace(0, np.pi, 19), "mirror_pair_phase_lag (rad)")]
    fig, axes = plt.subplots(len(GENOME_TYPES), 2 * len(strides), figsize=(3.2 * 2 * len(strides), 9),
                             sharex="col", squeeze=False)
    for r, genome in enumerate(GENOME_TYPES):
        for s, stride in enumerate(strides):
            sub = df[(df.genome == genome) & (df.stride == stride)]
            sub = sub[sub.gen > sub.gen.max() - LAST_GENS] if not sub.empty else sub
            for k, (col, bins, xlabel) in enumerate(specs):
                ax = axes[r, 2 * s + k]
                vals = sub[col].dropna()
                ax.hist(vals, bins=bins, color=GENOME_COLOR[genome], alpha=0.85, edgecolor="white")
                ax.text(0.97, 0.95, f"n={len(vals)}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
                ax.grid(alpha=0.3, axis="y")
                if r == 0:
                    ax.set_title(f"{stride_label(stride)}\n{xlabel}", fontsize=10)
                if r == len(GENOME_TYPES) - 1:
                    ax.set_xlabel(xlabel, fontsize=9)
                if s == 0 and k == 0:
                    ax.set_ylabel(f"{GENOME_LABEL[genome]}\nbodies")
    fig.suptitle(f"Mirror-pair coordination over the last {LAST_GENS} generations, pooled over reps "
                 "(bodies with ≥1 moving mirror pair)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_fitness_corr(df: pd.DataFrame, strides: list[int], out_path: Path) -> None:
    ok = df[~df.glitch]
    groups = [(g, s) for g in GENOME_TYPES for s in strides]
    mat = np.full((len(ALL_DESCRIPTORS), len(groups)), np.nan)
    for j, (genome, stride) in enumerate(groups):
        sub = ok[(ok.genome == genome) & (ok.stride == stride)]
        perf = -sub["stored_fitness"]
        for i, desc in enumerate(ALL_DESCRIPTORS):
            x = sub[f"d_{desc}"]
            valid = x.notna() & perf.notna()
            if valid.sum() >= MIN_CORR_POINTS and x[valid].nunique() > 1:
                mat[i, j] = stats.spearmanr(x[valid], perf[valid]).statistic

    fig, ax = plt.subplots(figsize=(1.0 * len(groups) + 4, 0.42 * len(ALL_DESCRIPTORS) + 2))
    im = ax.imshow(np.ma.masked_invalid(mat), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(mat[i, j]) > 0.55 else "#222222")
    ax.set_xticks(range(len(groups)), [f"{GENOME_LABEL[g]}\ns{s}" for g, s in groups], fontsize=8)
    ax.set_yticks(range(len(ALL_DESCRIPTORS)), ALL_DESCRIPTORS, fontsize=9)
    for k in range(1, len(GENOME_TYPES)):
        ax.axvline(k * len(strides) - 0.5, color="#222222", linewidth=1.5)
    y = -0.5
    for fam, ds in FAMILIES.items():
        y += len(ds)
        if y < len(ALL_DESCRIPTORS) - 0.5:
            ax.axhline(y, color="#222222", linewidth=0.8)
    fig.colorbar(im, ax=ax, label="Spearman ρ with performance (−fitness)\n> 0: higher value goes with better fitness")
    fig.suptitle("Which behaviours go with fitness? All non-glitch bodies across generations and reps")
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, strides: list[int], out_path: Path) -> None:
    ok = df[~df.glitch]
    fig, axes = plt.subplots(len(SCATTER_DESCRIPTORS), len(GENOME_TYPES),
                             figsize=(14, 3.0 * len(SCATTER_DESCRIPTORS)), sharex="row", sharey=True, squeeze=False)
    for c, genome in enumerate(GENOME_TYPES):
        for r, desc in enumerate(SCATTER_DESCRIPTORS):
            ax = axes[r, c]
            for stride in strides:
                sub = ok[(ok.genome == genome) & (ok.stride == stride)]
                ax.scatter(sub[f"d_{desc}"], sub["stored_fitness"], s=5, alpha=0.25,
                           color=STRIDE_COLOR[stride], marker=STRIDE_MARKER[stride],
                           label=stride_label(stride), rasterized=True)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(GENOME_LABEL[genome])
            ax.set_xlabel(desc, fontsize=9)
            if c == 0:
                ax.set_ylabel("fitness (lower is better)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for h in handles:
        h.set_alpha(1)
        h.set_sizes([30])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Key descriptors against fitness (every non-glitch body, all generations and reps)")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def print_table(df: pd.DataFrame, strides: list[int]) -> None:
    cols = ["has_pair", "d_mirror_pair_frac", "d_mirror_pair_corr_mean",
            "d_mirror_pair_phase_lag", "d_mirror_pair_amp_asym"]
    print(f"\nFinal-generation population means (mean across reps)")
    print(f"{'genome':<16}{'stride':>7}" + "".join(f"{c.removeprefix('d_mirror_pair_'):>12}" for c in cols))
    for genome in GENOME_TYPES:
        for stride in strides:
            sub = df[(df.genome == genome) & (df.stride == stride)]
            if sub.empty:
                continue
            final = sub[sub.gen == sub.gen.max()]
            vals = final.groupby("rep")[cols].mean().mean()
            print(f"{genome:<16}{stride:>7}" + "".join(f"{v:>12.3f}" for v in vals))


def main() -> None:
    df = load()
    strides = sorted(df.stride.unique())
    print("Runs loaded per (genome, stride):")
    print(df.groupby(["genome", "stride"]).rep.nunique().unstack("stride").to_string())
    print(f"{len(df)} bodies, {int(df.glitch.sum())} glitch episodes (excluded from fitness plots)")

    out_dir = args.stride_root / "analysis" / "behavior"
    out_dir.mkdir(parents=True, exist_ok=True)

    for family in FAMILIES:
        p = out_dir / f"gens_{family}.png"
        plot_family(df, family, strides, p)
        print(f"Saved -> {p}")
    for fn, name in [(plot_symmetry_by_stride, "symmetry_by_stride.png"),
                     (plot_symmetry_distributions, "symmetry_distributions.png"),
                     (plot_fitness_corr, "descriptor_fitness_corr.png"),
                     (plot_scatter, "descriptors_vs_fitness.png")]:
        p = out_dir / name
        fn(df, strides, p)
        print(f"Saved -> {p}")

    print_table(df, strides)


if __name__ == "__main__":
    main()
