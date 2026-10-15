"""
Penalty vs no-penalty comparison for the rowing-penalty experiment
(slurm/run_rowing_penalty.sh): forward task, stride 25 (20Hz), ANN brain, zoo
modules, tree vs tree_symmetric x {penalty, nopenalty} x 3 reps.

Inputs, per arm (<root>/<arm>/):
  ctrlstride_*/.../run_data.jsonl   fitness and raw_score (unpenalised forward displacement)
  rowing/individuals.csv            rowing_detect.py labels, from behavior_extract.py's 50Hz
                                    traces (--record-every-n 10 --out-dir <arm>/behavior_50hz/<run>)
The detector is re-run on both arms, so the no-penalty runs get rowing labels too.

"Penalised" displacement is what selection sees under the penalty:
-fitness = rowing_scaled_disp(raw, rowing_frac) - height/jerk penalties. For
the no-penalty arm it is the counterfactual, from the detector's rowing_frac.

Writes to <root>/analysis/:
  fitness_curves.png     rows = best raw displacement, mean raw displacement, best penalised
                         displacement per generation; cols = genome; colour = arm
                         (mean ± 1 SEM across reps, thin = each rep).
  fitness_by_genome_nopenalty.png
                         no-penalty arm only: best and mean fitness (-fitness, higher is better) per
                         generation, tree vs tree_symmetric.
  rowing_share.png       the same layout for rowing share (all / judgeable) and alternating share.
  label_composition.png  rows = arm, cols = genome: stacked gait classes per generation, reps pooled.
  final_tradeoff.png     final generation, one dot per individual: raw displacement vs rowing_frac.
  final_summary.png      per rep: best raw displacement and rowing share over the last
                         EARLY_LATE_GENS generations.
  individuals_merged.csv run_data joined with the detector output, both arms.
  summary.csv            the numbers behind final_summary.png (printed too).

Usage:
    python plot_rowing_penalty.py [--root __data__/ariel_rowing_penalty_ann]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_rowing_stride_sweep import (  # noqa: E402
    CI_ALPHA, CLASS_COLOR, CLASSES, EARLY_LATE_GENS, GENOME_COLOR, GENOME_LABEL, GENOMES, INK, INK_2, MUTED, gen_curve,
    shares_by_gen, style,
)
from shared import rowing_scaled_disp  # noqa: E402
from sweep_common import run_data_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK = "forward"
ARMS = ("nopenalty", "penalty")
ARM_LABEL = {"nopenalty": "no penalty (baseline)", "penalty": "rowing penalty"}
ARM_COLOR = {"nopenalty": "#898781", "penalty": "#4a3aa7"}


# ── Loading ───────────────────────────────────────────────────────────────────


def load_run_data(run_dir: Path) -> pd.DataFrame:
    rows = []
    for line in run_data_path(run_dir, TASK).open():
        rec = json.loads(line)
        sk = rec["per_skill"]["fwd"]
        rows.append({"gen": rec["gen"], "ind_id": rec["ind_id"], "fitness": rec["fitness"],
                     "raw": sk.get("raw_score", math.nan), "stored_rowing_frac": sk.get("rowing_frac", math.nan),
                     "stored_rowing_label": sk.get("rowing_label")})
    return pd.DataFrame(rows)


def load_arm(root: Path, arm: str) -> pd.DataFrame:
    ind = pd.read_csv(root / arm / "rowing" / "individuals.csv")
    parts = []
    for run, df in ind.groupby("run"):
        rd = load_run_data(root / arm / run)
        assert not rd.duplicated(["gen", "ind_id"]).any(), f"{run}: (gen, ind_id) not unique in run_data"
        m = df.merge(rd, on=["gen", "ind_id"], how="left", validate="one_to_one")
        missing = m["raw"].isna().sum()
        if missing:
            print(f"[warn] {run}: {missing} episodes without a run_data record")
        parts.append(m)
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "arm", arm)
    return df


def add_penalised(df: pd.DataFrame) -> pd.DataFrame:
    """penalised = what selection sees with the penalty on: -fitness for the penalty arm, and for
    the no-penalty arm -fitness with the raw displacement swapped for the scaled one."""
    scaled = [rowing_scaled_disp(r, f) for r, f in zip(df["raw"], df["rowing_frac"].fillna(0.0))]
    df["score"] = -df["fitness"]  # higher is better
    df["penalised"] = np.where(df["arm"] == "penalty", -df["fitness"], -df["fitness"] - df["raw"] + scaled)
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────


def per_gen(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["arm", "genome", "rep", "gen"])
    return g.agg(best_raw=("raw", "max"), mean_raw=("raw", "mean"), best_penalised=("penalised", "max"),
                 best_fitness=("score", "max"), mean_fitness=("score", "mean")).reset_index()


def plot_genome_fitness(pg: pd.DataFrame, arm: str, out: Path, dpi: int) -> None:
    """One arm: best and mean fitness per generation, one line per genome."""
    panels = [("best_fitness", "best fitness"), ("mean_fitness", "mean fitness")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        style(ax)
        for genome in GENOMES:
            df = pg[(pg["arm"] == arm) & (pg["genome"] == genome)]
            gens, m, sem, per_rep = gen_curve(df, col)
            color = GENOME_COLOR[genome]
            for rep in per_rep.columns:
                ax.plot(per_rep.index, per_rep[rep], color=color, linewidth=0.8, alpha=0.35)
            ax.fill_between(gens, m - sem, m + sem, color=color, alpha=CI_ALPHA, linewidth=0)
            ax.plot(gens, m, color=color, linewidth=2, label=f"{GENOME_LABEL[genome]} ({per_rep.shape[1]} reps)")
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("generation", fontsize=9, color=INK_2)
    axes[0].set_ylabel("fitness = forward displacement\n- height/jerk penalties (m)", fontsize=9, color=INK_2)
    axes[0].legend(fontsize=8.5, frameon=False, labelcolor=INK_2, loc="upper left")
    fig.suptitle(f"{ARM_LABEL[arm]}, stride 25: tree vs symmetric "
                 "(thick = mean ± 1 SEM across reps, thin = each rep)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def arm_panels(table: pd.DataFrame, rows: list[tuple[str, str]], title: str, out: Path, dpi: int,
               percent: bool = False) -> None:
    """rows x genome grid, one line per arm: mean ± 1 SEM across reps, thin line per rep."""
    fig, axes = plt.subplots(len(rows), len(GENOMES), figsize=(5.0 * len(GENOMES), 3.0 * len(rows)),
                             sharex=True, sharey="row", squeeze=False)
    for r, (col, ylabel) in enumerate(rows):
        for c, genome in enumerate(GENOMES):
            ax = axes[r, c]
            style(ax)
            for arm in ARMS:
                df = table[(table["arm"] == arm) & (table["genome"] == genome)]
                if df.empty:
                    continue
                gens, m, sem, per_rep = gen_curve(df, col)
                color = ARM_COLOR[arm]
                for rep in per_rep.columns:
                    ax.plot(per_rep.index, per_rep[rep], color=color, linewidth=0.8, alpha=0.35)
                ax.fill_between(gens, m - sem, m + sem, color=color, alpha=CI_ALPHA, linewidth=0)
                ax.plot(gens, m, color=color, linewidth=2, label=f"{ARM_LABEL[arm]} ({per_rep.shape[1]} reps)")
            if percent:
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            if r == 0:
                ax.set_title(GENOME_LABEL[genome], fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
            if r == len(rows) - 1:
                ax.set_xlabel("generation", fontsize=9, color=INK_2)
    axes[0, 0].legend(fontsize=8.5, frameon=False, labelcolor=INK_2, loc="upper left")
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_composition(sh: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(len(ARMS), len(GENOMES), figsize=(5.0 * len(GENOMES), 3.2 * len(ARMS)),
                             sharex=True, sharey=True, squeeze=False)
    cols = [f"class_{c}" for c in CLASSES]
    for r, arm in enumerate(ARMS):
        for c, genome in enumerate(GENOMES):
            ax = axes[r, c]
            style(ax)
            ax.grid(False)
            df = sh[(sh["arm"] == arm) & (sh["genome"] == genome)]
            if df.empty:
                continue
            w = df[cols].mul(df["n"], axis=0).assign(gen=df["gen"], n=df["n"]).groupby("gen").sum()
            frac = w[cols].div(w["n"], axis=0)
            ax.stackplot(frac.index, frac.T.to_numpy(), colors=[CLASS_COLOR[k] for k in CLASSES],
                         edgecolor="white", linewidth=1.0)
            last, bottom = frac.iloc[-1], 0.0
            for cls, col in zip(CLASSES, cols):
                h = last[col]
                if h >= 0.10:
                    ax.text(frac.index[-1] - 0.5, bottom + h / 2, f"{cls} {h:.0%}", ha="right", va="center",
                            fontsize=7.5, color="white" if cls in ("rowing", "alternating") else INK)
                bottom += h
            ax.set_xlim(frac.index[0], frac.index[-1])
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            if r == 0:
                ax.set_title(GENOME_LABEL[genome], fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"{ARM_LABEL[arm]}\nshare of population", fontsize=9, color=INK_2)
            if r == len(ARMS) - 1:
                ax.set_xlabel("generation", fontsize=9, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[k]) for k in CLASSES]
    labels = ["rowing", "alternating", "other gait (mixed / aperiodic)", "can't judge (unpaired / static)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Gait classes over evolution, stride 25 (3 reps pooled; labels: share in the final generation)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_tradeoff(df: pd.DataFrame, out: Path, dpi: int) -> None:
    last = df[df["gen"] == df.groupby("run")["gen"].transform("max")]
    fig, axes = plt.subplots(1, len(GENOMES), figsize=(5.0 * len(GENOMES), 3.8), sharex=True, sharey=True)
    for ax, genome in zip(axes, GENOMES):
        style(ax)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
        for arm in ARMS:
            d = last[(last["arm"] == arm) & (last["genome"] == genome)]
            ax.scatter(d["rowing_frac"], d["raw"], s=22, color=ARM_COLOR[arm], alpha=0.7, linewidths=0,
                       label=f"{ARM_LABEL[arm]} (n={len(d)})")
        ax.set_title(GENOME_LABEL[genome], fontsize=10, color=INK)
        ax.set_xlabel("rowing_frac (share of the stroke that rows)", fontsize=9, color=INK_2)
        ax.set_xlim(-0.03, 1.03)
    axes[0].set_ylabel("raw forward displacement (m)", fontsize=9, color=INK_2)
    axes[0].legend(fontsize=8.5, frameon=False, labelcolor=INK_2, loc="upper left")
    fig.suptitle("Final generation: displacement vs rowing, every individual of all 3 reps", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def summary(df: pd.DataFrame, sh: pd.DataFrame) -> pd.DataFrame:
    g1 = df["gen"].max()
    rows = []
    for (arm, genome, rep), d in df.groupby(["arm", "genome", "rep"]):
        last = d[d["gen"] == d["gen"].max()]
        s = sh[(sh["arm"] == arm) & (sh["genome"] == genome) & (sh["rep"] == rep) & (sh["gen"] > g1 - EARLY_LATE_GENS)]
        rows.append({"arm": arm, "genome": genome, "rep": rep,
                     "final_best_raw": last["raw"].max(), "final_mean_raw": last["raw"].mean(),
                     "final_best_penalised": last["penalised"].max(),
                     "late_rowing": s["rowing"].sum() / s["n"].sum(),
                     "late_alternating": s["alternating"].sum() / s["n"].sum(),
                     "late_rowing_judgeable": s["rowing"].sum() / max(s["n_judgeable"].sum(), 1)})
    return pd.DataFrame(rows)


def plot_summary(sm: pd.DataFrame, out: Path, dpi: int) -> None:
    panels = [("final_best_raw", "best raw displacement,\nfinal generation (m)", False),
              ("late_rowing", f"rowing share,\nlast {EARLY_LATE_GENS} generations", True)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6))
    groups = [(g, a) for g in GENOMES for a in ARMS]
    for ax, (col, ylabel, pct) in zip(axes, panels):
        style(ax)
        for x, (genome, arm) in enumerate(groups):
            v = sm[(sm["genome"] == genome) & (sm["arm"] == arm)][col].to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(v)) if len(v) > 1 else np.zeros(len(v))
            ax.scatter(x + jitter, v, s=30, color=ARM_COLOR[arm], alpha=0.75, linewidths=0, zorder=3)
            ax.hlines(v.mean(), x - 0.25, x + 0.25, color=ARM_COLOR[arm], linewidth=2.2, zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{'symmetric' if g == 'tree_symmetric' else 'tree'}\n"
                            f"{'penalty' if a == 'penalty' else 'no penalty'}" for g, a in groups],
                           fontsize=8, color=INK_2)
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
        if pct:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.axvline(1.5, color=MUTED, linewidth=0.6, alpha=0.5)
    fig.suptitle("Per rep (dots) and mean (bar)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "__data__" / "ariel_rowing_penalty_ann")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = add_penalised(pd.concat([load_arm(args.root, a) for a in ARMS], ignore_index=True))
    df.to_csv(out / "individuals_merged.csv", index=False)

    # Replay check: the detector on the replayed traces should agree with what the penalty saw in-run.
    p = df[df["arm"] == "penalty"]
    print(f"Penalty arm, detector rerun vs in-run: max|rowing_frac delta|="
          f"{(p['rowing_frac'] - p['stored_rowing_frac']).abs().max():.2e}, "
          f"label agreement={(p['rowing_label'] == p['stored_rowing_label']).mean():.3f}")

    pg = per_gen(df)
    arm_panels(pg, [("best_raw", "best raw forward\ndisplacement (m)"),
                    ("mean_raw", "mean raw forward\ndisplacement (m)"),
                    ("best_penalised", "best penalised\ndisplacement (m)")],
               "Forward displacement per generation, stride 25 "
               "(thick = mean ± 1 SEM across reps, thin = each rep)", out / "fitness_curves.png", args.dpi)
    plot_genome_fitness(pg, "nopenalty", out / "fitness_by_genome_nopenalty.png", args.dpi)

    sh = pd.concat([shares_by_gen(df[df["arm"] == a]).assign(arm=a) for a in ARMS], ignore_index=True)
    sh.to_csv(out / "rowing_share_by_gen.csv", index=False)
    arm_panels(sh, [("rowing_share", "rowing\n(share of all individuals)"),
                    ("rowing_share_judgeable", "rowing\n(share of judgeable individuals)"),
                    ("alternating_share", "alternating\n(share of all individuals)")],
               "Gait shares per generation, stride 25 (thick = mean ± 1 SEM across reps, thin = each rep)",
               out / "rowing_share.png", args.dpi, percent=True)
    plot_composition(sh, out / "label_composition.png", args.dpi)
    plot_tradeoff(df, out / "final_tradeoff.png", args.dpi)

    sm = summary(df, sh)
    sm.to_csv(out / "summary.csv", index=False)
    plot_summary(sm, out / "final_summary.png", args.dpi)
    pd.set_option("display.width", 200)
    print("\nPer rep:")
    print(sm.to_string(index=False, float_format="%.2f"))
    print("\nMean over reps:")
    print(sm.groupby(["genome", "arm"]).mean(numeric_only=True).drop(columns="rep").to_string(float_format="%.2f"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
