"""
E1: evolution at the default hinge by genome and controller (R1, R6).

{bang_bang, bang_bang_drift, bang_bang_free} x {tree, tree_symmetric,
tree_tied} x 20 seeds (slurm e1_coordination_ablation.sh, v2), plus DANN at
200 evaluations per body (pass the budget series' sb10 root as a second
--root). bang_bang_drift is the control for bang_bang_free's harder search.
Paired by seed:
  symmetric - tree   the symmetric advantage, with and without a shared frequency
  tied - tree        paired limbs without mirroring
  symmetric - tied   mirroring itself
Checks for the ablation: the spread of evolved hinge frequencies within each
bang_bang_free champion (if they converge, the ablation did not remove
coordination), and the absolute scores per arm (extra parameters can hurt
CMA-ES on every genome). With --behaviour (replay.py output for these runs,
--gens last), the gait class and phase locking of every run's champion.

Fitness curves: the population's best score after each outer generation
(plus-selection keeps it, so it never falls; v2: retest means).

Writes to --out (default <first root>/analysis):
  e1_runs.csv, e1_gaps.csv, e1_champions.csv, e1_curves.csv, e1_gap_by_gen.csv
  fig7_ablation.png             paired gaps per brain with 95% CI, share of
                                phase-locked champions per genome beside each bar
  fig_e1_curves_genome.png      Plot 1: tree / symmetric / tied under bang_bang
                                (mean +- SEM over seeds), and the paired gaps
                                to tree per generation (95% CI band)
  fig_e1_curves_controller.png  Plot 2: tree (dashed) and symmetric (solid)
                                per controller, and sym - tree per generation

Usage:
    python analyse_e1.py --root $FINAL_ROOT/e1 $FINAL_ROOT/e0_budget/sb10 [--behaviour $FINAL_ROOT/e3/e1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runs  # noqa: E402
from brains import BangBangDriftBrain, BangBangFreeBrain  # noqa: E402

COMPARISONS = [("tree_symmetric", "tree"), ("tree_tied", "tree"), ("tree_symmetric", "tree_tied")]
FREQ_BRAINS = ("bang_bang_free", "bang_bang_drift")  # brains with per-hinge frequencies
LOCKED_MIN = 0.5  # a champion is phase-locked if locked pairs carry at least this share of its stroke


def frequency_spread(run_dir: Path, brain_kind: str) -> float:
    """Coefficient of variation of the hinges' mean evolved frequency in a
    bang_bang_free or bang_bang_drift champion (0 = all hinges share one frequency)."""
    ckpt = runs.best_checkpoint(run_dir)
    w = np.load(ckpt / "fwd_weights.npy")
    brain = BangBangFreeBrain(len(w) // 9) if brain_kind == "bang_bang_free" else BangBangDriftBrain((len(w) - 4) // 9)
    brain.set_params(w)
    f = (np.asarray(brain.freq.start) + np.asarray(brain.freq.end)) / 2
    return float(f.std() / f.mean())


def champions(behaviour: Path) -> pd.DataFrame:
    """The best (lowest stored fitness) last-generation episode of every replayed run."""
    eps = pd.concat([pd.read_csv(p) for p in sorted(behaviour.glob("*/episodes.csv"))], ignore_index=True)
    best = eps.loc[eps.groupby("run")["stored_fitness"].idxmin()].copy()
    best["phase_locked"] = best["phase_locked_frac"] >= LOCKED_MIN
    return best


def load_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Long form: brain, genome, seed, gen, best_so_far."""
    frames = [runs.curve_by_gen(Path(r.path)).assign(brain=r.brain, genome=r.genome, seed=r.seed)
              for r in df.itertuples()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def gap_by_gen(cur: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> pd.DataFrame:
    """Paired a - b of best_so_far per brain and generation: mean, bootstrap 95% CI, n."""
    rows = []
    for (brain, gen), g in cur.groupby(["brain", "gen"]):
        wide = g.pivot_table(index="seed", columns="genome", values="best_so_far")
        if a not in wide or b not in wide:
            continue
        d = (wide[a] - wide[b]).dropna().to_numpy()
        if not len(d):
            continue
        boot = rng.choice(d, (2000, len(d))).mean(axis=1)
        rows.append({"brain": brain, "gen": gen, "comparison": f"{a} - {b}", "gap": d.mean(),
                     "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5), "n": len(d),
                     "mean_a": wide[a].mean(), "mean_b": wide[b].mean()})
    return pd.DataFrame(rows)


def _mean_sem(ax, g: pd.DataFrame, color: str, ls: str, label: str) -> None:  # noqa: ANN001
    m = g.groupby("gen")["best_so_far"].agg(["mean", "sem", "count"])
    ax.plot(m.index, m["mean"], color=color, linestyle=ls, linewidth=2, label=f"{label} (n={int(m['count'].max())})")
    ax.fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"], color=color, alpha=0.15, linewidth=0)


def curves(df: pd.DataFrame, out: Path) -> None:
    cur = load_curves(df)
    if cur.empty:
        return
    cur.to_csv(out / "e1_curves.csv", index=False)
    rng = np.random.default_rng(0)
    gaps = pd.concat([gap_by_gen(cur, a, b, rng) for a, b in COMPARISONS], ignore_index=True)
    gaps.to_csv(out / "e1_gap_by_gen.csv", index=False)
    unit = runs.unit_suffix(df)
    last = gaps.loc[gaps.groupby(["brain", "comparison"])["gen"].idxmax()]
    first = gaps.loc[gaps.groupby(["brain", "comparison"])["gen"].idxmin()]
    print("\ngap per generation, first vs last (relative lead = gap / mean of the second genome):")
    for (f, l) in zip(first.itertuples(), last.itertuples()):
        print(f"  {f.brain:16s} {f.comparison:28s} gen {f.gen}: {f.gap:+.3f} ({f.gap / f.mean_b:+.0%})   "
              f"gen {l.gen}: {l.gap:+.3f} ({l.gap / l.mean_b:+.0%})")

    # Plot 1: genomes under bang_bang.
    brain = "bang_bang" if "bang_bang" in set(cur["brain"]) else cur["brain"].iloc[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for ax in (a, b):
        runs.style(ax)
    for genome in ("tree", "tree_symmetric", "tree_tied"):
        g = cur[(cur["brain"] == brain) & (cur["genome"] == genome)]
        if len(g):
            _mean_sem(a, g, runs.GENOME_COLOR[genome], "-", runs.GENOME_LABEL[genome])
    a.set_xlabel("generation", fontsize=9, color=runs.INK_2)
    a.set_ylabel(f"best score{unit}", fontsize=9, color=runs.INK_2)
    a.set_title(f"(a) {runs.BRAIN_LABEL[brain]}: best score by genome", fontsize=9, color=runs.INK, loc="left")
    a.legend(frameon=False, fontsize=7)
    b.axhline(0, color=runs.INK_2, linewidth=0.8)
    for comp, genome in (("tree_symmetric - tree", "tree_symmetric"), ("tree_tied - tree", "tree_tied")):
        g = gaps[(gaps["brain"] == brain) & (gaps["comparison"] == comp)]
        b.plot(g["gen"], g["gap"], color=runs.GENOME_COLOR[genome], linewidth=2,
               label=f"{runs.GENOME_LABEL[genome]} − tree")
        b.fill_between(g["gen"], g["ci_lo"], g["ci_hi"], color=runs.GENOME_COLOR[genome], alpha=0.15, linewidth=0)
    b.set_xlabel("generation", fontsize=9, color=runs.INK_2)
    b.set_ylabel(f"paired gap to tree{unit}", fontsize=9, color=runs.INK_2)
    b.set_title("(b) paired gap to tree (95% CI over seeds)", fontsize=9, color=runs.INK, loc="left")
    b.legend(frameon=False, fontsize=7)
    fig.savefig(out / "fig_e1_curves_genome.png", dpi=200)
    plt.close(fig)

    # Plot 2: controllers, tree vs symmetric.
    brains = [x for x in ("bang_bang", "bang_bang_drift", "bang_bang_free", "dann") if x in set(cur["brain"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for ax in (a, b):
        runs.style(ax)
    for x in brains:
        for genome, ls in (("tree", "--"), ("tree_symmetric", "-")):
            g = cur[(cur["brain"] == x) & (cur["genome"] == genome)]
            if len(g):
                _mean_sem(a, g, runs.BRAIN_COLOR[x], ls, f"{runs.BRAIN_SHORT[x]}, {runs.GENOME_LABEL[genome]}")
        g = gaps[(gaps["brain"] == x) & (gaps["comparison"] == "tree_symmetric - tree")]
        b.plot(g["gen"], g["gap"], color=runs.BRAIN_COLOR[x], linewidth=2, label=runs.BRAIN_SHORT[x])
        b.fill_between(g["gen"], g["ci_lo"], g["ci_hi"], color=runs.BRAIN_COLOR[x], alpha=0.12, linewidth=0)
    b.axhline(0, color=runs.INK_2, linewidth=0.8)
    a.set_xlabel("generation", fontsize=9, color=runs.INK_2)
    a.set_ylabel(f"best score{unit}", fontsize=9, color=runs.INK_2)
    a.set_title("(a) best score: symmetric (solid), tree (dashed)", fontsize=9, color=runs.INK, loc="left")
    a.legend(frameon=False, fontsize=6.5, ncol=2)
    b.set_xlabel("generation", fontsize=9, color=runs.INK_2)
    b.set_ylabel(f"symmetric − tree{unit}", fontsize=9, color=runs.INK_2)
    b.set_title("(b) paired symmetric − tree (95% CI)", fontsize=9, color=runs.INK, loc="left")
    b.legend(frameon=False, fontsize=7)
    fig.savefig(out / "fig_e1_curves_controller.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", "--roots", type=Path, nargs="+", required=True,
                        help="E1 root(s); add the budget series' sb10 root for the DANN arm")
    parser.add_argument("--behaviour", type=Path, help="replay.py output root for the E1 runs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.root[0] / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = runs.load_runs(*args.root)
    df = df[df["hinge"] == runs.ZOO_HINGE.label]
    df = df[(df["brain"] != "dann") | (df["evals_per_body"] == 200)]
    df["freq_cv"] = [frequency_spread(Path(p), b) if b in FREQ_BRAINS else 0.0 for p, b in zip(df["path"], df["brain"])]
    df.to_csv(out / "e1_runs.csv", index=False)
    print("mean score per arm (and frequency spread of free / drift champions):")
    print(df.groupby(["brain", "genome"])[["score", "freq_cv"]].agg(["mean", "count"]).round(3).to_string())

    gaps = pd.concat([runs.gap_table(df, ["brain"], a=a, b=b).assign(comparison=f"{a} - {b}")
                      .rename(columns={f"mean_{a}": "mean_a", f"mean_{b}": "mean_b"}) for a, b in COMPARISONS])
    gaps.to_csv(out / "e1_gaps.csv", index=False)
    print("\npaired gaps:")
    print(gaps[["brain", "comparison", "gap", "ci_lo", "ci_hi", "p", "wins", "n"]].round(3).to_string(index=False))

    locked = {}
    if args.behaviour:
        champ = champions(args.behaviour)
        champ.to_csv(out / "e1_champions.csv", index=False)
        tab = champ.groupby(["brain", "genome"]).agg(
            n=("run", "size"), phase_locked=("phase_locked", "mean"), phase_quality=("phase_quality", "mean"),
            alternating=("rowing_label", lambda s: (s == "alternating").mean()), score=("score", "mean"))
        print(f"\nchampions (phase-locked: locked pairs >= {LOCKED_MIN:.0%} of the stroke):")
        print(tab.round(2).to_string())
        locked = tab["phase_locked"].to_dict()

    curves(df, out)
    brains = [b for b in ("bang_bang", "bang_bang_free", "bang_bang_drift") if b in set(df["brain"])]
    fig, axes = plt.subplots(1, len(brains), figsize=(4.4 * len(brains), 3.4), sharey=True, squeeze=False,
                             constrained_layout=True)
    for ax, brain in zip(axes[0], brains):
        runs.style(ax)
        ax.axhline(0, color=runs.INK_2, linewidth=0.8)
        g = gaps[gaps["brain"] == brain]
        x = np.arange(len(g))
        colors = [runs.GENOME_COLOR[c.split(" - ")[0]] for c in g["comparison"]]
        ax.bar(x, g["gap"], width=0.6, color=colors, edgecolor="white", linewidth=2)
        ax.errorbar(x, g["gap"], yerr=[g["gap"] - g["ci_lo"], g["ci_hi"] - g["gap"]], fmt="none",
                    ecolor=runs.INK, capsize=3, linewidth=1)
        labels = [c.replace("tree_symmetric", "sym").replace("tree_tied", "tied") for c in g["comparison"]]
        ax.set_xticks(x, labels, fontsize=8)
        for xi, (comp, p) in enumerate(zip(g["comparison"], g["p"])):
            a, b = comp.split(" - ")
            note = f"p={p:.2g}"
            if locked:
                note += f"\nlocked {locked.get((brain, a), np.nan):.0%} vs {locked.get((brain, b), np.nan):.0%}"
            ax.annotate(note, (xi, g["ci_hi"].iloc[xi]), textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=7, color=runs.INK_2)
        ax.set_title(runs.BRAIN_LABEL[brain], fontsize=10, color=runs.INK)
    axes[0, 0].set_ylabel("paired score gap (95% CI)", fontsize=9, color=runs.INK_2)
    fig.savefig(out / "fig7_ablation.png", dpi=200)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
