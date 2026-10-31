"""
Phase-shift replay (phase_shift.py): does breaking the left-right lock remove
the symmetric advantage?

Checkpoints are grouped by genome and, for symmetric bodies, by whether the
unperturbed episode is phase-locked (locked pairs >= LOCKED_MIN of the stroke,
as in analyse_e1). Per checkpoint and condition: the score change from the
baseline (random-subset conditions averaged over their draws). The test is
side - random, the score after a side perturbation minus the score after
perturbing the same number of lateral hinges at random (both break k (n - k)
hinge-pair timings; the side one breaks exactly the left-right ones).
Negative: the left-right timing matters more than the others. If the lock
causes the gap, it is negative for locked symmetric bodies and near zero for
tree bodies, and perturbed locked symmetric bodies fall to the tree baseline.

Writes to --out (default <root>/analysis):
  phase_shift_ckpts.csv     one row per checkpoint: group, baseline, score per condition
  phase_shift_summary.csv   per brain x group: mean score per condition, side - random
                            (mean, bootstrap 95% CI, Wilcoxon p, n)
  fig8_phase_shift.png      score change vs perturbation size, side (solid) vs random
                            (dashed); one column per brain, a row for shift and detune

Usage:
    python analyse_phase_shift.py --roots /scratch/$USER/ariel_paper/e3/phase_shift_e2_random [...]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import runs  # noqa: E402
from phase_shift import DETUNES, SHIFTS  # noqa: E402

LOCKED_MIN = 0.5
KINDS = [("shift", SHIFTS, "left-side phase shift (cycles)"), ("detune", DETUNES, "left-side detuning (frequency x 1 + D)")]
SIDE_VS_RAND = [(f"side_{k}_{m}", f"rand_{k}_{m}") for k, mags, _ in KINDS for m in mags]
PERTURBED = [c for pair in SIDE_VS_RAND for c in pair]
GROUPS = ["tree", "tree_tied", "symmetric unlocked", "symmetric locked"]
GROUP_STYLE = {"tree": (runs.GENOME_COLOR["tree"], 1.0), "tree_tied": (runs.GENOME_COLOR["tree_tied"], 1.0),
               "symmetric unlocked": (runs.GENOME_COLOR["tree_symmetric"], 0.4),
               "symmetric locked": (runs.GENOME_COLOR["tree_symmetric"], 1.0)}


def load(roots: list[Path]) -> pd.DataFrame:
    """One row per checkpoint: identity, group, baseline and per-condition score."""
    eps = pd.concat([pd.read_csv(p) for r in roots for p in sorted(r.glob("*/phase_shift.csv"))], ignore_index=True)
    eps = eps[~eps["skipped"].astype(bool)]
    key = ["run", "ckpt"]
    base = eps[eps["condition"] == "baseline"].set_index(key)
    wide = eps.pivot_table(index=key, columns="condition", values="score", aggfunc="mean")
    out = base[["brain", "genome", "body_source", "hinges_left", "hinges_midline", "phase_locked_frac"]].join(wide)
    locked = out["phase_locked_frac"] >= LOCKED_MIN
    out["group"] = np.where(out["genome"] != "tree_symmetric", out["genome"],
                            np.where(locked, "symmetric locked", "symmetric unlocked"))
    return out.reset_index()


def mean_ci_p(d: np.ndarray, rng: np.random.Generator) -> dict:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"mean": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "p": math.nan, "n": 0}
    boot = rng.choice(d, (runs.N_BOOT, len(d))).mean(axis=1)
    p = wilcoxon(d).pvalue if len(d) >= 2 and np.any(d != 0) else math.nan
    return {"mean": float(d.mean()), "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)), "p": float(p), "n": len(d)}


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for (brain, group), g in df.groupby(["brain", "group"]):
        row = {"brain": brain, "group": group, "n": len(g), "baseline": g["baseline"].mean()}
        row |= {c: g[c].mean() for c in PERTURBED if c in g}
        for side, rand in SIDE_VS_RAND:
            if side in g and rand in g:
                s = mean_ci_p((g[side] - g[rand]).to_numpy(), rng)
                row |= {f"{side}-rand_{k}": v for k, v in s.items()}
        rows.append(row)
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame, path: Path) -> None:
    brains = [b for b in runs.BRAIN_LABEL if b in set(df["brain"])]
    groups = [g for g in GROUPS if g in set(df["group"])]
    fig, axes = plt.subplots(len(KINDS), len(brains), figsize=(4.4 * len(brains), 3.4 * len(KINDS)),
                             squeeze=False, sharey=True)
    for row, (kind, mags, xlabel) in enumerate(KINDS):
        x = np.arange(len(mags))
        for ax, brain in zip(axes[row], brains):
            d = df[df["brain"] == brain]
            for group in groups:
                g = d[d["group"] == group]
                if g.empty:
                    continue
                color, alpha = GROUP_STYLE[group]
                for subset, ls in (("side", "-"), ("rand", "--")):
                    change = [(g[f"{subset}_{kind}_{m}"] - g["baseline"]) for m in mags]
                    ax.errorbar(x, [c.mean() for c in change], yerr=[c.sem() for c in change], color=color,
                                alpha=alpha, linestyle=ls, linewidth=2, marker="o", markersize=4, capsize=2,
                                label=f"{group} (n={len(g)})" if subset == "side" else None)
            ax.axhline(0, color=runs.INK_2, linewidth=0.8)
            ax.set_xticks(x, [str(m) for m in mags])
            ax.set_xlabel(xlabel, fontsize=9, color=runs.INK_2)
            if row == 0:
                ax.set_title(runs.BRAIN_LABEL[brain], fontsize=10, color=runs.INK)
            runs.style(ax)
        axes[row][0].set_ylabel("score change (± SEM)", fontsize=9, color=runs.INK_2)
    axes[0][0].legend(fontsize=7, frameon=False, labelcolor=runs.INK_2, title="solid: side, dashed: random",
                      title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=Path, nargs="+", required=True, help="phase_shift.py output roots")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.roots[0] / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.roots)
    df.to_csv(out / "phase_shift_ckpts.csv", index=False)
    s = summary(df)
    s.to_csv(out / "phase_shift_summary.csv", index=False)
    with pd.option_context("display.width", 250):
        print("mean score per condition:")
        print(s[["brain", "group", "n", "baseline", *[c for c in PERTURBED if c in s]]].round(2).to_string(index=False))
        print("\nside - random per checkpoint: mean [95% CI] p")
        rows = []
        for _, r in s.iterrows():
            row = {"brain": r["brain"], "group": r["group"]}
            for side, _ in SIDE_VS_RAND:
                k = f"{side}-rand"
                if f"{k}_mean" in r:
                    row[side.removeprefix("side_")] = (f"{r[f'{k}_mean']:+.2f} [{r[f'{k}_ci_lo']:+.2f}, {r[f'{k}_ci_hi']:+.2f}]"
                                                       f" p={r[f'{k}_p']:.2g}")
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
    figure(df, out / "fig8_phase_shift.png")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
