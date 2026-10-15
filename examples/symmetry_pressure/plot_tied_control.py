"""
Dimensionality control for symmetry enforcement (slurm/run_tied_control.sh):
tree vs tree_symmetric vs tree_tied, forward task, stride 25, ANN brain, zoo
modules. tree_tied ties ~64% of modules to randomly placed twins
(ariel.ec.genotypes.tree.tying), the share tree_symmetric mirrors, with no
geometric symmetry:
  tree_tied ~ tree_symmetric  -> the advantage is dimensionality
  tree_tied ~ tree            -> symmetry itself matters.

Reads every ctrlstride_forward_<genome>_s25_rep* run under --roots (by default
the rowing-penalty no-penalty arm, which holds reps 0-2 of tree and
tree_symmetric, and the tied-control runs), one row per evaluated individual
from its checkpoint (meta.json fitness + best_genome.json body).

Writes to --out (default __data__/ariel_tied_control_ann/analysis/):
  fitness_by_genome.png   best and mean fitness per generation, one line per genome
                          (mean ± 1 SEM across reps, thin = each rep).
  final_fitness.png       final generation's best and mean fitness, one dot per rep.
  dimensionality_check.png
                          per generation: share of modules sharing a gene with another
                          (mirrored pair / tie), module count, hinge count (= the
                          controller's output size) and yz symmetry. tree_tied is a fair
                          control only if it matches tree_symmetric on the first three
                          and stays low on the last.
  individuals.csv, summary.csv
                          per individual; per rep at the final generation. Mann-Whitney U
                          tests of tree_tied against the other two are printed.

Usage:
    python plot_tied_control.py [--roots ROOT ...] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from itertools import starmap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_rowing_stride_sweep import (
    CI_ALPHA,
    GENOME_COLOR,
    GENOME_LABEL,
    INK,
    INK_2,
    gen_curve,
    style,
)
from scipy import stats
from shared import bilateral_symmetry_score
from sweep_common import checkpoint_generations

from ariel.ec.genotypes.tree.collision import mirror_node
from ariel.ec.genotypes.tree.symmetry import MirrorAxis
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.tying import tied_fraction

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK = "forward"
STRIDE = 25
GENOMES = ("tree", "tree_symmetric", "tree_tied")
RUN_RE = re.compile(rf"^ctrlstride_forward_(?P<genome>{'|'.join(sorted(GENOMES, key=len, reverse=True))})"
                    rf"_s{STRIDE}_rep(?P<rep>\d+)_\d+_\d+$")


# ── Loading ───────────────────────────────────────────────────────────────────


def shared_gene_fraction(genome: str, d: dict) -> float:
    """Share of non-core modules whose gene another module also carries."""
    if genome == "tree_tied":
        return tied_fraction(d)
    if genome == "tree":
        return 0.0
    t = TreeGenome.from_dict(d)
    nodes = [n for n in t.nodes if n != 0]
    paired = sum(1 for n in nodes if mirror_node(t, n, MirrorAxis.Y_ZERO) not in {None, n})
    return paired / len(nodes) if nodes else 0.0


def load_run(run_dir: Path, genome: str, rep: int) -> pd.DataFrame:
    rows = []
    for gen, ckpts in checkpoint_generations(run_dir, TASK).items():
        for c in ckpts:
            meta = json.loads((c / "meta.json").read_text())
            d = json.loads((c / "best_genome.json").read_text())
            rows.append({
                "run": run_dir.name, "genome": genome, "rep": rep, "gen": gen, "ckpt": c.name,
                "score": -meta["fitness"],  # fitness is minimised; score = displacement - penalties
                "n_modules": len(d["nodes"]) - 1,
                "n_hinges": sum(1 for n in d["nodes"].values() if n["type"] == "HINGE"),
                "shared_gene_frac": shared_gene_fraction(genome, d),
                "yz_symmetry": bilateral_symmetry_score(d),
            })
    return pd.DataFrame(rows)


def discover(roots: list[Path]) -> list[tuple[Path, str, int]]:
    """(run dir, genome, rep); a (genome, rep) found twice keeps the first root's run."""
    found: dict[tuple[str, int], Path] = {}
    for root in roots:
        for d in sorted(root.glob("ctrlstride_forward_*")):
            m = RUN_RE.match(d.name)
            if m and d.is_dir():
                key = (m["genome"], int(m["rep"]))
                if key in found:
                    print(f"[info] {d} skipped: {key} already from {found[key]}")
                    continue
                found[key] = d
    return [(d, g, r) for (g, r), d in sorted(found.items())]


# ── Plots ─────────────────────────────────────────────────────────────────────


def curves(table: pd.DataFrame, panels: list[tuple[str, str]], title: str, out: Path, dpi: int,
           ylabel: str | None = None) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8), squeeze=False)
    for ax, (col, sub) in zip(axes[0], panels, strict=False):
        style(ax)
        for genome in GENOMES:
            df = table[table["genome"] == genome]
            if df.empty:
                continue
            gens, m, sem, per_rep = gen_curve(df, col)
            color = GENOME_COLOR[genome]
            for rep in per_rep.columns:
                ax.plot(per_rep.index, per_rep[rep], color=color, linewidth=0.8, alpha=0.3)
            ax.fill_between(gens, m - sem, m + sem, color=color, alpha=CI_ALPHA, linewidth=0)
            ax.plot(gens, m, color=color, linewidth=2, label=f"{GENOME_LABEL[genome]} ({per_rep.shape[1]} reps)")
        ax.set_title(sub, fontsize=10, color=INK)
        ax.set_xlabel("generation", fontsize=9, color=INK_2)
    if ylabel:
        axes[0, 0].set_ylabel(ylabel, fontsize=9, color=INK_2)
    axes[0, 0].legend(fontsize=8.5, frameon=False, labelcolor=INK_2, loc="best")
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_final(sm: pd.DataFrame, out: Path, dpi: int) -> None:
    panels = [("final_best", "best fitness, final generation"), ("final_mean", "mean fitness, final generation")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8))
    present = [g for g in GENOMES if g in set(sm["genome"])]
    for ax, (col, title) in zip(axes, panels, strict=False):
        style(ax)
        for x, genome in enumerate(present):
            v = sm.loc[sm["genome"] == genome, col].to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(v)) if len(v) > 1 else np.zeros(len(v))
            ax.scatter(x + jitter, v, s=30, color=GENOME_COLOR[genome], alpha=0.75, linewidths=0, zorder=3)
            ax.hlines(v.mean(), x - 0.25, x + 0.25, color=GENOME_COLOR[genome], linewidth=2.2, zorder=4)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([GENOME_LABEL[g].replace(" (", "\n(") for g in present], fontsize=8.5, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK)
    axes[0].set_ylabel("fitness = forward displacement\n- height/jerk penalties (m)", fontsize=9, color=INK_2)
    fig.suptitle("Per rep (dots) and mean (bar)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def mann_whitney(sm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("final_best", "final_mean"):
        tied = sm.loc[sm["genome"] == "tree_tied", col]
        for other in ("tree", "tree_symmetric"):
            o = sm.loc[sm["genome"] == other, col]
            if len(tied) < 2 or len(o) < 2:
                continue
            res = stats.mannwhitneyu(tied, o, alternative="two-sided")
            rows.append({"metric": col, "comparison": f"tree_tied vs {other}", "n": f"{len(tied)} vs {len(o)}",
                         "median_tied": tied.median(), "median_other": o.median(), "U": res.statistic,
                         "p": res.pvalue})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=Path, nargs="+",
                        default=[REPO_ROOT / "__data__" / "ariel_rowing_penalty_ann" / "nopenalty",
                                 REPO_ROOT / "__data__" / "ariel_tied_control_ann"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "__data__" / "ariel_tied_control_ann" / "analysis")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = discover([r for r in args.roots if r.exists()])
    if not runs:
        msg = f"No stride-{STRIDE} runs under {args.roots}"
        raise SystemExit(msg)
    ind = pd.concat(list(starmap(load_run, runs)), ignore_index=True)
    ind.to_csv(args.out / "individuals.csv", index=False)
    print(ind.groupby("genome")["rep"].nunique().rename("reps").to_string())

    pg = (ind.groupby(["genome", "rep", "gen"])
          .agg(best=("score", "max"), mean=("score", "mean"), shared_gene_frac=("shared_gene_frac", "mean"),
               n_modules=("n_modules", "mean"), n_hinges=("n_hinges", "mean"), yz_symmetry=("yz_symmetry", "mean"))
          .reset_index())
    curves(pg, [("best", "best fitness"), ("mean", "mean fitness")],
           f"Stride {STRIDE}, zoo modules: tree vs symmetric vs random-tied "
           "(thick = mean ± 1 SEM across reps, thin = each rep)", args.out / "fitness_by_genome.png", args.dpi,
           ylabel="fitness = forward displacement\n- height/jerk penalties (m)")
    curves(pg, [("shared_gene_frac", "modules sharing a gene\n(mirrored pair / tie)"),
                ("n_modules", "modules"), ("n_hinges", "hinges (controller outputs)"),
                ("yz_symmetry", "bilateral (yz) symmetry")],
           "Is tree_tied a fair control? Population means per generation", args.out / "dimensionality_check.png",
           args.dpi)

    last = pg[pg["gen"] == pg.groupby(["genome", "rep"])["gen"].transform("max")]
    sm = last.rename(columns={"best": "final_best", "mean": "final_mean"})
    sm.to_csv(args.out / "summary.csv", index=False)
    plot_final(sm, args.out / "final_fitness.png", args.dpi)

    pd.set_option("display.width", 200)
    print("\nFinal generation, mean over reps:")
    print(sm.groupby("genome")[["final_best", "final_mean", "shared_gene_frac", "n_modules", "n_hinges",
                                "yz_symmetry"]].mean().to_string(float_format="%.2f"))
    mw = mann_whitney(sm)
    if not mw.empty:
        mw.to_csv(args.out / "mann_whitney.csv", index=False)
        print("\nMann-Whitney U (two-sided):")
        print(mw.to_string(index=False, float_format="%.3f"))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
