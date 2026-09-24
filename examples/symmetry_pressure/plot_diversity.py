"""
Plot population genome diversity as mean pairwise tree edit distance (TED)
per generation, comparing symmetry-enforced vs no-symmetry runs.

Tree edit distance (Zhang-Shasha) is computed between every pair of
evaluated individuals within each generation.  Children are ordered by face
name (alphabetical) to produce a deterministic ordered tree; node labels are
the module type initial (C / B / H).

Usage:
    python plot_diversity.py \\
        --sym-dirs   ../../__data__/food_skills/food_skills_..._35117 ... \\
        --nosym-dirs ../../__data__/food_skills/food_skills_..._35301 ... \\
        --out-dir    ../../__data__/food_skills_comparison
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--sym-dirs",   nargs="+", type=Path, required=True)
parser.add_argument("--nosym-dirs", nargs="+", type=Path, required=True)
parser.add_argument("--out-dir",    type=Path, default=Path("__data__/food_skills_comparison"))
parser.add_argument("--dpi",        type=int, default=150)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

SYM_COLOR   = "#4C72B0"
NOSYM_COLOR = "#DD8452"
CI_ALPHA    = 0.20


# ── Tree edit distance (Zhang-Shasha) ─────────────────────────────────────────

def _genome_to_ordered_tree(genome: dict) -> dict:
    """Convert genome dict to a nested ordered labeled tree.

    Children are sorted by attachment face name (alphabetically) to give a
    deterministic ordering.  Node labels are the first character of the module
    type: C, B, or H.
    """
    nodes = genome["nodes"]
    edges = genome["edges"]

    children: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in edges:
        children[str(e["parent"])].append((e["face"], str(e["child"])))
    for pid in children:
        children[pid].sort(key=lambda x: x[0])

    core_id = next(nid for nid, n in nodes.items() if n["type"] == "CORE")

    def build(nid: str) -> dict:
        return {
            "label": nodes[nid]["type"][0],
            "children": [build(cid) for _, cid in children.get(nid, [])],
        }

    return build(core_id)


def _prepare(tree: dict) -> tuple[list[str], list[int], list[int]]:
    """Index a labeled ordered tree for Zhang-Shasha.

    Returns (labels, lml, keyroots) where:
      labels[i]  – module-type label of post-order node i
      lml[i]     – post-order index of the leftmost leaf of the subtree at i
      keyroots   – sorted list of keyroot post-order indices
    """
    labels: list[str] = []
    lml: list[int] = []
    lml_rightmost: dict[int, int] = {}

    def visit(node: dict) -> int:
        child_idxs = [visit(c) for c in node["children"]]
        i = len(labels)
        labels.append(node["label"])
        lml.append(lml[child_idxs[0]] if child_idxs else i)
        lml_rightmost[lml[i]] = i
        return i

    visit(tree)
    return labels, lml, sorted(lml_rightmost.values())


def _zhang_shasha(
    t1: tuple[list[str], list[int], list[int]],
    t2: tuple[list[str], list[int], list[int]],
) -> int:
    """Compute tree edit distance using the Zhang-Shasha algorithm.

    Edit costs: insert = delete = 1;  rename = 0 if same label, 1 otherwise.
    """
    labs1, lml1, kr1 = t1
    labs2, lml2, kr2 = t2
    n1, n2 = len(labs1), len(labs2)

    if n1 == 0:
        return n2
    if n2 == 0:
        return n1

    td = [[0] * n2 for _ in range(n1)]

    for k1 in kr1:
        for k2 in kr2:
            l1, l2 = lml1[k1], lml2[k2]
            s1, s2 = k1 - l1 + 2, k2 - l2 + 2
            fd = [[0] * s2 for _ in range(s1)]

            for i in range(1, s1):
                fd[i][0] = fd[i - 1][0] + 1
            for j in range(1, s2):
                fd[0][j] = fd[0][j - 1] + 1

            for i in range(1, s1):
                for j in range(1, s2):
                    ni, nj = l1 + i - 1, l2 + j - 1
                    c = 0 if labs1[ni] == labs2[nj] else 1

                    if lml1[ni] == l1 and lml2[nj] == l2:
                        fd[i][j] = min(
                            fd[i - 1][j] + 1,
                            fd[i][j - 1] + 1,
                            fd[i - 1][j - 1] + c,
                        )
                        td[ni][nj] = fd[i][j]
                    else:
                        pi = lml1[ni] - l1
                        pj = lml2[nj] - l2
                        fd[i][j] = min(
                            fd[i - 1][j] + 1,
                            fd[i][j - 1] + 1,
                            fd[pi][pj] + td[ni][nj],
                        )

    return td[n1 - 1][n2 - 1]


def mean_pairwise_ted(genomes: list[dict]) -> float:
    """Mean pairwise tree edit distance for a collection of genomes."""
    if len(genomes) < 2:
        return 0.0
    prepared = [_prepare(_genome_to_ordered_tree(g)) for g in genomes]
    total, n = 0, 0
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            total += _zhang_shasha(prepared[i], prepared[j])
            n += 1
    return total / n if n else 0.0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run_diversity(run_dir: Path) -> dict[int, float]:
    """Return gen -> mean pairwise TED for every generation in the run."""
    ckpt_base = run_dir / "__data__" / "gecko_food_skills" / "checkpoints"

    by_gen: dict[int, list[dict]] = defaultdict(list)
    for ckpt in sorted(ckpt_base.iterdir()):
        meta_path = ckpt / "meta.json"
        genome_path = ckpt / "best_genome.json"
        if not meta_path.exists() or not genome_path.exists():
            continue
        gen = json.loads(meta_path.read_text())["gen"]
        genome = json.loads(genome_path.read_text())
        by_gen[gen].append(genome)

    result: dict[int, float] = {}
    for gen, genomes in sorted(by_gen.items()):
        result[gen] = mean_pairwise_ted(genomes)
    return result


# ── Statistics ────────────────────────────────────────────────────────────────

def group_stats(
    runs_diversity: list[dict[int, float]],
) -> tuple[list[int], list[float], list[float], list[float]]:
    all_series: dict[int, list[float]] = defaultdict(list)
    for run in runs_diversity:
        for g, v in run.items():
            all_series[g].append(v)
    gens = sorted(all_series)
    means, lo, hi = [], [], []
    for g in gens:
        vs = np.array(all_series[g])
        m = float(np.mean(vs))
        ci = (stats.sem(vs) * stats.t.ppf(0.975, max(len(vs) - 1, 1))
              if len(vs) > 1 else 0.0)
        means.append(m)
        lo.append(m - ci)
        hi.append(m + ci)
    return gens, means, lo, hi


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_diversity(
    sym_div: list[dict], nosym_div: list[dict],
    out_path: Path, dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for runs_div, color, label in [
        (sym_div,   SYM_COLOR,   "Symmetry enforced (y_zero)"),
        (nosym_div, NOSYM_COLOR, "No symmetry (none)"),
    ]:
        gens, means, lo, hi = group_stats(runs_div)
        ax.plot(gens, means, color=color, linewidth=2.5, label=label)
        ax.fill_between(gens, lo, hi, color=color, alpha=CI_ALPHA)

        for run in runs_div:
            g_list = sorted(run)
            ax.plot(g_list, [run[g] for g in g_list],
                    color=color, linewidth=0.7, alpha=0.2)

    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Mean pairwise tree edit distance", fontsize=11)
    ax.set_title(
        "Morphological diversity over evolution\n"
        "(mean pairwise TED within evaluated batch per generation;\n"
        "thick = group mean ± 95% CI,  faint = individual runs)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Computing TED diversity for symmetric runs...")
    sym_div = []
    for i, d in enumerate(args.sym_dirs, 1):
        print(f"  [{i}/{len(args.sym_dirs)}] {d.name} ...", flush=True)
        sym_div.append(load_run_diversity(d))

    print("Computing TED diversity for no-symmetry runs...")
    nosym_div = []
    for i, d in enumerate(args.nosym_dirs, 1):
        print(f"  [{i}/{len(args.nosym_dirs)}] {d.name} ...", flush=True)
        nosym_div.append(load_run_diversity(d))

    print("Plotting...")
    plot_diversity(sym_div, nosym_div,
                   args.out_dir / "diversity.png", args.dpi)
    print(f"Done. Output -> {args.out_dir}/diversity.png")


if __name__ == "__main__":
    main()
