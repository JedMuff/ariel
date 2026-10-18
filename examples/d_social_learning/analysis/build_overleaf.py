"""Fill the Overleaf folder with exactly the figures its reports use.

The reports live in <overleaf-dir> (default __data__/social/figures/overleaf)
and are the source of truth: edit them there. Each report's \\graphicspath
names folders under figures/; GRAPHICS_ROOTS maps those to the analysis
output they come from. This script wipes <overleaf-dir>/figures and copies in
every \\includegraphics target of every report (PDF only, no PNGs or
manifests), so the whole folder uploads to Overleaf as one compilable project.

Usage (from the repo root):
    uv run examples/d_social_learning/analysis/build_overleaf.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

FIGURES = Path("__data__/social/figures")
# \graphicspath entry (relative to the Overleaf folder) -> source folder.
GRAPHICS_ROOTS = {
    "figures/parta/paper/": FIGURES / "parta/paper/pdf",
    "figures/parta/budget/": FIGURES / "parta/learning_budget/pdf",
    "figures/zoo_budget/paper/": FIGURES / "zoo_budget/paper/pdf",
    "figures/zoo_budget/budget/": FIGURES / "zoo_budget/learning_budget/pdf",
    "figures/zoo_budget_fe10/paper/": FIGURES / "zoo_budget_fe10/paper/pdf",
    "figures/zoo_budget_fe10/budget/": FIGURES / "zoo_budget_fe10/learning_budget/pdf",
    "figures/zoo_budget_fe10_g60/paper/": FIGURES / "zoo_budget_fe10_g60/paper/pdf",
    "figures/zoo_budget_fe10_g60/budget/": FIGURES / "zoo_budget_fe10_g60/learning_budget/pdf",
}


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overleaf-dir", default=str(FIGURES / "overleaf"))
    args = parser.parse_args()
    out = Path(args.overleaf_dir)

    copies: dict[Path, Path] = {}
    missing = []
    for tex_path in sorted(out.glob("*.tex")):
        tex = strip_comments(tex_path.read_text())
        gp = re.search(r"\\graphicspath\{((?:\{[^}]*\})+)\}", tex)
        roots = re.findall(r"\{([^}]*)\}", gp[1]) if gp else [""]
        unknown = [r for r in roots if r not in GRAPHICS_ROOTS]
        if unknown:
            sys.exit(f"{tex_path.name}: graphicspath entries {unknown} not in GRAPHICS_ROOTS")
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
            name = name if name.endswith(".pdf") else f"{name}.pdf"
            for root in roots:  # first match wins, as in LaTeX
                src = GRAPHICS_ROOTS[root] / name
                if src.exists():
                    copies[out / root / name] = src
                    break
            else:
                missing.append(f"{tex_path.name}: {name}")
    if missing:
        sys.exit("Missing figures (run the make_*_figures.sh scripts first):\n  " + "\n  ".join(missing))

    shutil.rmtree(out / "figures", ignore_errors=True)
    for dst, src in copies.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"Copied {len(copies)} figures into {out / 'figures'}")


if __name__ == "__main__":
    main()
