# Paper draft (LaTeX, Overleaf-ready)

Upload: zip this folder and use Overleaf's *New project → Upload project*.

```bash
cd examples/symmetry_paper/paper && rm -f ../paper_overleaf.zip && zip -r ../paper_overleaf.zip main.tex references.bib sections figures
```

- `main.tex`: preamble and macros. `\todo{…}` marks work left; `\draftfalse` hides the notes.
- `sections/`: one file per section. The introduction and related work are placeholders for the author.
- `figures/`: copied from `__data__/ariel_paper_v2/paper/` by `make_figures.sh` (`FAST=1` skips the two body-settle scripts).
- Style: no underscores in names, no bold, no run-in `\paragraph` headings, no "zoo" (the macros `\bb`, `\dann`, `\sym`, … print plain names).
- Numbers in the text are typed by hand from `__data__/ariel_paper_v2/paper/logs/` (`stats.txt` is the main source) and `extra/inline_numbers.txt`. The E0 runs are not in the paper.
