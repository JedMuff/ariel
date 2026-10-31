# Paper draft (LaTeX, Overleaf-ready)

Upload: zip this folder (`main.tex`, `sections/`, `figures/`, `references.bib`) and use Overleaf's *New project → Upload project*. `make_figures.sh` and this README are harmless if included.

```bash
cd examples/symmetry_paper/paper && zip -r ../paper_overleaf.zip main.tex references.bib sections figures
```

- `main.tex`: preamble and macros. `\drafttrue` / `\draftfalse` shows or hides every draft note.
  - `\stat{…}` (blue): where a significance test or CI belongs, and which one.
  - `\caveat{…}` (red): a claim the current data do not fully support.
  - `\todo{…}` (yellow): work left.
  - `\pending{…}` (orange): numbers that will change with more runs.
- `sections/`: one file per section. The introduction and related work are placeholders for the author.
- `figures/`: copied from `__data__/ariel_paper/paper/` by `make_figures.sh`. Rerun it after adding runs (`FAST=1` skips the two body-settle scripts, which take a few minutes). `V2=<root>` reads the v2 runs instead of the v1 data.
- Style: no underscores in names, no bold, no run-in `\paragraph` headings, no "zoo" (the macros `\bb`, `\dann`, `\sym`, … print plain names).
- Numbers in the text are typed by hand from the logs in `__data__/ariel_paper/paper/logs/` and `…/extra/inline_numbers.txt`. After new runs, rerun `make_figures.sh` and update them.
