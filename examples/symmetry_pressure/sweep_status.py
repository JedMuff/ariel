"""
Report completion status of the symmetry-pressure sweep
(slurm/run_symmetry_pressure_sweep.sh): 4 tasks x 3 genome types x 5 reps
= 60 expected runs, decoded with the same array-index math as the slurm
script (rep is slowest-varying).

Usage:
    python sweep_status.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from sweep_common import (
    GENOME_TYPES,
    N_REPS,
    TASKS,
    discover_run_dirs,
    last_generation,
    parse_run_tag,
    run_config,
)


def _latest_mtime(run_dir: Path) -> dt.datetime | None:
    mtimes = [p.stat().st_mtime for p in run_dir.rglob("meta.json")]
    return dt.datetime.fromtimestamp(max(mtimes)) if mtimes else None


def main() -> None:
    present: dict[tuple[str, str, int], Path] = {}
    for d in discover_run_dirs():
        info = parse_run_tag(d.name)
        present[(info["task"], info["genome"], info["rep"])] = d

    header = ("idx", "task", "genome", "rep", "seed", "status", "last_gen", "last_write", "run_dir")
    rows: list[tuple] = []

    for idx in range(len(TASKS) * len(GENOME_TYPES) * N_REPS):
        task = TASKS[idx % len(TASKS)]
        genome = GENOME_TYPES[(idx // len(TASKS)) % len(GENOME_TYPES)]
        rep = (idx // (len(TASKS) * len(GENOME_TYPES))) % N_REPS
        seed = 42 + rep

        run_dir = present.get((task, genome, rep))
        if run_dir is None:
            rows.append((idx, task, genome, rep, seed, "not_started", "-", "-", "-"))
            continue

        budget = run_config(run_dir, task).get("budget", 30)
        last_gen = last_generation(run_dir, task)
        if last_gen is None:
            status = "no_checkpoints"
        elif last_gen >= budget:
            status = "complete"
        else:
            status = f"stalled ({last_gen}/{budget})"
        mtime = _latest_mtime(run_dir)

        rows.append((
            idx, task, genome, rep, seed, status,
            last_gen if last_gen is not None else "-",
            mtime.strftime("%Y-%m-%d %H:%M") if mtime else "-",
            run_dir.name,
        ))

    widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(len(header))]

    def fmt_row(r: tuple) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(r, widths))

    print(fmt_row(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))

    n_complete = sum(1 for r in rows if r[5] == "complete")
    n_stalled = sum(1 for r in rows if str(r[5]).startswith("stalled"))
    n_no_ckpt = sum(1 for r in rows if r[5] == "no_checkpoints")
    n_not_started = sum(1 for r in rows if r[5] == "not_started")

    print()
    print(
        f"Total expected: {len(rows)}   complete: {n_complete}   "
        f"stalled: {n_stalled}   no_checkpoints: {n_no_ckpt}   "
        f"not_started: {n_not_started}"
    )


if __name__ == "__main__":
    main()
