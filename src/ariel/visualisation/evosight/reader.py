"""Read a finished (or running) ARIEL EA database, independent of experiment.

Every ARIEL run stores its individuals in one ``individual`` table
(:class:`ariel.ec.individual.Individual`): one row per individual with
``time_of_birth`` / ``time_of_death`` generations, ``fitness_`` and free-form
``genotype_`` / ``tags_`` JSON. This module reads that table with plain
:mod:`sqlite3` (read-only, so it never touches the SQLModel schema and is safe
on a database another process is still writing) and reconstructs the
per-generation populations.

A run continued with ``EA(restart=...)`` into a second file (e.g.
``database.db`` + ``database_part2.db``) is read as one run: the files are
merged by ``id`` and the later file's row wins, because only there are the
restored individuals' ``time_of_death`` / ``alive`` current.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_PART_RE = re.compile(r"_part(\d+)\.db$")


class StoredIndividual:
    """One ``individual`` row. ``genotype`` and ``tags`` are parsed lazily:
    tags can carry large vectors (learned weights, learning curves) that most
    of the export never looks at.
    """

    def __init__(
        self,
        id: int,  # noqa: A002
        alive: bool,
        time_of_birth: int,
        time_of_death: int,
        fitness: float | None,
        genotype_raw: str | None,
        tags_raw: str | None,
    ) -> None:
        self.id = id
        self.alive = alive
        self.time_of_birth = time_of_birth
        self.time_of_death = time_of_death
        self.fitness = fitness
        self._genotype_raw = genotype_raw
        self._tags_raw = tags_raw

    @cached_property
    def genotype(self) -> Any:
        return json.loads(self._genotype_raw) if self._genotype_raw else None

    @cached_property
    def tags(self) -> dict:
        tags = json.loads(self._tags_raw) if self._tags_raw else {}
        return tags if isinstance(tags, dict) else {}

    def __repr__(self) -> str:
        return (
            f"StoredIndividual(id={self.id}, born={self.time_of_birth}, "
            f"died={self.time_of_death}, fitness={self.fitness})"
        )


@dataclass
class RunData:
    """All individuals of one run plus its generation structure."""

    run_dir: Path
    db_paths: list[Path]
    individuals: dict[int, StoredIndividual]
    #: Generations present, from the first birth to the last death.
    first_generation: int = 0
    last_generation: int = 0
    _members: dict[int, list[int]] = field(default_factory=dict, repr=False)

    def generations(self) -> range:
        return range(self.first_generation, self.last_generation + 1)

    def members(self, generation: int) -> list[int]:
        """Ids alive during ``generation``: everyone with
        ``time_of_birth <= generation <= time_of_death`` (as
        :meth:`ariel.ec.archive.Archive.by_generation`). For a (mu+lambda)
        EA this is the selection pool — last generation's survivors plus
        this generation's offspring — so offspring that lose selection
        right away still appear once.
        """
        return self._members.get(generation, [])

    def survived(self, ind: StoredIndividual, generation: int) -> bool:
        """Did ``ind`` make it through selection at the end of ``generation``?"""
        if generation < ind.time_of_death:
            return True
        return generation == self.last_generation and ind.alive

    def __iter__(self) -> Iterator[StoredIndividual]:
        return iter(self.individuals.values())

    def __len__(self) -> int:
        return len(self.individuals)


def find_db_parts(run_dir: Path, db_name: str = "database.db") -> list[Path]:
    """``db_name`` followed by its ``*_partN.db`` continuations in N order.

    Falls back to every ``*.db`` in ``run_dir`` (sorted) when ``db_name`` is
    absent, so runs that use a timestamped file name still load.
    """
    run_dir = Path(run_dir)
    main = run_dir / db_name
    stem = main.stem
    parts = sorted(
        (
            p
            for p in run_dir.glob(f"{stem}_part*.db")
            if _PART_RE.search(p.name)
        ),
        key=lambda p: int(_PART_RE.search(p.name).group(1)),  # type: ignore[union-attr]
    )
    if main.exists():
        return [main, *parts]
    return sorted(run_dir.glob("*.db"))


def _read_rows(db: Path) -> list[tuple]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(
            "SELECT id, alive, time_of_birth, time_of_death, fitness_, "
            "genotype_, tags_ FROM individual",
        ).fetchall()
    finally:
        con.close()


def load_run(
    db_paths: Sequence[Path | str],
    run_dir: Path | str | None = None,
    *,
    evaluated_only: bool = True,
) -> RunData:
    """Load one run from its database file(s), merged in the given order.

    ``evaluated_only`` drops rows with no ``fitness_`` (individuals created
    but never evaluated, e.g. when a run was killed mid-generation).
    """
    db_paths = [Path(p) for p in db_paths]
    if not db_paths:
        msg = "load_run: no database files given"
        raise FileNotFoundError(msg)
    individuals: dict[int, StoredIndividual] = {}
    for db in db_paths:
        for row in _read_rows(db):
            iid, alive, born, died, fitness, geno, tags = row
            if evaluated_only and fitness is None:
                continue
            individuals[int(iid)] = StoredIndividual(
                id=int(iid),
                alive=bool(alive),
                time_of_birth=int(born),
                time_of_death=int(died)
                if died is not None and died >= 0
                else int(born),
                fitness=None if fitness is None else float(fitness),
                genotype_raw=geno,
                tags_raw=tags,
            )
    run = RunData(
        run_dir=Path(run_dir) if run_dir is not None else db_paths[0].parent,
        db_paths=db_paths,
        individuals=dict(sorted(individuals.items())),
    )
    if individuals:
        run.first_generation = min(
            i.time_of_birth for i in individuals.values()
        )
        run.last_generation = max(i.time_of_death for i in individuals.values())
        members: dict[int, list[int]] = {g: [] for g in run.generations()}
        for ind in run.individuals.values():
            for g in range(ind.time_of_birth, ind.time_of_death + 1):
                members[g].append(ind.id)
        run._members = members  # noqa: SLF001
    return run
