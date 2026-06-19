"""Standalone SQLite persistence layer for EvoGym experiments.

A stripped-down copy of ariel.ec (Individual, Population, Archive) that
imports nothing from the ariel package. Runs on Python 3.10 with sqlmodel
installed.

Install once:
    evogym-venv/bin/pip install sqlmodel pydantic-settings
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
from sqlalchemy import JSON, Column, Engine, Select, create_engine, func
from sqlmodel import Field, SQLModel, Session, col, select


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------


class Individual(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: int | None = Field(default=None, primary_key=True)

    alive: bool = Field(default=True, index=True)
    time_of_birth: int = Field(default=-1, index=True)
    time_of_death: int = Field(default=-1, index=True)

    requires_eval: bool = Field(default=True, index=True)
    fitness_: float | None = Field(default=None, index=True)

    requires_init: bool = Field(default=True, index=True)
    genotype_: dict = Field(default=None, sa_column=Column(JSON))
    tags_: dict = Field(default={}, sa_column=Column(JSON))

    @property
    def fitness(self) -> float:
        if self.fitness_ is None:
            raise ValueError(f"fitness accessed before evaluation: {self.id=}")
        return self.fitness_

    @fitness.setter
    def fitness(self, value: float) -> None:
        self.requires_eval = False
        self.fitness_ = float(value)

    @property
    def genotype(self):
        if self.genotype_ is None:
            raise ValueError(f"genotype accessed before initialization: {self.id=}")
        return self.genotype_

    @genotype.setter
    def genotype(self, value) -> None:
        self.requires_init = not bool(value)
        self.genotype_ = value

    @property
    def tags(self) -> dict:
        return self.tags_

    @tags.setter
    def tags(self, update: dict) -> None:
        self.tags_ = {**self.tags_, **update}


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def _safe_attr(ind: Individual, attr: str) -> float:
    try:
        val = getattr(ind, attr)
        return float(val) if val is not None else float("-inf")
    except (ValueError, AttributeError, TypeError):
        return float("-inf")


class Population:
    def __init__(self, individuals: list[Individual]) -> None:
        self.population: list[Individual] = list(individuals)

    def __len__(self) -> int:
        return len(self.population)

    def __iter__(self):
        return iter(self.population)

    def __bool__(self) -> bool:
        return bool(self.population)

    def __add__(self, other: "Population") -> "Population":
        return Population(self.population + other.population)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Population(self.population[index])
        return self.population[index]

    def append(self, ind: Individual) -> None:
        self.population.append(ind)

    def extend(self, other) -> None:
        self.population.extend(
            other.population if isinstance(other, Population) else other
        )

    def to_list(self) -> list[Individual]:
        return list(self.population)

    @property
    def size(self) -> int:
        return len(self.population)

    def where(self, predicate) -> "Population":
        return Population([ind for ind in self.population if predicate(ind)])

    @property
    def alive(self) -> "Population":
        return self.where(lambda ind: ind.alive)

    @property
    def unevaluated(self) -> "Population":
        return self.where(lambda ind: ind.requires_eval)

    @property
    def evaluated(self) -> "Population":
        return self.where(lambda ind: not ind.requires_eval)

    def best(self, *, n: int = 1, attribute: str = "fitness_") -> "Population":
        return Population(
            sorted(self.population, key=lambda ind: _safe_attr(ind, attribute), reverse=True)[:n]
        )

    def sample(self, n: int) -> "Population":
        return Population(random.sample(self.population, min(n, len(self.population))))

    @classmethod
    def empty(cls) -> "Population":
        return cls([])


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


class Archive:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Archive database not found: {self.db_path}")
        self.engine: Engine = create_engine(f"sqlite:///{self.db_path}")

    def _fetch_all(self, statement) -> list[Individual]:
        with Session(self.engine, expire_on_commit=False) as session:
            rows = list(session.exec(statement).all())
            # access all fields while still in session to populate instance cache
            for ind in rows:
                _ = ind.id, ind.fitness_, ind.genotype_, ind.tags_, ind.time_of_birth, ind.time_of_death
        for ind in rows:
            ind.id = None
        return rows

    def by_generation(self, generation: int) -> Population:
        statement = (
            select(Individual)
            .where(Individual.requires_eval == False)  # noqa: E712
            .where(Individual.time_of_birth <= generation)
            .where(Individual.time_of_death >= generation)
        )
        return Population(self._fetch_all(statement))

    @property
    def generation_range(self) -> tuple[int, int]:
        with Session(self.engine) as session:
            min_birth = session.exec(select(func.min(Individual.time_of_birth))).one()
            max_death = session.exec(select(func.max(Individual.time_of_death))).one()
        return (min_birth or 0, max_death or 0)

    def fitness_stats(self, birth_range=None) -> dict[str, float]:
        statement = select(Individual).where(Individual.requires_eval == False)  # noqa: E712
        if birth_range is not None:
            lo, hi = birth_range
            statement = statement.where(Individual.time_of_birth >= lo).where(
                Individual.time_of_birth <= hi
            )
        inds = self._fetch_all(statement)
        if not inds:
            nan = float("nan")
            return {"min": nan, "max": nan, "mean": nan, "std": nan, "median": nan}
        fitnesses = np.array([ind.fitness_ for ind in inds], dtype=float)
        return {
            "min": float(np.min(fitnesses)),
            "max": float(np.max(fitnesses)),
            "mean": float(np.mean(fitnesses)),
            "std": float(np.std(fitnesses)),
            "median": float(np.median(fitnesses)),
        }

    def __repr__(self) -> str:
        return f"Archive(path={self.db_path!r})"


# ---------------------------------------------------------------------------
# Minimal EA engine
# ---------------------------------------------------------------------------


class SimpleEA:
    """Minimal generational EA engine for EvoGym (no rich, no EAOperation wrapper)."""

    def __init__(
        self,
        population: Population,
        db_file_path: Path,
        *,
        db_handling: Literal["delete", "halt"] = "delete",
    ) -> None:
        self.population = population
        self.current_generation = 0
        self.engine = self._setup_db(db_file_path, db_handling)
        self._commit()

    def _setup_db(self, path: Path, handling: str) -> Engine:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if handling == "delete":
                path.unlink()
            else:
                raise FileExistsError(f"DB exists: {path}")
        engine = create_engine(f"sqlite:///{path}")
        SQLModel.metadata.create_all(engine)
        return engine

    def _commit(self) -> None:
        # expire_on_commit=False keeps instances usable after the session closes
        with Session(self.engine, expire_on_commit=False) as session:
            for ind in self.population:
                if ind.time_of_birth == -1:
                    ind.time_of_birth = self.current_generation
                ind.time_of_death = self.current_generation
                session.add(ind)
            session.commit()

    def step(self, step_fn) -> None:
        self.current_generation += 1
        self.population = step_fn(self.population, self.current_generation)
        self._commit()

    def run(self, step_fn, num_steps: int) -> None:
        for gen in range(num_steps):
            print(f"  gen {self.current_generation + 1}/{num_steps}", flush=True)
            self.step(step_fn)
