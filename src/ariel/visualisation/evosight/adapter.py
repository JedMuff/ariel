"""How one ARIEL experiment maps onto EvoSight.

ARIEL fixes the database schema but not what goes in it: the genotype layout,
the ``tags_`` keys (parents, descriptors, learned weights), the body decoder,
the brain and the episode loop all belong to the experiment. Subclass
:class:`ArielEvosightAdapter` and override the hooks that differ; the
defaults handle the common ARIEL conventions (a tree-genome body under
``genotype_["morph"]``, the parent id under ``tags_["parent_id"]``, the EA's
``fitness_`` as the single objective).

The one hook without a default is :meth:`ArielEvosightAdapter.make_rollout`
(only needed for videos and behavioural descriptors): rebuild the robot and
its brain from a stored individual and return a
:class:`~ariel.visualisation.evosight.rollout.Rollout`.

Adapters are loaded by spec — ``package.module:Class`` or
``path/to/file.py:Class`` — so they can live next to the experiment.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ariel.visualisation.evosight.reader import (
    RunData,
    StoredIndividual,
    find_db_parts,
    load_run,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import networkx as nx

    from ariel.visualisation.evosight.rollout import Rollout

#: Lists longer than this are summarised in ``genotype_str`` / dropped from
#: default metadata: they are weight vectors, not something to read or diff.
_LONG_LIST = 32


class ArielEvosightAdapter:
    """Default adapter; subclass per experiment.

    ``options`` are free-form ``key=value`` strings from the command line
    (``-o key=value``), for adapter-specific switches.
    """

    #: Objective keys the EA optimised (shown in Pareto / effectiveness views).
    objectives_optimized: ClassVar[Sequence[str]] = ("fitness",)
    #: Recorded-only objective keys; each must start with ``.``.
    objectives_tracked: ClassVar[Sequence[str]] = ()
    algorithm: ClassVar[str] = "ARIEL EA"
    #: Main database file of a run directory (continuations: ``<stem>_partN.db``).
    db_name: ClassVar[str] = "database.db"

    def __init__(self, **options: str) -> None:
        self.options = options
        self.run_dir: Path | None = None
        self.run_config: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Run level
    # ------------------------------------------------------------------

    def find_runs(self, root: Path) -> list[Path]:
        """Run directories under ``root`` (directories holding ``db_name``)."""
        root = Path(root)
        if (root / self.db_name).exists():
            return [root]
        return sorted(p.parent for p in root.rglob(self.db_name))

    def run_name(self, run_dir: Path, root: Path) -> str:
        """Name (and output sub-path) of a run found under ``root`` by
        ``batch``. Default: its path relative to ``root``.
        """
        run_dir, root = Path(run_dir), Path(root)
        if run_dir == root:
            return run_dir.name
        return run_dir.relative_to(root).as_posix()

    def prepare(self, run_dir: Path) -> None:
        """Load run-level context (no database access) before any other
        hook is used for ``run_dir``. Called in the main process and again
        in every render worker. Default: read ``run_config.json`` if any.
        """
        self.run_dir = Path(run_dir)
        cfg = self.run_dir / "run_config.json"
        self.run_config = json.loads(cfg.read_text()) if cfg.exists() else {}

    def load(self, run_dir: Path) -> RunData:
        return load_run(find_db_parts(run_dir, self.db_name), run_dir)

    def run_params(self, run: RunData) -> dict[str, Any]:
        """EvoSight run params (shown in the sidebar / compare view).
        Default: the run's ``run_config.json``, flattened one level.
        """
        params: dict[str, Any] = {
            "algorithm": self.algorithm,
            "domain": "ARIEL",
        }
        for k, v in self.run_config.items():
            if isinstance(v, dict):
                params.update({
                    kk: vv for kk, vv in v.items() if _is_scalar(vv)
                })
            elif _is_scalar(v):
                params[k] = v
        if len(run):
            # The initial population; later generations also hold offspring.
            params.setdefault(
                "population_size", len(run.members(run.first_generation))
            )
        return params

    def run_note(self, run: RunData) -> str | None:  # noqa: ARG002
        return None

    def setup_worker(self) -> None:
        """Per-process setup before :meth:`make_rollout` (e.g. applying a
        simulation preset). Called after :meth:`prepare` in every worker.
        """

    # ------------------------------------------------------------------
    # Individual level
    # ------------------------------------------------------------------

    def parents(self, ind: StoredIndividual) -> list[int]:
        """Ids of the individual's parents, from the usual tag names."""
        for key in ("parent_ids", "parents", "parent_id", "ps"):
            raw = ind.tags.get(key)
            if raw is None:
                continue
            values = raw if isinstance(raw, (list, tuple)) else [raw]
            out = []
            for v in values:
                try:
                    out.append(int(v))
                except (TypeError, ValueError):
                    continue
            return out
        return []

    def operation(self, ind: StoredIndividual) -> str:
        """Variation operator that created ``ind`` (a lineage-edge colour)."""
        n = len(self.parents(ind))
        if n == 0:
            return "Initialization"
        return "Crossover" if n > 1 else "Mutation"

    def objectives(self, ind: StoredIndividual) -> dict[str, float]:
        """Objective values; tracked ones use ``.``-prefixed keys."""
        return {"fitness": ind.fitness} if ind.fitness is not None else {}

    def metadata(self, ind: StoredIndividual) -> dict[str, Any]:
        """Small per-individual JSON for the inspector. Default: every scalar
        tag (vectors and nested structures are left out).
        """
        return {k: v for k, v in ind.tags.items() if _is_scalar(v)}

    def genotype_str(self, ind: StoredIndividual) -> str | None:
        """Text shown (and diffed against the parent) in the inspector.
        Default: the genotype as JSON with long numeric vectors summarised.
        """
        if ind.genotype is None:
            return None
        return json.dumps(
            _summarise(ind.genotype), separators=(",", ":"), sort_keys=True
        )

    def phenotype_str(self, ind: StoredIndividual) -> str | None:  # noqa: ARG002
        return None

    def build_graph(self, ind: StoredIndividual) -> nx.DiGraph | None:
        """The robot body as a robogen-lite graph, or None. Default: a
        :class:`~ariel.ec.genotypes.tree.tree_genome.TreeGenome` dict under
        ``genotype["morph"]`` (or the genotype itself).
        """
        geno = ind.genotype
        tree = geno.get("morph", geno) if isinstance(geno, dict) else None
        if not (isinstance(tree, dict) and "nodes" in tree and "edges" in tree):
            return None
        from ariel.ec.genotypes.tree.tree_genome import TreeGenome

        return TreeGenome.from_dict(tree).to_networkx()

    def morph_descriptors(self, ind: StoredIndividual) -> dict[str, float]:
        """Morphological descriptors (``morph.*``). Default:
        :class:`~ariel.utils.morphological_descriptor.MorphologicalMeasures`
        of :meth:`build_graph` — the paper letters B L E C J S D P plus
        module counts and bounding-box size.
        """
        graph = self.build_graph(ind)
        if graph is None:
            return {}
        from ariel.utils.morphological_descriptor import MorphologicalMeasures

        m = MorphologicalMeasures(graph)
        # descriptor name -> MorphologicalMeasures attribute
        names = {
            "branching": "B",
            "limbs": "L",
            "limb_length": "E",
            "coverage": "C",
            "joints": "J",
            "symmetry": "S",
            "module_diversity": "D",
            "footprint_proportion": "P",
            "num_modules": "num_modules",
            "num_bricks": "num_bricks",
            "num_active_hinges": "num_active_hinges",
            "bounding_box_depth": "bounding_box_depth",
            "bounding_box_width": "bounding_box_width",
            "bounding_box_height": "bounding_box_height",
        }
        out: dict[str, float] = {}
        for label, attr in names.items():
            try:
                value = float(getattr(m, attr))
            except Exception:  # noqa: BLE001 - one bad measure must not sink the export
                continue
            if math.isfinite(value):
                out[f"morph.{label}"] = value
        return out

    def stored_behaviour(self, ind: StoredIndividual) -> dict[str, float]:  # noqa: ARG002
        """Behavioural descriptors (``behav.*``) already stored in the
        database, merged with any computed from re-simulation.
        """
        return {}

    def make_rollout(self, ind: StoredIndividual) -> Rollout:
        """Rebuild ``ind``'s robot and brain and return the episode to
        record. Required for videos and computed behavioural descriptors.
        """
        msg = (
            f"{type(self).__name__} cannot simulate individuals: implement "
            "make_rollout() to render videos / compute behaviour"
        )
        raise NotImplementedError(msg)

    def verify(
        self, ind: StoredIndividual, result: dict[str, Any]
    ) -> tuple[float, float] | None:
        """``(stored, replayed)`` value pair to check a replay against, or
        None. Default: the stored ``fitness_`` against a ``fitness`` result.
        """
        if "fitness" in result and ind.fitness is not None:
            return ind.fitness, float(result["fitness"])
        return None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _is_scalar(v: Any) -> bool:
    if isinstance(v, float):
        return math.isfinite(v)
    return isinstance(v, (str, int, bool)) or v is None


def _summarise(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _summarise(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > _LONG_LIST and all(
            isinstance(x, (int, float)) for x in value
        ):
            return f"<{len(value)} numbers>"
        return [_summarise(v) for v in value]
    return value


def load_adapter(spec: str, **options: str) -> ArielEvosightAdapter:
    """Instantiate an adapter from ``module.path:Class`` or
    ``path/to/file.py:Class`` (``Class`` defaults to ``Adapter``).
    """
    target, _, cls_name = spec.partition(":")
    cls_name = cls_name or "Adapter"
    if target.endswith(".py") or "/" in target:
        path = Path(target).resolve()
        name = f"_evosight_adapter_{path.stem}"
        module = sys.modules.get(name)
        if module is None:
            module_spec = importlib.util.spec_from_file_location(name, path)
            if module_spec is None or module_spec.loader is None:
                msg = f"cannot load adapter file {path}"
                raise ImportError(msg)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    cls = getattr(module, cls_name)
    if not (isinstance(cls, type) and issubclass(cls, ArielEvosightAdapter)):
        msg = f"{spec} is not an ArielEvosightAdapter subclass"
        raise TypeError(msg)
    return cls(**options)


DEFAULT_ADAPTER = "ariel.visualisation.evosight.adapter:ArielEvosightAdapter"
