r"""EvoSight adapter for the social-learning experiment (experiment.py).

Bodies are tree genomes under ``genotype_["morph"]``; each body's
DistributedMLP brain was learned by CMA-ES during its lifetime and stored as
``tags_["theta"]`` (initial mean: ``tags_["init_theta"]``, inherited from
``tags_["donor_ids"]`` by the run's scheme). The EA selected on ``fitness_``,
a normalised blend of distance and novelty, so ``distance`` and friends are
exported as tracked objectives.

A replay re-runs one of the individual's stored final evaluations
(evaluate.py ``--final-evals``): the same jittered start drawn from
``eval_seed``, so ``--verify`` compares it with ``final_eval_fitness``.

    uv run --group evosight python -m ariel.visualisation.evosight render \\
        __data__/social/ariel_zoo_fe10/ig10/best/x05/k5/rep_0_<ts> \\
        __data__/evosight/ariel_zoo_fe10/ig10/best/x05/k5/rep_0 \\
        --adapter examples/d_social_learning/evosight_adapter.py:SocialLearningAdapter \\
        --workers 16
    uv run --group evosight evosight serve --local __data__/evosight/ariel_zoo_fe10

Adapter options (``-o key=value``): ``episode`` — which stored final
evaluation to replay (default 0); ``weights`` — ``learned`` (default) or
``init`` to watch the brain the individual started learning from.
"""

from __future__ import annotations

import math
import operator
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ariel.visualisation.evosight.adapter import ArielEvosightAdapter
from ariel.visualisation.evosight.reader import (
    StoredIndividual,
    find_db_parts,
    load_run,
)
from ariel.visualisation.evosight.rollout import Rollout

HERE = Path(__file__).resolve().parent
# Appended, not prepended: d_social_learning/ariel would shadow the ariel package.
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

RUN_RE = re.compile(
    r"(?P<budget>ig\d+)/(?P<scheme>[^/]+)/x(?P<x>\d+)/k(?P<k>\d+)/rep_(?P<rep>\d+)_",
)

# Run-config args that describe the machine/job rather than the experiment.
_NOISE_ARGS = {
    "workers",
    "eval_timeout",
    "out_root",
    "resume_dir",
    "allow_novelty_change",
}
# Stored per-individual scalars that describe the behaviour of the learned brain.
_STORED_BEHAVIOUR = (
    "distance",
    "displacement",
    "mean_jerk",
    "c_hinge",
    "init_fitness",
)


class SocialLearningAdapter(ArielEvosightAdapter):
    objectives_optimized = ("fitness",)
    objectives_tracked = (
        ".distance",
        ".displacement",
        ".novelty",
        ".init_fitness",
        ".learning_gain",
        ".mean_jerk",
    )
    algorithm = "(mu+lambda) body evolution + CMA-ES brain learning"

    def find_runs(self, root: Path) -> list[Path]:
        return sorted(
            p.parent
            for p in Path(root).rglob("database.db")
            if (p.parent / "run_config.json").exists()
        )

    def run_name(self, run_dir: Path, root: Path) -> str:
        # ig10/best/x05/k5/rep_0_fe10_novTOPDOWN_<timestamp> -> .../rep_0
        rel = super().run_name(run_dir, root)
        return re.sub(r"(^|/)(rep_\d+)_[^/]*$", r"\1\2", rel)

    def load(self, run_dir: Path):
        # database.db + database_part<N>.db (experiment.py --resume), merged by id.
        return load_run(find_db_parts(run_dir), run_dir)

    def run_params(self, run) -> dict[str, Any]:
        args = dict(self.run_config.get("args", {}))
        params: dict[str, Any] = {
            k: v
            for k, v in args.items()
            if k not in _NOISE_ARGS
            and (v is None or isinstance(v, (str, int, float, bool)))
        }
        params.update(
            algorithm=self.algorithm,
            domain=f"ARIEL {args.get('sim_preset', '')} locomotion".replace(
                "  ", " ",
            ),
            population_size=args.get("pop"),
            selection_method=args.get("selection"),
            gens=run.last_generation,
            seed=self.run_config.get("seed"),
            git_commit=self.run_config.get("git_commit"),
            resumes=len(self.run_config.get("resumes", [])),
        )
        if m := RUN_RE.search(Path(self.run_dir).as_posix()):
            params.update(
                budget=m["budget"],
                x=int(m["x"]) / 10,
                k_inherit=int(m["k"]),
                rep=int(m["rep"]),
            )
        return params

    def run_note(self, run) -> str | None:
        args = self.run_config.get("args", {})
        return (
            f"scheme={args.get('scheme')} x={args.get('x')} inner_gens={args.get('inner_gens')} "
            f"rep={args.get('rep')}: brains inherited by '{args.get('scheme')}', learned by CMA-ES"
        )

    def setup_worker(self) -> None:
        import simulator_dependent_functions as sdf

        sdf.simulator = sdf.SIMULATOR_ARIEL
        sdf.set_sim_preset(
            self.run_config.get("args", {}).get("sim_preset", "legacy"),
        )

    # -- individuals -------------------------------------------------------

    def parents(self, ind: StoredIndividual) -> list[int]:
        pid = ind.tags.get("parent_id")
        return [] if pid is None else [int(pid)]

    def operation(self, ind: StoredIndividual) -> str:
        if not self.parents(ind):
            return "Initialization"
        op = (ind.tags.get("mutation") or {}).get("op")
        return f"Mutation:{op}" if op else "Mutation"

    def objectives(self, ind: StoredIndividual) -> dict[str, float]:
        t = ind.tags
        out = {"fitness": ind.fitness}
        for key in (
            "distance",
            "displacement",
            "novelty",
            "init_fitness",
            "mean_jerk",
        ):
            if t.get(key) is not None:
                out[f".{key}"] = t[key]
        if t.get("distance") is not None and t.get("init_fitness") is not None:
            out[".learning_gain"] = t["distance"] - t["init_fitness"]
        return {k: v for k, v in out.items() if v is not None}

    def metadata(self, ind: StoredIndividual) -> dict[str, Any]:
        t = ind.tags
        meta: dict[str, Any] = {
            "parent_id": t.get("parent_id"),
            "donor_ids": t.get("donor_ids"),
            "mutation": t.get("mutation"),
            "eval_seed": t.get("eval_seed"),
            "distance_cma": t.get("distance_cma"),
            "c_hinge": t.get("c_hinge"),
        }
        finals = t.get("final_eval_fitness") or []
        if finals:
            meta["final_eval_fitness_std"] = float(np.std(finals))
        curve = t.get("learning_curve") or []
        if curve:
            # Best candidate per CMA-ES generation, the shape of the learning curve.
            meta["learning_curve_best"] = [
                round(max(gen), 5) for gen in curve if gen
            ]
        return {k: v for k, v in meta.items() if v is not None}

    def genotype_str(self, ind: StoredIndividual) -> str | None:
        """The body tree, one module per line in depth-first order, so the
        inspector's parent diff shows exactly which modules a mutation
        changed (the learned weights are not shown).
        """
        morph = (ind.genotype or {}).get("morph")
        if not morph:
            return None
        nodes = morph.get("nodes", {})
        children: dict[str, list[dict]] = {}
        for e in morph.get("edges", []):
            children.setdefault(str(e["parent"]), []).append(e)
        lines: list[str] = []

        def walk(nid: str, face: str, depth: int) -> None:
            node = nodes.get(nid) or nodes.get(int(nid)) or {}
            rot = node.get("rotation", "")
            lines.append(
                f"{'  ' * depth}{face}{node.get('type', '?')} {rot}".rstrip(),
            )
            for e in sorted(
                children.get(nid, []), key=operator.itemgetter("face"),
            ):
                walk(str(e["child"]), f"{e['face']}: ", depth + 1)

        root = next(
            (nid for nid, n in nodes.items() if n.get("type") == "CORE"), "0",
        )
        walk(str(root), "", 0)
        return "\n".join(lines)

    def stored_behaviour(self, ind: StoredIndividual) -> dict[str, float]:
        out = {}
        for key in _STORED_BEHAVIOUR:
            v = ind.tags.get(key)
            if isinstance(v, (int, float)) and math.isfinite(v):
                out[f"behav.{key}"] = float(v)
        return out

    # -- simulation --------------------------------------------------------

    def _episode(self) -> int:
        return int(self.options.get("episode", 0))

    def make_rollout(self, ind: StoredIndividual) -> Rollout:
        import mujoco
        import simulator_dependent_functions as sdf
        from sim_ariel import evaluator

        from ariel.simulation.controllers.distributed_mlp import DistributedMLP

        key = "init_theta" if self.options.get("weights") == "init" else "theta"
        theta = np.asarray(ind.tags[key], dtype=np.float64)
        spec = evaluator.initialize_world(ind.genotype["morph"])
        hidden = int(self.run_config.get("args", {}).get("hidden", 16))
        brain = DistributedMLP(n_neighbors=sdf.n_neighbours(), hidden=hidden)

        model = spec["model"]
        n_hinges = sum(
            1
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        )

        def run(recorder) -> dict[str, Any]:
            # evaluate.py's final evaluations draw episode i's start jitter
            # from this stream (run_episode: 2 position draws, then one per
            # hinge); skip the earlier episodes' draws without simulating them.
            rng = np.random.default_rng(
                np.random.SeedSequence([ind.tags["eval_seed"], 1]),
            )
            for _ in range(self._episode()):
                rng.normal(0.0, 1.0, 2)
                for _ in range(n_hinges):
                    rng.normal(0.0, 1.0)
            return evaluator.run_episode(
                theta, brain, spec, perturb_rng=rng, recorder=recorder,
            )

        return Rollout(model=spec["model"], data=spec["data"], run=run)

    def verify(
        self, ind: StoredIndividual, result: dict[str, Any],
    ) -> tuple[float, float] | None:
        finals = ind.tags.get("final_eval_fitness") or []
        ep = self._episode()
        if self.options.get("weights") == "init" or ep >= len(finals):
            return None
        return float(finals[ep]), float(result["fitness"])
