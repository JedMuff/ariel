"""Browse ARIEL runs in EvoSight (https://github.com/JedMuff/evosight).

Pipeline, per run::

    database.db(+_partN.db) --reader--> RunData --adapter--> EvoSight local run
                                          |                     ^
                                          +--render (rollout)---+  videos, behaviour

* :mod:`.reader` — generic, read-only access to any ARIEL EA database.
* :mod:`.adapter` — :class:`ArielEvosightAdapter`, the per-experiment hooks
  (parents, objectives, descriptors, and ``make_rollout`` for re-simulation).
* :mod:`.rollout` / :mod:`.recorders` — replay an individual once and record
  a compressed video and its behavioural descriptors.
* :mod:`.selection` — which individuals to re-simulate.
* :mod:`.export` / :mod:`.render` — write the EvoSight run and its media.

Command line: ``python -m ariel.visualisation.evosight --help``. Needs the
``evosight`` dependency group (``uv sync --group evosight``).
"""

import os

# Rendering videos on a headless machine (cluster node, ssh) needs EGL, and
# MuJoCo picks its GL backend once, on import — so choose it before the
# imports below. An explicit MUJOCO_GL always wins.
if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

from ariel.visualisation.evosight.adapter import (
    ArielEvosightAdapter,
    load_adapter,
)
from ariel.visualisation.evosight.reader import (
    RunData,
    StoredIndividual,
    find_db_parts,
    load_run,
)
from ariel.visualisation.evosight.rollout import Rollout, play, stepped_rollout

__all__ = [
    "ArielEvosightAdapter",
    "Rollout",
    "RunData",
    "StoredIndividual",
    "find_db_parts",
    "load_adapter",
    "load_run",
    "play",
    "stepped_rollout",
]
