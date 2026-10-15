"""kgd's zoo modules (apets-ariel) for the ARIEL social-learning runs.

Ported from ariel_symmetry's examples/symmetry_pressure/shared.py
(``--zoo-modules``), which keeps these values in sync with the vendored
examples/zoo_benchmark/zoo_modules.py. Only the values differ from the local
modules: the local module classes, and so their face layout, are kept, and
every site position scales with the dimensions.

`use_zoo_modules_globally` switches the module-level constants the local
module classes read when instantiated, for the rest of the process, so the
genome operators' collision checks (random_tree, prune_colliding_subtrees)
use the zoo geometry too. The hinge actuator is hard-coded in hinge.py, so
`apply_zoo_modules` sets it on each built spec.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco

ZOO_HINGE_KP          = 1.3605113362038874
ZOO_HINGE_KV          = 0.3555497840137177
ZOO_HINGE_ARMATURE    = 0.08117047736037042
ZOO_HINGE_RANGE       = (-math.pi / 2, math.pi / 2)  # hard joint limit, same as the ctrl range
ZOO_CORE_MASS         = 1.363
ZOO_STATOR_MASS       = 0.065
ZOO_ROTOR_MASS        = 0.040
ZOO_BRICK_MASS        = 0.055
ZOO_CORE_DIMENSIONS   = (0.075, 0.075, 0.075)  # half-extents (m)
ZOO_BRICK_DIMENSIONS  = (0.0375, 0.0375, 0.0375)
ZOO_STATOR_DIMENSIONS = (0.026, 0.022, 0.026)
ZOO_ROTOR_DIMENSIONS  = (0.026, 0.0155, 0.026)


def _zoo_module_overrides() -> list[tuple[Any, str, Any]]:
    """(module, attribute, zoo value) for the module-level constants the local
    module classes read when they are instantiated."""
    from ariel.body_phenotypes.robogen_lite.modules import brick, core, hinge  # noqa: PLC0415

    brick_config = brick.ariel_modules_config.model_copy(
        update={"BRICK_MASS": ZOO_BRICK_MASS, "BRICK_DIMENSIONS": ZOO_BRICK_DIMENSIONS},
    )
    return [
        (core, "CORE_MASS", ZOO_CORE_MASS),
        (core, "CORE_DIMENSIONS", ZOO_CORE_DIMENSIONS),
        (hinge, "STATOR_MASS", ZOO_STATOR_MASS),
        (hinge, "ROTOR_MASS", ZOO_ROTOR_MASS),
        (hinge, "STATOR_DIMENSIONS", ZOO_STATOR_DIMENSIONS),
        (hinge, "ROTOR_DIMENSIONS", ZOO_ROTOR_DIMENSIONS),
        (brick, "ariel_modules_config", brick_config),
    ]


def _set_module_constants(values: list[tuple[Any, str, Any]]) -> None:
    from ariel.body_phenotypes.robogen_lite.collision_utils import module_template  # noqa: PLC0415

    for mod, name, value in values:
        setattr(mod, name, value)
    module_template.cache_clear()  # collision geometry is compiled from the module classes and cached


def use_zoo_modules_globally() -> None:
    """Zoo sizes and masses for the rest of this process."""
    _set_module_constants(_zoo_module_overrides())


def apply_zoo_modules(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Give a robot spec kgd's zoo hinge actuator (kp, kv, armature, a +-90 deg
    joint limit), in place."""
    for a in spec.actuators:
        a.gainprm[0] = ZOO_HINGE_KP
        a.biasprm[1] = -ZOO_HINGE_KP
        a.biasprm[2] = -ZOO_HINGE_KV
    scale = 180.0 / math.pi if spec.compiler.degree else 1.0
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.armature = ZOO_HINGE_ARMATURE
            j.range = [r * scale for r in ZOO_HINGE_RANGE]
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    return spec
