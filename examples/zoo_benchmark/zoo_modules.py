"""Robogen-lite modules vendored from kgd-al/ariel, as used by apets-ariel's zoo.

Source: https://github.com/kgd-al/ariel/tree/24c3f4109ae2b57136237a230892b3c22fc87797
        src/ariel/body_phenotypes/robogen_lite/modules/{module,core,brick,hinge}.py

Copied verbatim apart from imports: ArielModulesConfig's brick constants are
inlined and `printable` is fixed to True (kgd's default), so no TOP/BOTTOM
sites. These differ from the local ariel modules (masses, dimensions and the
hinge actuator: kp/kv/armature and a +-90 deg joint range), which is why they
are vendored here instead of reusing src/ariel. Do not edit by hand; re-vendor.
"""

# ruff: noqa

import typing
from abc import ABC
from functools import lru_cache

import mujoco
import numpy as np
import quaternion as qnp
from mujoco import MjsBody, MjsSite

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType

printable = True

type WeightType = float
type DimensionType = tuple[float, float, float]

# From kgd-al/ariel src/ariel/parameters/ariel_modules.py (ArielModulesConfig)
BRICK_MASS: WeightType = 0.055  # 55 grams
BRICK_DIMENSIONS: DimensionType = (0.0375, 0.0375, 0.0375)


# ======================================================================
# module.py
# ======================================================================

# Standard library


class Module(ABC):
    """Base class for all modules."""

    def __init__(self):
        self.body: typing.Optional[MjsBody] = None
        self.sites: typing.Mapping[ModuleFaces, MjsSite] = dict()

    @staticmethod
    def add_site(body: MjsBody, *args, **kwargs):
        return body.add_site(*args, **kwargs, group=5)

    @property
    def children(self):
        for site in self.sites.values():
            print(site)
        return {}

    def rotate(
        self,
        angle: float,
    ) -> None:
        """
        Rotate the brick module by a specified angle.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the brick.
        """
        # Convert angle to quaternion
        quat = qnp.from_euler_angles([
            np.deg2rad(180),
            -np.deg2rad(180 - angle),
            np.deg2rad(0),
        ])
        quat = np.roll(qnp.as_float_array(quat), shift=-1)

        # Set the quaternion for the brick body
        self.body.quat = np.round(quat, decimals=3)


# ======================================================================
# core.py
# ======================================================================

# Third-party libraries

# Local libraries

# Type Aliases

# --- Robogen Configuration --- #
# Module weights (kg)
CORE_MASS: WeightType = 1.363

# Module dimensions (length, width, height) in meters
CORE_DIMENSIONS: DimensionType = (0.075, 0.075, 0.075)
# ------------------------------ #


class CoreModule(Module):
    """Core module specifications."""

    module_type: ModuleType = ModuleType.CORE

    def __init__(self) -> None:
        super().__init__()

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # ========= Core =========
        core_name = self.module_type.name.lower()
        core = spec.worldbody.add_body(
            name=core_name,
        )
        core.add_geom(
            name=core_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=CORE_MASS,
            size=CORE_DIMENSIONS,
            # pos=[0, CORE_DIMENSIONS[0], 0],
            pos=[0, 0, 0],
            rgba=(253 / 255, 202 / 255, 64 / 255, 1),
        )

        core.add_camera(
            name=f"{core_name}_mycamera",
            pos=[0, 0, CORE_DIMENSIONS[0]-0.02],
            euler=[-90, 0, 180],
        )

        # ========= Attachment Points =========
        self.sites = {}
        shift = -1  # mujoco uses xyzw instead of wxyz
        self.sites[ModuleFaces.FRONT] = core.add_site(
            name=f"{core_name}-front",
            pos=[CORE_DIMENSIONS[0], 0, -CORE_DIMENSIONS[1] / 2],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(90),
                            np.deg2rad(90),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.BACK] = self.add_site(
            core,
            name=f"{core_name}-back",
            pos=[-CORE_DIMENSIONS[0], 0, -CORE_DIMENSIONS[1] / 2],

            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(90),
                            -np.deg2rad(90),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.LEFT] = self.add_site(
            core,
            name=f"{core_name}-left",
            pos=[0, CORE_DIMENSIONS[1], -CORE_DIMENSIONS[1] / 2],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(180),
                            np.deg2rad(180),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.RIGHT] = self.add_site(
            core,
            name=f"{core_name}-right",
            pos=[0, -CORE_DIMENSIONS[1], -CORE_DIMENSIONS[1] / 2],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(0),
                            np.deg2rad(0),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),

        )

        if not printable:
            self.sites[ModuleFaces.TOP] = self.add_site(
                core,
                name=f"{core_name}-top",
                pos=[0, 0, CORE_DIMENSIONS[2]],
                quat=np.round(
                    np.roll(
                        qnp.as_float_array(
                            qnp.from_euler_angles([
                                np.deg2rad(0),
                                np.deg2rad(180),
                                np.deg2rad(90),
                            ]),
                        ),
                        shift=shift,
                    ),
                    decimals=3,
                ),
            )
            self.sites[ModuleFaces.BOTTOM] = self.add_site(
                core,
                name=f"{core_name}-bottom",
                pos=[0, 0, -CORE_DIMENSIONS[2]],
                quat=np.round(
                    np.roll(
                        qnp.as_float_array(
                            qnp.from_euler_angles([
                                np.deg2rad(0),
                                np.deg2rad(0),
                                -np.deg2rad(90),
                            ]),
                        ),
                        shift=shift,
                    ),
                    decimals=3,
                ),
            )

        # Save model specifications
        self.spec = spec

    def rotate(self, angle: float) -> None:
        """
        Rotate the core module by a specified angle.

        Parameters
        ----------
        angle : float
            The angle in radians to rotate the core.

        Raises
        ------
        AttributeError
            Core module does not support rotation.
        """
        if angle != 0:
            msg = f"Attempted to rotate the core module by: {angle}."
            msg += f"Core ({self.index}) module does not support rotation."
            raise AttributeError(msg)

    @property
    @lru_cache
    def hinges(self):
        return self.spec.actuators


# ======================================================================
# brick.py
# ======================================================================

# Third-party libraries

# Local libraries

# Global functions


class BrickModule(Module):
    """Brick module specifications."""

    module_type: ModuleType = ModuleType.BRICK

    def __init__(self) -> None:
        super().__init__()

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # ========= BRICK =========
        brick_name = self.module_type.name.lower()
        brick = spec.worldbody.add_body(
            name=brick_name,
        )
        brick.add_geom(
            name=brick_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=BRICK_MASS,
            size=BRICK_DIMENSIONS,
            pos=[0, BRICK_DIMENSIONS[0], 0],
            rgba=(28 / 255, 119 / 255, 195 / 255, 1),
        )

        # ========= Attachment Points =========
        self.sites = {}
        shift = -1  # mujoco uses xyzw instead of wxyz
        self.sites[ModuleFaces.FRONT] = self.add_site(
            brick,
            name=f"{brick_name}-front",
            pos=[0, BRICK_DIMENSIONS[1] * 2, 0],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(0),
                            np.deg2rad(180),
                            np.deg2rad(180),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.LEFT] = self.add_site(
            brick,
            name=f"{brick_name}-left",
            pos=[
                -BRICK_DIMENSIONS[0],
                BRICK_DIMENSIONS[1],
                0,
            ],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(90),
                            -np.deg2rad(90),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )
        self.sites[ModuleFaces.RIGHT] = self.add_site(
            brick,
            name=f"{brick_name}-right",
            pos=[
                BRICK_DIMENSIONS[0],
                BRICK_DIMENSIONS[1],
                0,
            ],
            quat=np.round(
                np.roll(
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(90),
                            np.deg2rad(90),
                            -np.deg2rad(90),
                        ]),
                    ),
                    shift=shift,
                ),
                decimals=3,
            ),
        )

        if not printable:
            self.sites[ModuleFaces.TOP] = self.add_site(
                brick,
                name=f"{brick_name}-top",
                pos=[
                    0,
                    BRICK_DIMENSIONS[1],
                    BRICK_DIMENSIONS[2],
                ],
                quat=np.round(
                    np.roll(
                        qnp.as_float_array(
                            qnp.from_euler_angles([
                                np.deg2rad(0),
                                np.deg2rad(180),
                                np.deg2rad(90),
                            ]),
                        ),
                        shift=shift,
                    ),
                    decimals=3,
                ),
            )
            self.sites[ModuleFaces.BOTTOM] = self.add_site(
                brick,
                name=f"{brick_name}-bottom",
                pos=[
                    0,
                    BRICK_DIMENSIONS[1],
                    -BRICK_DIMENSIONS[2],
                ],
                quat=np.round(
                    np.roll(
                        qnp.as_float_array(
                            qnp.from_euler_angles([
                                np.deg2rad(0),
                                np.deg2rad(0),
                                -np.deg2rad(90),
                            ]),
                        ),
                        shift=shift,
                    ),
                    decimals=3,
                ),
            )

        # Save model specifications
        self.spec = spec
        self.body = brick
        self.rotate(angle=0)  # Initialize with no rotation


# ======================================================================
# hinge.py
# ======================================================================

# Third-party libraries

# Local libraries

# Global constants
SHRINK = 0.99

# Type Aliases

# --- Robogen Configuration --- #
# Module weights (kg)
STATOR_MASS: WeightType = 0.065  # 20 grams
ROTOR_MASS: WeightType = 0.040  # 40 grams

# Module dimensions (length, width, height) in meters
STATOR_DIMENSIONS: DimensionType = (0.026, 0.022, 0.026)
ROTOR_DIMENSIONS: DimensionType = (0.026, 0.0155, 0.026)
# ------------------------------ #


# HINGE_KP = 4.0   # 1  # Ariel says `1`, revolve said 5
# HINGE_KV = 0.5  # 0.05  # 1  # Same
# HINGE_ARMATURE = 0.1  # 0.002  # Armature, new value for ariel
# After Optuna rough optimisation (single hinge + brick)
HINGE_KP = 1.3605113362038874
HINGE_KV = 0.3555497840137177
HINGE_ARMATURE = 0.08117047736037042
CTRL_RANGE = (-np.pi / 2, np.pi / 2)  # [-90, 90] degrees (range of 180)


class HingeModule(Module):
    """Hinge module specifications."""

    module_type: ModuleType = ModuleType.HINGE

    def __init__(self) -> None:
        super().__init__()

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # ========= Hinge =========
        hinge_name = self.module_type.name.lower()
        hinge = spec.worldbody.add_body(
            name=hinge_name,
            mass=STATOR_MASS + ROTOR_MASS,
        )

        # ========= Stator =========
        stator_name = "stator"
        stator = hinge.add_body(
            name=stator_name,
            pos=[0, STATOR_DIMENSIONS[1], 0],
        )
        stator.add_geom(
            name=stator_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=STATOR_MASS,
            size=np.array(STATOR_DIMENSIONS) * SHRINK,  # z-fighting
            rgba=(223 / 255, 41 / 255, 53 / 255, 1),
        )

        # ========= Rotor =========
        rotor_name = "rotor"
        rotor = hinge.add_body(
            name=rotor_name,
            pos=[0, STATOR_DIMENSIONS[1] * 2 + ROTOR_DIMENSIONS[1], 0],
        )
        rotor.add_geom(
            name=rotor_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=ROTOR_MASS,
            size=ROTOR_DIMENSIONS,
            rgba=(160 / 255, 24 / 255, 33 / 255, 1),
        )

        # ======== Attachment Points =========
        self.sites = {}
        self.sites[ModuleFaces.FRONT] = self.add_site(
            rotor,
            name=f"{hinge_name}-front",
            pos=[0, ROTOR_DIMENSIONS[1], 0],
        )

        # ========= Servo =========
        # Robot actuators
        kp = HINGE_KP
        kv = HINGE_KV
        a = HINGE_ARMATURE
        servo_axis = (0, 0, 1)

        servo_name = "servo"
        rotor.add_joint(
            name=servo_name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=servo_axis,
            pos=[0, -ROTOR_DIMENSIONS[1], 0],
            armature=a,
            range=(np.degrees(d) for d in CTRL_RANGE),
        )

        # Actuator parameters are defined over a range of 10...
        dynprm = np.zeros(10)
        gainprm = np.zeros(10)
        biasprm = np.zeros(10)

        # ... but only a few of the parameters are actually used
        gainprm[0] = kp
        biasprm[:3] = [0, -kp, -kv]

        # Contact exclusion
        spec.add_exclude(
            bodyname1=stator_name,
            bodyname2=rotor_name,
        )

        # --- Actuator(s) --- #
        dyntype = mujoco.mjtDyn.mjDYN_NONE
        gaintype = mujoco.mjtGain.mjGAIN_FIXED
        biastype = mujoco.mjtBias.mjBIAS_AFFINE
        trntype = mujoco.mjtTrn.mjTRN_JOINT
        spec.add_actuator(
            name=servo_name,
            dyntype=dyntype,
            gaintype=gaintype,
            biastype=biastype,
            dynprm=dynprm,
            gainprm=gainprm,
            biasprm=biasprm,
            trntype=trntype,
            target=servo_name,
            ctrlrange=CTRL_RANGE,
        )

        # Save model specifications
        self.spec = spec
        self.body = hinge
        self.rotate(angle=0)  # Initialize with no rotation
