"""Custom configuration for ARIEL-robots body phenotypes.

Notes
-----
    * Imports enums from ARIEL to ensure compatibility
    * CORE module excludes TOP and BOTTOM attachment points
    * All rotations allowed for BRICK and HINGE modules

"""

# Import ARIEL's enums to ensure we use the same enum instances
from ariel.body_phenotypes.robogen_lite.config import (
    ModuleType,
    ModuleFaces,
    ModuleRotationsIdx,
)


# Define allowed faces for each module type
# NOTE: CORE excludes TOP and BOTTOM attachment points
ALLOWED_FACES: dict[ModuleType, list[ModuleFaces]] = {
    ModuleType.CORE: [
        ModuleFaces.FRONT,
        ModuleFaces.BACK,
        ModuleFaces.RIGHT,
        ModuleFaces.LEFT,
    ],
    ModuleType.BRICK: [
        ModuleFaces.FRONT,
        ModuleFaces.RIGHT,
        ModuleFaces.LEFT,
        ModuleFaces.TOP,
        ModuleFaces.BOTTOM,
    ],
    ModuleType.HINGE: [ModuleFaces.FRONT],
    ModuleType.NONE: [],
}

# Define allowed rotations for each module type
ALLOWED_ROTATIONS: dict[ModuleType, list[ModuleRotationsIdx]] = {
    ModuleType.CORE: [ModuleRotationsIdx.DEG_0],
    ModuleType.BRICK: [
        ModuleRotationsIdx.DEG_0,
        ModuleRotationsIdx.DEG_45,
        ModuleRotationsIdx.DEG_90,
        ModuleRotationsIdx.DEG_135,
        ModuleRotationsIdx.DEG_180,
        ModuleRotationsIdx.DEG_225,
        ModuleRotationsIdx.DEG_270,
        ModuleRotationsIdx.DEG_315,
    ],
    ModuleType.HINGE: [
        ModuleRotationsIdx.DEG_0,
        ModuleRotationsIdx.DEG_45,
        ModuleRotationsIdx.DEG_90,
        ModuleRotationsIdx.DEG_135,
        ModuleRotationsIdx.DEG_180,
        ModuleRotationsIdx.DEG_225,
        ModuleRotationsIdx.DEG_270,
        ModuleRotationsIdx.DEG_315,
    ],
    ModuleType.NONE: [ModuleRotationsIdx.DEG_0],
}
