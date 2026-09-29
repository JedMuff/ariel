# Vendored from kgd-al/apets-ariel@70a79fb6cad355eee5ed332984615145e13cc363
#   src/aapets/common/canonical_bodies.py
# Only the module imports changed (-> zoo_modules). Do not edit by hand; re-vendor.
# ruff: noqa
"""
Implements the canonical bodies from revolve2

Warning: while the joints have been respected (axis of rotation), their orientation is not guaranteed. These robots
should have the same range of movements as their original counterparts but controllers are **not**
expected to transfer (which should be obvious).

Includes robots from https://arxiv.org/abs/2203.03967 + ant & park (because?) and rotated spider (spider45)
"""

import math
import sys
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Type, Tuple, Generator, Optional

import numpy as np

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from zoo_modules import BrickModule
from zoo_modules import CoreModule
from zoo_modules import HingeModule

current_module = sys.modules[__name__]

Hinge, Brick = HingeModule, BrickModule

SUBMODULES = Hinge | Brick
MODULES = CoreModule | SUBMODULES

faces = [
    ModuleFaces.FRONT, ModuleFaces.LEFT, ModuleFaces.BACK, ModuleFaces.RIGHT,
]
F, L, B, R = faces


def get(name: "str | CanonicalBodies") -> CoreModule:
    if isinstance(name, CanonicalBodies):
        name = name.value
    body_fn = getattr(current_module, f"body_{name}", None)
    if body_fn is None:
        raise RuntimeError(f"'{name}' is not a valid canonical body name")

    return body_fn()


def get_all() -> dict[str, Callable]:
    return {
        name[5:]: getattr(current_module, name)
        for name in current_module.__dir__()
        if name.startswith("body_")
    }


_DEFAULT_COLORS = {
    CoreModule: (["core"], (.1, .1, .9)),
    Hinge: (["stator", "rotor"], (.9, .1, .1)),
    Brick: (["brick"], (.1, .7, .1)),
}


def apply_color(colors, module):
    geom_names, geom_color = colors[module.__class__]
    for name in geom_names:
        module.spec.geom(name).rgba = (*geom_color, 1)


def make_core():
    core = CoreModule()
    core.name = "C"
    apply_color(_DEFAULT_COLORS, core)
    return core


class Attacher:
    def __init__(
            self,
            colors=None):
        self.colors = colors or _DEFAULT_COLORS

    def __call__(self,
                 parent: MODULES,
                 face: ModuleFaces,
                 module_t: Type[SUBMODULES],
                 name: Optional[str] = None,
                 rotation: float = 0):

        module = module_t()
        apply_color(self.colors, module)

        if name is None:
            name = module_t.__name__[0]
            if face != ModuleFaces.FRONT:
                name = face.name[0].lower() + name
        name = f"{parent.name}-{name}"

        parent.sites[face].attach_body(body=module.body, prefix=name + "-")

        module.name = name

        if rotation != 0:
            module.rotate(rotation)

        return module

    @contextmanager
    def branch(self, *args, **kwargs):
        yield self.__call__(*args, **kwargs)


def body_spider() -> CoreModule:
    core, attacher = make_core(), Attacher()

    for f in faces:
        h0 = attacher(core, f, Hinge)
        b0 = attacher(h0, F, Brick)
        h1 = attacher(b0, F, Hinge, rotation=90)
        b1 = attacher(h1, F, Brick)

    return core


def body_spider45() -> CoreModule:
    core = body_spider()
    core.spec.body("core").quat = (math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8))
    return core


def body_gecko() -> CoreModule:
    core, attacher = make_core(), Attacher()

    al = attacher(core, L, Hinge, "LH", rotation=90)
    attacher(al, F, Brick, "LB")

    ar = attacher(core, R, Hinge, "RH", rotation=90)
    attacher(ar, F, Brick, "RB")

    sh0 = attacher(core, B, Hinge, "S")
    sb0 = attacher(sh0, F, Brick, "S")
    sh1 = attacher(sb0, F, Hinge, "S")
    sb1 = attacher(sh1, F, Brick, "S")

    ll = attacher(sb1, L, Hinge, "LH", rotation=90)
    attacher(ll, F, Brick, "LB")

    lr = attacher(sb1, R, Hinge, "RH", rotation=90)
    attacher(lr, F, Brick, "RB")

    return core


def body_babya() -> CoreModule:
    core, attacher = make_core(), Attacher()

    al = attacher(core, L, Hinge, "LH", rotation=90)
    attacher(al, F, Brick, "LB")

    h0 = attacher(core, R, Hinge, f"RH", rotation=90)
    h1 = attacher(h0, F, Hinge, "H", rotation=-90)
    b0 = attacher(h1, F, Brick, "B")
    h2 = attacher(b0, F, Hinge, "H", rotation=90)
    b1 = attacher(h2, F, Brick, "B")

    sh0 = attacher(core, B, Hinge, "S")
    sb0 = attacher(sh0, F, Brick, "S")
    sh1 = attacher(sb0, F, Hinge, "S")
    sb1 = attacher(sh1, F, Brick, "S")

    ll = attacher(sb1, L, Hinge, "LH", rotation=90)
    attacher(ll, F, Brick, "LB")

    lr = attacher(sb1, R, Hinge, "RH", rotation=90)
    attacher(lr, F, Brick, "RB")

    return core


def body_ant() -> CoreModule:
    core, attacher = make_core(), Attacher()

    def limb(src, face):
        _h = attacher(src, face, Hinge, rotation=90)
        attacher(_h, F, Brick)

    def limbs(src):
        limb(src, L)
        limb(src, R)

    sh0 = attacher(core, B, Hinge)
    sb0 = attacher(sh0, F, Brick)
    sh1 = attacher(sb0, F, Hinge)
    sb1 = attacher(sh1, F, Brick)

    limbs(core)
    limbs(sb0)
    limbs(sb1)

    return core


def body_salamander() -> CoreModule:
    core, attacher = make_core(), Attacher()

    # left arm
    l_h = attacher(core, L, Hinge)
    attacher(l_h, F, Hinge, rotation=90)

    # right arm
    attacher(core, R, Hinge, rotation=90)

    # first block (two bricks, two hinges)
    b_h = attacher(core, B, Hinge)
    with attacher.branch(b_h, F, Brick) as b_hb:
        attacher(b_hb, L, Hinge, "bHBrH", rotation=90)
    with attacher.branch(b_hb, F, Brick) as b_hbb:
        attacher(b_hbb, L, Hinge, "LH", rotation=90)
    b_hbbh = attacher(b_hbb, F, Hinge)

    # second block (four bricks, one 'limb', two hinges)
    with attacher.branch(b_hbbh, F, Brick) as b_hbbh_b:
        b_hbbh_l_h = attacher(b_hbbh_b, L, Hinge, "bHBBHlH", rotation=90)
        with attacher.branch(b_hbbh_l_h, F, Brick, "bHBBHlHB", rotation=-90) as b_hbbh_l_hb:
            attacher(b_hbbh_l_hb, L, Brick, "bHBBHlHBlB")
        b_hbbh_l_hbh = attacher(b_hbbh_l_hb, F, Hinge, "bHBBHlHBH")
        attacher(b_hbbh_l_hbh, F, Hinge, "bHBBHlHBHH", rotation=90)
    with attacher.branch(b_hbbh_b, F, Brick) as b_hbbh_bb:
        attacher(b_hbbh_bb, L, Hinge, "bHBBBH_BBH", rotation=90)
    with attacher.branch(b_hbbh_bb, F, Brick) as b_hbbh_bbb:
        attacher(b_hbbh_bbb, L, Hinge, "bHBBBH_BBBH", rotation=90)
    b_hbbh_bbbb = attacher(b_hbbh_bbb, F, Brick)

    # tail
    b_hbbh_bbbb_h = attacher(b_hbbh_bbbb, F, Hinge)
    with attacher.branch(b_hbbh_bbbb_h, F, Brick) as b_hbbh_bbbb_hb:
        attacher(b_hbbh_bbbb_hb, L, Brick, "bHBBBH_BBBB_HBlB")
    b_hbbh_bbbb_hbh = attacher(b_hbbh_bbbb_hb, F, Hinge)

    return core


def body_blokky() -> CoreModule:
    core, attacher = make_core(), Attacher()

    attacher(core, L, Hinge)

    b_b = attacher(core, B, Brick)
    attacher(b_b, R, Hinge)
    b_bh = attacher(b_b, F, Hinge)
    b_bhh = attacher(b_bh, F, Hinge, rotation=90)

    b_bhh_b0 = attacher(b_bhh, F, Brick, "tail_B0", rotation=-90)

    b_bhh_b1 = attacher(b_bhh_b0, F, Brick, "tail_B1")
    b_bhh_b2 = attacher(b_bhh_b1, R, Brick, "tail_B2")
    b_bhh_b3 = attacher(b_bhh_b2, L, Brick, "tail_B3")
    b_bhh_b4 = attacher(b_bhh_b2, F, Brick, "tail_B4")

    b_bhh_b5 = attacher(b_bhh_b0, R, Brick, "tail_B5")
    b_bhh_b6 = attacher(b_bhh_b5, F, Brick, "tail_B6")
    b_bhh_b7 = attacher(b_bhh_b6, R, Brick, "tail_B7")
    b_bhh_b8 = attacher(b_bhh_b6, F, Hinge, "tail_H", rotation=90)

    return core


def body_park() -> CoreModule:
    core, attacher = make_core(), Attacher()

    h = attacher(core, B, Hinge)
    hh = attacher(h, F, Hinge, rotation=90)

    hhb = attacher(hh, F, Brick, rotation=-90)
    attacher(hhb, R, Brick)
    attacher(hhb, L, Hinge, rotation=90)
    hhbb = attacher(hhb, F, Brick)

    attacher(hhbb, R, Hinge)
    attacher(hhbb, F, Hinge)
    hhbbh = attacher(hhbb, L, Hinge, rotation=90)

    hhbbhb = attacher(hhbbh, F, Brick, rotation=90)
    attacher(hhbbhb, R, Brick)
    attacher(hhbbhb, L, Hinge)
    hhbbhbh = attacher(hhbbhb, F, Hinge, rotation=90)
    hhbbhbhb = attacher(hhbbhbh, F, Brick, rotation=-90)

    return core


def body_babyb() -> CoreModule:
    core, attacher = make_core(), Attacher()

    for f in faces:
        h0 = attacher(core, f, Hinge)
        b0 = attacher(h0, F, Brick)
        if f is B:
            continue
        h1 = attacher(b0, F, Hinge, rotation=90)
        b1 = attacher(h1, F, Brick, rotation=-90)
        h2 = attacher(b1, F, Hinge)
        b2 = attacher(h2, F, Brick)

    return core


def body_garrix() -> CoreModule:
    core, attacher = make_core(), Attacher()

    attacher(core, F, Hinge)

    h0 = attacher(core, L, Hinge)
    h1 = attacher(h0, F, Hinge)
    h2 = attacher(h1, F, Hinge, rotation=90)
    with attacher.branch(h2, F, Brick, rotation=-90) as b0:
        attacher(b0, F, Brick)
    h3 = attacher(b0, L, Hinge, rotation=90)
    with attacher.branch(h3, F, Brick, rotation=-90) as b1:
        attacher(b1, R, Hinge)
        attacher(b1, F, Hinge)
    h4 = attacher(b1, L, Hinge, rotation=90)
    h5 = attacher(h4, F, Hinge, rotation=-90)
    h6 = attacher(h5, F, Hinge, rotation=90)
    b2 = attacher(h6, F, Brick, rotation=-90)

    return core


def body_insect() -> CoreModule:
    core, attacher = make_core(), Attacher()

    h0 = attacher(core, R, Hinge)
    h1 = attacher(h0, F, Hinge, rotation=90)
    with attacher.branch(h1, F, Brick, rotation=-90) as b0:
        attacher(b0, R, Hinge, rotation=90)
        attacher(b0, F, Hinge)
    h2 = attacher(b0, L, Hinge)
    with attacher.branch(h2, F, Brick) as b1:
        attacher(b1, F, Hinge)
    h3 = attacher(b1, R, Hinge, rotation=90)
    h4 = attacher(h3, F, Hinge)
    h5 = attacher(h4, F, Hinge, rotation=-90)

    return core


def body_linkin() -> CoreModule:
    core, attacher = make_core(), Attacher()

    attacher(core, B, Hinge, rotation=90)

    h0 = attacher(core, R, Hinge)
    h1 = attacher(h0, F, Hinge)
    h2 = attacher(h1, F, Hinge)
    h3 = attacher(h2, F, Hinge, rotation=90)
    b0 = attacher(h3, F, Brick, rotation=-90)
    with attacher.branch(b0, L, Hinge, rotation=90) as lh:
        attacher(lh, F, Hinge)
    attacher(b0, F, Brick)
    with attacher.branch(b0, R, Hinge) as rh0:
        rh1 = attacher(rh0, F, Hinge, rotation=90)
        rh2 = attacher(rh1, F, Hinge)
        rh3 = attacher(rh2, F, Hinge, rotation=-90)
        rh4 = attacher(rh3, F, Hinge)

    return core


def body_longleg() -> CoreModule:
    core, attacher = make_core(), Attacher()

    h0 = attacher(core, L, Hinge)
    h1 = attacher(h0, F, Hinge)
    h2 = attacher(h1, F, Hinge)
    h3 = attacher(h2, F, Hinge, rotation=90)
    h4 = attacher(h3, F, Hinge)
    with attacher.branch(h4, F, Brick, rotation=-90) as b0:
        attacher(b0, R, Hinge, rotation=90)
        attacher(b0, F, Hinge, rotation=90)
    h5 = attacher(b0, L, Hinge)
    h6 = attacher(h5, F, Hinge, rotation=90)
    with attacher.branch(h6, F, Brick, rotation=-90) as b1:
        attacher(b1, R, Hinge)
    h7 = attacher(b1, L, Hinge)
    h8 = attacher(h7, F, Hinge)

    return core


def body_penguin() -> CoreModule:
    core, attacher = make_core(), Attacher()

    b0 = attacher(core, R, Brick)
    h0 = attacher(b0, L, Hinge)
    h1 = attacher(h0, F, Hinge, rotation=90)
    with attacher.branch(h1, F, Brick, rotation=-90) as b1:
        attacher(b1, R, Hinge, rotation=90)
    h2 = attacher(b1, L, Hinge)
    h3 = attacher(h2, F, Hinge, rotation=90)
    h4 = attacher(h3, F, Hinge, rotation=-90)
    with attacher.branch(h4, F, Brick) as b2:
        lh = attacher(b2, F, Hinge)
        attacher(lh, F, Brick)
    h5 = attacher(b2, R, Hinge, rotation=90)
    h6 = attacher(h5, F, Hinge)
    h7 = attacher(h6, F, Hinge, rotation=-90)
    with attacher.branch(h7, F, Brick) as b3:
        attacher(b3, L, Hinge)
    b4 = attacher(b3, R, Brick)
    h8 = attacher(b4, F, Hinge)

    return core


def body_pentapod() -> CoreModule:
    core, attacher = make_core(), Attacher()

    h0 = attacher(core, R, Hinge)
    h1 = attacher(h0, F, Hinge)
    h2 = attacher(h1, F, Hinge)
    h3 = attacher(h2, F, Hinge, rotation=90)
    with attacher.branch(h3, F, Brick, rotation=-90) as b0:
        attacher(b0, L, Hinge, rotation=90)
    h4 = attacher(b0, F, Hinge)
    with attacher.branch(h4, F, Brick) as b1:
        attacher(b1, L, Brick)
        attacher(b1, R, Hinge, rotation=90)
    h5 = attacher(b1, F, Hinge)
    with attacher.branch(h5, F, Brick) as b2:
        attacher(b2, L, Hinge, rotation=90)
        attacher(b2, R, Hinge, rotation=90)

    return core


def body_queen() -> CoreModule:
    core, attacher = make_core(), Attacher()

    attacher(core, B, Hinge)

    h0 = attacher(core, R, Hinge)
    h1 = attacher(h0, F, Hinge)
    h2 = attacher(h1, F, Hinge, rotation=90)
    with attacher.branch(h2, F, Brick, rotation=-90) as b0:
        attacher(b0, L, Hinge, rotation=90)
    with attacher.branch(b0, R, Brick) as b1:
        with attacher.branch(b1, F, Brick) as bf:
            attacher(bf, L, Hinge, rotation=90)
            attacher(bf, R, Hinge, rotation=90)
        b2 = attacher(b1, R, Brick)
        h3 = attacher(b2, F, Hinge)
        h4 = attacher(h3, F, Hinge)

    return core


def body_squarish() -> CoreModule:
    core, attacher = make_core(), Attacher()

    h0 = attacher(core, B, Hinge, rotation=90)
    with attacher.branch(h0, F, Brick, rotation=-90) as b0:
        attacher(b0, F, Hinge, rotation=90)
    h1 = attacher(b0, L, Hinge)
    b1 = attacher(h1, F, Brick)
    with attacher.branch(b1, L, Brick) as b2:
        attacher(b2, L, Hinge)
        attacher(b2, F, Hinge, rotation=90)
    h2 = attacher(b2, R, Hinge)
    b3 = attacher(h2, F, Brick)
    b4 = attacher(b3, L, Brick)
    b5 = attacher(b4, L, Brick)

    return core


def body_snake() -> CoreModule:
    core, attacher = make_core(), Attacher()

    prev = attacher(core, L, Hinge, rotation=90)
    for _ in range(7):
        b = attacher(prev, F, Brick)
        prev = attacher(b, F, Hinge, rotation=90)

    return core


def body_stingray() -> CoreModule:
    core, attacher = make_core(), Attacher()

    attacher(core, B, Hinge)

    h0 = attacher(core, R, Hinge)
    h1 = attacher(h0, F, Hinge, rotation=90)
    with attacher.branch(h1, F, Brick, rotation=-90) as b0:
        attacher(b0, L, Hinge, rotation=90)
        attacher(b0, R, Brick)
    with attacher.branch(b0, F, Brick) as b1:
        attacher(b1, R, Hinge)
        attacher(b1, F, Hinge)
    h2 = attacher(b1, L, Hinge, rotation=90)
    with attacher.branch(h2, F, Brick, rotation=-90) as b2:
        attacher(b2, R, Hinge)
    h3 = attacher(b2, F, Hinge, rotation=90)
    b3 = attacher(h3, F, Brick, rotation=-90)

    return core


def body_tinlicker() -> CoreModule:
    core, attacher = make_core(), Attacher()

    h0 = attacher(core, R, Hinge)
    h1 = attacher(h0, F, Hinge)
    h2 = attacher(h1, F, Hinge)
    h3 = attacher(h2, F, Hinge, rotation=90)
    b0 = attacher(h3, F, Brick, rotation=-90)
    with attacher.branch(b0, L, Brick) as b1:
        attacher(b1, F, Hinge)
    with attacher.branch(b1, R, Brick) as b2:
        attacher(b2, L, Brick)
    h4 = attacher(b2, F, Hinge, rotation=90)
    with attacher.branch(h4, F, Brick, rotation=-90) as b3:
        attacher(b3, F, Hinge)
    b4 = attacher(b3, R, Brick)
    h5 = attacher(b4, R, Hinge)

    return core


def body_turtle() -> CoreModule:
    core, attacher = make_core(), Attacher()

    with attacher.branch(core, L, Brick) as b0:
        attacher(b0, R, Hinge, rotation=90)
    h0 = attacher(b0, L, Hinge)
    h1 = attacher(h0, F, Hinge, rotation=90)
    with attacher.branch(h1, F, Brick, rotation=-90) as b1:
        attacher(b1, L, Hinge)
        attacher(b1, F, Brick)
    h2 = attacher(b1, R, Hinge, rotation=90)
    with attacher.branch(h2, F, Brick, rotation=-90) as b2:
        attacher(b2, F, Brick)

        sh = attacher(b2, L, Hinge)
        attacher(sh, F, Hinge, rotation=90)

    h3 = attacher(b2, R, Hinge, rotation=90)
    with attacher.branch(h3, F, Brick, rotation=-90) as b3:
        attacher(b3, R, Hinge, rotation=90)
    h4 = attacher(b3, L, Hinge)
    h5 = attacher(h4, F, Hinge, rotation=90)
    h6 = attacher(h5, F, Hinge)
    h7 = attacher(h6, F, Hinge)

    return core


def body_ww() -> CoreModule:
    core, attacher = make_core(), Attacher()

    attacher(core, B, Hinge, rotation=90)

    h0 = attacher(core, R, Hinge)
    h1 = attacher(h0, F, Hinge)
    h2 = attacher(h1, F, Hinge, rotation=90)
    b0 = attacher(h2, F, Brick, rotation=-90)
    h3 = attacher(b0, L, Hinge, rotation=90)
    with attacher.branch(h3, F, Brick, rotation=-90) as b1:
        attacher(b1, L, Hinge, rotation=90)
    b2 = attacher(b1, F, Brick)
    h4 = attacher(b2, R, Hinge)
    b3 = attacher(h4, F, Brick)
    h5 = attacher(b3, L, Hinge)
    h6 = attacher(h5, F, Hinge)
    h7 = attacher(h6, F, Hinge, rotation=90)

    return core


def body_zappa() -> CoreModule:
    core, attacher = make_core(), Attacher()

    attacher(core, B, Hinge, rotation=90)

    h0 = attacher(core, R, Hinge)
    h1 = attacher(h0, F, Hinge)
    h2 = attacher(h1, F, Hinge)
    h3 = attacher(h2, F, Hinge, rotation=90)
    h4 = attacher(h3, F, Hinge)
    with attacher.branch(h4, F, Brick, rotation=-90) as b0:
        sh = attacher(b0, F, Hinge, rotation=90)
        attacher(sh, F, Hinge)
    h5 = attacher(b0, L, Hinge)
    with attacher.branch(h5, F, Brick) as b1:
        attacher(b1, F, Hinge, rotation=90)
    h6 = attacher(b1, L, Hinge, rotation=90)
    b2 = attacher(h6, F, Brick, rotation=-90)

    return core


CanonicalBodies = Enum('CanonicalBodies', [(k.upper(), k.lower()) for k in get_all()])
CanonicalBodies.get = lambda self: get(self)
