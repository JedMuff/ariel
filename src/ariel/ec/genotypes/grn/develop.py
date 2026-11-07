"""Development of a body from a genetic regulatory network (GRN).

Port of the GRN of Miras and Feenstra, "Epistatic gene interactions in the
evolution of robots", IEEE Transactions on Evolutionary Computation, 2026
(doi:10.1109/TEVC.2026.3659703): ``geno_body_GRN_v3.py`` on branch
``plasticoding_v3`` of karinemiras/revolve2 (commit 593bfc2), included with the
author's permission. The revolve2 module classes are replaced by plain data;
the development itself (order of operations, constants, float arithmetic) is
unchanged, and ``tests/test_grn.py`` checks it against bodies grown by the
original code.

Genome: a list of values in [0, 1] (two decimals). The first is the
concentration threshold for growth; the rest is read left to right, and every
value below 0.8 is a promoter followed by a gene of 6 values: regulatory
factor, two regulatory bounds, product factor, amount and diffusion site.

Development: a maternal injection of the first gene's regulatory factor starts
the core cell; then for up to 100 steps every cell, oldest first, expresses its
genes, diffuses its factors within itself and to its neighbours, tries to grow
a module, and decays. A cell grows the module whose structural factor (brick
or hinge) is most concentrated, if above the threshold, at its free slot with
the most of that factor, unless the 2D grid cell there is taken; a hinge is
rotated by 90 degrees when the rotation factor exceeds 0.5. Development stops
when the body has max_modules modules, the core included.

Slots follow revolve2: core 0 front, 1 right, 2 left, 3 back; brick 0 front,
1 right, 2 left; hinge 0 (its attachment). Diffusion sites use the same
indices (3 = back, towards the parent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FRONT, RIGHT, LEFT, BACK = 0, 1, 2, 3
N_SITES = 4
N_SLOTS = {"CORE": 4, "BRICK": 3, "HINGE": 1}


@dataclass(frozen=True)
class GrnDesign:
    """A GRN design: its structural factors (one per module type, then rotation)
    and number of regulatory factors."""

    name: str
    structural: tuple[str, ...]
    regulatory: int
    genome_size: int = 151
    promoter_threshold: float = 0.8
    gene_length: int = 6
    concentration_decay: float = 0.005
    increase_scaling: float = 100.0
    dev_steps: int = 100

    @property
    def intra_diffusion(self) -> float:
        return self.concentration_decay / 2

    @property
    def inter_diffusion(self) -> float:
        return self.intra_diffusion / 8


# R2 ("reg2m2"): brick, hinge and rotation factors, two regulatory factors.
GRN_R2 = GrnDesign("reg2m2", ("BRICK", "HINGE", "rotation"), 2)


@dataclass
class Module:
    id: int
    type: str                       # CORE, BRICK or HINGE
    parent: "Module | None"
    slot: int | None                # slot in the parent
    rotation: int = 0               # 0, or 1 for 90 degrees (hinges, and bricks on rotated hinges)
    absolute_rotation: int = 0
    coords: tuple[int, int] = (0, 0)
    turtle: int = FRONT
    children: list = field(default_factory=list)
    cell: "Cell | None" = None


@dataclass
class Cell:
    module: Module | None = None
    factors: dict[str, list[float]] = field(default_factory=dict)
    original_genes: list[int] = field(default_factory=list)


def _label(value: float, limits: list[float], prefix: bool) -> str | int:
    """The original's interval lookup, kept as written (it reassigns the label
    for every interval, so the last applicable branch wins)."""
    label = None
    for idx in range(len(limits) - 1):
        if limits[idx] <= value < limits[idx + 1]:
            label = idx + 1 if prefix else idx
        elif value >= limits[idx + 1]:
            label = len(limits) if prefix else len(limits) - 1
    return f"TF{label}" if prefix else label


def parse_genes(values: list[float], design: GrnDesign) -> list[tuple]:
    """Genes as (regulatory label, v1, v2, product label, amount, site)."""
    total = len(design.structural) + design.regulatory
    tf_limits = [round(x, 2) for x in np.linspace(0, 1 - 1 / total, total)]
    site_limits = [round(x / 100, 2) for x in range(0, 100, int(100 / N_SITES))]
    genes, i = [], 0
    while i < len(values):
        if values[i] < design.promoter_threshold and len(values) - 1 - i >= design.gene_length:
            reg, v1, v2, product, amount, site = values[i + 1:i + 1 + design.gene_length]
            genes.append((_label(reg, tf_limits, True), v1, v2, _label(product, tf_limits, True), amount,
                          _label(site, site_limits, False)))
            i += design.gene_length
        i += 1
    return genes


class Development:
    def __init__(self, values: list[float], design: GrnDesign, max_modules: int) -> None:
        self.design = design
        self.max_modules = max_modules
        self.threshold = values[0]
        self.genes = parse_genes(values[1:], design)
        self.products = [f"TF{k}" for k in range(1, len(design.structural) + 1)]
        self.cells: list[Cell] = []
        self.grid: dict[tuple[int, int], Module] = {}
        self.n_modules = 0                 # modules besides the core
        self.core: Module | None = None

    # ── development ───────────────────────────────────────────────────────────

    def run(self) -> Module | None:
        """The developed core (None if the genome has no genes)."""
        if not self.genes:
            return None
        self.maternal_injection()
        for _ in range(self.design.dev_steps):
            for idx in range(len(self.cells)):
                cell = self.cells[idx]
                self.increase(cell)
                for tf in cell.factors:
                    self.intra_diffusion(tf, cell)
                    self.inter_diffusion(tf, cell)
                self.place_module(cell)
                if self.n_modules == self.max_modules - 1:
                    return self.core
                for tf in cell.factors:
                    self.decay(tf, cell)
        return self.core

    def maternal_injection(self) -> None:
        reg, v1 = self.genes[0][0], self.genes[0][1]
        cell = Cell()
        cell.factors[reg] = [v1 / N_SITES] * N_SITES
        self.express(cell)
        self.cells.append(cell)
        self.core = Module(0, "CORE", None, None, children=[None] * N_SLOTS["CORE"], cell=cell)
        cell.module = self.core
        self.grid[(0, 0)] = self.core

    def express(self, cell: Cell) -> None:
        for idg, (reg, v1, v2, product, amount, site) in enumerate(self.genes):
            if cell.factors.get(reg):
                total = sum(cell.factors[reg])
                if min(v1, v2) <= total <= max(v1, v2):
                    if cell.factors.get(product):
                        cell.factors[product][site] += amount
                    else:
                        cell.factors[product] = [0] * N_SITES
                        cell.factors[product][site] = amount
                    cell.original_genes.append(idg)

    def increase(self, cell: Cell) -> None:
        for idg, (reg, v1, v2, product, amount, site) in enumerate(self.genes):
            if idg in cell.original_genes and cell.factors.get(reg):
                total = sum(cell.factors[reg])
                if min(v1, v2) <= total <= max(v1, v2):
                    cell.factors[product][site] += amount / self.design.increase_scaling

    def intra_diffusion(self, tf: str, cell: Cell) -> None:
        rate, c = self.design.intra_diffusion, cell.factors[tf]
        for ds in range(N_SITES):
            left = ds - 1 if ds - 1 >= 0 else N_SITES - 1
            right = ds + 1 if ds + 1 <= N_SITES - 1 else 0
            if c[ds] >= rate:
                c[ds] -= rate
                c[right] += rate
            if c[ds] >= rate:
                c[ds] -= rate
                c[left] += rate

    @staticmethod
    def _give(target: Cell, tf: str, site: int, amount: float) -> None:
        if target.factors.get(tf):
            target.factors[tf][site] += amount
        else:
            target.factors[tf] = [0] * N_SITES
            target.factors[tf][site] += amount

    def inter_diffusion(self, tf: str, cell: Cell) -> None:
        rate, c, m = self.design.inter_diffusion, cell.factors[tf], cell.module
        for ds in range(N_SITES):
            if ds == BACK and m.type in ("HINGE", "BRICK"):
                # the back site of every module but the core shares with the parent
                if c[BACK] >= rate:
                    c[BACK] -= rate
                    self._give(m.parent.cell, tf, m.slot, rate)
            elif ds in (LEFT, FRONT, RIGHT) and m.type == "HINGE":
                # a hinge's slot-less sites also share with its single child
                if m.children[FRONT] is not None and c[ds] >= rate:
                    c[ds] -= rate
                    self._give(m.children[FRONT].cell, tf, BACK, rate)
            elif m.children[ds] is not None and c[ds] >= rate:
                c[ds] -= rate
                self._give(m.children[ds].cell, tf, BACK, rate)

    def decay(self, tf: str, cell: Cell) -> None:
        c = cell.factors[tf]
        for ds in range(N_SITES):
            c[ds] = max(0, c[ds] - self.design.concentration_decay)

    def place_module(self, cell: Cell) -> None:
        conc = [sum(cell.factors[p]) if cell.factors.get(p) else 0 for p in self.products[:-1]]
        idx_max = conc.index(max(conc))
        rotation_conc = sum(cell.factors[self.products[-1]]) if cell.factors.get(self.products[-1]) else 0
        if not conc[idx_max] > self.threshold:
            return
        m = cell.module
        free = np.array([c is None for c in m.children] + [False] * (N_SITES - len(m.children)))
        if not free.any():
            return
        true_idx = np.where(free)[0]
        slot = int(true_idx[np.argmax(np.array(cell.factors[self.products[idx_max]])[true_idx])])
        coords, turtle = self.coordinates(m, slot)
        if coords in self.grid:
            return
        mtype = self.design.structural[idx_max]
        orientation = 1 if rotation_conc > 0.5 and mtype == "HINGE" else 0
        parent_rotated = m.type == "HINGE" and m.absolute_rotation == 1
        if mtype == "HINGE" and orientation == 1:
            absolute = 0 if parent_rotated else 1
        else:
            absolute = 1 if parent_rotated else 0
        if mtype == "BRICK" and parent_rotated:
            orientation = 1
        self.n_modules += 1
        new = Module(self.n_modules, mtype, m, slot, orientation, absolute, coords, turtle,
                     children=[None] * N_SLOTS[mtype])
        m.children[slot] = new
        self.grid[coords] = new
        self.new_cell(cell, new, slot)

    def new_cell(self, source: Cell, module: Module, slot: int) -> None:
        cell = Cell()
        for tf in source.factors:
            s = source.factors[tf]
            if source.module.type == "HINGE":
                # a hinge passes on its slot-less sites too
                conc = []
                for site in (LEFT, FRONT, RIGHT):
                    if s[site] > 0:
                        half = s[site] / 2
                        s[site] = half
                        conc.append(half)
                if conc:
                    cell.factors[tf] = [0, 0, 0, 0]
                    cell.factors[tf][BACK] = sum(conc) / len(conc)
            elif s[slot] > 0:
                half = s[slot] / 2
                s[slot] = half
                cell.factors[tf] = [0, 0, 0, 0]
                cell.factors[tf][BACK] = half
        self.express(cell)
        self.cells.append(cell)
        cell.module = module
        module.cell = cell

    @staticmethod
    def coordinates(parent: Module, slot: int) -> tuple[tuple[int, int], int]:
        """Grid cell and heading of a module at the parent's slot (turtle
        geometry in the plane; headings in the core's slot indices)."""
        to_turn = {FRONT: 0, LEFT: 1, BACK: 2, RIGHT: 3}
        from_turn = {0: FRONT, 1: LEFT, 2: BACK, 3: RIGHT}
        turtle = from_turn[(to_turn[parent.turtle] + to_turn[slot]) % 4]
        x, y = parent.coords
        step = {LEFT: (-1, 0), RIGHT: (1, 0), FRONT: (0, 1), BACK: (0, -1)}[turtle]
        return (x + step[0], y + step[1]), turtle


def develop(values: list[float], design: GrnDesign = GRN_R2, max_modules: int = 26) -> Module | None:
    """The body grown from a GRN genome, as a tree of Modules from the core
    (None if the genome has no genes). max_modules counts the core."""
    return Development(list(values), design, max_modules).run()


def modules_of(core: Module | None) -> list[Module]:
    """Every module, in creation (id) order."""
    if core is None:
        return []
    out, queue = [], [core]
    while queue:
        m = queue.pop(0)
        out.append(m)
        queue += [c for c in m.children if c is not None]
    return sorted(out, key=lambda m: m.id)
