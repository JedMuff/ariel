"""Random-tied tree genomes: the dimensionality control for symmetry enforcement.

`symmetrize_genome` derives every mirrored module from its twin, so a
symmetric genome has fewer free genes than a plain tree. This module gives the
same kind of reduction without any geometric symmetry: when a module is
placed, with probability ``tie_prob`` a twin with the same type and rotation is
placed at a random free attachment site anywhere in the body (never at the
module's own mirror position about y=0). From then on the two share one gene:

* a point or rotation mutation of either changes both;
* a subtree replacement or shrink at a tied module is replayed at its twin,
  and the new modules there are tied to their counterparts;
* a hoist is replayed at the twin when the twin has a child on the same face;
* removing a module also removes its twin's subtree (one level: twins of the
  twin subtree's own modules are left in place);
* subtree crossover carries each swapped module's tie along.

Being tied is a heritable property of a module, like its type: a tied module
that loses its twin (its twin was cut away by crossover, a hoist, a one-level
removal or face pruning) gets a new twin at a random free site (`settle`).
Without this the tied fraction erodes under crossover (to ~0.06 after 30
generations of random drift), while a symmetric genome's mirrored fraction
cannot. A tie whose other end is gone is kept with that end dead until
`settle` re-places it.

Ties live in the genome dict next to ``nodes`` and ``edges`` as
``"ties": [{"a", "b", "type", "rotation"}, ...]``; `TreeGenome.from_dict`
ignores the key, so a tied genome dict builds a body like any tree genome. The
tree holds the *expressed* body: tied modules take the tie's type and
rotation, except that a leaf HINGE is expressed as a BRICK (as
`_fix_terminal_hinges` does for every tree genome). The tie keeps the HINGE
gene, so the module becomes a hinge again once something attaches to it.
Free faces are always those of the gene type, so expression never flips.
"""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)

from .collision import core_checker, first_collision, mirror_node, try_place
from .operators import _fix_terminal_hinges, _prune_invalid_edges, add_node
from .symmetry import MirrorAxis, mirror_face
from .tree_genome import TreeGenome
from .validation import validate_genome_dict

MODULE_TYPES = [t.name for t in ModuleType if t not in {ModuleType.CORE, ModuleType.NONE}]
MUTATION_OPS = ("point", "subtree", "shrink", "hoist")
MUTATION_P = (0.45, 0.35, 0.10, 0.10)  # as shared.mutate_morph
ROTATION_MUTATION_P = 0.25
MAX_BRANCH = 3  # mutate_subtree_replacement grafts 1-3 modules


@dataclass
class Tie:
    a: int
    b: int
    type: str
    rotation: str


class TiedGenome:
    """A tree genome plus its ties; see the module docstring."""

    def __init__(self, tree: TreeGenome, ties: list[Tie]) -> None:
        self.tree = tree
        self.ties = ties

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TiedGenome:
        return cls(TreeGenome.from_dict(data), [Tie(**t) for t in data.get("ties", [])])

    def to_dict(self) -> dict[str, Any]:
        return {**self.tree.to_dict(), "ties": [asdict(t) for t in self.ties]}

    def tie_of(self, nid: int) -> Tie | None:
        return next((t for t in self.ties if nid in {t.a, t.b}), None)

    def twin(self, nid: int) -> int | None:
        t = self.tie_of(nid)
        return None if t is None else (t.b if t.a == nid else t.a)

    def gene(self, nid: int) -> tuple[str, str]:
        t = self.tie_of(nid)
        node = self.tree.nodes[nid]
        return (t.type, t.rotation) if t else (node["type"], node["rotation"])

    def set_gene(self, nid: int, mtype: str, rotation: str) -> None:
        t = self.tie_of(nid)
        if t:
            t.type, t.rotation = mtype, rotation
        else:
            self.tree.nodes[nid] = {"type": mtype, "rotation": rotation}


def tied_fraction(data: dict[str, Any]) -> float:
    """Share of non-core modules that are in a tie."""
    g = TiedGenome.from_dict(data)
    n = len(g.tree.nodes) - 1
    return 2 * len(g.ties) / n if n > 0 else 0.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def _random_gene(exclude_type: str | None = None) -> tuple[str, str]:
    mtype = random.choice([t for t in MODULE_TYPES if t != exclude_type])
    return mtype, random.choice([r.name for r in ALLOWED_ROTATIONS[ModuleType[mtype]]])


def _next_id(g: TiedGenome) -> int:
    """A fresh id; ids a dead tie end still names are not reused, or a new module
    would inherit that tie.
    """
    return max([*g.tree.nodes, *(n for t in g.ties for n in (t.a, t.b))]) + 1


def _site_of(tree: TreeGenome, nid: int) -> tuple[int, str] | None:
    return next(((e["parent"], e["face"]) for e in tree.edges if e["child"] == nid), None)


def _subtree(tree: TreeGenome, nid: int) -> set[int]:
    return {nid, *nx.descendants(tree.to_networkx(), nid)} if nid in tree.nodes else set()


def _free_sites(g: TiedGenome, parents: set[int] | None = None) -> list[tuple[int, str]]:
    """(parent, face) pairs with nothing attached, by the parent's gene type."""
    used = {(e["parent"], e["face"]) for e in g.tree.edges}
    sites = []
    for pid in g.tree.nodes:
        if parents is not None and pid not in parents:
            continue
        ptype = g.gene(pid)[0] if pid != IDX_OF_CORE else "CORE"
        sites += [(pid, f.name) for f in ALLOWED_FACES[ModuleType[ptype]] if (pid, f.name) not in used]
    return sites


def _mirror_site(tree: TreeGenome, site: tuple[int, str]) -> tuple[int, str] | None:
    pid, face = site
    mp = IDX_OF_CORE if pid == IDX_OF_CORE else mirror_node(tree, pid, MirrorAxis.Y_ZERO)
    if mp is None:
        return None
    return mp, mirror_face(face, MirrorAxis.Y_ZERO, is_outer=pid == IDX_OF_CORE)


def _body_checker(tree: TreeGenome):
    """Collision checker holding every module that can be placed (breadth-first)."""
    checker = core_checker(tree)
    graph = tree.to_networkx()
    for pid, cid in nx.bfs_edges(graph, IDX_OF_CORE):
        if pid in checker.frames:
            node = tree.nodes[cid]
            try_place(checker, cid, pid, graph.edges[pid, cid]["face"], node["type"], node["rotation"])
    return checker


def place_twin(g: TiedGenome, nid: int, checker) -> int | None:
    """Tie a new copy of untied module ``nid`` to it, at a random free site that
    does not collide and is not ``nid``'s mirror position. None if no site fits.
    """
    mtype, rot = g.gene(nid)
    site = _site_of(g.tree, nid)
    excluded = {_mirror_site(g.tree, site)} if site else set()
    sites = [s for s in _free_sites(g) if s not in excluded and s[0] in checker.frames]
    random.shuffle(sites)
    for pid, face in sites:
        new = _next_id(g)
        if try_place(checker, new, pid, face, mtype, rot):
            add_node(g.tree, pid, face, new, mtype, rot)
            g.ties.append(Tie(nid, new, mtype, rot))
            return new
    return None


def remove_tied(g: TiedGenome, nid: int) -> set[int]:
    """Remove ``nid``'s subtree and, for each tied module in it, its twin's
    subtree (one level). Returns the removed ids.
    """
    gone = _subtree(g.tree, nid)
    for n in list(gone):
        tw = g.twin(n)
        if tw is not None and tw not in gone:
            gone |= _subtree(g.tree, tw)
    gone.discard(IDX_OF_CORE)
    for n in gone:
        g.tree.nodes.pop(n, None)
    g.tree.edges = [e for e in g.tree.edges if e["parent"] not in gone and e["child"] not in gone]
    return gone


def _drop_broken(g: TiedGenome) -> list[Tie]:
    """Remove ties with a dead end; return those whose other end still lives."""
    alive = set(g.tree.nodes)
    orphans = [t for t in g.ties if (t.a in alive) != (t.b in alive)]
    g.ties = [t for t in g.ties if t.a in alive and t.b in alive]
    return orphans


def express(g: TiedGenome) -> list[Tie]:
    """Write each tie's gene onto both modules, then the usual clean-up (invalid
    faces, leaf hinges -> bricks). Returns the ties broken on the way.
    """
    orphans = []
    for _ in range(2):  # pruning can break ties; a second pass settles it
        orphans += _drop_broken(g)
        for t in g.ties:
            for n in (t.a, t.b):
                g.tree.nodes[n] = {"type": t.type, "rotation": t.rotation}
        _prune_invalid_edges(g.tree)
    orphans += _drop_broken(g)
    _fix_terminal_hinges(g.tree)
    return orphans


def settle(g: TiedGenome, rounds: int = 3) -> None:
    """Express, remove collisions, and give each tied module that lost its twin a
    new one at a random free site (a module stays untied if no site fits).
    """
    orphans = express(g)
    prune_colliding_tied(g)
    orphans += express(g)
    for _ in range(rounds):
        alive = set(g.tree.nodes)
        lone = {(t.a if t.a in alive else t.b): t for t in orphans}
        lone = {n: t for n, t in lone.items() if n in alive and g.tie_of(n) is None}
        if not lone:
            break
        checker = _body_checker(g.tree)
        for n, t in lone.items():
            g.tree.nodes[n] = {"type": t.type, "rotation": t.rotation}
            place_twin(g, n, checker)
        orphans = express(g)
        prune_colliding_tied(g)
        orphans += express(g)


def prune_colliding_tied(g: TiedGenome) -> int:
    """Remove colliding subtrees together with their twins' subtrees."""
    removed = 0
    while (nid := first_collision(g.tree)) is not None:
        remove_tied(g, nid)
        removed += 1
    return removed


def _graft_copy(g: TiedGenome, nodes: list[int], site: tuple[int, str]) -> None:
    """Copy the branch ``nodes`` (root first) onto ``site``, tying each copy to its original."""
    pid, face = site
    if pid not in g.tree.nodes or (pid, face) not in _free_sites(g, {pid}):
        return
    inside = set(nodes)
    mapping = {n: _next_id(g) + i for i, n in enumerate(nodes)}
    for n in nodes:
        mtype, rot = g.gene(n)
        g.tree.nodes[mapping[n]] = {"type": mtype, "rotation": rot}
        g.ties.append(Tie(n, mapping[n], mtype, rot))
    g.tree.edges += [{**e, "parent": mapping[e["parent"]], "child": mapping[e["child"]]}
                     for e in g.tree.edges if e["parent"] in inside and e["child"] in inside]
    g.tree.edges.append({"parent": pid, "child": mapping[nodes[0]], "face": face})


# ── Growth ───────────────────────────────────────────────────────────────────


def _grow(g: TiedGenome, n: int, tie_prob: float, checker, root_site: tuple[int, str] | None = None,
          count_twins: bool = True) -> list[int]:
    """Add up to ``n`` random modules; each gets a twin with probability
    ``tie_prob``. With ``root_site`` the first goes there and the rest grow
    below it (a branch); otherwise anywhere. Returns the modules added (not
    their twins), in placement order.
    """
    added: list[int] = []
    blocked: set[tuple[int, str]] = set()
    count = 0
    while count < n:
        if root_site is not None and not added:
            free = [root_site] if root_site not in blocked and root_site[0] in checker.frames else []
        else:
            free = [s for s in _free_sites(g, set(added) if root_site else None)
                    if s not in blocked and s[0] in checker.frames]
        if not free:
            break
        pid, face = random.choice(free)
        mtype, rot = _random_gene()
        nid = _next_id(g)
        if not try_place(checker, nid, pid, face, mtype, rot):
            blocked.add((pid, face))
            continue
        add_node(g.tree, pid, face, nid, mtype, rot)
        added.append(nid)
        count += 1
        if (count < n or not count_twins) and random.random() < tie_prob:
            if place_twin(g, nid, checker) is not None and count_twins:
                count += 1
    return added


def random_tree_tied(max_modules: int, tie_prob: float) -> TiedGenome:
    """Grow ``max_modules // 2`` modules plus their twins, as `random_tree_symmetric`
    grows half the budget and then mirrors it, so both start at a similar size.
    """
    g = TiedGenome(TreeGenome(nodes={IDX_OF_CORE: {"type": "CORE", "rotation": "DEG_0"}}, edges=[]), [])
    _grow(g, max(1, max_modules // 2), tie_prob, core_checker(g.tree), count_twins=False)
    # a leaf hinge's twin was placed against its brick expression; settle any overlap
    settle(g)
    return g


# ── Mutation ─────────────────────────────────────────────────────────────────


def _mutate_point(g: TiedGenome, x: int) -> None:
    g.set_gene(x, *_random_gene(exclude_type=g.gene(x)[0]))


def _mutate_rotation(g: TiedGenome, x: int) -> None:
    mtype = g.gene(x)[0]
    g.set_gene(x, mtype, random.choice([r.name for r in ALLOWED_ROTATIONS[ModuleType[mtype]]]))


def _mutate_subtree(g: TiedGenome, x: int, num_modules: int, tie_prob: float, single: bool) -> None:
    """Subtree replacement (``single``: shrink, one new module) at x, replayed at its twin."""
    site, tw = _site_of(g.tree, x), g.twin(x)
    twin_site = _site_of(g.tree, tw) if tw is not None else None
    remove_tied(g, x)
    if site is None or site[0] not in g.tree.nodes:
        return
    if single:
        # shrink: no collision check, as mutate_shrink; prune_colliding_tied runs after
        nid = _next_id(g)
        add_node(g.tree, *site, nid, *_random_gene())
        branch = [nid]
        if twin_site is None and random.random() < tie_prob:
            place_twin(g, nid, _body_checker(g.tree))
    else:
        k = random.randint(1, min(MAX_BRANCH, num_modules))
        branch = _grow(g, k, 0.0 if twin_site else tie_prob, _body_checker(g.tree), root_site=site,
                       count_twins=False)
    if twin_site is not None and branch:
        _graft_copy(g, branch, twin_site)


def _hoist_one(g: TiedGenome, x: int, face: str) -> None:
    site = _site_of(g.tree, x)
    child = next((e["child"] for e in g.tree.edges if e["parent"] == x and e["face"] == face), None)
    if site is None or child is None:
        return
    for e in [e for e in g.tree.edges if e["parent"] == x and e["child"] != child]:
        remove_tied(g, e["child"])
    if x not in g.tree.nodes or child not in g.tree.nodes or site[0] not in g.tree.nodes:
        return
    del g.tree.nodes[x]
    g.tree.edges = [e for e in g.tree.edges if x not in {e["parent"], e["child"]}]
    g.tree.edges.append({"parent": site[0], "child": child, "face": site[1]})


def _mutate_hoist(g: TiedGenome, x: int) -> None:
    faces = [e["face"] for e in g.tree.edges if e["parent"] == x]
    if not faces:
        return
    face, tw = random.choice(faces), g.twin(x)
    _hoist_one(g, x, face)
    if tw is not None and tw in g.tree.nodes:
        _hoist_one(g, tw, face)


def _valid(g: TiedGenome) -> bool:
    try:
        validate_genome_dict(g.tree.to_dict())
    except ValueError:
        return False
    return True


def mutate_morph_tied(data: dict[str, Any], rng, num_modules: int, tie_prob: float) -> dict[str, Any]:
    """`shared.mutate_morph` for tied genomes: same operator mix, each mutation
    of a tied module applied to its twin too. Invalid results are no-ops.
    """
    g = TiedGenome.from_dict(copy.deepcopy(data))
    op = rng.choice(MUTATION_OPS, p=MUTATION_P)
    steps = [op] + (["rotation"] if rng.random() < ROTATION_MUTATION_P else [])
    for step in steps:
        noncore = [n for n in g.tree.nodes if n != IDX_OF_CORE]
        if not noncore:
            break
        x = random.choice(noncore)
        backup = copy.deepcopy(g)
        if step == "point":
            _mutate_point(g, x)
        elif step == "rotation":
            _mutate_rotation(g, x)
        elif step in {"subtree", "shrink"}:
            _mutate_subtree(g, x, num_modules, tie_prob, single=step == "shrink")
        else:
            _mutate_hoist(g, x)
        settle(g)
        if not _valid(g):
            g = backup
    return g.to_dict()


# ── Crossover ────────────────────────────────────────────────────────────────


def _swap_in(dst: TiedGenome, src: TiedGenome, src_root: int, site: tuple[int, str]) -> None:
    """Graft a fresh-id copy of src's subtree at ``site`` in dst, with the ties of
    its modules (a twin outside the subtree becomes a dead end for `settle`).
    """
    inside = _subtree(src.tree, src_root)
    start = _next_id(dst)
    mapping = {n: start + i for i, n in enumerate(sorted(inside))}
    for n in inside:
        dst.tree.nodes[mapping[n]] = copy.deepcopy(src.tree.nodes[n])
    dst.tree.edges += [{**e, "parent": mapping[e["parent"]], "child": mapping[e["child"]]}
                       for e in src.tree.edges if e["parent"] in inside and e["child"] in inside]
    dst.tree.edges.append({"parent": site[0], "child": mapping[src_root], "face": site[1]})
    dst.ties += [Tie(mapping.get(t.a, -1), mapping.get(t.b, -1), t.type, t.rotation)
                 for t in src.ties if t.a in inside or t.b in inside]


def _drop_subtree(g: TiedGenome, nid: int) -> None:
    gone = _subtree(g.tree, nid)
    for n in gone:
        g.tree.nodes.pop(n, None)
    g.tree.edges = [e for e in g.tree.edges if e["parent"] not in gone and e["child"] not in gone]


def crossover_tied(da: dict[str, Any], db: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Subtree crossover: swap a random subtree between the parents. Ties inside a
    swapped subtree move with it; a module whose twin is left behind gets a new
    twin in its new body (`settle`). Invalid children fall back to the parents.
    """
    a, b = TiedGenome.from_dict(copy.deepcopy(da)), TiedGenome.from_dict(copy.deepcopy(db))
    ca = [n for n in a.tree.nodes if n != IDX_OF_CORE]
    cb = [n for n in b.tree.nodes if n != IDX_OF_CORE]
    if not ca or not cb:
        return copy.deepcopy(da), copy.deepcopy(db)
    x, y = random.choice(ca), random.choice(cb)
    sa, sb = _site_of(a.tree, x), _site_of(b.tree, y)
    a0, b0 = copy.deepcopy(a), copy.deepcopy(b)
    _drop_subtree(a, x)
    _drop_subtree(b, y)
    _swap_in(a, b0, y, sa)
    _swap_in(b, a0, x, sb)
    children = []
    for child, parent in ((a, da), (b, db)):
        settle(child)
        children.append(child.to_dict() if _valid(child) else copy.deepcopy(parent))
    return children[0], children[1]
