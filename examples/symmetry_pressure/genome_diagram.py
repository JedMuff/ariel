"""Genome-image renderers, shared across symmetry-pressure scripts.

Two genome representations are evolved in this experiment family:
  - "tree" / "tree_symmetric": a `TreeGenome` dict {"nodes", "edges"} that
    *is* the phenotype body graph (see `shared.genome_to_spec`) -- so its
    diagram is a schematic 2D top-down module layout.
  - "cppn": a NEAT-style `Genome` dict {"nodes", "connections"} (see
    `ariel.body_phenotypes.robogen_lite.cppn_neat.genome.Genome`) that must
    be *decoded* (via `genome_adapter.cppn_genome_to_spec`) to get a body --
    the raw genome itself is a small feed-forward/recurrent network with no
    spatial meaning, so its diagram is a layered network graph instead.

`genome_layout`/`draw_tree_genome` were moved here from `compare_food_skills.py`
verbatim (behavior-preserving refactor); `draw_cppn_genome` is new.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Tree-genome layout + plotting ──────────────────────────────────────────────

MODULE_COLORS = {
    "CORE":  "#59A14F",
    "BRICK": "#4E79A7",
    "HINGE": "#E15759",
}
MODULE_LABELS = {"CORE": "C", "BRICK": "B", "HINGE": "H"}
# `s` is marker *area* in points^2, so a 4x smaller-looking (linear) circle
# needs area/16.
MODULE_SIZES = {"CORE": 520 / 16, "BRICK": 420 / 16, "HINGE": 360 / 16}

FACE_LABELS = {
    "FRONT": "F", "BACK": "Bk", "LEFT": "L",
    "RIGHT": "R", "TOP": "T", "BOTTOM": "Bt",
}


def genome_layout(genome: dict) -> dict[str, np.ndarray]:
    """Reingold-Tilford-style tree layout from the CORE root: every leaf gets
    a unique x slot, internal nodes sit at the mean x of their children, and
    y is `-depth` -- so nodes never overlap regardless of how many siblings
    share a face (unlike the old face-direction layout this replaces)."""
    nodes = genome["nodes"]
    edges = genome["edges"]

    children: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        children[str(e["parent"])].append(str(e["child"]))

    core_id = next(nid for nid, n in nodes.items() if n["type"] == "CORE")

    x_of: dict[str, float] = {}
    depth_of: dict[str, int] = {}
    next_leaf_x = [0.0]

    def assign(nid: str, depth: int, visited: set[str]) -> float:
        depth_of[nid] = depth
        visited.add(nid)
        kids = [c for c in children.get(nid, []) if c not in visited]
        if not kids:
            x = next_leaf_x[0]
            next_leaf_x[0] += 1.0
        else:
            xs = [assign(cid, depth + 1, visited) for cid in kids]
            x = sum(xs) / len(xs)
        x_of[nid] = x
        return x

    assign(core_id, 0, set())

    return {nid: np.array([x_of[nid], -depth_of[nid]]) for nid in x_of}


# Shared parent/child diff-highlight styling, used by both draw functions.
# "new"/"changed" ring colors highlight a node without changing its normal
# fill color; "deleted" (tree only -- see `build_tree_diff`) overrides fill
# entirely to grey, since a removed node has no "current" type to show off.
DIFF_RING_COLORS = {"new": "#F2C14E", "changed": "#E58A00"}
DELETED_FILL = "#C9CDD4"   # matches ancestry_tree_viewer.css's --dim-fill
DELETED_RING = "#8A8F98"
DELETED_ALPHA = 0.55


def build_tree_diff(
    child_genome: dict, parent_genome: dict
) -> tuple[dict, dict[str, str], dict[tuple[str, str], str]]:
    """Diff a tree genome against its parent for visual highlighting.

    Every tree-genome mutation path starts from a deepcopy of the parent
    (`shared.mutate_morph`), so an untouched node keeps its exact id and
    contents -- dict-key identity is a safe way to detect new/changed/
    deleted nodes (except for a crossover-donated subtree, which gets fresh
    ids unrelated to either parent; callers diff against `parent_ids[0]`
    only, a known simplification for that case).

    Returns `(union_genome, node_status, edge_status)`:
    - `union_genome` is `child_genome` plus every node/edge the parent had
      that the child no longer does (a whole deleted subtree) -- feeding it
      through `genome_layout` places the removed material for free, since
      it's just extra branches to the same tree-layout algorithm.
    - `node_status[node_id]` is `"new"` / `"changed"` / `"deleted"` (nodes
      with no entry are unchanged).
    - `edge_status[(parent_id, child_id)]` is `"deleted"` for edges whose
      child is a deleted node (nodes with no entry render normally).
    """
    child_nodes: dict[str, dict] = {str(k): v for k, v in child_genome["nodes"].items()}
    parent_nodes: dict[str, dict] = {str(k): v for k, v in parent_genome["nodes"].items()}
    child_ids = set(child_nodes)
    parent_ids = set(parent_nodes)
    deleted_ids = parent_ids - child_ids

    node_status: dict[str, str] = {nid: "deleted" for nid in deleted_ids}
    for nid in child_ids & parent_ids:
        if (child_nodes[nid].get("type") != parent_nodes[nid].get("type")
                or child_nodes[nid].get("rotation") != parent_nodes[nid].get("rotation")):
            node_status[nid] = "changed"
    for nid in child_ids - parent_ids:
        node_status[nid] = "new"

    union_nodes = dict(child_nodes)
    union_edges = list(child_genome["edges"])
    for e in parent_genome["edges"]:
        cid = str(e["child"])
        if cid in deleted_ids:
            union_nodes[cid] = parent_nodes[cid]
            union_edges.append(e)

    edge_status: dict[tuple[str, str], str] = {
        (str(e["parent"]), str(e["child"])): "deleted"
        for e in union_edges
        if str(e["child"]) in deleted_ids
    }

    return {"nodes": union_nodes, "edges": union_edges}, node_status, edge_status


def build_tree_diff_best(
    child_genome: dict, parent_genomes: list[dict]
) -> Optional[tuple[dict, dict[str, str], dict[tuple[str, str], str]]]:
    """`build_tree_diff` against whichever of 1-2 given parents looks like
    the actual crossover host (fewest changed+deleted nodes).

    `parent_ids[0]` is *not* reliably the host: `shared.make_offspring`'s
    crossover produces two candidate children (`c1` host-is-p1, `c2`
    host-is-p2) and picks one with a coin flip independent of parent
    listing order (`child_morph = c1 if rng.random() < 0.5 else c2`), while
    `parent_ids` always lists `[p1.id, p2.id]`. Diffing against the wrong
    parent produces mostly-spurious "changed" nodes (same id, coincidentally
    different content) instead of a clean new/deleted diff -- picking the
    lower-edit-distance parent side-steps needing to know which one is
    actually inherited from without touching the reproduction code.
    """
    if not parent_genomes:
        return None
    best = None
    for pg in parent_genomes:
        union, node_status, edge_status = build_tree_diff(child_genome, pg)
        edits = sum(1 for v in node_status.values() if v in ("changed", "deleted"))
        if best is None or edits < best[0]:
            best = (edits, union, node_status, edge_status)
    _, union, node_status, edge_status = best
    return union, node_status, edge_status


# Fixed padding (data units) added around a genome's own layout extent when
# rendering onto a shared, batch-wide canvas -- see `tree_layout_extent`/
# `cppn_layout_extent` and `render_genome_thumbnail`'s `xlim`/`ylim`. Sized to
# comfortably clear a node's own radius plus its label (the CPPN's bottom
# margin is larger to fit the activation/bias text drawn under each node).
TREE_LAYOUT_MARGIN = 0.65
CPPN_LAYOUT_MARGIN = (0.5, 0.5, 0.5, 0.9)  # left, right, top, bottom


def tree_layout_extent(genome: dict) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of a tree genome's own layout, no margin."""
    pos = genome_layout(genome)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    return min(xs), max(xs), min(ys), max(ys)


def cppn_layout_extent(genome: dict) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of a CPPN genome's own layout, no margin."""
    pos = _cppn_layout(genome)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    return min(xs), max(xs), min(ys), max(ys)


def compute_shared_window(
    genomes: list[dict],
    genome_type: str,
    target_px: int,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """One shared (xlim, ylim, figsize_px) sized to fit every genome in
    `genomes` at the same scale -- the larger side of the window maps to
    `target_px`, the other is scaled down to match its aspect ratio. Pass
    the result straight through to `render_genome_thumbnail` for each
    genome so a whole batch renders at an identical scale and identical
    output pixel dimensions (see that function's docstring for why)."""
    if genome_type in ("tree", "tree_symmetric"):
        extents = [tree_layout_extent(g) for g in genomes]
        margin_l = margin_r = margin_t = margin_b = TREE_LAYOUT_MARGIN
    elif genome_type == "cppn":
        extents = [cppn_layout_extent(g) for g in genomes]
        margin_l, margin_r, margin_t, margin_b = CPPN_LAYOUT_MARGIN
    else:
        raise ValueError(f"Unknown genome_type: {genome_type!r}")

    xmin = min(e[0] for e in extents)
    xmax = max(e[1] for e in extents)
    ymin = min(e[2] for e in extents)
    ymax = max(e[3] for e in extents)
    xlim = (xmin - margin_l, xmax + margin_r)
    ylim = (ymin - margin_b, ymax + margin_t)

    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]
    aspect = width / height
    if aspect >= 1:
        figsize_px = (float(target_px), target_px / aspect)
    else:
        figsize_px = (target_px * aspect, float(target_px))
    return xlim, ylim, figsize_px


def draw_tree_genome(
    ax: plt.Axes,
    genome: dict,
    title: str = "",
    fitness: Optional[float] = None,
    compact: bool = False,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    node_status: Optional[dict[str, str]] = None,
    edge_status: Optional[dict[tuple[str, str], str]] = None,
) -> None:
    """Draw a top-down 2D module layout of a tree genome on ax.

    `node_status`/`edge_status` (see `build_tree_diff`) optionally highlight
    a parent/child diff: "new"/"changed" nodes get a colored ring, "deleted"
    nodes (present only because `genome` is a `build_tree_diff` union, not a
    genome that was ever actually evaluated) render as grey ghosts.
    """
    node_status = node_status or {}
    edge_status = edge_status or {}
    nodes = genome["nodes"]
    pos = genome_layout(genome)

    # Draw edges first, each labeled with the face it attaches on.
    edges = genome["edges"]
    for e in edges:
        pid, cid = str(e["parent"]), str(e["child"])
        if pid in pos and cid in pos:
            x0, y0 = pos[pid]
            x1, y1 = pos[cid]
            deleted = edge_status.get((pid, cid)) == "deleted"
            if deleted:
                ax.plot([x0, x1], [y0, y1], linestyle="--", color=DELETED_RING,
                        linewidth=1.0, alpha=0.6, zorder=1)
            else:
                ax.plot([x0, x1], [y0, y1], "k-", linewidth=1.0, alpha=0.5, zorder=1)
            face_label = FACE_LABELS.get(e.get("face"), "")
            if face_label:
                ax.text(
                    (x0 + x1) / 2, (y0 + y1) / 2, face_label,
                    fontsize=2.75, ha="center", va="center", color="#444444", zorder=3,
                    alpha=0.6 if deleted else 1.0,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.7),
                    clip_on=False,
                )

    # Draw nodes: uniform circles, colored + labeled by module type.
    for nid, n_data in nodes.items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        ntype = n_data["type"]
        status = node_status.get(nid)
        if status == "deleted":
            face_color, ring_color, ring_width, ring_style, node_alpha = (
                DELETED_FILL, DELETED_RING, 1.2, "--", DELETED_ALPHA)
        elif status in DIFF_RING_COLORS:
            face_color, ring_color, ring_width, ring_style, node_alpha = (
                MODULE_COLORS.get(ntype, "gray"), DIFF_RING_COLORS[status], 2.0, "-", 1.0)
        else:
            face_color, ring_color, ring_width, ring_style, node_alpha = (
                MODULE_COLORS.get(ntype, "gray"), "white", 0.8, "-", 1.0)
        ax.scatter(
            x, y,
            s=MODULE_SIZES.get(ntype, 300),
            c=face_color,
            marker="o",
            zorder=2, edgecolors=ring_color, linewidths=ring_width,
            linestyle=ring_style, alpha=node_alpha, clip_on=False,
        )
        ax.text(
            x, y, MODULE_LABELS.get(ntype, "?"),
            fontsize=3.5, fontweight="bold", ha="center", va="center",
            color="white", zorder=4, alpha=node_alpha, clip_on=False,
        )

    ax.set_aspect("equal")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if not compact:
        title_text = title
        if fitness is not None:
            title_text = f"{title}\nfit={fitness:.3f}"
        ax.set_title(title_text, fontsize=9)
        legend_handles = [
            mpatches.Patch(color=MODULE_COLORS[t], label=t)
            for t in ["CORE", "BRICK", "HINGE"]
        ]
        ax.legend(handles=legend_handles, fontsize=6, loc="lower right",
                  framealpha=0.7, handlelength=1.0)
    ax.axis("off")


# Backwards-compatible alias for the name compare_food_skills.py used before
# the refactor (fitness was a required positional arg there).
def draw_morphology(ax: plt.Axes, genome: dict, title: str, fitness: float) -> None:
    draw_tree_genome(ax, genome, title=title, fitness=fitness, compact=False)


# ── CPPN-genome layout + plotting ──────────────────────────────────────────────

CPPN_NODE_COLORS = {
    "input":  "#4E79A7",
    "hidden": "#999999",
    "output": "#F28E2B",
}


def _cppn_layers(genome: dict) -> dict[str, int]:
    """Assign each node a layer via a longest-path pass over *all* connections
    (enabled or not -- disabled ones are still real topology): inputs are
    layer 0, every other node is 1 + max(layer of its predecessors), and
    outputs are then pinned to the last layer so they stay the rightmost
    column even if some hidden node has no path to an output. This spreads
    hidden nodes across as many columns as the network is actually deep,
    instead of cramming them into one."""
    nodes = genome["nodes"]
    preds: dict[str, list[str]] = defaultdict(list)
    for conn in genome.get("connections", []):
        preds[str(conn["out_id"])].append(str(conn["in_id"]))

    layer: dict[str, int] = {}

    def compute(nid: str, visiting: set[str]) -> int:
        if nid in layer:
            return layer[nid]
        n_data = nodes.get(nid, {})
        if n_data.get("typ") == "input" or not preds.get(nid) or nid in visiting:
            layer[nid] = 0
            return 0
        visiting.add(nid)
        depth = 1 + max((compute(p, visiting) for p in preds[nid] if p in nodes), default=-1)
        visiting.discard(nid)
        layer[nid] = max(depth, 0)
        return layer[nid]

    for nid in nodes:
        compute(nid, set())

    max_layer = max(layer.values(), default=0)
    for nid, n_data in nodes.items():
        if n_data.get("typ") == "output":
            layer[nid] = max_layer + 1

    return layer


def _cppn_layout(genome: dict) -> dict[str, tuple[float, float]]:
    """Depth-layered layout: x = computed layer, evenly spaced in y within
    each layer (hidden nodes may span several layers, not just one)."""
    layer_of = _cppn_layers(genome)

    by_layer: dict[int, list[str]] = defaultdict(list)
    for nid in genome["nodes"]:
        by_layer[layer_of.get(nid, 0)].append(nid)

    pos: dict[str, tuple[float, float]] = {}
    for x, ids in by_layer.items():
        n = len(ids)
        for i, nid in enumerate(sorted(ids, key=int)):
            y = (i - (n - 1) / 2.0) if n > 1 else 0.0
            pos[nid] = (float(x), y)
    return pos


def _cppn_node_label(n_data: dict) -> Optional[str]:
    activation = n_data.get("activation")
    if activation is None:
        return None
    bias = n_data.get("bias", 0.0)
    return f"{activation}\nb={bias:.2f}"


def build_cppn_diff(
    child_genome: dict, parent_genomes: list[dict]
) -> tuple[dict[str, str], dict[int, str]]:
    """Diff a CPPN genome against 1-2 parents for visual highlighting.

    Unlike tree genomes, CPPN mutation never deletes a node/connection, and
    crossover reconstructs genes from whichever parent had them while
    keeping their original `_id`/`innov_id` -- so diffing against the
    *union* of all parents' ids is exactly correct, no simplification
    needed (there is no "deleted" case for CPPN at all).

    Returns `(node_status, edge_status)`:
    - `node_status[node_id] = "new"` for a node absent from every parent.
    - `edge_status[innov_id] = "new"` for a connection absent from every
      parent, else `"changed"` if it's enabled in some parent but disabled
      in the child -- the only in-place edit any CPPN mutation makes: an
      add-node mutation disables the connection it splits.
    """
    parent_node_ids: set[str] = set()
    parent_enabled_by_innov: dict[int, bool] = {}
    for pg in parent_genomes:
        parent_node_ids.update(str(k) for k in pg["nodes"])
        for conn in pg.get("connections", []):
            innov = conn["innov_id"]
            was_enabled = parent_enabled_by_innov.get(innov, False)
            parent_enabled_by_innov[innov] = was_enabled or conn.get("enabled", True)

    node_status: dict[str, str] = {
        str(nid): "new" for nid in child_genome["nodes"] if str(nid) not in parent_node_ids
    }

    edge_status: dict[int, str] = {}
    for conn in child_genome.get("connections", []):
        innov = conn["innov_id"]
        if innov not in parent_enabled_by_innov:
            edge_status[innov] = "new"
        elif parent_enabled_by_innov[innov] and not conn.get("enabled", True):
            edge_status[innov] = "changed"

    return node_status, edge_status


def draw_cppn_genome(
    ax: plt.Axes,
    genome: dict,
    title: str = "",
    compact: bool = False,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    node_status: Optional[dict[str, str]] = None,
    edge_status: Optional[dict[int, str]] = None,
) -> None:
    """Draw a layered NEAT-style network diagram of a CPPN genome on ax.

    Disabled connections and hidden nodes with no enabled connection are
    still drawn, just faded, so the full topology stays visible alongside
    what's actually active.

    `node_status`/`edge_status` (see `build_cppn_diff`) optionally highlight
    a parent/child diff: "new" nodes/connections get a colored ring/pop,
    "changed" connections (newly disabled this generation) render orange
    dashed instead of the generic long-disabled grey fade. CPPN has no
    "deleted" case -- mutation never removes a node or connection.
    """
    node_status = node_status or {}
    edge_status = edge_status or {}
    pos = _cppn_layout(genome)

    active: dict[str, bool] = {
        nid: n_data.get("typ") in ("input", "output")
        for nid, n_data in genome["nodes"].items()
    }
    for conn in genome.get("connections", []):
        if conn.get("enabled", True):
            for nid in (str(conn["in_id"]), str(conn["out_id"])):
                if nid in active:
                    active[nid] = True

    for conn in genome.get("connections", []):
        in_id, out_id = str(conn["in_id"]), str(conn["out_id"])
        if in_id not in pos or out_id not in pos:
            continue
        x0, y0 = pos[in_id]
        x1, y1 = pos[out_id]
        weight = conn.get("weight", 0.0)
        enabled = conn.get("enabled", True)
        status = edge_status.get(conn["innov_id"])
        if status == "changed":
            color, linestyle, alpha, linewidth = "#E58A00", "--", 0.9, 1.4
        else:
            color = "#4C72B0" if weight > 0 else "#C44E52"
            linestyle = "-"
            alpha = 0.12 if not enabled else min(0.3 + abs(weight) * 0.2, 0.95)
            linewidth = min(abs(weight), 3.0) + 0.4
            if status == "new":
                alpha = 1.0
                linewidth += 0.6
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth,
                linestyle=linestyle, alpha=alpha, zorder=1)

    for nid, n_data in genome["nodes"].items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        typ = n_data.get("typ", "hidden")
        node_alpha = 1.0 if active.get(nid, True) else 0.35
        is_new = node_status.get(nid) == "new"
        ring_color = DIFF_RING_COLORS["new"] if is_new else "white"
        ring_width = 2.0 if is_new else 0.8
        ax.scatter(x, y, s=260 / 16, c=CPPN_NODE_COLORS.get(typ, "gray"),
                   marker="o", zorder=2, edgecolors=ring_color, linewidths=ring_width,
                   alpha=node_alpha, clip_on=False)
        label = _cppn_node_label(n_data)
        if label:
            ax.text(x, y - 0.32, label, fontsize=2.5, ha="center", va="top",
                    color="black", alpha=node_alpha, zorder=3, clip_on=False)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(-0.5, max((p[0] for p in pos.values()), default=2.0) + 0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect("auto")
    if not compact:
        ax.set_title(title, fontsize=9)
        legend_handles = [
            mpatches.Patch(color=CPPN_NODE_COLORS[t], label=t)
            for t in ["input", "hidden", "output"]
        ]
        ax.legend(handles=legend_handles, fontsize=6, loc="lower right",
                  framealpha=0.7, handlelength=1.0)
    ax.axis("off")


# ── Dispatcher ──────────────────────────────────────────────────────────────────

_LAYOUT_REFERENCE_DPI = 100  # figsize is always computed against this, so bumping
                              # `dpi` below changes output sharpness/pixel-count only
                              # -- it does not rescale marker/font sizes relative to
                              # the diagram, since those are fixed in points.


def render_genome_thumbnail(
    genome: dict,
    genome_type: str,
    out_path: Path,
    size_px: int = 220,
    dpi: int = 220,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    figsize_px: Optional[tuple[float, float]] = None,
    node_status: Optional[dict] = None,
    edge_status: Optional[dict] = None,
) -> None:
    """Render a compact genome diagram (tree schematic or CPPN network) to a PNG.

    Output pixel dimensions are `size_px * dpi / _LAYOUT_REFERENCE_DPI` per side
    -- e.g. size_px=220 (the default `--image-size`) with dpi=220 renders at
    ~2.2x the pixel density of the original 100dpi thumbnails, at the same
    on-page layout.

    Pass `xlim`/`ylim` (and matching `figsize_px`) to render onto a *fixed*,
    caller-chosen window/canvas instead of auto-fitting to this genome's own
    content -- this is how `ancestry_tree_prepare.py` gets every genome in a
    run to render at an identical scale and identical output pixel
    dimensions (see `tree_layout_extent`/`cppn_layout_extent`), instead of
    each genome's PNG being independently tight-cropped to its own extent,
    which otherwise makes node/edge sizes look inconsistent once images of
    different native dimensions are scaled to fit the same on-page box.
    """
    fixed_canvas = xlim is not None or ylim is not None
    fig_w_px, fig_h_px = figsize_px if figsize_px is not None else (size_px, size_px)
    fig = plt.figure(
        figsize=(fig_w_px / _LAYOUT_REFERENCE_DPI, fig_h_px / _LAYOUT_REFERENCE_DPI),
        dpi=dpi,
    )
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0)) if fixed_canvas else fig.add_subplot(111)
    try:
        if genome_type in ("tree", "tree_symmetric"):
            draw_tree_genome(ax, genome, compact=True, xlim=xlim, ylim=ylim,
                              node_status=node_status, edge_status=edge_status)
        elif genome_type == "cppn":
            draw_cppn_genome(ax, genome, compact=True, xlim=xlim, ylim=ylim,
                              node_status=node_status, edge_status=edge_status)
        else:
            raise ValueError(f"Unknown genome_type: {genome_type!r}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fixed_canvas:
            # Full, uncropped canvas -- every genome in the batch gets the
            # exact same output pixel dimensions at the exact same scale.
            fig.savefig(out_path, dpi=dpi)
        else:
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(fig)
