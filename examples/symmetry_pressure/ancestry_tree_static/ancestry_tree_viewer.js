(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const MAX_SLOTS = 5;
  const COL_PITCH = 100;
  const ROW_PITCH = 26;
  const DOT_R = 10;
  const HIT_R = 14;
  const MARGIN_LEFT = 50;
  const MARGIN_TOP = 30;
  const LABEL_OFFSET_X = 16;

  const VIRIDIS_STOPS = [
    [0.0, [68, 1, 84]], [0.1, [72, 40, 120]], [0.2, [62, 74, 137]],
    [0.3, [49, 104, 142]], [0.4, [38, 130, 142]], [0.5, [31, 158, 137]],
    [0.6, [53, 183, 121]], [0.7, [110, 206, 88]], [0.8, [181, 222, 43]],
    [1.0, [253, 231, 37]],
  ];

  function viridis(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 0; i < VIRIDIS_STOPS.length - 1; i++) {
      const [t0, c0] = VIRIDIS_STOPS[i];
      const [t1, c1] = VIRIDIS_STOPS[i + 1];
      if (t >= t0 && t <= t1) {
        const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
        const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
        const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
        const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
        return `rgb(${r},${g},${b})`;
      }
    }
    return "rgb(68,1,84)";
  }

  function el(tag, attrs, parent) {
    const e = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function loadManifest() {
    if (window.ANCESTRY_MANIFEST) {
      return Promise.resolve(window.ANCESTRY_MANIFEST);
    }
    return fetch("manifest.json").then((r) => r.json());
  }

  loadManifest().then(init).catch((err) => {
    document.body.innerHTML =
      '<div style="padding:24px;font-family:sans-serif;color:#b00">' +
      "Failed to load manifest: " + String(err) + "</div>";
  });

  function init(manifest) {
    const run = manifest.run;
    const nodes = manifest.nodes;
    const edges = manifest.edges;
    const nodesById = new Map(nodes.map((n) => [n.node_id, n]));

    renderHeader(run);
    const layout = computeLayout(run, nodes);
    const svgEls = renderTree(nodes, edges, layout);
    setupSlots(nodesById);
    setupInteractions(nodes, nodesById, svgEls);
  }

  // ── Header ──────────────────────────────────────────────────────────────

  function renderHeader(run) {
    document.getElementById("run-title").textContent =
      `${run.task || "?"} / ${run.genome_type || "?"}`;
    const bits = [];
    if (run.strategy_type) bits.push(`strategy=${run.strategy_type}`);
    if (run.pop != null && run.lam != null) bits.push(`pop=${run.pop} lam=${run.lam}`);
    if (run.num_generations != null) bits.push(`${run.num_generations} generations`);
    if (run.num_nodes != null) bits.push(`${run.num_nodes} individuals`);
    if (run.num_nodes_with_images != null) {
      bits.push(`${run.num_nodes_with_images}/${run.num_nodes} with images`);
    }
    if (run.num_invalid_nodes) {
      bits.push(`${run.num_invalid_nodes} invalid (never evaluated)`);
    }
    if (run.num_passthrough_nodes) {
      bits.push(`${run.num_passthrough_nodes} passthrough (alive, not re-evaluated)`);
    }
    document.getElementById("run-subtitle").textContent = bits.join("  ·  ");

    const fmin = run.fitness_min, fmax = run.fitness_max;
    document.getElementById("legend-min").textContent = fmin != null ? fmin.toFixed(3) : "";
    document.getElementById("legend-max").textContent = fmax != null ? fmax.toFixed(3) : "";
    document.getElementById("legend-note").textContent = run.fitness_lower_is_better
      ? "(darker = better)" : "";
  }

  // ── Layout ──────────────────────────────────────────────────────────────

  function computeLayout(run, nodes) {
    // Each node carries a stable `lane` (row index), reused across all of an
    // individual's generations, assigned server-side (ancestry_tree_common.py)
    // so that a persisting individual draws as a straight horizontal line.
    // Position is therefore a direct function of (gen, lane) -- no
    // per-generation grouping/centering needed.
    const numGenerations = run.num_generations != null
      ? run.num_generations
      : (nodes.length ? Math.max(...nodes.map((n) => n.gen)) + 1 : 0);
    const numLanes = run.num_lanes != null
      ? run.num_lanes
      : (nodes.length ? Math.max(...nodes.map((n) => n.lane)) + 1 : 0);

    const contentHeight = ROW_PITCH * numLanes + 2 * MARGIN_TOP;
    const width = MARGIN_LEFT * 2 + COL_PITCH * Math.max(0, numGenerations - 1) + 70;

    const positions = new Map(); // node_id -> {x, y}
    for (const n of nodes) {
      const x = MARGIN_LEFT + COL_PITCH * n.gen;
      const y = MARGIN_TOP + ROW_PITCH * n.lane;
      positions.set(n.node_id, { x, y });
    }

    const fmin = run.fitness_min != null ? run.fitness_min : 0;
    const fmax = run.fitness_max != null ? run.fitness_max : 1;
    const lowerIsBetter = run.fitness_lower_is_better !== false;

    return {
      positions, width, height: contentHeight,
      colorFor(fitness) {
        if (fitness == null || fmax === fmin) return "#8a94a6";
        // viridis(0) is dark purple, viridis(1) is bright yellow. With
        // lowerIsBetter, low fitness should map to t=0 (dark=best) -- which
        // (fitness-fmin)/(fmax-fmin) already does, so no inversion needed
        // there; invert only when higher fitness is the better direction.
        let t = (fitness - fmin) / (fmax - fmin);
        if (!lowerIsBetter) t = 1 - t;
        return viridis(t);
      },
    };
  }

  // ── Tree rendering ──────────────────────────────────────────────────────

  function renderTree(nodes, edges, layout) {
    const svg = document.getElementById("tree-svg");
    svg.setAttribute("width", layout.width);
    svg.setAttribute("height", layout.height);
    svg.innerHTML = "";

    const edgeLayer = el("g", { class: "edge-layer" }, svg);
    const nodeLayer = el("g", { class: "node-layer" }, svg);

    // Every edge spans exactly one generation-column by construction
    // (ancestry_tree_common.py bridges any gap with passthrough nodes), so
    // "survival" edges are always a perfectly horizontal one-column hop
    // (same lane at both ends) and "reproduction" edges span one column at
    // whatever lane difference the parent/child happen to sit at.
    for (const e of edges) {
      const p0 = layout.positions.get(e.from_node_id);
      const p1 = layout.positions.get(e.to_node_id);
      if (!p0 || !p1) continue;
      const cls = "edge-line" + (e.kind === "survival" ? " survival" : "");
      el("path", {
        class: cls,
        d: `M ${p0.x} ${p0.y} C ${p0.x + COL_PITCH * 0.5} ${p0.y}, ` +
           `${p1.x - COL_PITCH * 0.5} ${p1.y}, ${p1.x} ${p1.y}`,
      }, edgeLayer);
    }

    const nodeGroups = new Map();
    for (const n of nodes) {
      const p = layout.positions.get(n.node_id);
      if (!p) continue;
      const isInvalid = !!n.is_invalid;
      const isPassthrough = !!n.is_passthrough;
      const hasImages = !!n.has_images;

      let cls = "node-dot";
      if (isInvalid) cls += " invalid";
      else if (isPassthrough) cls += " passthrough";
      else if (!hasImages) cls += " no-images";

      const g = el("g", {
        class: cls,
        "data-node-id": n.node_id,
        transform: `translate(${p.x},${p.y})`,
      }, nodeLayer);

      el("circle", { class: "dot-hit", r: HIT_R }, g);

      let fill, stroke, strokeWidth, dash, radius = DOT_R;
      if (isInvalid) {
        fill = "var(--invalid-fill)"; stroke = "var(--invalid-stroke)";
        strokeWidth = 1; dash = "";
      } else if (isPassthrough) {
        // Synthetic "known alive, not re-evaluated this round" placeholder
        // -- no real data exists for this generation, so it's drawn small
        // and hollow to read as clearly distinct from a real evaluation.
        fill = "none"; stroke = "var(--passthrough-stroke)";
        strokeWidth = 1.2; dash = ""; radius = DOT_R * 0.5;
      } else if (hasImages) {
        fill = layout.colorFor(n.fitness); stroke = "#ffffff";
        strokeWidth = 1.2; dash = "";
      } else {
        fill = "var(--dim-fill)"; stroke = "#9aa1ad";
        strokeWidth = 1; dash = "2 2";
      }
      el("circle", {
        class: "dot-fill", r: radius, fill, stroke,
        "stroke-width": strokeWidth, "stroke-dasharray": dash,
      }, g);

      let labelText = "";
      let labelDim = false;
      if (isInvalid) { labelText = "invalid"; labelDim = true; }
      else if (isPassthrough) { labelText = ""; }
      else if (n.fitness_display != null) { labelText = n.fitness_display; labelDim = !hasImages; }

      el("text", {
        class: "node-label" + (labelDim ? " dim" : ""),
        x: LABEL_OFFSET_X, y: 0,
      }, g).textContent = labelText;

      nodeGroups.set(n.node_id, g);
    }

    return { svg, nodeGroups };
  }

  // ── Hover + click interaction ────────────────────────────────────────────

  const state = { lockedNodeIds: [] };

  function setupInteractions(nodes, nodesById, svgEls) {
    const hoverCard = document.getElementById("hover-card");

    for (const [nodeId, g] of svgEls.nodeGroups) {
      const node = nodesById.get(nodeId);

      g.addEventListener("mouseenter", (ev) => showHoverCard(node, ev));
      g.addEventListener("mousemove", (ev) => positionHoverCard(ev));
      g.addEventListener("mouseleave", () => hideHoverCard());
      g.addEventListener("click", () => toggleLock(nodeId, nodesById, svgEls));
    }

    function showHoverCard(node, ev) {
      hoverCard.innerHTML = "";
      if (node.has_images) {
        if (node.genome_image) {
          const img = document.createElement("img");
          img.src = node.genome_image;
          hoverCard.appendChild(img);
        }
        if (node.phenotype_image) {
          const img = document.createElement("img");
          img.src = node.phenotype_image;
          hoverCard.appendChild(img);
        }
        const cap = document.createElement("div");
        cap.className = "hover-caption";
        cap.textContent = captionFor(node);
        hoverCard.appendChild(cap);
      } else {
        const p = document.createElement("div");
        p.className = "hover-plain";
        if (node.is_invalid) {
          p.textContent = "failed genome validity check — never evaluated (no data logged for this individual)";
        } else if (node.is_passthrough) {
          p.textContent = "still alive this generation, not re-evaluated — no data logged";
        } else {
          p.textContent = "no rendered images for this individual";
        }
        hoverCard.appendChild(p);
      }
      hoverCard.classList.remove("hidden");
      positionHoverCard(ev);
    }

    function positionHoverCard(ev) {
      const pad = 16;
      const rect = hoverCard.getBoundingClientRect();
      let left = ev.clientX + pad;
      let top = ev.clientY + pad;
      if (left + rect.width > window.innerWidth) left = ev.clientX - rect.width - pad;
      if (top + rect.height > window.innerHeight) top = ev.clientY - rect.height - pad;
      hoverCard.style.left = Math.max(4, left) + "px";
      hoverCard.style.top = Math.max(4, top) + "px";
    }

    function hideHoverCard() {
      hoverCard.classList.add("hidden");
    }
  }

  function captionFor(node) {
    return `gen ${node.gen} · ind ${node.ind_id} · fitness ${node.fitness_display}`;
  }

  function toggleLock(nodeId, nodesById, svgEls) {
    const node = nodesById.get(nodeId);
    if (!node.has_images) return;

    const idx = state.lockedNodeIds.indexOf(nodeId);
    if (idx >= 0) {
      state.lockedNodeIds.splice(idx, 1);
    } else if (state.lockedNodeIds.length < MAX_SLOTS) {
      state.lockedNodeIds.push(nodeId);
    } else {
      showToast(`${MAX_SLOTS} slots full — unlock one first`);
      return;
    }

    state.lockedNodeIds.sort((a, b) => {
      const na = nodesById.get(a), nb = nodesById.get(b);
      return na.gen - nb.gen || na.ind_id - nb.ind_id;
    });

    for (const [id, g] of svgEls.nodeGroups) {
      g.classList.toggle("locked", state.lockedNodeIds.includes(id));
    }
    renderSlots(nodesById);
  }

  let toastTimer = null;
  function showToast(msg) {
    const toast = document.getElementById("toast");
    toast.textContent = msg;
    toast.classList.remove("hidden");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.add("hidden"), 1800);
  }

  // ── Slots panel ───────────────────────────────────────────────────────────

  function setupSlots(nodesById) {
    renderSlots(nodesById);
  }

  function renderSlots(nodesById) {
    const panel = document.getElementById("slots-panel");
    const oldGrid = document.getElementById("slots-grid");
    if (oldGrid) oldGrid.remove();
    const grid = document.createElement("div");
    grid.id = "slots-grid";

    function unpin(nodeId) {
      const svg = document.getElementById("tree-svg");
      const g = svg.querySelector(`[data-node-id="${cssEscape(nodeId)}"]`);
      const idx = state.lockedNodeIds.indexOf(nodeId);
      if (idx >= 0) state.lockedNodeIds.splice(idx, 1);
      if (g) g.classList.remove("locked");
      renderSlots(nodesById);
    }

    for (let i = 0; i < MAX_SLOTS; i++) {
      const nodeId = state.lockedNodeIds[i];
      const col = i + 1;

      if (!nodeId) {
        const empty = document.createElement("div");
        empty.className = "slot-cell empty";
        empty.style.gridColumn = String(col);
        empty.style.gridRow = "1 / span 3";
        empty.textContent = "click a dot to pin it here";
        grid.appendChild(empty);
        continue;
      }

      const node = nodesById.get(nodeId);

      const genomeCell = document.createElement("div");
      genomeCell.className = "slot-cell genome" + (node.genome_image ? "" : " missing");
      genomeCell.style.gridColumn = String(col);
      genomeCell.style.gridRow = "1";
      if (node.genome_image) {
        const im = document.createElement("img");
        im.src = node.genome_image;
        genomeCell.appendChild(im);
      }
      const removeBtn = document.createElement("button");
      removeBtn.className = "slot-remove";
      removeBtn.textContent = "×";
      removeBtn.title = "Unpin";
      removeBtn.addEventListener("click", (ev) => { ev.stopPropagation(); unpin(nodeId); });
      genomeCell.appendChild(removeBtn);
      grid.appendChild(genomeCell);

      const phenotypeCell = document.createElement("div");
      phenotypeCell.className = "slot-cell phenotype" + (node.phenotype_image ? "" : " missing");
      phenotypeCell.style.gridColumn = String(col);
      phenotypeCell.style.gridRow = "2";
      if (node.phenotype_image) {
        const im = document.createElement("img");
        im.src = node.phenotype_image;
        phenotypeCell.appendChild(im);
      }
      grid.appendChild(phenotypeCell);

      const cap = document.createElement("div");
      cap.className = "slot-caption";
      cap.style.gridColumn = String(col);
      cap.style.gridRow = "3";
      cap.textContent = captionFor(node);
      grid.appendChild(cap);
    }

    panel.appendChild(grid);
  }

  function cssEscape(s) {
    return window.CSS && CSS.escape ? CSS.escape(s) : s.replace(/"/g, '\\"');
  }
})();
