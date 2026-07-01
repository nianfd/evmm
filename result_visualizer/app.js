const state = {
  raw: null,
  l1Results: null,
  evidenceIndex: null,
  atomIndex: new Map(),
  nodes: [],
  edges: [],
  filteredNodes: [],
  filteredEdges: [],
  selected: null,
};

const els = {
  fileInput: document.getElementById("fileInput"),
  l1Input: document.getElementById("l1Input"),
  evidenceInput: document.getElementById("evidenceInput"),
  svg: document.getElementById("graphSvg"),
  emptyState: document.getElementById("emptyState"),
  detail: document.getElementById("detailContent"),
  paperTitle: document.getElementById("paperTitle"),
  qualitySummary: document.getElementById("qualitySummary"),
  statProblems: document.getElementById("statProblems"),
  statMethods: document.getElementById("statMethods"),
  statLinks: document.getElementById("statLinks"),
  statConfidence: document.getElementById("statConfidence"),
  nodeKindFilter: document.getElementById("nodeKindFilter"),
  granularityFilter: document.getElementById("granularityFilter"),
  confidenceFilter: document.getElementById("confidenceFilter"),
  confidenceValue: document.getElementById("confidenceValue"),
  searchInput: document.getElementById("searchInput"),
  fitButton: document.getElementById("fitButton"),
  resetButton: document.getElementById("resetButton"),
};

els.fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  loadResult(JSON.parse(text), file.name);
});

els.l1Input.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  state.l1Results = JSON.parse(await file.text());
  state.atomIndex = buildAtomIndex(state.l1Results);
  els.qualitySummary.textContent = `${els.qualitySummary.textContent} · L1 已加载`;
  if (state.selected?.type === "node") {
    const node = state.nodes.find((item) => item.id === state.selected.id);
    if (node) showNode(node);
  }
});

els.evidenceInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  state.evidenceIndex = JSON.parse(await file.text());
  els.qualitySummary.textContent = `${els.qualitySummary.textContent} · 证据索引已加载`;
  if (state.selected?.type === "node") {
    const node = state.nodes.find((item) => item.id === state.selected.id);
    if (node) showNode(node);
  }
});

for (const el of [els.nodeKindFilter, els.granularityFilter, els.confidenceFilter, els.searchInput]) {
  el.addEventListener("input", () => {
    els.confidenceValue.textContent = Number(els.confidenceFilter.value).toFixed(2);
    applyFilters();
  });
}

els.fitButton.addEventListener("click", renderGraph);
els.resetButton.addEventListener("click", () => {
  els.nodeKindFilter.value = "all";
  els.granularityFilter.value = "all";
  els.confidenceFilter.value = "0";
  els.confidenceValue.textContent = "0.00";
  els.searchInput.value = "";
  applyFilters();
});

function loadResult(data, filename) {
  state.raw = data;
  state.selected = null;
  state.nodes = buildNodes(data);
  state.edges = buildEdges(data, state.nodes);
  els.paperTitle.textContent = filename || "04_final_extraction.json";
  updateStats(data);
  updateQuality(data);
  applyFilters();
}

function buildNodes(data) {
  const problems = Array.isArray(data.final_research_problems) ? data.final_research_problems : [];
  const methods = Array.isArray(data.final_methods) ? data.final_methods : [];
  return [
    ...problems.map((item, index) => ({
      id: String(item.id || `RP${index + 1}`),
      kind: "problem",
      title: item.problem || "",
      type: item.problem_type || "other",
      granularity: item.granularity || "unknown",
      confidence: toNumber(item.confidence),
      evidence: array(item.evidence_refs),
      risk: item.risk_note || "",
      raw: item,
    })),
    ...methods.map((item, index) => ({
      id: String(item.id || `M${index + 1}`),
      kind: "method",
      title: item.method || "",
      type: item.method_type || "other",
      granularity: item.granularity || "unknown",
      confidence: toNumber(item.confidence),
      evidence: array(item.evidence_refs),
      risk: item.risk_note || "",
      raw: item,
    })),
  ];
}

function buildEdges(data, nodes) {
  const ids = new Set(nodes.map((node) => node.id));
  const links = Array.isArray(data.problem_method_links) ? data.problem_method_links : [];
  return links
    .filter((link) => ids.has(String(link.problem_id)) && ids.has(String(link.method_id)))
    .map((link, index) => ({
      id: `E${index + 1}`,
      source: String(link.problem_id),
      target: String(link.method_id),
      relation: link.relation || "related",
      linkType: link.link_type || "evidence_supported",
      confidence: toNumber(link.confidence),
      rationale: link.rationale || "",
      raw: link,
    }));
}

function updateStats(data) {
  const all = state.nodes;
  const avg = all.length ? all.reduce((sum, node) => sum + node.confidence, 0) / all.length : 0;
  els.statProblems.textContent = String((data.final_research_problems || []).length);
  els.statMethods.textContent = String((data.final_methods || []).length);
  els.statLinks.textContent = String((data.problem_method_links || []).length);
  els.statConfidence.textContent = avg.toFixed(2);
}

function updateQuality(data) {
  const report = data.quality_report || {};
  const parts = [
    `证据覆盖: ${report.evidence_coverage || "unknown"}`,
    `跨模态使用: ${report.cross_modal_usage || "unknown"}`,
  ];
  els.qualitySummary.textContent = parts.join(" · ");
}

function applyFilters() {
  const kind = els.nodeKindFilter.value;
  const granularity = els.granularityFilter.value;
  const minConfidence = Number(els.confidenceFilter.value);
  const query = els.searchInput.value.trim().toLowerCase();

  state.filteredNodes = state.nodes.filter((node) => {
    if (kind !== "all" && node.kind !== kind) return false;
    if (granularity !== "all" && node.granularity !== granularity) return false;
    if (node.confidence < minConfidence) return false;
    if (query && !nodeSearchText(node).includes(query)) return false;
    return true;
  });

  const ids = new Set(state.filteredNodes.map((node) => node.id));
  state.filteredEdges = state.edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target));
  renderGraph();
}

function renderGraph() {
  const svg = els.svg;
  const width = svg.clientWidth || 900;
  const nodeCount = state.filteredNodes.length;
  const height = Math.max(640, nodeCount * 44 + 120);
  svg.style.height = `${height}px`;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  els.emptyState.style.display = state.filteredNodes.length ? "none" : "grid";
  if (!state.filteredNodes.length) return;

  const problemNodes = state.filteredNodes.filter((node) => node.kind === "problem");
  const methodNodes = state.filteredNodes.filter((node) => node.kind === "method");
  const positions = new Map();
  placeColumn(problemNodes, width * 0.27, height, positions);
  placeColumn(methodNodes, width * 0.73, height, positions);

  const defs = svgEl("defs");
  defs.innerHTML = `
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b93a1"></path>
    </marker>`;
  svg.appendChild(defs);

  for (const edge of state.filteredEdges) {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) continue;
    const edgeClass = edgeClassFor(edge);
    const line = svgEl("path", {
      d: curvedPath(source.x + 34, source.y, target.x - 34, target.y),
      y1: source.y,
      class: edgeClass,
      fill: "none",
      "marker-end": "url(#arrow)",
    });
    line.appendChild(svgEl("title", {}, `${edge.source} -> ${edge.target}: ${edge.relation}`));
    line.addEventListener("click", () => showEdge(edge));
    svg.appendChild(line);
  }

  for (const node of state.filteredNodes) {
    const point = positions.get(node.id);
    if (!point) continue;
    svg.appendChild(drawNode(node, point));
  }
}

function placeColumn(nodes, x, height, positions) {
  const top = 60;
  const bottom = 60;
  const available = Math.max(1, height - top - bottom);
  const gap = available / Math.max(1, nodes.length - 1);
  nodes.forEach((node, index) => {
    positions.set(node.id, { x, y: nodes.length === 1 ? height / 2 : top + gap * index });
  });
}

function drawNode(node, point) {
  const group = svgEl("g", { class: "node", tabindex: "0" });
  group.addEventListener("click", () => showNode(node));
  group.addEventListener("keydown", (event) => {
    if (event.key === "Enter") showNode(node);
  });

  const color = node.kind === "problem" ? "#2364aa" : "#2a9d6f";
  if (node.kind === "problem") {
    group.appendChild(svgEl("circle", { cx: point.x, cy: point.y, r: 23, fill: color }));
  } else {
    group.appendChild(svgEl("rect", { x: point.x - 26, y: point.y - 20, width: 52, height: 40, rx: 8, fill: color }));
  }
  group.appendChild(svgEl("text", { x: point.x, y: point.y + 4, "text-anchor": "middle" }, compactId(node.id)));
  group.appendChild(svgEl("title", {}, `${node.id} · ${node.title}`));
  return group;
}

function curvedPath(x1, y1, x2, y2) {
  const mid = Math.max(80, Math.abs(x2 - x1) * 0.45);
  return `M ${x1} ${y1} C ${x1 + mid} ${y1}, ${x2 - mid} ${y2}, ${x2} ${y2}`;
}

function compactId(id) {
  return String(id).replace(/^RP/, "R").replace(/^M/, "M");
}

function showNode(node) {
  state.selected = { type: "node", id: node.id };
  renderGraph();
  const fields = node.kind === "method" ? methodFields(node.raw) : "";
  els.detail.innerHTML = `
    <h3>${escapeHtml(node.id)} · ${node.kind === "problem" ? "研究问题" : "方法"}</h3>
    <div class="badge-row">
      <span class="badge">${escapeHtml(node.type)}</span>
      <span class="badge">${escapeHtml(node.granularity)}</span>
      <span class="badge">confidence ${node.confidence.toFixed(2)}</span>
    </div>
    <p>${escapeHtml(node.title)}</p>
    ${node.risk ? `<p><strong>风险:</strong> ${escapeHtml(node.risk)}</p>` : ""}
    ${fields}
    <div>
      <strong>证据</strong>
      ${renderEvidence(node.evidence)}
    </div>
  `;
}

function showEdge(edge) {
  state.selected = { type: "edge", id: edge.id };
  renderGraph();
  els.detail.innerHTML = `
    <h3>${escapeHtml(edge.source)} → ${escapeHtml(edge.target)}</h3>
    <div class="badge-row">
      <span class="badge">${escapeHtml(edge.relation)}</span>
      <span class="badge">${escapeHtml(edge.linkType)}</span>
      ${edge.linkType === "inferred" ? `<span class="badge">confidence ${edge.confidence.toFixed(2)}</span>` : ""}
    </div>
    <p>${escapeHtml(edge.rationale || "无关系说明")}</p>
    <pre class="mono">${escapeHtml(JSON.stringify(edge.raw, null, 2))}</pre>
  `;
}

function edgeClassFor(edge) {
  const classes = ["edge"];
  if (edge.linkType === "inferred") classes.push("inferred");
  if (!state.selected) return classes.join(" ");
  if (state.selected.type === "edge") {
    classes.push(state.selected.id === edge.id ? "highlight" : "dimmed");
  }
  if (state.selected.type === "node") {
    classes.push(edge.source === state.selected.id || edge.target === state.selected.id ? "highlight" : "dimmed");
  }
  return classes.join(" ");
}

function methodFields(raw) {
  const fields = raw.reproducibility_fields || {};
  return `
    <div>
      <strong>可复现字段</strong>
      <pre class="mono">${escapeHtml(JSON.stringify(fields, null, 2))}</pre>
    </div>
  `;
}

function renderEvidence(evidence) {
  if (!evidence.length) return "<p>无证据引用</p>";
  return evidence.map((item) => renderEvidenceItem(item)).join("");
}

function renderEvidenceItem(item) {
  if (typeof item === "string") {
    const resolved = resolveEvidenceRef(item);
    if (!resolved) {
      return `
        <div class="evidence">
          <h4>${escapeHtml(item)}</h4>
          <p>这是证据索引。请额外加载 <span class="mono">01_l1_chunk_results.json</span> 和 <span class="mono">02_evidence_index.json</span> 后查看具体内容。</p>
        </div>
      `;
    }
    return `
      <div class="evidence">
        <h4>${escapeHtml(item)} · ${escapeHtml(resolved.section || "")}</h4>
        ${resolved.atom ? `<p><strong>原子内容:</strong> ${escapeHtml(resolved.atom.claim || resolved.atom.problem || resolved.atom.method || "")}</p>` : ""}
        ${resolved.atomEvidence ? `<p><strong>L1 证据:</strong></p><pre class="mono">${escapeHtml(JSON.stringify(resolved.atomEvidence, null, 2))}</pre>` : ""}
        ${resolved.images?.length ? `<p><strong>相关图片:</strong> ${escapeHtml(resolved.images.join(", "))}</p>` : ""}
        ${resolved.textPreview ? `<p><strong>文本预览:</strong></p><div class="preview">${escapeHtml(resolved.textPreview)}</div>` : ""}
      </div>
    `;
  }
  return `<div class="evidence"><pre class="mono">${escapeHtml(JSON.stringify(item, null, 2))}</pre></div>`;
}

function resolveEvidenceRef(ref) {
  const [chunkId, atomId] = String(ref).split(":");
  if (!chunkId) return null;
  const chunkInfo = state.evidenceIndex ? state.evidenceIndex[chunkId] : null;
  const atom = state.atomIndex.get(ref) || null;
  if (!chunkInfo && !atom) return null;
  return {
    section: chunkInfo?.section || atom?.section || "",
    textPreview: chunkInfo?.text_preview || "",
    images: array(chunkInfo?.images),
    atom,
    atomEvidence: atom?.evidence || null,
  };
}

function buildAtomIndex(l1Results) {
  const map = new Map();
  if (!Array.isArray(l1Results)) return map;
  for (const chunk of l1Results) {
    const chunkId = String(chunk.chunk_id || "");
    if (!chunkId) continue;
    for (const atom of array(chunk.research_problem_atoms)) {
      if (atom && atom.id) {
        map.set(`${chunkId}:${atom.id}`, { ...atom, section: chunk.section, atom_kind: "research_problem" });
      }
    }
    for (const atom of array(chunk.method_atoms)) {
      if (atom && atom.id) {
        map.set(`${chunkId}:${atom.id}`, { ...atom, section: chunk.section, atom_kind: "method" });
      }
    }
  }
  return map;
}

function nodeSearchText(node) {
  return JSON.stringify(node.raw || {}, null, 0).toLowerCase();
}

function array(value) {
  return Array.isArray(value) ? value : [];
}

function toNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function truncate(text, max) {
  if (!text) return "";
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}

function svgEl(name, attrs = {}, text = "") {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  if (text) el.textContent = text;
  return el;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
