"""Render a comparison as a single self-contained HTML file.

Everything the page needs — data, styles, script, layout coordinates — is
inlined. The output loads no fonts, no scripts, no stylesheets, and makes no
network requests of any kind, so it works unchanged on an air-gapped machine
and can be copied around as one file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._types import CHANGED_ATTRS, ETYPE, LABEL, SOURCE, STATUS, STATUS_ORDER, TARGET
from ..core.union import UnionDiffGraph
from ..report.report import ComparisonReport
from .focus import select_focus
from .layout import LayoutParams, force_directed_layout

__all__ = ["STATUS_COLORS", "render_html", "write_html"]

#: Status -> colour. SHARED is deliberately muted so differences carry the eye.
STATUS_COLORS: dict[str, str] = {
    "SHARED": "#b4bcc8",
    "A_ONLY": "#2f6fd0",
    "B_ONLY": "#e2761b",
    "CHANGED": "#d6336c",
}

_STATUS_INDEX = {name: i for i, name in enumerate(STATUS_ORDER)}


def _changed_summary(value: Any) -> str:
    """One-line 'attr: old -> new' description for a tooltip."""
    if not isinstance(value, dict):
        return ""
    parts = []
    for attr, change in list(value.items())[:4]:
        if isinstance(change, dict):
            parts.append(f"{attr}: {change.get('a')!r} → {change.get('b')!r}")
        else:  # pragma: no cover - defensive
            parts.append(f"{attr}: {change}")
    return "; ".join(parts)


def _build_payload(
    union: UnionDiffGraph,
    report: ComparisonReport | None,
    *,
    max_nodes: int,
    context_hops: int,
    layout: LayoutParams,
) -> dict[str, Any]:
    focus = select_focus(union, max_nodes=max_nodes, context_hops=context_hops)
    labels = focus.labels
    index_of = pd.Series(np.arange(len(labels)), index=labels)

    node_rows = union.nodes.reindex(labels)
    node_status = node_rows[STATUS].astype(str).map(_STATUS_INDEX).fillna(0).astype(int)

    if len(focus.edges):
        src = index_of.reindex(focus.edges[SOURCE]).to_numpy()
        dst = index_of.reindex(focus.edges[TARGET]).to_numpy()
        edge_status = (
            focus.edges[STATUS].astype(str).map(_STATUS_INDEX).fillna(0).astype(int).to_numpy()
        )
        edge_types = focus.edges[ETYPE].astype(str)
        type_names = sorted(edge_types.unique().tolist())
        type_index = {name: i for i, name in enumerate(type_names)}
        edge_type_idx = edge_types.map(type_index).to_numpy()
        keep = ~(pd.isna(src) | pd.isna(dst))
        edge_array = np.stack(
            [
                src[keep].astype(np.int64),
                dst[keep].astype(np.int64),
                edge_status[keep],
                edge_type_idx[keep],
            ],
            axis=1,
        )
    else:
        type_names = []
        edge_array = np.zeros((0, 4), dtype=np.int64)

    coords = force_directed_layout(
        len(labels), edge_array[:, :2] if len(edge_array) else np.zeros((0, 2)), params=layout
    )

    changed_notes: dict[str, str] = {}
    if CHANGED_ATTRS in node_rows.columns:
        for label, value in node_rows[CHANGED_ATTRS].items():
            summary = _changed_summary(value)
            if summary:
                changed_notes[str(label)] = summary

    edge_notes: list[str] = []
    if len(focus.edges) and CHANGED_ATTRS in focus.edges.columns:
        notes = focus.edges[CHANGED_ATTRS].map(_changed_summary)
        edge_notes = notes[keep].tolist() if len(edge_array) else []

    metrics: list[list[Any]] = []
    top_changed: list[list[Any]] = []
    if report is not None:
        for name, values in report.scalar_scores().items():
            metrics.append([name, _num(values["raw"]), _num(values["shared"])])
        table = report.neighborhood.top_changed(30)
        for row in table.to_dict("records"):
            top_changed.append(
                [
                    str(row[LABEL]),
                    round(float(row["jaccard"]), 4),
                    int(row["n_added"]),
                    int(row["n_removed"]),
                ]
            )

    return {
        "meta": {
            "nameA": union.name_a,
            "nameB": union.name_b,
            "directed": bool(union.directed),
            "truncated": focus.truncated,
            "drawnNodes": len(labels),
            "drawnEdges": len(edge_array),
            "totalNodes": focus.n_nodes_total,
            "totalEdges": focus.n_edges_total,
            "contextHops": focus.context_hops,
            "nodeCounts": union.node_status_counts(),
            "edgeCounts": union.edge_status_counts(),
        },
        "statuses": STATUS_ORDER,
        "colors": [STATUS_COLORS[s] for s in STATUS_ORDER],
        "labels": [str(x) for x in labels],
        "nodeStatus": node_status.tolist(),
        "x": [round(float(v), 5) for v in coords[:, 0]],
        "y": [round(float(v), 5) for v in coords[:, 1]],
        "edges": edge_array.tolist(),
        "edgeTypes": type_names,
        "edgeNotes": edge_notes,
        "changedNotes": changed_notes,
        "metrics": metrics,
        "topChanged": top_changed,
    }


def _num(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else round(value, 6)


def render_html(
    source: ComparisonReport | UnionDiffGraph,
    *,
    max_nodes: int = 2000,
    context_hops: int = 1,
    layout: LayoutParams | None = None,
    title: str | None = None,
) -> str:
    """Render a comparison to a self-contained HTML document.

    Parameters
    ----------
    source:
        A :class:`~graphdiff.report.ComparisonReport` (preferred — its metrics
        populate the side panel) or a bare
        :class:`~graphdiff.core.UnionDiffGraph`.
    max_nodes:
        Cap on drawn nodes. Differing nodes are kept ahead of context.
    context_hops:
        Hops of unchanged neighborhood drawn around the differences.
    layout:
        Force-directed layout tuning; see :class:`~graphdiff.viewer.LayoutParams`.
    title:
        Document title; defaults to ``"graphdiff: A vs B"``.

    Returns
    -------
    str
        A complete HTML document that loads no external resources.

    Raises
    ------
    ValueError
        If a report is passed that no longer carries its union diff graph
        (``compare(..., keep_union=False)``).
    """
    if isinstance(source, ComparisonReport):
        report: ComparisonReport | None = source
        if source.union is None:
            raise ValueError(
                "report carries no union diff graph; re-run compare() with keep_union=True"
            )
        union = source.union
    else:
        report = None
        union = source

    payload = _build_payload(
        union,
        report,
        max_nodes=max_nodes,
        context_hops=context_hops,
        layout=layout or LayoutParams(),
    )
    doc_title = title or f"graphdiff: {union.name_a} vs {union.name_b}"
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return _TEMPLATE.replace("__TITLE__", _escape(doc_title)).replace("__DATA__", data)


def write_html(source: ComparisonReport | UnionDiffGraph, path: str | Path, **kwargs: Any) -> Path:
    """Render and write to ``path``. Returns the path written."""
    path = Path(path)
    path.write_text(render_html(source, **kwargs), encoding="utf-8")
    return path


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #ffffff; --panel: #f7f8fa; --line: #e3e6ea; --ink: #1c2530;
    --muted: #667085; --canvas: #fcfcfd;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font: 13px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    color: var(--ink); background: var(--bg); display: flex; overflow: hidden;
  }
  #stage { position: relative; flex: 1 1 auto; background: var(--canvas); }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas.dragging { cursor: grabbing; }
  #side {
    flex: 0 0 320px; border-left: 1px solid var(--line); background: var(--panel);
    overflow-y: auto; padding: 16px;
  }
  h1 { font-size: 15px; margin: 0 0 2px; }
  h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
       color: var(--muted); margin: 20px 0 8px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 11px; margin-bottom: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; font-weight: 600; color: var(--muted); font-size: 10px;
       text-transform: uppercase; letter-spacing: .05em; padding: 4px 6px 4px 0; }
  td { padding: 3px 6px 3px 0; border-top: 1px solid var(--line); }
  td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
  .legend { display: flex; flex-direction: column; gap: 4px; }
  .legend label { display: flex; align-items: center; gap: 8px; cursor: pointer;
                  padding: 3px 4px; border-radius: 4px; }
  .legend label:hover { background: #eceef2; }
  .swatch { width: 11px; height: 11px; border-radius: 3px; flex: 0 0 auto; }
  .count { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
  input[type=search] {
    width: 100%; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px;
    font: inherit; background: #fff;
  }
  #hits { margin-top: 6px; max-height: 130px; overflow-y: auto; }
  .hit, .row { cursor: pointer; padding: 3px 4px; border-radius: 4px; }
  .hit:hover, .row:hover { background: #eceef2; }
  #tip {
    position: absolute; pointer-events: none; background: #1c2530; color: #fff;
    padding: 6px 9px; border-radius: 6px; font-size: 12px; max-width: 300px;
    opacity: 0; transition: opacity .08s; z-index: 5;
  }
  #hud {
    position: absolute; left: 12px; top: 12px; background: rgba(255,255,255,.92);
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 11px; font-size: 11px;
    color: var(--muted);
  }
  #hud b { color: var(--ink); font-variant-numeric: tabular-nums; }
  button {
    font: inherit; padding: 5px 10px; border: 1px solid var(--line); background: #fff;
    border-radius: 6px; cursor: pointer; color: var(--ink);
  }
  button:hover { background: #eceef2; }
  .note { color: var(--muted); font-size: 11px; margin-top: 8px; }
</style>
</head>
<body>
<div id="stage">
  <canvas id="c"></canvas>
  <div id="hud"></div>
  <div id="tip"></div>
</div>
<aside id="side">
  <h1 id="ttl"></h1>
  <div class="sub" id="scale"></div>

  <h2>Show</h2>
  <div class="legend" id="legend"></div>
  <div style="margin-top:10px"><button id="reset">Reset view</button></div>

  <h2>Find node</h2>
  <input type="search" id="q" placeholder="Type a label…" autocomplete="off">
  <div id="hits"></div>

  <h2>Most-changed nodes</h2>
  <div class="sub">Neighborhood Jaccard — click to focus</div>
  <table id="changed"><thead><tr>
    <th>Node</th><th class="n">Jac</th><th class="n">+</th><th class="n">&minus;</th>
  </tr></thead><tbody></tbody></table>

  <h2>Scores</h2>
  <table id="metrics"><thead><tr>
    <th>Metric</th><th class="n">Raw</th><th class="n">Shared</th>
  </tr></thead><tbody></tbody></table>
  <div class="note">“Shared” restricts each metric to nodes present in both graphs.</div>
</aside>

<script>
const D = __DATA__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip'), stage = document.getElementById('stage');
const N = D.labels.length;
const on = D.statuses.map(() => true);
let view = {k: 1, x: 0, y: 0}, hover = -1, W = 0, H = 0, dpr = 1;

document.getElementById('ttl').textContent = D.meta.nameA + '  vs  ' + D.meta.nameB;
document.getElementById('scale').textContent =
  D.meta.drawnNodes.toLocaleString() + ' of ' + D.meta.totalNodes.toLocaleString() + ' nodes, ' +
  D.meta.drawnEdges.toLocaleString() + ' of ' + D.meta.totalEdges.toLocaleString() + ' edges drawn' +
  (D.meta.truncated ? ' (differences prioritized)' : '');

// ---- legend -------------------------------------------------------------
const legend = document.getElementById('legend');
D.statuses.forEach((s, i) => {
  const nc = D.meta.nodeCounts[s] || 0, ec = D.meta.edgeCounts[s] || 0;
  const lab = document.createElement('label');
  lab.innerHTML = '<input type="checkbox" checked data-i="' + i + '">' +
    '<span class="swatch" style="background:' + D.colors[i] + '"></span>' +
    '<span>' + s + '</span><span class="count">' + nc.toLocaleString() + 'n / ' +
    ec.toLocaleString() + 'e</span>';
  lab.querySelector('input').addEventListener('change', e => {
    on[+e.target.dataset.i] = e.target.checked; draw();
  });
  legend.appendChild(lab);
});

// ---- tables -------------------------------------------------------------
const mt = document.querySelector('#metrics tbody');
D.metrics.forEach(([name, raw, shared]) => {
  const tr = document.createElement('tr');
  const f = v => v === null ? '—' : v.toFixed(4);
  tr.innerHTML = '<td>' + name + '</td><td class="n">' + f(raw) + '</td><td class="n">' +
    f(shared) + '</td>';
  mt.appendChild(tr);
});

const ct = document.querySelector('#changed tbody');
D.topChanged.forEach(([label, jac, add, rem]) => {
  const tr = document.createElement('tr');
  tr.className = 'row';
  tr.innerHTML = '<td>' + esc(label) + '</td><td class="n">' + jac.toFixed(3) +
    '</td><td class="n">' + add + '</td><td class="n">' + rem + '</td>';
  tr.addEventListener('click', () => focusLabel(label));
  ct.appendChild(tr);
});

// ---- search -------------------------------------------------------------
const q = document.getElementById('q'), hits = document.getElementById('hits');
q.addEventListener('input', () => {
  const term = q.value.trim().toLowerCase();
  hits.innerHTML = '';
  if (!term) { draw(); return; }
  let shown = 0;
  for (let i = 0; i < N && shown < 25; i++) {
    if (D.labels[i].toLowerCase().includes(term)) {
      const d = document.createElement('div');
      d.className = 'hit'; d.textContent = D.labels[i];
      d.addEventListener('click', () => focusLabel(D.labels[i]));
      hits.appendChild(d); shown++;
    }
  }
  draw();
});

function focusLabel(label) {
  const i = D.labels.indexOf(label);
  if (i < 0) return;
  view.k = 4;
  view.x = W / 2 - D.x[i] * span() * view.k - pad();
  view.y = H / 2 - D.y[i] * span() * view.k - pad();
  hover = i; draw(); showTip(i, W / 2, H / 2);
}

// ---- geometry -----------------------------------------------------------
function pad() { return 30; }
function span() { return Math.min(W, H) - 2 * pad(); }
function sx(i) { return D.x[i] * span() * view.k + view.x + pad(); }
function sy(i) { return D.y[i] * span() * view.k + view.y + pad(); }

function resize() {
  dpr = window.devicePixelRatio || 1;
  W = stage.clientWidth; H = stage.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  const term = q.value.trim().toLowerCase();

  ctx.lineWidth = Math.min(1.4, 0.5 + view.k * 0.12);
  for (const [a, b, st] of D.edges) {
    if (!on[st]) continue;
    ctx.strokeStyle = D.colors[st];
    ctx.globalAlpha = st === 0 ? 0.22 : 0.62;
    ctx.beginPath(); ctx.moveTo(sx(a), sy(a)); ctx.lineTo(sx(b), sy(b)); ctx.stroke();
  }

  ctx.globalAlpha = 1;
  const r = Math.max(1.6, Math.min(5, 1.7 + view.k * 0.5));
  for (let i = 0; i < N; i++) {
    const st = D.nodeStatus[i];
    if (!on[st]) continue;
    const match = term && D.labels[i].toLowerCase().includes(term);
    // Differing nodes are drawn larger: the whole point of the view is that
    // they should be findable without hunting.
    const rr = (st === 0 ? r : r * 1.8) + (match ? 3 : 0);
    ctx.beginPath();
    ctx.arc(sx(i), sy(i), rr, 0, 6.2832);
    ctx.fillStyle = D.colors[st];
    ctx.globalAlpha = st === 0 && term ? 0.35 : 1;
    ctx.fill();
    if (match || i === hover) {
      ctx.lineWidth = 2; ctx.strokeStyle = '#1c2530'; ctx.globalAlpha = 1; ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  if (view.k > 2.2) {
    ctx.fillStyle = '#1c2530';
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
    for (let i = 0; i < N; i++) {
      if (!on[D.nodeStatus[i]]) continue;
      const x = sx(i), y = sy(i);
      if (x < -40 || x > W + 40 || y < -20 || y > H + 20) continue;
      ctx.fillText(D.labels[i], x + 7, y + 3);
    }
  }

  document.getElementById('hud').innerHTML =
    'zoom <b>' + view.k.toFixed(1) + '&times;</b> &middot; drag to pan &middot; scroll to zoom' +
    (view.k > 2.2 ? '' : ' &middot; zoom in for labels');
}

// ---- interaction --------------------------------------------------------
let drag = null;
cv.addEventListener('mousedown', e => {
  drag = {x: e.offsetX - view.x, y: e.offsetY - view.y}; cv.classList.add('dragging');
});
window.addEventListener('mouseup', () => { drag = null; cv.classList.remove('dragging'); });
cv.addEventListener('mousemove', e => {
  if (drag) { view.x = e.offsetX - drag.x; view.y = e.offsetY - drag.y; draw(); return; }
  const i = pick(e.offsetX, e.offsetY);
  if (i !== hover) { hover = i; draw(); }
  if (i >= 0) showTip(i, e.offsetX, e.offsetY); else tip.style.opacity = 0;
});
cv.addEventListener('mouseleave', () => { tip.style.opacity = 0; hover = -1; draw(); });
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.16 : 1 / 1.16;
  const nk = Math.max(0.3, Math.min(60, view.k * f));
  const ratio = nk / view.k;
  view.x = e.offsetX - (e.offsetX - view.x) * ratio;
  view.y = e.offsetY - (e.offsetY - view.y) * ratio;
  view.k = nk; draw();
}, {passive: false});
document.getElementById('reset').addEventListener('click', () => {
  view = {k: 1, x: 0, y: 0}; q.value = ''; hits.innerHTML = ''; draw();
});

function pick(px, py) {
  let best = -1, bd = 100;
  for (let i = 0; i < N; i++) {
    if (!on[D.nodeStatus[i]]) continue;
    const dx = sx(i) - px, dy = sy(i) - py, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function showTip(i, px, py) {
  const st = D.statuses[D.nodeStatus[i]];
  const note = D.changedNotes[D.labels[i]];
  tip.innerHTML = '<b>' + esc(D.labels[i]) + '</b><br>' + st +
    (note ? '<br>' + esc(note) : '');
  tip.style.left = Math.min(px + 14, W - 310) + 'px';
  tip.style.top = (py + 14) + 'px';
  tip.style.opacity = 1;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
}

window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""
