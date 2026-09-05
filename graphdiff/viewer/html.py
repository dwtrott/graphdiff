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
from .theme import DARK, LIGHT, SHAPES, STATUS_COLORS, status_labels

__all__ = ["STATUS_COLORS", "render_html", "write_html"]

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

    edge_notes: list[str] = []
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
        if CHANGED_ATTRS in focus.edges.columns:
            edge_notes = focus.edges.loc[keep, CHANGED_ATTRS].map(_changed_summary).tolist()
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

    metrics: list[list[Any]] = []
    top_changed: list[list[Any]] = []
    headline: dict[str, Any] = {}
    if report is not None:
        for name, values in report.scalar_scores().items():
            metrics.append([name, _num(values["raw"]), _num(values["shared"])])
        for row in report.neighborhood.top_changed(40).to_dict("records"):
            top_changed.append(
                [
                    str(row[LABEL]),
                    round(float(row["jaccard"]), 4),
                    int(row["n_added"]),
                    int(row["n_removed"]),
                ]
            )
        headline = {
            "similarity": _num(report.score("ged_similarity")),
            "edgeJaccard": _num(report.score("jaccard_typed_edges")),
        }

    node_counts = union.node_status_counts()
    edge_counts = union.edge_status_counts()

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
            "nodeCounts": node_counts,
            "edgeCounts": edge_counts,
            "headline": headline,
        },
        "statuses": STATUS_ORDER,
        "statusLabels": [status_labels(union.name_a, union.name_b)[s] for s in STATUS_ORDER],
        "shapes": [SHAPES[s] for s in STATUS_ORDER],
        "themes": {
            "light": {**LIGHT, "series": [LIGHT[s] for s in STATUS_ORDER]},
            "dark": {**DARK, "series": [DARK[s] for s in STATUS_ORDER]},
        },
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
    color-scheme: light;
    --surface: #fcfcfb; --panel: #f4f4f2; --line: #e0e0dc;
    --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #7b7a75;
    --hover: rgba(0,0,0,.05); --tip-bg: #0b0b0b; --tip-ink: #fcfcfb;
    --shadow: 0 1px 2px rgba(0,0,0,.06), 0 4px 12px rgba(0,0,0,.05);
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #232322; --line: #373735;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8d8c84;
    --hover: rgba(255,255,255,.07); --tip-bg: #f4f4f2; --tip-ink: #0b0b0b;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 4px 14px rgba(0,0,0,.35);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font: 13px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    color: var(--ink); background: var(--surface); display: flex; overflow: hidden;
    -webkit-font-smoothing: antialiased;
  }
  #stage { position: relative; flex: 1 1 auto; min-width: 0; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas.dragging { cursor: grabbing; }

  #side {
    flex: 0 0 336px; border-left: 1px solid var(--line); background: var(--panel);
    overflow-y: auto; overscroll-behavior: contain;
  }
  .pad { padding: 14px 16px; }
  .divide { border-top: 1px solid var(--line); }

  h1 { font-size: 14px; margin: 0; font-weight: 600; letter-spacing: -.01em; }
  .sub { color: var(--ink-3); font-size: 11px; margin-top: 3px; }
  h2 {
    font-size: 10px; text-transform: uppercase; letter-spacing: .07em;
    color: var(--ink-3); margin: 0 0 9px; font-weight: 600;
  }

  /* stat tiles ---------------------------------------------------------- */
  .tiles { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
  .tile {
    background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
    padding: 8px 9px;
  }
  .tile .k { font-size: 9px; text-transform: uppercase; letter-spacing: .06em;
             color: var(--ink-3); font-weight: 600; }
  .tile .v { font-size: 17px; font-weight: 600; margin-top: 2px;
             font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
  .tile .d { font-size: 10px; color: var(--ink-3); font-variant-numeric: tabular-nums; }

  /* legend -------------------------------------------------------------- */
  .legend { display: flex; flex-direction: column; gap: 1px; }
  .legend label {
    display: flex; align-items: center; gap: 9px; cursor: pointer;
    padding: 5px 6px; border-radius: 6px; user-select: none;
  }
  .legend label:hover { background: var(--hover); }
  .legend input { margin: 0; flex: 0 0 auto; accent-color: var(--ink-2); }
  .legend .name { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis;
                  white-space: nowrap; }
  .legend .count { color: var(--ink-3); font-size: 11px; font-variant-numeric: tabular-nums;
                   flex: 0 0 auto; }
  .legend label.off .name, .legend label.off .count { opacity: .42; }
  svg.glyph { flex: 0 0 auto; display: block; }

  /* controls ------------------------------------------------------------ */
  .row { display: flex; gap: 6px; flex-wrap: wrap; }
  button {
    font: inherit; font-size: 12px; padding: 5px 10px; border: 1px solid var(--line);
    background: var(--surface); border-radius: 7px; cursor: pointer; color: var(--ink);
  }
  button:hover { background: var(--hover); }
  button[aria-pressed="true"] { background: var(--ink); color: var(--surface);
                                border-color: var(--ink); }
  input[type=search] {
    width: 100%; padding: 7px 9px; border: 1px solid var(--line); border-radius: 7px;
    font: inherit; background: var(--surface); color: var(--ink);
  }
  input[type=search]:focus { outline: 2px solid var(--ink-3); outline-offset: -1px; }

  /* tables -------------------------------------------------------------- */
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; font-weight: 600; color: var(--ink-3); font-size: 9px;
       text-transform: uppercase; letter-spacing: .06em; padding: 0 6px 5px 0; }
  td { padding: 4px 6px 4px 0; border-top: 1px solid var(--line); }
  td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: var(--hover); }
  tbody tr.sel { background: var(--hover); }
  .hit { cursor: pointer; padding: 4px 6px; border-radius: 5px; }
  .hit:hover { background: var(--hover); }
  #hits { margin-top: 6px; max-height: 132px; overflow-y: auto; }

  /* selection card ------------------------------------------------------ */
  #card { display: none; }
  #card.on { display: block; }
  .chip {
    display: inline-flex; align-items: center; gap: 6px; font-size: 11px;
    border: 1px solid var(--line); border-radius: 999px; padding: 2px 9px 2px 6px;
    background: var(--surface);
  }
  .nbr { font-size: 11px; color: var(--ink-2); margin-top: 6px; }
  .nbr b { color: var(--ink); font-weight: 600; }
  .nbr .list { color: var(--ink-3); }

  details > summary { cursor: pointer; list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary h2 { margin: 0; }
  details > summary::after { content: " ▸"; color: var(--ink-3); }
  details[open] > summary::after { content: " ▾"; }

  /* canvas overlays ----------------------------------------------------- */
  #tip {
    position: absolute; pointer-events: none; background: var(--tip-bg); color: var(--tip-ink);
    padding: 7px 10px; border-radius: 7px; font-size: 12px; max-width: 320px;
    opacity: 0; transition: opacity .09s; z-index: 5; box-shadow: var(--shadow);
    line-height: 1.4;
  }
  #hud {
    position: absolute; left: 14px; bottom: 14px; color: var(--ink-3); font-size: 11px;
    background: var(--surface); border: 1px solid var(--line); border-radius: 7px;
    padding: 5px 10px; opacity: .93;
  }
  #hud b { color: var(--ink-2); font-variant-numeric: tabular-nums; }
  #banner {
    position: absolute; left: 14px; top: 14px; font-size: 11px; color: var(--ink-3);
    background: var(--surface); border: 1px solid var(--line); border-radius: 7px;
    padding: 5px 10px; opacity: .93;
  }
</style>
</head>
<body>
<div id="stage">
  <canvas id="c"></canvas>
  <div id="banner"></div>
  <div id="hud"></div>
  <div id="tip"></div>
</div>

<aside id="side">
  <div class="pad">
    <h1 id="ttl"></h1>
    <div class="sub" id="scale"></div>
  </div>

  <div class="pad divide">
    <h2>Summary</h2>
    <div class="tiles" id="tiles"></div>
  </div>

  <div class="pad divide">
    <h2>Legend &mdash; click to show or hide</h2>
    <div class="legend" id="legend"></div>
    <div class="row" style="margin-top:10px">
      <button id="diffonly" aria-pressed="false">Differences only</button>
      <button id="reset">Reset view</button>
      <button id="theme" title="Toggle light/dark">Dark</button>
    </div>
  </div>

  <div class="pad divide" id="card">
    <h2>Selected</h2>
    <div id="cardbody"></div>
  </div>

  <div class="pad divide">
    <h2>Find node</h2>
    <input type="search" id="q" placeholder="Type a label…" autocomplete="off" spellcheck="false">
    <div id="hits"></div>
  </div>

  <div class="pad divide">
    <h2>Most-changed nodes</h2>
    <table id="changed"><thead><tr>
      <th>Node</th><th class="n">Jac</th><th class="n">+</th><th class="n">&minus;</th>
    </tr></thead><tbody></tbody></table>
  </div>

  <div class="pad divide">
    <details>
      <summary><h2 style="display:inline">Scores</h2></summary>
      <table id="metrics" style="margin-top:9px"><thead><tr>
        <th>Metric</th><th class="n">Raw</th><th class="n">Shared</th>
      </tr></thead><tbody></tbody></table>
      <div class="sub" style="margin-top:7px">
        “Shared” restricts each metric to nodes present in both graphs.
      </div>
    </details>
  </div>
</aside>

<script>
const D = __DATA__;
const N = D.labels.length;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip'), stage = document.getElementById('stage');
const q = document.getElementById('q'), hits = document.getElementById('hits');

const on = D.statuses.map(() => true);
let T = D.themes.light, mode = 'light';
let view = {k: 1, x: 0, y: 0}, hover = -1, sel = -1, W = 0, H = 0;

// Incident edges per node, so hover/selection can dim everything else.
const inc = Array.from({length: N}, () => []);
D.edges.forEach(([a, b], i) => { inc[a].push(i); inc[b].push(i); });

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const fmt = v => v === null || v === undefined ? '—' : v.toFixed(4);

// ---- shapes -------------------------------------------------------------
function path(g, shape, x, y, r) {
  g.beginPath();
  if (shape === 'circle') { g.arc(x, y, r, 0, 6.2832); return; }
  if (shape === 'square') { g.rect(x - r * .92, y - r * .92, r * 1.84, r * 1.84); return; }
  if (shape === 'triangle') {
    g.moveTo(x, y - r * 1.24); g.lineTo(x + r * 1.12, y + r * .84);
    g.lineTo(x - r * 1.12, y + r * .84); g.closePath(); return;
  }
  g.moveTo(x, y - r * 1.3); g.lineTo(x + r * 1.3, y);   // diamond
  g.lineTo(x, y + r * 1.3); g.lineTo(x - r * 1.3, y); g.closePath();
}

// Built as an HTML string rather than createElementNS: the SVG namespace URI is
// never fetched, but keeping it out of the file lets the offline test stay a
// strict "no URLs at all" check instead of a check with an exception list.
function glyphHTML(i, size) {
  const s = size || 13, r = s * .38, c = D.themes[mode].series[i];
  const rec = {
    d: '',
    beginPath() { this.d = ''; },
    arc(x, y, rr) {
      this.d += `M${x - rr} ${y}a${rr} ${rr} 0 1 0 ${rr * 2} 0a${rr} ${rr} 0 1 0 ${-rr * 2} 0Z`;
    },
    rect(x, y, w, h) { this.d += `M${x} ${y}h${w}v${h}h${-w}Z`; },
    moveTo(x, y) { this.d += `M${x} ${y}`; },
    lineTo(x, y) { this.d += `L${x} ${y}`; },
    closePath() { this.d += 'Z'; }
  };
  path(rec, D.shapes[i], s / 2, s / 2, r);
  return `<svg class="glyph" width="${s}" height="${s}" viewBox="0 0 ${s} ${s}" ` +
    `aria-hidden="true"><path d="${rec.d}" fill="${c}"/></svg>`;
}

function glyph(i, size) {
  const t = document.createElement('template');
  t.innerHTML = glyphHTML(i, size);
  return t.content.firstChild;
}

// ---- side panel ---------------------------------------------------------
document.getElementById('ttl').textContent = D.meta.nameA + '  vs  ' + D.meta.nameB;
document.getElementById('scale').textContent =
  (D.meta.directed ? 'Directed' : 'Undirected') + ' · ' +
  D.meta.totalNodes.toLocaleString() + ' nodes, ' +
  D.meta.totalEdges.toLocaleString() + ' edges in the union';

function tiles() {
  const nc = D.meta.nodeCounts, ec = D.meta.edgeCounts, h = D.meta.headline || {};
  const box = document.getElementById('tiles');
  const sim = h.similarity === null || h.similarity === undefined
    ? '—' : (h.similarity * 100).toFixed(0) + '%';
  box.innerHTML =
    tile('Node changes', (nc.B_ONLY + nc.A_ONLY).toLocaleString(),
         `+${nc.B_ONLY} added / &minus;${nc.A_ONLY} gone`) +
    tile('Edge changes', (ec.B_ONLY + ec.A_ONLY + ec.CHANGED).toLocaleString(),
         `+${ec.B_ONLY} / &minus;${ec.A_ONLY} / ~${ec.CHANGED} reweighted`) +
    tile('Similarity', sim, 'normalized edit distance');
}
function tile(k, v, d) {
  return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div>` +
         `<div class="d">${d}</div></div>`;
}
tiles();

const legend = document.getElementById('legend');
D.statuses.forEach((s, i) => {
  const nc = D.meta.nodeCounts[s] || 0, ec = D.meta.edgeCounts[s] || 0;
  const lab = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.checked = true;
  const name = document.createElement('span');
  name.className = 'name'; name.textContent = D.statusLabels[i];
  const count = document.createElement('span');
  count.className = 'count'; count.textContent = `${nc.toLocaleString()} · ${ec.toLocaleString()}`;
  count.title = `${nc} nodes, ${ec} edges`;
  lab.append(cb, glyph(i), name, count);
  cb.addEventListener('change', () => {
    on[i] = cb.checked; lab.classList.toggle('off', !cb.checked); syncDiffOnly(); draw();
  });
  legend.appendChild(lab);
});

const mt = document.querySelector('#metrics tbody');
D.metrics.forEach(([name, raw, shared]) => {
  const tr = document.createElement('tr');
  tr.style.cursor = 'default';
  tr.innerHTML = `<td>${name}</td><td class="n">${fmt(raw)}</td><td class="n">${fmt(shared)}</td>`;
  mt.appendChild(tr);
});

const ct = document.querySelector('#changed tbody');
D.topChanged.forEach(([label, jac, add, rem]) => {
  const tr = document.createElement('tr');
  tr.dataset.label = label;
  tr.innerHTML = `<td>${esc(label)}</td><td class="n">${jac.toFixed(3)}</td>` +
    `<td class="n">${add}</td><td class="n">${rem}</td>`;
  tr.addEventListener('click', () => focusLabel(label));
  ct.appendChild(tr);
});
if (!D.topChanged.length) {
  ct.innerHTML = '<tr style="cursor:default"><td colspan="4" style="color:var(--ink-3)">' +
    'No shared nodes to compare.</td></tr>';
}

// ---- selection card -----------------------------------------------------
function showCard(i) {
  const card = document.getElementById('card'), body = document.getElementById('cardbody');
  if (i < 0) { card.classList.remove('on'); return; }
  const st = D.nodeStatus[i];
  const buckets = {};
  for (const ei of inc[i]) {
    const [a, b, es] = D.edges[ei];
    const other = a === i ? b : a;
    (buckets[es] = buckets[es] || new Set()).add(D.labels[other]);
  }
  let html = `<div class="chip"><span id="cs"></span><b>${esc(D.labels[i])}</b></div>`;
  html += `<div class="nbr" style="margin-top:8px">${D.statusLabels[st]}</div>`;
  const note = D.changedNotes[D.labels[i]];
  if (note) html += `<div class="nbr">${esc(note)}</div>`;
  D.statuses.forEach((s, si) => {
    const set = buckets[si];
    if (!set || !set.size) return;
    const names = [...set].slice(0, 8).map(esc).join(', ');
    html += `<div class="nbr"><b>${set.size}</b> ${D.statusLabels[si].toLowerCase()}` +
      ` <span class="list">— ${names}${set.size > 8 ? '…' : ''}</span></div>`;
  });
  body.innerHTML = html;
  document.getElementById('cs').appendChild(glyph(st, 12));
  card.classList.add('on');
  [...ct.children].forEach(tr => tr.classList.toggle('sel', tr.dataset.label === D.labels[i]));
}

// ---- search -------------------------------------------------------------
q.addEventListener('input', () => {
  const term = q.value.trim().toLowerCase();
  hits.innerHTML = '';
  if (term) {
    let shown = 0;
    for (let i = 0; i < N && shown < 25; i++) {
      if (D.labels[i].toLowerCase().includes(term)) {
        const d = document.createElement('div');
        d.className = 'hit'; d.textContent = D.labels[i];
        d.addEventListener('click', () => focusLabel(D.labels[i]));
        hits.appendChild(d); shown++;
      }
    }
    if (!shown) hits.innerHTML = '<div class="sub">No match.</div>';
  }
  draw();
});

function focusLabel(label) {
  const i = D.labels.indexOf(label);
  if (i < 0) return;
  view.k = Math.max(view.k, 4.5);
  view.x = W / 2 - D.x[i] * span() * view.k - pad();
  view.y = H / 2 - D.y[i] * span() * view.k - pad();
  sel = i; showCard(i); draw();
}

// ---- controls -----------------------------------------------------------
const diffBtn = document.getElementById('diffonly');
diffBtn.addEventListener('click', () => {
  const nextOn = !(diffBtn.getAttribute('aria-pressed') === 'true');
  const shared = D.statuses.indexOf('SHARED');
  on[shared] = !nextOn;
  [...legend.children][shared].querySelector('input').checked = !nextOn;
  [...legend.children][shared].classList.toggle('off', nextOn);
  diffBtn.setAttribute('aria-pressed', String(nextOn));
  draw();
});
function syncDiffOnly() {
  const shared = D.statuses.indexOf('SHARED');
  diffBtn.setAttribute('aria-pressed', String(!on[shared]));
}
document.getElementById('reset').addEventListener('click', () => {
  view = {k: 1, x: 0, y: 0}; q.value = ''; hits.innerHTML = '';
  sel = -1; showCard(-1); draw();
});
const themeBtn = document.getElementById('theme');
themeBtn.addEventListener('click', () => setTheme(mode === 'light' ? 'dark' : 'light'));
function setTheme(next) {
  mode = next; T = D.themes[next];
  document.documentElement.setAttribute('data-theme', next);
  themeBtn.textContent = next === 'light' ? 'Dark' : 'Light';
  [...legend.children].forEach((lab, i) => {
    lab.replaceChild(glyph(i), lab.querySelector('svg.glyph'));
  });
  if (sel >= 0) showCard(sel);
  draw();
}
// ---- geometry -----------------------------------------------------------
const pad = () => 34;
const span = () => Math.min(W, H) - 2 * pad();
const sx = i => D.x[i] * span() * view.k + view.x + pad();
const sy = i => D.y[i] * span() * view.k + view.y + pad();

function resize() {
  const dpr = window.devicePixelRatio || 1;
  W = stage.clientWidth; H = stage.clientHeight;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

// ---- drawing ------------------------------------------------------------
function draw() {
  ctx.clearRect(0, 0, W, H);
  const term = q.value.trim().toLowerCase();
  const focusNode = hover >= 0 ? hover : sel;
  const near = new Set();
  if (focusNode >= 0) {
    near.add(focusNode);
    for (const ei of inc[focusNode]) {
      const [a, b] = D.edges[ei]; near.add(a); near.add(b);
    }
  }
  const dimming = focusNode >= 0;

  // edges under nodes
  for (let ei = 0; ei < D.edges.length; ei++) {
    const [a, b, st] = D.edges[ei];
    if (!on[st]) continue;
    const lit = !dimming || (near.has(a) && near.has(b));
    const shared = st === D.statuses.indexOf('SHARED');
    ctx.strokeStyle = T.series[st];
    ctx.globalAlpha = lit ? (shared ? .3 : .75) : .06;
    ctx.lineWidth = (shared ? .8 : 1.5) * Math.min(1.7, .75 + view.k * .18);
    ctx.setLineDash(st === D.statuses.indexOf('CHANGED') ? [4, 3] : []);
    ctx.beginPath(); ctx.moveTo(sx(a), sy(a)); ctx.lineTo(sx(b), sy(b)); ctx.stroke();
  }
  ctx.setLineDash([]);

  // nodes, differences last so they land on top
  const base = Math.max(2.2, Math.min(6.5, 2.4 + view.k * .62));
  const order = [...Array(N).keys()].sort(
    (i, j) => (D.nodeStatus[i] === 0 ? 0 : 1) - (D.nodeStatus[j] === 0 ? 0 : 1));
  for (const i of order) {
    const st = D.nodeStatus[i];
    if (!on[st]) continue;
    const match = term && D.labels[i].toLowerCase().includes(term);
    const lit = !dimming || near.has(i);
    const r = (st === 0 ? base : base * 1.5) + (match ? 3 : 0);
    ctx.globalAlpha = lit ? 1 : .12;
    path(ctx, D.shapes[st], sx(i), sy(i), r);
    ctx.fillStyle = T.series[st];
    ctx.fill();
    if (i === sel || i === hover || match) {
      ctx.lineWidth = 2; ctx.strokeStyle = T.ink; ctx.globalAlpha = lit ? 1 : .3; ctx.stroke();
    } else if (!shrunk()) {
      ctx.lineWidth = 1; ctx.strokeStyle = T.surface; ctx.globalAlpha = lit ? .9 : .1; ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  // labels: everything when zoomed in, else just the focused node
  ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif';
  ctx.lineJoin = 'round';
  const labelled = view.k > 2.4 ? order : (focusNode >= 0 ? [focusNode] : []);
  for (const i of labelled) {
    if (!on[D.nodeStatus[i]]) continue;
    if (dimming && !near.has(i)) continue;
    const x = sx(i) + 9, y = sy(i) + 4;
    if (x < -60 || x > W + 60 || y < -20 || y > H + 20) continue;
    ctx.lineWidth = 3; ctx.strokeStyle = T.surface;
    ctx.strokeText(D.labels[i], x, y);
    ctx.fillStyle = T.ink; ctx.fillText(D.labels[i], x, y);
  }

  document.getElementById('hud').innerHTML =
    'zoom <b>' + view.k.toFixed(1) + '&times;</b> · drag to pan · scroll to zoom' +
    (view.k > 2.4 ? '' : ' · zoom in for labels');
  document.getElementById('banner').textContent =
    D.meta.drawnNodes.toLocaleString() + ' nodes / ' + D.meta.drawnEdges.toLocaleString() +
    ' edges shown' + (D.meta.truncated ? ' — differences prioritized' : '');
}
const shrunk = () => view.k < 1.2;

// ---- interaction --------------------------------------------------------
let drag = null, moved = false;
cv.addEventListener('mousedown', e => {
  drag = {x: e.offsetX - view.x, y: e.offsetY - view.y}; moved = false;
  cv.classList.add('dragging');
});
window.addEventListener('mouseup', () => { drag = null; cv.classList.remove('dragging'); });
cv.addEventListener('mousemove', e => {
  if (drag) {
    view.x = e.offsetX - drag.x; view.y = e.offsetY - drag.y; moved = true; draw(); return;
  }
  const i = pick(e.offsetX, e.offsetY);
  if (i !== hover) { hover = i; draw(); }
  if (i >= 0) showTip(i, e.offsetX, e.offsetY); else tip.style.opacity = 0;
});
cv.addEventListener('mouseleave', () => { tip.style.opacity = 0; hover = -1; draw(); });
cv.addEventListener('click', e => {
  if (moved) return;
  const i = pick(e.offsetX, e.offsetY);
  sel = (i === sel) ? -1 : i;
  showCard(sel); draw();
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.16 : 1 / 1.16;
  const nk = Math.max(.3, Math.min(60, view.k * f)), ratio = nk / view.k;
  view.x = e.offsetX - (e.offsetX - view.x) * ratio;
  view.y = e.offsetY - (e.offsetY - view.y) * ratio;
  view.k = nk; draw();
}, {passive: false});
window.addEventListener('keydown', e => {
  if (e.key === 'Escape') { sel = -1; showCard(-1); q.value = ''; hits.innerHTML = ''; draw(); }
});

function pick(px, py) {
  let best = -1, bd = 144;
  for (let i = 0; i < N; i++) {
    if (!on[D.nodeStatus[i]]) continue;
    const dx = sx(i) - px, dy = sy(i) - py, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function showTip(i, px, py) {
  const note = D.changedNotes[D.labels[i]];
  tip.innerHTML = '<b>' + esc(D.labels[i]) + '</b><br>' +
    D.statusLabels[D.nodeStatus[i]] + ' · ' + inc[i].length + ' edges' +
    (note ? '<br>' + esc(note) : '');
  tip.style.left = Math.min(px + 16, W - 330) + 'px';
  tip.style.top = Math.min(py + 16, H - 60) + 'px';
  tip.style.opacity = 1;
}

// Theme last: setTheme draws, and draw() depends on the geometry helpers above.
const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
setTheme(prefersDark ? 'dark' : 'light');

window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""
