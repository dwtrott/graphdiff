"""The timeline page: a series of snapshots on one screen.

Above: a churn strip (edges removed / added per step, similarity as a line)
and the timeline findings. Below: a heatmap of the nodes that were notable in
any step, and the full pairwise viewer for whichever step is selected —
embedded as a same-page ``srcdoc`` frame, so the whole thing is still one
file that loads nothing from anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..batch.timeline import TimelineReport
from .html import _escape, render_html
from .theme import CHANGE_RAMP, DARK, LIGHT

__all__ = ["render_timeline_html", "write_timeline_html"]


def render_timeline_html(
    timeline: TimelineReport,
    *,
    title: str | None = None,
    max_history_nodes: int = 40,
    **render_kwargs: Any,
) -> str:
    """Render a :class:`~graphdiff.batch.timeline.TimelineReport` to one HTML file.

    ``render_kwargs`` go to :func:`render_html` for each step's embedded viewer
    (``max_nodes``, ``context_hops``, ...).
    """
    steps_html = [render_html(rep, **render_kwargs) for rep in timeline.steps]
    hist = timeline.node_history.head(max_history_nodes)
    churn = timeline.churn().replace({float("nan"): None}).to_dict("records")
    payload = {
        "names": timeline.names,
        "stepLabels": timeline.step_labels(),
        "churn": churn,
        "findings": [f.to_dict() for f in timeline.findings],
        "hist": {
            "labels": [str(x) for x in hist.index],
            "sig": [[round(float(v), 2) for v in row] for row in hist.to_numpy()],
        },
        "flickers": timeline.flickers().head(20).to_dict("records"),
        "themes": {"light": LIGHT, "dark": DARK},
        "ramp": CHANGE_RAMP,
        "docs": steps_html,
    }
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    doc_title = title or f"graphdiff timeline: {' → '.join(timeline.names)}"
    return _TEMPLATE.replace("__TITLE__", _escape(doc_title)).replace("__DATA__", data)


def write_timeline_html(timeline: TimelineReport, path: str | Path, **kwargs: Any) -> Path:
    """Render and write to ``path``."""
    path = Path(path)
    path.write_text(render_timeline_html(timeline, **kwargs), encoding="utf-8")
    return path


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
    --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #7b7a75; --hover: rgba(0,0,0,.05);
    --a: #2a78d6; --b: #eb6834; --chg: #199e70;
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #232322; --line: #373735;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8d8c84; --hover: rgba(255,255,255,.07);
    --a: #3987e5; --b: #d95926;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { font: 13px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink); background: var(--surface); display: grid;
    grid-template-rows: auto 1fr; overflow: hidden; }
  header { display: grid; grid-template-columns: 1fr 380px; gap: 0; border-bottom: 1px solid var(--line); }
  #left { padding: 12px 16px 8px; display: flex; flex-direction: column; gap: 6px; min-width: 0; }
  #ttl { font-weight: 650; font-size: 15px; margin: 0; }
  #sub { color: var(--ink-3); font-size: 12px; }
  #strip { width: 100%; height: 120px; display: block; }
  #steps { display: flex; gap: 4px; flex-wrap: wrap; }
  #steps button { font: inherit; font-size: 12px; padding: 4px 10px; border: 1px solid var(--line);
    border-radius: 999px; background: var(--panel); color: var(--ink-2); cursor: pointer; }
  #steps button[aria-pressed="true"] { background: var(--ink); color: var(--surface); border-color: var(--ink); }
  #right { border-left: 1px solid var(--line); padding: 12px 16px; overflow: auto; max-height: 230px; }
  h2 { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3);
    margin: 0 0 6px; font-weight: 600; }
  ol { margin: 0; padding-left: 18px; }
  li { margin: 0 0 6px; }
  .sev { font-size: 10px; font-weight: 700; letter-spacing: .04em; padding: 1px 5px; border-radius: 3px;
    background: var(--panel); color: var(--ink-2); margin-right: 4px; }
  .sev.high { background: var(--ink); color: var(--surface); }
  .go { color: var(--ink-3); cursor: pointer; font-size: 11px; white-space: nowrap; }
  .go:hover { color: var(--ink); text-decoration: underline; }
  main { display: grid; grid-template-columns: 300px 1fr; min-height: 0; }
  #heat { border-right: 1px solid var(--line); overflow: auto; padding: 10px 12px; }
  table { border-collapse: collapse; font-size: 11px; width: 100%; }
  th { font-weight: 600; color: var(--ink-3); text-align: left; padding: 2px 4px; position: sticky; top: 0;
    background: var(--surface); }
  td { padding: 0; }
  td.lab { padding: 0 6px 0 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 130px;
    cursor: pointer; }
  td.lab:hover { text-decoration: underline; }
  td.c { width: 22px; height: 16px; border: 1px solid var(--surface); cursor: pointer; }
  td.c.on { outline: 2px solid var(--ink); outline-offset: -2px; }
  iframe { width: 100%; height: 100%; border: 0; background: var(--surface); }
  #legend { display: flex; gap: 12px; font-size: 11px; color: var(--ink-3); align-items: center; }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
  #theme { font: inherit; font-size: 11px; border: 1px solid var(--line); background: var(--panel);
    color: var(--ink-2); border-radius: 6px; padding: 2px 8px; cursor: pointer; margin-left: auto; }
</style>
</head>
<body>
<header>
  <div id="left">
    <h1 id="ttl"></h1>
    <div id="sub"></div>
    <canvas id="strip"></canvas>
    <div id="legend">
      <span><i class="sw" style="background:var(--a)"></i>edges removed</span>
      <span><i class="sw" style="background:var(--b)"></i>edges added</span>
      <span><i class="sw" style="background:var(--ink);height:2px;vertical-align:2px"></i>similarity</span>
      <button id="theme">Dark</button>
    </div>
    <div id="steps"></div>
  </div>
  <div id="right">
    <h2>Findings across the series</h2>
    <ol id="findings"></ol>
  </div>
</header>
<main>
  <div id="heat">
    <h2>Where and when &mdash; significance per step</h2>
    <table id="ht"><thead></thead><tbody></tbody></table>
  </div>
  <iframe id="frame" title="Step viewer" sandbox="allow-scripts allow-same-origin"></iframe>
</main>
<script>
const D = __DATA__;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let mode = 'light', T = D.themes.light, step = 0;
const frame = document.getElementById('frame');
const NS = D.stepLabels.length;

document.getElementById('ttl').textContent = D.names.join('  →  ');
document.getElementById('sub').textContent = `${D.names.length} snapshots · ${NS} steps · click a step, a finding, or a cell`;

// ---- step buttons -------------------------------------------------------
const stepsEl = document.getElementById('steps');
D.stepLabels.forEach((lab, k) => {
  const b = document.createElement('button');
  b.textContent = lab; b.dataset.k = k;
  b.addEventListener('click', () => setStep(k));
  stepsEl.appendChild(b);
});

function setStep(k, pick) {
  step = k;
  [...stepsEl.children].forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.k === k)));
  document.querySelectorAll('#ht td.c').forEach(td => td.classList.toggle('on', +td.dataset.k === k && td.dataset.l === pick));
  frame.srcdoc = D.docs[k];
  frame.onload = () => {
    frame.contentWindow.postMessage({theme: mode}, '*');
    if (pick) setTimeout(() => frame.contentWindow.postMessage({pick}, '*'), 60);
  };
  paint();
}

// ---- findings -----------------------------------------------------------
const fl = document.getElementById('findings');
D.findings.forEach(f => {
  const li = document.createElement('li');
  let go = '';
  if (f.kind === 'step' && f.target != null) go = `<span class="go" data-k="${f.target}">→ open step</span>`;
  else if ((f.kind === 'recurrent' || f.kind === 'flicker') && f.target) {
    const k = bestStepFor(f.target);
    go = `<span class="go" data-k="${k}" data-l="${esc(f.target)}">→ ${esc(f.target)}</span>`;
  }
  li.innerHTML = `<span class="sev ${f.severity}">${f.severity}</span>${esc(f.text)} ${go}`;
  const g = li.querySelector('.go');
  if (g) g.addEventListener('click', () => setStep(+g.dataset.k, g.dataset.l));
  fl.appendChild(li);
});
function bestStepFor(label) {
  const i = D.hist.labels.indexOf(label);
  if (i < 0) return 0;
  let best = 0; D.hist.sig[i].forEach((v, k) => { if (v > D.hist.sig[i][best]) best = k; });
  return best;
}

// ---- heatmap ------------------------------------------------------------
const ramp = () => D.ramp[mode];
function shade(v, vmax) {
  const r = ramp(); if (!(v > 0)) return 'transparent';
  return r[Math.min(r.length - 1, Math.floor((v / vmax) * r.length))];
}
function buildHeat() {
  const thead = document.querySelector('#ht thead'), tbody = document.querySelector('#ht tbody');
  thead.innerHTML = '<tr><th>node</th>' + D.stepLabels.map((s, k) => `<th title="${esc(s)}">${k + 1}</th>`).join('') + '</tr>';
  const vmax = Math.max(1e-9, ...D.hist.sig.flat());
  tbody.innerHTML = D.hist.labels.map((lab, i) =>
    `<tr><td class="lab" data-l="${esc(lab)}" title="${esc(lab)}">${esc(lab)}</td>` +
    D.hist.sig[i].map((v, k) =>
      `<td class="c" data-k="${k}" data-l="${esc(lab)}" style="background:${shade(v, vmax)}" title="${esc(lab)} · ${esc(D.stepLabels[k])} · significance ${v}"></td>`
    ).join('') + '</tr>').join('');
  tbody.querySelectorAll('td.c').forEach(td => td.addEventListener('click', () => setStep(+td.dataset.k, td.dataset.l)));
  tbody.querySelectorAll('td.lab').forEach(td => td.addEventListener('click', () => setStep(bestStepFor(td.dataset.l), td.dataset.l)));
}

// ---- churn strip --------------------------------------------------------
const cv = document.getElementById('strip');
function paint() {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const L = 36, R = 44, top = 18, bot = 22, iw = w - L - R, ih = h - top - bot;
  const maxE = Math.max(1, ...D.churn.map(c => c.edges_removed + c.edges_added));
  const bw = iw / NS;
  g.font = '10px ui-sans-serif, system-ui, sans-serif'; g.textBaseline = 'middle';
  g.strokeStyle = T.line; g.lineWidth = 1;
  g.beginPath(); g.moveTo(L, top + ih + .5); g.lineTo(L + iw, top + ih + .5); g.stroke();
  D.churn.forEach((c, k) => {
    const x = L + k * bw + bw * .18, ww = bw * .64;
    const hr = c.edges_removed / maxE * ih, ha = c.edges_added / maxE * ih;
    g.globalAlpha = k === step ? 1 : .55;
    g.fillStyle = T.A_ONLY; g.fillRect(x, top + ih - hr, ww, hr);
    g.fillStyle = T.B_ONLY; g.fillRect(x, top + ih - hr - ha, ww, ha);
    g.globalAlpha = 1;
    g.fillStyle = k === step ? T.ink : T.inkMuted; g.textAlign = 'center';
    g.fillText(String(k + 1), x + ww / 2, top + ih + 11);
    if (k === step) { g.strokeStyle = T.ink; g.strokeRect(x - 2.5, top - 2.5, ww + 5, ih + 5); }
  });
  // similarity line (right axis 0..1)
  g.strokeStyle = T.ink; g.lineWidth = 1.6; g.beginPath();
  let started = false;
  D.churn.forEach((c, k) => {
    if (c.similarity == null) return;
    const x = L + k * bw + bw / 2, y = top + ih - c.similarity * ih;
    started ? g.lineTo(x, y) : g.moveTo(x, y); started = true;
  });
  g.stroke();
  D.churn.forEach((c, k) => {
    if (c.similarity == null) return;
    const x = L + k * bw + bw / 2, y = top + ih - c.similarity * ih;
    g.fillStyle = T.surface; g.beginPath(); g.arc(x, y, 3.5, 0, 6.2832); g.fill();
    g.strokeStyle = T.ink; g.lineWidth = 1.6; g.stroke();
    g.fillStyle = T.ink; g.textAlign = 'center'; g.fillText(c.similarity.toFixed(2), x, y - 10);
  });
  g.fillStyle = T.inkMuted; g.textAlign = 'right';
  g.fillText(String(maxE), L - 4, top + 6); g.fillText('0', L - 4, top + ih);
  g.textAlign = 'left'; g.fillText('1.0', L + iw + 4, top + 6); g.fillText('0', L + iw + 4, top + ih);
}

// ---- theme --------------------------------------------------------------
const themeBtn = document.getElementById('theme');
function setTheme(next) {
  mode = next; T = D.themes[next];
  document.documentElement.setAttribute('data-theme', next);
  themeBtn.textContent = next === 'light' ? 'Dark' : 'Light';
  buildHeat(); paint();
  if (frame.contentWindow) frame.contentWindow.postMessage({theme: next}, '*');
}
themeBtn.addEventListener('click', () => setTheme(mode === 'light' ? 'dark' : 'light'));
window.addEventListener('resize', paint);

const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
setTheme(prefersDark ? 'dark' : 'light');
// Open on the busiest step: that is the one a reader wants first.
let busiest = 0;
D.churn.forEach((c, k) => { const t = c.edges_removed + c.edges_added + c.nodes_added + c.nodes_removed;
  const b = D.churn[busiest]; if (t > b.edges_removed + b.edges_added + b.nodes_added + b.nodes_removed) busiest = k; });
setStep(busiest);
</script>
</body>
</html>
"""
