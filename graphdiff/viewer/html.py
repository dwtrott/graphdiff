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
from ..metrics.cluster import cluster_union
from ..report.report import ComparisonReport
from .ego import ego_networks
from .focus import select_focus
from .layout import LayoutParams, force_directed_layout
from .theme import CHANGE_RAMP, DARK, LIGHT, SHAPES, STATUS_COLORS, status_labels

__all__ = ["STATUS_COLORS", "render_html", "write_html"]

_STATUS_INDEX = {name: i for i, name in enumerate(STATUS_ORDER)}
_DIFFERING_KEYS = ("A_ONLY", "B_ONLY", "CHANGED")


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
    max_cards: int,
    max_card_neighbors: int,
    cluster_by: str | None,
    max_clusters: int,
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

    pair_index = edge_array[:, :2] if len(edge_array) else np.zeros((0, 2))
    coords = force_directed_layout(len(labels), pair_index, params=layout)
    # A separate 3D layout rather than a 2D one with a random z: the third axis
    # has to be earned by the structure or rotation shows nothing but jitter.
    coords3 = force_directed_layout(
        len(labels),
        pair_index,
        params=LayoutParams(
            iterations=max(120, int(layout.iterations * 0.6)),
            grid=layout.grid,
            seed=layout.seed,
            gravity=layout.gravity,
            initial_temperature=layout.initial_temperature,
        ),
        dims=3,
    )

    changed_notes: dict[str, str] = {}
    if CHANGED_ATTRS in node_rows.columns:
        for label, value in node_rows[CHANGED_ATTRS].items():
            summary = _changed_summary(value)
            if summary:
                changed_notes[str(label)] = summary

    matches: dict[str, list[Any]] = {}
    if "match_method" in node_rows.columns:
        inexact = node_rows[node_rows["match_method"].isin(["normalized", "fuzzy"])]
        for label, row in inexact.iterrows():
            matches[str(label)] = [
                str(row["match_method"]),
                round(float(row["match_confidence"]), 3),
                str(row["label_b"]),
            ]
    align_counts = union.alignment.counts if union.alignment is not None else None

    metrics: list[list[Any]] = []
    top_changed: list[list[Any]] = []
    egos: list[dict[str, Any]] = []
    headline: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    node_sig: list[float] = [0.0] * len(labels)
    if report is not None:
        for name, values in report.scalar_scores().items():
            metrics.append([name, _num(values["raw"]), _num(values["shared"])])
        if report.significance is not None:
            sig_series = report.significance.table.set_index(LABEL)["significance"]
            node_sig = [
                round(float(v), 3) for v in sig_series.reindex(labels).fillna(0.0).to_numpy()
            ]
            for row in report.significance.top(max_cards).to_dict("records"):
                top_changed.append(
                    [
                        str(row[LABEL]),
                        round(float(row["significance"]), 2),
                        int(row["n_added"]),
                        int(row["n_removed"]),
                        int(row["n_changed"]),
                        str(row["status"]),
                    ]
                )
        else:
            for row in report.neighborhood.top_changed(max_cards).to_dict("records"):
                top_changed.append(
                    [
                        str(row[LABEL]),
                        round(1.0 - float(row["jaccard"]), 4),
                        int(row["n_added"]),
                        int(row["n_removed"]),
                        0,
                        "SHARED",
                    ]
                )
        findings = [f.to_dict() for f in report.findings]
        headline = {
            "similarity": _num(report.score("ged_similarity")),
            "edgeJaccard": _num(report.score("jaccard_typed_edges")),
        }
        # Cards come from the full union, not the drawn subset, so a card is
        # never missing a neighbour the overview happened to cap away.
        ranked = [row[0] for row in top_changed[:max_cards]]
        for net in ego_networks(union, ranked, max_neighbors=max_card_neighbors):
            egos.append(
                {
                    "l": net.label,
                    "s": _STATUS_INDEX.get(net.status, 0),
                    "t": net.truncated,
                    "n": [
                        [n.label, _STATUS_INDEX.get(n.status, 0), int(n.outgoing)]
                        for n in net.neighbors
                    ],
                }
            )

    # ---- aggregate view: one mark per cluster, shaded by change density ----
    cmap = (
        report.clusters
        if (report is not None and report.clusters is not None and cluster_by is None)
        else cluster_union(union, attribute=cluster_by, max_clusters=max_clusters)
    )
    cluster_index = {c.key: i for i, c in enumerate(cmap.clusters)}
    cl_edges = (
        np.array([[i, j] for i, j, _ in cmap.links], dtype=np.int64)
        if cmap.links
        else np.zeros((0, 2), dtype=np.int64)
    )
    cl_coords = force_directed_layout(len(cmap.clusters), cl_edges, params=layout)
    clusters = [
        {
            "k": c.key,
            "name": c.name,
            "size": c.size,
            "nc": c.node_counts,
            "ec": c.edge_counts,
            "d": round(c.change_density, 5),
            "x": round(float(cl_coords[i, 0]), 5),
            "y": round(float(cl_coords[i, 1]), 5),
        }
        for i, c in enumerate(cmap.clusters)
    ]
    cluster_links = [
        [i, j, int(sum(counts.values())), int(sum(counts[s] for s in _DIFFERING_KEYS))]
        for i, j, counts in cmap.links
    ]
    node_cluster = (
        cmap.membership.reindex(labels).map(cluster_index).fillna(-1).astype(int).tolist()
    )

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
            # A few hundred nodes still reads as a picture; beyond that the
            # overview is a hairball and the aggregate view is the honest start.
            "initialView": "overview",
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
        "p3": [[round(float(v), 4) for v in row] for row in coords3],
        "edges": edge_array.tolist(),
        "edgeTypes": type_names,
        "edgeNotes": edge_notes,
        "changedNotes": changed_notes,
        "matches": matches,
        "alignCounts": align_counts,
        "metrics": metrics,
        "topChanged": top_changed,
        "nodeSig": node_sig,
        "findings": findings,
        "egos": egos,
        "clusters": clusters,
        "clusterLinks": cluster_links,
        "nodeCluster": node_cluster,
        "clusterMethod": cmap.method,
        "ramp": CHANGE_RAMP,
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
    max_cards: int = 40,
    max_card_neighbors: int = 24,
    cluster_by: str | None = None,
    max_clusters: int = 60,
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
    max_cards:
        How many ego-network cards the Changes view holds, in most-changed order.
    max_card_neighbors:
        Neighbour cap per card; differences are kept ahead of unchanged ones.
    cluster_by:
        Node attribute to aggregate by in the Clusters view. ``None`` detects
        communities instead.
    max_clusters:
        Cluster cap; the long tail folds into a single ``"(other)"`` mark.

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
        max_cards=max_cards,
        max_card_neighbors=max_card_neighbors,
        cluster_by=cluster_by,
        max_clusters=max_clusters,
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

  /* stage & views ------------------------------------------------------- */
  #stage { position: relative; flex: 1 1 auto; min-width: 0; display: flex;
           flex-direction: column; }
  #tabs { flex: 0 0 auto; display: flex; align-items: center; gap: 4px;
          padding: 8px 12px; border-bottom: 1px solid var(--line); }
  #tabs button {
    font: inherit; font-size: 12px; padding: 5px 12px; border: 1px solid transparent;
    background: none; border-radius: 7px; cursor: pointer; color: var(--ink-2);
  }
  #tabs button:hover { background: var(--hover); }
  #tabs button.on { background: var(--ink); color: var(--surface); }
  #tabnote { margin-left: auto; font-size: 11px; color: var(--ink-3); }
  .pane { display: none; flex: 1 1 auto; position: relative; min-height: 0; }
  .pane.on { display: flex; flex-direction: column; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas.dragging { cursor: grabbing; }

  /* cards --------------------------------------------------------------- */
  #cards {
    flex: 1 1 auto; overflow-y: auto; padding: 14px;
    display: grid; gap: 12px; align-content: start;
    grid-template-columns: repeat(auto-fill, minmax(196px, 1fr));
  }
  .card {
    border: 1px solid var(--line); border-radius: 10px; background: var(--panel);
    padding: 9px 10px 4px; scroll-margin: 16px;
  }
  .card.hot { outline: 2px solid var(--ink); }
  .card h3 {
    margin: 0; font-size: 12px; font-weight: 600; display: flex; align-items: baseline;
    gap: 6px; overflow: hidden;
  }
  .card h3 .rank { color: var(--ink-3); font-variant-numeric: tabular-nums;
                   font-weight: 500; flex: 0 0 auto; }
  .card h3 .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card .meta { font-size: 10px; color: var(--ink-2); margin: 3px 0 1px;
                font-variant-numeric: tabular-nums; display: flex; align-items: center;
                flex-wrap: wrap; }
  .card svg.ego { display: block; width: 100%; height: auto; }
  .card .meta svg.glyph { width: 9px; height: 9px; flex: 0 0 auto; }
  .card .omit { font-size: 10px; color: var(--ink-3); text-align: center; padding-bottom: 4px; }
  #cardsEmpty { color: var(--ink-3); padding: 20px; }

  /* compare ------------------------------------------------------------- */
  #cmpbar {
    flex: 0 0 auto; display: flex; align-items: center; gap: 10px;
    padding: 8px 14px; border-bottom: 1px solid var(--line); font-size: 11px;
    color: var(--ink-3);
  }
  #cmpbar .nm { font-weight: 600; color: var(--ink-2); white-space: nowrap; }
  #blend { flex: 1 1 auto; max-width: 340px; accent-color: var(--ink-2); }
  #cmpwrap { flex: 1 1 auto; display: flex; min-height: 0; }
  #cmpwrap canvas { flex: 1 1 0; min-width: 0; }
  #cmpwrap.split #cB { display: block; border-left: 1px solid var(--line); }
  #cmpwrap #cB { display: none; }
  .cmplabel {
    position: absolute; top: 8px; font-size: 10px; letter-spacing: .06em;
    text-transform: uppercase; color: var(--ink-3); pointer-events: none;
  }

  /* side panel ---------------------------------------------------------- */
  #side {
    flex: 0 0 336px; border-left: 1px solid var(--line); background: var(--panel);
    overflow-y: auto; overscroll-behavior: contain;
  }
  @media (max-width: 860px) {
    body { flex-direction: column; }
    #stage { flex: 0 0 62vh; }
    #side { flex: 1 1 auto; border-left: 0; border-top: 1px solid var(--line); }
  }
  .pad { padding: 14px 16px; }
  .divide { border-top: 1px solid var(--line); }
  h1 { font-size: 14px; margin: 0; font-weight: 600; letter-spacing: -.01em; }
  .sub { color: var(--ink-3); font-size: 11px; margin-top: 3px; }
  h2 { font-size: 10px; text-transform: uppercase; letter-spacing: .07em;
       color: var(--ink-3); margin: 0 0 9px; font-weight: 600; }
  #findings { margin: 0; padding-left: 18px; font-size: 12px; line-height: 1.45; }
  #findings li { margin: 0 0 7px; cursor: pointer; border-radius: 5px; padding: 3px 5px 3px 2px; }
  #findings li:hover { background: var(--hover); }
  #findings li.low { color: var(--ink-3); }
  #findings li .sev {
    display: inline-block; font-size: 9px; text-transform: uppercase; letter-spacing: .06em;
    font-weight: 600; border-radius: 3px; padding: 0 5px; margin-right: 5px;
    vertical-align: 1px; color: var(--surface); background: var(--ink-3);
  }
  #findings li.high .sev { background: var(--ink); }
  #findings li .go { color: var(--ink-3); font-size: 11px; white-space: nowrap; }
  .tiles { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
  .tile { background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
          padding: 8px 9px; }
  .tile .k { font-size: 9px; text-transform: uppercase; letter-spacing: .06em;
             color: var(--ink-3); font-weight: 600; }
  .tile .v { font-size: 17px; font-weight: 600; margin-top: 2px;
             font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
  .tile .d { font-size: 10px; color: var(--ink-3); font-variant-numeric: tabular-nums; }
  .legend { display: flex; flex-direction: column; gap: 1px; }
  .legend label { display: flex; align-items: center; gap: 9px; cursor: pointer;
                  padding: 5px 6px; border-radius: 6px; user-select: none; }
  .legend label:hover { background: var(--hover); }
  .legend input { margin: 0; flex: 0 0 auto; accent-color: var(--ink-2); }
  .legend .name { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis;
                  white-space: nowrap; }
  .legend .count { color: var(--ink-3); font-size: 11px; flex: 0 0 auto;
                   font-variant-numeric: tabular-nums; }
  .legend label.off .name, .legend label.off .count { opacity: .42; }
  svg.glyph { flex: 0 0 auto; display: block; }
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
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; font-weight: 600; color: var(--ink-3); font-size: 9px;
       text-transform: uppercase; letter-spacing: .06em; padding: 0 6px 5px 0; }
  td { padding: 4px 6px 4px 0; border-top: 1px solid var(--line); }
  td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
  tbody tr { cursor: pointer; }
  tbody tr:hover, tbody tr.sel { background: var(--hover); }
  .hit { cursor: pointer; padding: 4px 6px; border-radius: 5px; }
  .hit:hover { background: var(--hover); }
  #hits { margin-top: 6px; max-height: 132px; overflow-y: auto; }
  #card { display: none; } #card.on { display: block; }
  .chip { display: inline-flex; align-items: center; gap: 6px; font-size: 11px;
          border: 1px solid var(--line); border-radius: 999px; padding: 2px 9px 2px 6px;
          background: var(--surface); }
  .nbr { font-size: 11px; color: var(--ink-2); margin-top: 6px; }
  .nbr b { color: var(--ink); font-weight: 600; }
  .nbr .list { color: var(--ink-3); }
  details > summary { cursor: pointer; list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::after { content: " ▸"; color: var(--ink-3); }
  details[open] > summary::after { content: " ▾"; }

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
  #mapbar {
    flex: 0 0 auto; display: flex; align-items: center; gap: 4px;
    padding: 7px 14px; border-bottom: 1px solid var(--line);
  }
  #mapbar .seg { padding: 4px 10px; font-size: 11px; border-radius: 6px; }
  #mapbar .seg.on { background: var(--ink); color: var(--surface); border-color: var(--ink); }
  #drill {
    position: absolute; left: 14px; top: 56px; font-size: 11px; display: none;
    background: var(--surface); border: 1px solid var(--line); border-radius: 7px;
    padding: 5px 6px 5px 10px; align-items: center; gap: 8px; color: var(--ink-2);
  }
  #drill.on { display: flex; }
  #drill button { padding: 2px 8px; font-size: 11px; }
  #rampLegend {
    position: absolute; left: 14px; bottom: 14px; background: var(--surface);
    border: 1px solid var(--line); border-radius: 7px; padding: 7px 10px;
    font-size: 10px; color: var(--ink-3);
  }
  #rampLegend .bar { display: flex; margin: 4px 0 3px; }
  #rampLegend .bar i { width: 22px; height: 9px; display: block; }
  #rampLegend .ends { display: flex; justify-content: space-between;
                      font-variant-numeric: tabular-nums; }
  #threebar {
    flex: 0 0 auto; display: flex; align-items: center; gap: 10px;
    padding: 8px 14px; border-bottom: 1px solid var(--line);
  }
  #threebar .sub { margin: 0; }
  #tip3 {
    position: absolute; pointer-events: none; background: var(--tip-bg); color: var(--tip-ink);
    padding: 7px 10px; border-radius: 7px; font-size: 12px; opacity: 0;
    transition: opacity .09s; z-index: 5; box-shadow: var(--shadow);
  }
  #clusterHint {
    position: absolute; right: 14px; bottom: 14px; font-size: 11px; color: var(--ink-3);
    background: var(--surface); border: 1px solid var(--line); border-radius: 7px;
    padding: 5px 10px;
  }
</style>
</head>
<body>
<div id="stage">
  <nav id="tabs">
    <button data-v="overview" class="on">Map</button>
    <button data-v="clusters">Regions</button>
    <button data-v="cards">Top changes</button>
    <button data-v="compare">Before / After</button>
    <button data-v="three">3D</button>
    <span id="tabnote"></span>
  </nav>

  <section id="pane-overview" class="pane on">
    <div id="mapbar">
      <span class="sub" style="margin:0 6px 0 0">Show</span>
      <button class="seg on" data-show="all">Everything</button>
      <button class="seg" data-show="changes">Changes only</button>
      <button class="seg" data-show="sig">Significant only</button>
      <span class="sub" style="margin-left:14px">Node size = significance · ring = top 10</span>
    </div>
    <canvas id="cO"></canvas>
    <div id="drill"></div>
    <div id="hud"></div>
    <div id="tip"></div>
  </section>

  <section id="pane-clusters" class="pane">
    <canvas id="cC"></canvas>
    <div id="rampLegend"></div>
    <div id="clusterHint"></div>
  </section>

  <section id="pane-cards" class="pane">
    <div id="cards"></div>
  </section>

  <section id="pane-three" class="pane">
    <div id="threebar">
      <button id="spin" aria-pressed="true">Auto-rotate</button>
      <button id="threeReset">Reset</button>
      <span class="sub">drag to rotate · scroll to zoom · hover for labels</span>
    </div>
    <canvas id="c3"></canvas>
    <div id="tip3"></div>
  </section>

  <section id="pane-compare" class="pane">
    <div id="cmpbar">
      <span class="nm" id="cmpNameA"></span>
      <input type="range" id="blend" min="0" max="100" value="50"
             aria-label="Blend between the two graphs">
      <span class="nm" id="cmpNameB"></span>
      <button id="flick">Flicker</button>
      <button id="splitBtn" aria-pressed="false">Split</button>
    </div>
    <div id="cmpwrap"><canvas id="cA"></canvas><canvas id="cB"></canvas></div>
  </section>
</div>

<aside id="side">
  <div class="pad">
    <h1 id="ttl"></h1>
    <div class="sub" id="scale"></div>
  </div>
  <div class="pad divide" id="findingsBox">
    <h2>Findings</h2>
    <ol id="findings"></ol>
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
      <button id="theme">Dark</button>
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
    <h2>Most significant nodes</h2>
    <div class="sub" style="margin:-4px 0 7px">Ranked by how unlikely each node's change is
      under the graph-wide background rate (&minus;log<sub>10</sub> p).</div>
    <table id="changed"><thead><tr>
      <th>Node</th><th class="n">Sig</th><th class="n">+</th><th class="n">&minus;</th>
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
const S_SHARED = D.statuses.indexOf('SHARED'), S_CHANGED = D.statuses.indexOf('CHANGED');
const S_A = D.statuses.indexOf('A_ONLY'), S_B = D.statuses.indexOf('B_ONLY');

const cO = document.getElementById('cO'), gO = cO.getContext('2d');
const cA = document.getElementById('cA'), gA = cA.getContext('2d');
const cB = document.getElementById('cB'), gB = cB.getContext('2d');
const tip = document.getElementById('tip');
const q = document.getElementById('q'), hits = document.getElementById('hits');

const on = D.statuses.map(() => true);
let T = D.themes.light, mode = 'light', vw = 'overview';
let view = {k: 1, x: 0, y: 0}, hover = -1, sel = -1, blend = .5, split = false;
let drill = -1, hoverCl = -1, cview = {k: 1, x: 0, y: 0}, showMode = 'all';
const SIG_MAX = Math.max(1e-9, ...D.nodeSig);
const TOP_SIG = new Set(D.topChanged.slice(0, 10).map(r => D.labels.indexOf(r[0])).filter(i => i >= 0));
// Radius grows with sqrt(significance) so a p = 1e-12 node is visibly bigger
// than a p = 1e-3 one without a hub swallowing the map.
const sigScale = i => Math.sqrt(Math.max(0, D.nodeSig[i]) / SIG_MAX);
const cC = document.getElementById('cC'), gC = cC.getContext('2d');
const maxDensity = Math.max(1e-9, ...D.clusters.map(c => c.d));

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
  g.moveTo(x, y - r * 1.3); g.lineTo(x + r * 1.3, y);
  g.lineTo(x, y + r * 1.3); g.lineTo(x - r * 1.3, y); g.closePath();
}

// Built as an HTML string rather than createElementNS: the SVG namespace URI is
// never fetched, but keeping it out of the file lets the offline test stay a
// strict "no URLs at all" check instead of a check with an exception list.
function shapePath(shape, x, y, r) {
  const rec = {
    d: '',
    beginPath() { this.d = ''; },
    arc(cx, cy, rr) {
      this.d += `M${cx - rr} ${cy}a${rr} ${rr} 0 1 0 ${rr * 2} 0a${rr} ${rr} 0 1 0 ${-rr * 2} 0Z`;
    },
    rect(rx, ry, w, h) { this.d += `M${rx} ${ry}h${w}v${h}h${-w}Z`; },
    moveTo(mx, my) { this.d += `M${mx} ${my}`; },
    lineTo(lx, ly) { this.d += `L${lx} ${ly}`; },
    closePath() { this.d += 'Z'; }
  };
  path(rec, shape, x, y, r);
  return rec.d;
}
function glyphHTML(i, size) {
  const s = size || 13;
  return `<svg class="glyph" width="${s}" height="${s}" viewBox="0 0 ${s} ${s}" ` +
    `aria-hidden="true"><path d="${shapePath(D.shapes[i], s / 2, s / 2, s * .38)}" ` +
    `fill="${D.themes[mode].series[i]}"/></svg>`;
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
document.getElementById('cmpNameA').textContent = D.meta.nameA;
document.getElementById('cmpNameB').textContent = D.meta.nameB;

function tile(k, v, d) {
  return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div>` +
         `<div class="d">${d}</div></div>`;
}
(function tiles() {
  const nc = D.meta.nodeCounts, ec = D.meta.edgeCounts, h = D.meta.headline || {};
  const sim = h.similarity === null || h.similarity === undefined
    ? '—' : (h.similarity * 100).toFixed(0) + '%';
  document.getElementById('tiles').innerHTML =
    tile('Node changes', (nc.B_ONLY + nc.A_ONLY).toLocaleString(),
         `+${nc.B_ONLY} added / &minus;${nc.A_ONLY} gone`) +
    tile('Edge changes', (ec.B_ONLY + ec.A_ONLY + ec.CHANGED).toLocaleString(),
         `+${ec.B_ONLY} / &minus;${ec.A_ONLY} / ~${ec.CHANGED} reweighted`) +
    tile('Similarity', sim, 'normalized edit distance');
})();

(function findings() {
  const ol = document.getElementById('findings');
  if (!D.findings.length) {
    document.getElementById('findingsBox').style.display = 'none'; return;
  }
  const label = {overview: 'map', clusters: 'regions', cards: 'top changes', compare: 'before/after'};
  D.findings.forEach(f => {
    const li = document.createElement('li');
    li.className = f.severity;
    li.innerHTML = `<span class="sev">${f.severity}</span>${esc(f.text)}` +
      (f.view ? ` <span class="go">→ ${label[f.view] || f.view}</span>` : '');
    li.title = f.view ? 'Click to open in the ' + (label[f.view] || f.view) + ' view' : '';
    li.addEventListener('click', () => goTo(f));
    ol.appendChild(li);
  });
})();

function goTo(f) {
  if (f.view === 'clusters' && f.target) {
    const i = D.clusters.findIndex(c => c.k === f.target);
    if (i >= 0) { setDrill(i); document.querySelector('#tabs button[data-v="overview"]').click();
      const members = []; for (let n = 0; n < N; n++) if (D.nodeCluster[n] === i) members.push(n);
      fitTo(members); paintAll(); return; }
  }
  if (f.view) document.querySelector(`#tabs button[data-v="${f.view}"]`).click();
  if (f.target && f.view !== 'clusters') pickLabel(f.target);
}

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
    on[i] = cb.checked; lab.classList.toggle('off', !cb.checked);
    document.getElementById('diffonly').setAttribute('aria-pressed', String(!on[S_SHARED]));
    paintAll();
  });
  legend.appendChild(lab);
});
if (D.alignCounts && (D.alignCounts.normalized + D.alignCounts.fuzzy) > 0) {
  // Inexact matches: a dashed ring, and the count so the reader knows how
  // much of "in both" rests on the matcher rather than on identical labels.
  const lab = document.createElement('label');
  lab.style.cursor = 'default';
  const ring = document.createElement('span');
  ring.style.cssText = 'display:inline-block;width:14px;height:14px;border-radius:50%;' +
    'border:1.5px dashed var(--ink-2);box-sizing:border-box;margin-left:20px;flex:0 0 auto';
  const name = document.createElement('span');
  name.className = 'name'; name.textContent = 'Matched despite differing labels';
  const count = document.createElement('span');
  count.className = 'count';
  count.textContent = `${D.alignCounts.normalized + D.alignCounts.fuzzy}`;
  count.title = `${D.alignCounts.normalized} by normalization, ${D.alignCounts.fuzzy} fuzzy`;
  lab.append(ring, name, count);
  legend.appendChild(lab);
}

const mt = document.querySelector('#metrics tbody');
D.metrics.forEach(([name, raw, shared]) => {
  const tr = document.createElement('tr');
  tr.style.cursor = 'default';
  tr.innerHTML = `<td>${name}</td><td class="n">${fmt(raw)}</td><td class="n">${fmt(shared)}</td>`;
  mt.appendChild(tr);
});

const ct = document.querySelector('#changed tbody');
D.topChanged.forEach(([label, sig, add, rem, chg, st]) => {
  const tr = document.createElement('tr');
  tr.dataset.label = label;
  tr.title = `${D.statusLabels[D.statuses.indexOf(st)] || st} · ${chg} reweighted`;
  tr.innerHTML = `<td>${esc(label)}</td><td class="n">${sig.toFixed(1)}</td>` +
    `<td class="n">${add}</td><td class="n">${rem}</td>`;
  tr.addEventListener('click', () => pickLabel(label));
  ct.appendChild(tr);
});
if (!D.topChanged.length) {
  ct.innerHTML = '<tr style="cursor:default"><td colspan="4" style="color:var(--ink-3)">' +
    'No shared nodes to compare.</td></tr>';
}

// ---- ego cards ----------------------------------------------------------
// Neighbours sit in fixed angular sectors — removed left, added right, changed
// below, unchanged above — so the same kind of change lands in the same place on
// every card. That is what makes a wall of cards scannable rather than 40
// separate puzzles.
const SECTOR = {};
SECTOR[S_B] = [-52, 52]; SECTOR[S_A] = [128, 232];
SECTOR[S_CHANGED] = [62, 118]; SECTOR[S_SHARED] = [242, 298];

function egoCardHTML(e, rank) {
  const W = 190, H = 150, cx = W / 2, cy = H / 2, R = 52;
  const groups = {}; D.statuses.forEach((_, i) => { groups[i] = []; });
  e.n.forEach(n => groups[n[1]].push(n));

  let edges = '', nodes = '';
  for (const si of Object.keys(groups)) {
    const list = groups[si]; if (!list.length) continue;
    const [a0, a1] = SECTOR[si]; const n = list.length;
    list.forEach((nb, i) => {
      const t = n === 1 ? .5 : i / (n - 1);
      const ang = (a0 + (a1 - a0) * t) * Math.PI / 180;
      const ring = n > 8 && i % 2 ? R * .72 : R;
      const x = cx + Math.cos(ang) * ring, y = cy + Math.sin(ang) * ring;
      const col = D.themes[mode].series[si];
      const dash = Number(si) === S_CHANGED ? ' stroke-dasharray="3 2"' : '';
      const wdt = Number(si) === S_SHARED ? .8 : 1.6;
      const op = Number(si) === S_SHARED ? .45 : .9;
      edges += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" ` +
        `stroke="${col}" stroke-width="${wdt}" stroke-opacity="${op}"${dash}/>`;
      const r = Number(si) === S_SHARED ? 2.6 : 4;
      nodes += `<path d="${shapePath(D.shapes[si], x, y, r)}" fill="${col}">` +
        `<title>${esc(nb[0])} — ${esc(D.statusLabels[si])}</title></path>`;
    });
  }
  const centre = `<path d="${shapePath(D.shapes[e.s], cx, cy, 6)}" ` +
    `fill="${D.themes[mode].series[e.s]}" stroke="${D.themes[mode].surface}" ` +
    `stroke-width="1.5"><title>${esc(e.l)}</title></path>`;

  // Counts wear ink; the coloured glyph beside each carries the identity. A
  // number tinted with the series colour reads as decoration and fails against
  // the surface at small sizes.
  const sigRow = D.topChanged.find(r => r[0] === e.l);
  const c = {}; D.statuses.forEach((_, i) => { c[i] = (groups[i] || []).length; });
  const pair = (si, txt) =>
    `<span style="display:inline-flex;align-items:center;gap:3px;margin-right:8px">` +
    glyphHTML(si, 9) + `<span>${txt}</span></span>`;
  const meta = pair(S_B, '+' + c[S_B]) + pair(S_A, '&minus;' + c[S_A]) +
    pair(S_CHANGED, '~' + c[S_CHANGED]) + `<span>${c[S_SHARED]} kept</span>`;

  return `<div class="card" data-label="${esc(e.l)}">
    <h3><span class="rank">${rank}</span><span class="nm" title="${esc(e.l)}">${esc(e.l)}</span>
      ${sigRow ? `<span class="rank" title="significance, -log10 p">${sigRow[1].toFixed(1)}</span>` : ''}</h3>
    <div class="meta">${meta}</div>
    <svg class="ego" viewBox="0 0 ${W} ${H}" role="img" aria-label="Neighbourhood of ${esc(e.l)}">
      ${edges}${nodes}${centre}
    </svg>
    ${e.t ? `<div class="omit">+${e.t} more neighbours not shown</div>` : ''}
  </div>`;
}

function buildCards() {
  const box = document.getElementById('cards');
  if (!D.egos.length) {
    box.innerHTML = '<div id="cardsEmpty">No shared nodes, so there are no ' +
      'neighbourhoods to compare. Try the Compare view.</div>';
    return;
  }
  box.innerHTML = D.egos.map((e, i) => egoCardHTML(e, i + 1)).join('');
  box.querySelectorAll('.card').forEach(el => {
    el.addEventListener('click', () => pickLabel(el.dataset.label));
  });
}

// ---- geometry (per target canvas) ---------------------------------------
const PAD = 34;
const spanOf = (w, h) => Math.min(w, h) - 2 * PAD;
const sxOf = (i, w, h) => D.x[i] * spanOf(w, h) * view.k + view.x + PAD;
const syOf = (i, w, h) => D.y[i] * spanOf(w, h) * view.k + view.y + PAD;
const sx = i => sxOf(i, cO.clientWidth, cO.clientHeight);
const sy = i => syOf(i, cO.clientWidth, cO.clientHeight);

// ---- generic painter ----------------------------------------------------
function paint(g, cv, opts) {
  const w = cv.clientWidth, h = cv.clientHeight;
  const alpha = opts.alpha || D.statuses.map(() => 1);
  g.clearRect(0, 0, w, h);
  const term = q.value.trim().toLowerCase();
  const focusNode = opts.interactive ? (hover >= 0 ? hover : sel) : -1;
  const near = new Set();
  if (focusNode >= 0) {
    near.add(focusNode);
    for (const ei of inc[focusNode]) { const [a, b] = D.edges[ei]; near.add(a); near.add(b); }
  }
  const dimming = focusNode >= 0;

  const inDrill = i => drill < 0 || D.nodeCluster[i] === drill;
  const shown = i => inDrill(i) && (!opts.interactive || visibleNode(i));
  for (const [a, b, st] of D.edges) {
    if (!on[st] || alpha[st] <= 0.01) continue;
    if (!shown(a) || !shown(b)) continue;
    if (opts.interactive && showMode !== 'all' && st === S_SHARED) continue;
    const lit = !dimming || (near.has(a) && near.has(b));
    const shared = st === S_SHARED;
    g.strokeStyle = T.series[st];
    g.globalAlpha = alpha[st] * (lit ? (shared ? .3 : .75) : .06);
    g.lineWidth = (shared ? .8 : 1.5) * Math.min(1.7, .75 + view.k * .18);
    g.setLineDash(st === S_CHANGED ? [4, 3] : []);
    g.beginPath();
    g.moveTo(sxOf(a, w, h), syOf(a, w, h)); g.lineTo(sxOf(b, w, h), syOf(b, w, h));
    g.stroke();
  }
  g.setLineDash([]);

  const base = Math.max(2.2, Math.min(6.5, 2.4 + view.k * .62));
  const order = [...Array(N).keys()].sort(
    (i, j) => (D.nodeStatus[i] === S_SHARED ? 0 : 1) - (D.nodeStatus[j] === S_SHARED ? 0 : 1));
  for (const i of order) {
    const st = D.nodeStatus[i];
    if (!on[st] || alpha[st] <= 0.01 || !shown(i)) continue;
    const match = term && D.labels[i].toLowerCase().includes(term);
    const lit = !dimming || near.has(i);
    const r = (st === S_SHARED ? base : base * 1.3) + base * 2.2 * sigScale(i) + (match ? 3 : 0);
    const x = sxOf(i, w, h), y = syOf(i, w, h);
    g.globalAlpha = alpha[st] * (lit ? 1 : .12);
    path(g, D.shapes[st], x, y, r);
    g.fillStyle = T.series[st]; g.fill();
    const mt = D.matches[D.labels[i]];
    if (mt && lit) {
      // Dashed ring = this node is "in both" by the matcher's judgement, not
      // by identical labels; the fainter the ring the surer the match.
      g.beginPath(); g.arc(x, y, r + 3, 0, 6.2832);
      g.setLineDash([2.5, 2.5]); g.lineWidth = 1.3; g.strokeStyle = T.ink;
      g.globalAlpha = alpha[st] * (mt[0] === 'fuzzy' ? (.35 + .6 * (1 - mt[1])) : .35);
      g.stroke(); g.setLineDash([]);
      g.globalAlpha = alpha[st] * (lit ? 1 : .12);
    }
    if (opts.interactive && (i === sel || i === hover || match)) {
      g.lineWidth = 2; g.strokeStyle = T.ink; g.globalAlpha = lit ? 1 : .3; g.stroke();
    } else if (TOP_SIG.has(i) && lit) {
      // Halo ring on the top-10: the one thing that should be findable from
      // across the room, before any colour or shape is read.
      g.beginPath(); g.arc(x, y, r + 4.5, 0, 6.2832);
      g.lineWidth = 1.8; g.strokeStyle = T.ink; g.globalAlpha = alpha[st] * .85; g.stroke();
    }
  }
  g.globalAlpha = 1;

  g.font = '600 11px ui-sans-serif, system-ui, sans-serif';
  g.lineJoin = 'round';
  const labelled = view.k > 2.4 ? order
    : [...new Set([...(focusNode >= 0 ? [focusNode] : []), ...(opts.interactive ? TOP_SIG : [])])];
  // Greedy declutter: a label that would sit on top of one already drawn is
  // skipped. Significant nodes are labelled first so they always win.
  const placed = [];
  const ordered = [...labelled].sort((i, j) => D.nodeSig[j] - D.nodeSig[i]);
  for (const i of ordered) {
    const st = D.nodeStatus[i];
    if (!on[st] || alpha[st] <= 0.01 || !shown(i)) continue;
    if (dimming && !near.has(i)) continue;
    const x = sxOf(i, w, h) + 9, y = syOf(i, w, h) + 4;
    if (x < -60 || x > w + 60 || y < -20 || y > h + 20) continue;
    const tw = g.measureText(D.labels[i]).width;
    if (placed.some(b => x < b[0] + b[2] && x + tw > b[0] && y - 11 < b[1] + 13 && y > b[1])) continue;
    placed.push([x, y - 11, tw]);
    g.lineWidth = 3; g.strokeStyle = T.surface; g.strokeText(D.labels[i], x, y);
    g.fillStyle = T.ink; g.fillText(D.labels[i], x, y);
  }
}

// ---- 3D -----------------------------------------------------------------
// A presentation mode. Depth is real (a separate 3D force layout), but the
// analysis views stay flat on purpose: rotation reads well in a room and costs
// accuracy at a desk. Everything here is a plain perspective projection onto
// the 2D canvas — no WebGL, no library, nothing to bundle.
const c3 = document.getElementById('c3'), g3 = c3.getContext('2d');
const tip3 = document.getElementById('tip3');
let yaw = .6, pitch = .35, zoom3 = 1, spin = true, spinHandle = null, hover3 = -1;
let proj = null;  // per-node [sx, sy, depth, scale] for the last frame

function project(w, h) {
  const cy = Math.cos(yaw), sy_ = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const fov = 2.6, S = Math.min(w, h) * .78 * zoom3;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const p = D.p3[i];
    const x0 = p[0] - .5, y0 = p[1] - .5, z0 = p[2] - .5;
    const x1 = x0 * cy + z0 * sy_, z1 = -x0 * sy_ + z0 * cy;         // yaw about Y
    const y2 = y0 * cp - z1 * sp, z2 = y0 * sp + z1 * cp;             // pitch about X
    const f = fov / (fov + z2);
    out[i] = [w / 2 + x1 * f * S, h / 2 + y2 * f * S, z2, f];
  }
  return out;
}

function paint3D() {
  const w = c3.clientWidth, h = c3.clientHeight;
  g3.clearRect(0, 0, w, h);
  if (!N) return;
  proj = project(w, h);
  const inDrill = i => drill < 0 || D.nodeCluster[i] === drill;
  const term = q.value.trim().toLowerCase();

  // Depth cue: far things are small and faint, near things large and solid.
  const cue = z => .3 + .7 * (1 - (z + .7) / 1.4);

  const edges = [];
  for (const [a, b, st] of D.edges) {
    if (!on[st] || !inDrill(a) || !inDrill(b)) continue;
    edges.push([a, b, st, (proj[a][2] + proj[b][2]) / 2]);
  }
  edges.sort((p, q) => q[3] - p[3]);
  for (const [a, b, st, z] of edges) {
    const shared = st === S_SHARED;
    g3.strokeStyle = T.series[st];
    g3.globalAlpha = Math.max(.04, cue(z)) * (shared ? .28 : .8);
    g3.lineWidth = shared ? .8 : 1.6;
    g3.setLineDash(st === S_CHANGED ? [4, 3] : []);
    g3.beginPath(); g3.moveTo(proj[a][0], proj[a][1]); g3.lineTo(proj[b][0], proj[b][1]); g3.stroke();
  }
  g3.setLineDash([]);

  const order = [];
  for (let i = 0; i < N; i++) if (on[D.nodeStatus[i]] && inDrill(i) && visibleNode(i)) order.push(i);
  order.sort((i, j) => proj[j][2] - proj[i][2]);
  for (const i of order) {
    const st = D.nodeStatus[i], [x, y, z, f] = proj[i];
    const match = term && D.labels[i].toLowerCase().includes(term);
    const r = ((st === S_SHARED ? 2.6 : 3.6) + 6 * sigScale(i)) * f * zoom3 + (match ? 3 : 0);
    g3.globalAlpha = cue(z);
    path(g3, D.shapes[st], x, y, r);
    g3.fillStyle = T.series[st]; g3.fill();
    const mt3 = D.matches[D.labels[i]];
    if (mt3) {
      g3.beginPath(); g3.arc(x, y, r + 3, 0, 6.2832);
      g3.setLineDash([2.5, 2.5]); g3.lineWidth = 1.2; g3.strokeStyle = T.ink;
      g3.globalAlpha = cue(z) * (mt3[0] === 'fuzzy' ? (.35 + .6 * (1 - mt3[1])) : .35);
      g3.stroke(); g3.setLineDash([]); g3.globalAlpha = cue(z);
    }
    if (i === hover3 || match) { g3.lineWidth = 2; g3.strokeStyle = T.ink; g3.globalAlpha = 1; g3.stroke(); }
    else if (TOP_SIG.has(i)) {
      g3.beginPath(); g3.arc(x, y, r + 4, 0, 6.2832);
      g3.lineWidth = 1.6; g3.strokeStyle = T.ink; g3.globalAlpha = cue(z) * .9; g3.stroke();
    }
  }
  g3.globalAlpha = 1;
  g3.font = '600 11px ui-sans-serif, system-ui, sans-serif';
  const lab3 = new Set([...TOP_SIG]); if (hover3 >= 0) lab3.add(hover3);
  for (const i of lab3) {
    if (!inDrill(i) || !on[D.nodeStatus[i]] || !proj[i]) continue;
    const [x, y] = proj[i];
    g3.lineWidth = 3; g3.strokeStyle = T.surface; g3.strokeText(D.labels[i], x + 9, y + 4);
    g3.fillStyle = T.ink; g3.fillText(D.labels[i], x + 9, y + 4);
  }
}

function pick3(px, py) {
  if (!proj) return -1;
  let best = -1, bd = 121;
  for (let i = 0; i < N; i++) {
    if (!on[D.nodeStatus[i]]) continue;
    const dx = proj[i][0] - px, dy = proj[i][1] - py, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function startSpin() {
  if (spinHandle) return;
  let last = performance.now();
  const step = now => {
    if (!spin || vw !== 'three') { spinHandle = null; return; }
    yaw += (now - last) * .00035; last = now;
    paint3D(); spinHandle = requestAnimationFrame(step);
  };
  spinHandle = requestAnimationFrame(step);
}
const spinBtn = document.getElementById('spin');
spinBtn.addEventListener('click', () => {
  spin = !spin; spinBtn.setAttribute('aria-pressed', String(spin));
  if (spin) startSpin();
});
document.getElementById('threeReset').addEventListener('click', () => {
  yaw = .6; pitch = .35; zoom3 = 1; paint3D();
});
let drag3 = null;
c3.addEventListener('mousedown', e => { drag3 = {x: e.offsetX, y: e.offsetY, yaw, pitch}; });
window.addEventListener('mouseup', () => { drag3 = null; });
c3.addEventListener('mousemove', e => {
  if (drag3) {
    yaw = drag3.yaw + (e.offsetX - drag3.x) * .008;
    pitch = Math.max(-1.4, Math.min(1.4, drag3.pitch + (e.offsetY - drag3.y) * .008));
    paint3D(); return;
  }
  const i = pick3(e.offsetX, e.offsetY);
  if (i !== hover3) { hover3 = i; paint3D(); }
  if (i >= 0) {
    tip3.innerHTML = '<b>' + esc(D.labels[i]) + '</b><br>' + D.statusLabels[D.nodeStatus[i]];
    tip3.style.left = Math.min(e.offsetX + 16, c3.clientWidth - 220) + 'px';
    tip3.style.top = (e.offsetY + 16 + 42) + 'px';
    tip3.style.opacity = 1;
  } else tip3.style.opacity = 0;
});
c3.addEventListener('mouseleave', () => { hover3 = -1; tip3.style.opacity = 0; paint3D(); });
c3.addEventListener('wheel', e => {
  e.preventDefault();
  zoom3 = Math.max(.4, Math.min(6, zoom3 * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
  paint3D();
}, {passive: false});

// ---- clusters -----------------------------------------------------------
const rampOf = d => D.ramp[mode][Math.min(D.ramp[mode].length - 1,
  Math.round((d / maxDensity) * (D.ramp[mode].length - 1)))];
const rampIdx = d => Math.min(D.ramp[mode].length - 1,
  Math.round((d / maxDensity) * (D.ramp[mode].length - 1)));
// Bubble radius follows sqrt(size) so area tracks membership, then a global
// scale keeps the whole set inside the canvas when there are many big ones.
let clBase = [], clKey = '', clScale = 1;
const clRadius = c => (9 + Math.sqrt(c.size) * 4.5) * clScale;

// The force layout places cluster centres well but knows nothing about their
// radii, so bubbles land on top of each other. Resolve overlaps in pixel space
// with a few relaxation passes, then keep everything inside the frame.
function layoutClusters(w, h) {
  const key = w + 'x' + h;
  if (clKey === key) return;
  clKey = key;
  const area = D.clusters.reduce((s, c) => s + Math.PI * Math.pow(9 + Math.sqrt(c.size) * 4.5, 2), 0);
  clScale = Math.min(1, Math.sqrt((0.30 * w * h) / Math.max(area, 1)));
  const s = Math.min(w, h) - 130;
  const pts = D.clusters.map(c => ({x: c.x * s + 65, y: c.y * s + 65, r: clRadius(c) + 16}));
  for (let it = 0; it < 120; it++) {
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const a = pts[i], b = pts[j];
        let dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.hypot(dx, dy) || 0.01, min = a.r + b.r;
        if (d < min) {
          const push = (min - d) / 2; dx /= d; dy /= d;
          a.x -= dx * push; a.y -= dy * push; b.x += dx * push; b.y += dy * push;
        }
      }
    }
    for (const q of pts) {
      q.x = Math.max(q.r, Math.min(w - q.r, q.x));
      q.y = Math.max(q.r, Math.min(h - q.r - 26, q.y));
    }
  }
  clBase = pts;
}
const clX = (i, w, h) => clBase[i].x * cview.k + cview.x;
const clY = (i, w, h) => clBase[i].y * cview.k + cview.y;

function paintClusters() {
  const w = cC.clientWidth, h = cC.clientHeight;
  gC.clearRect(0, 0, w, h);
  if (!D.clusters.length) return;
  layoutClusters(w, h);

  // One encoding for the whole view: violet means "share of this that changed",
  // for links as much as for clusters. Reusing the categorical CHANGED hue here
  // would claim these links are reweighted edges, which they are not.
  for (const [i, j, total, changed] of D.clusterLinks) {
    gC.strokeStyle = rampOf(total ? changed / total : 0);
    gC.globalAlpha = .8;
    gC.lineWidth = Math.min(7, .6 + Math.log2(1 + total) * .8);
    gC.beginPath();
    gC.moveTo(clX(i, w, h), clY(i, w, h)); gC.lineTo(clX(j, w, h), clY(j, w, h));
    gC.stroke();
  }
  gC.globalAlpha = 1;

  D.clusters.forEach((c, i) => {
    const x = clX(i, w, h), y = clY(i, w, h), r = clRadius(c) * Math.min(1.6, cview.k);
    gC.beginPath(); gC.arc(x, y, r, 0, 6.2832);
    gC.fillStyle = rampOf(c.d); gC.fill();
    gC.lineWidth = i === hoverCl ? 2.5 : 1;
    gC.strokeStyle = i === hoverCl ? T.ink : T.line; gC.stroke();

    // Ink on the pale end of the ramp, surface on the dark end.
    gC.fillStyle = rampIdx(c.d) >= 3 ? T.surface : T.ink;
    gC.textAlign = 'center';
    gC.font = '600 ' + Math.max(9, Math.min(13, r * .42)) + 'px ui-sans-serif, system-ui, sans-serif';
    gC.fillText((c.d * 100).toFixed(0) + '%', x, y + 4);
    gC.fillStyle = T.ink;
    gC.font = '11px ui-sans-serif, system-ui, sans-serif';
    gC.fillText(c.name.length > 22 ? c.name.slice(0, 21) + '…' : c.name, x, y + r + 13);
    gC.fillStyle = T.inkMuted || T.ink;
    gC.font = '10px ui-sans-serif, system-ui, sans-serif';
    gC.fillText(c.size + ' nodes', x, y + r + 25);
  });
  gC.textAlign = 'left';

  document.getElementById('rampLegend').innerHTML =
    '<div>Share of the cluster that changed</div><div class="bar">' +
    D.ramp[mode].map(c => `<i style="background:${c}"></i>`).join('') +
    `</div><div class="ends"><span>0%</span><span>${(maxDensity * 100).toFixed(0)}%</span></div>`;
  document.getElementById('clusterHint').textContent =
    D.clusters.length + ' clusters · ' + D.clusterMethod + ' · click one to drill in';
}

function pickCluster(px, py) {
  const w = cC.clientWidth, h = cC.clientHeight;
  layoutClusters(w, h);
  for (let i = D.clusters.length - 1; i >= 0; i--) {
    const c = D.clusters[i];
    const dx = clX(i, w, h) - px, dy = clY(i, w, h) - py;
    const r = clRadius(c) * Math.min(1.6, cview.k);
    if (dx * dx + dy * dy <= r * r) return i;
  }
  return -1;
}

// Frame the drilled cluster instead of leaving it wherever it sat in the
// full-graph layout — nine nodes stranded in a corner is not a drill-down.
function fitTo(indices) {
  if (!indices.length) { view = {k: 1, x: 0, y: 0}; return; }
  let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
  for (const i of indices) {
    x0 = Math.min(x0, D.x[i]); x1 = Math.max(x1, D.x[i]);
    y0 = Math.min(y0, D.y[i]); y1 = Math.max(y1, D.y[i]);
  }
  const w = cO.clientWidth, h = cO.clientHeight, sp = spanOf(w, h);
  const dx = Math.max(x1 - x0, .02), dy = Math.max(y1 - y0, .02);
  view.k = Math.max(.5, Math.min(20, Math.min((w - 3 * PAD) / (sp * dx),
                                              (h - 3 * PAD) / (sp * dy))));
  view.x = w / 2 - ((x0 + x1) / 2) * sp * view.k - PAD;
  view.y = h / 2 - ((y0 + y1) / 2) * sp * view.k - PAD;
}

function setDrill(i) {
  drill = i;
  const chip = document.getElementById('drill');
  if (i < 0) { chip.classList.remove('on'); chip.innerHTML = ''; }
  else {
    const c = D.clusters[i];
    chip.classList.add('on');
    chip.innerHTML = `<span>Cluster <b>${esc(c.name)}</b> · ${c.size} nodes · ` +
      `${(c.d * 100).toFixed(0)}% changed</span><button id="undrill">Show all</button>`;
    chip.querySelector('#undrill').addEventListener('click', e => {
      e.stopPropagation(); setDrill(-1); paintAll();
    });
  }
}

function paintAll() {
  if (vw === 'clusters') { paintClusters(); }
  if (vw === 'three') { paint3D(); if (spin) startSpin(); }
  if (vw === 'overview') {
    paint(gO, cO, {interactive: true});
    document.getElementById('hud').innerHTML =
      'zoom <b>' + view.k.toFixed(1) + '&times;</b> · drag to pan · scroll to zoom' +
      (view.k > 2.4 ? '' : ' · zoom in for labels');
  } else if (vw === 'compare') {
    const a = [], b = [];
    D.statuses.forEach((_, i) => { a[i] = 1; b[i] = 1; });
    if (split) {
      a[S_B] = 0; b[S_A] = 0;
      paint(gA, cA, {alpha: a}); paint(gB, cB, {alpha: b});
    } else {
      a[S_A] = 1 - blend; a[S_B] = blend;
      paint(gA, cA, {alpha: a});
    }
  }
  const note = document.getElementById('tabnote');
  if (drill >= 0 && vw === 'overview') {
    note.textContent = D.clusters[drill].size.toLocaleString() + ' nodes in cluster ' +
      D.clusters[drill].name;
  } else {
    note.textContent = D.meta.drawnNodes.toLocaleString() + ' of ' +
      D.meta.totalNodes.toLocaleString() + ' nodes drawn' +
      (D.meta.truncated ? ' · differences prioritized' : '');
  }
}

// ---- selection ----------------------------------------------------------
function showCard(i) {
  const card = document.getElementById('card'), body = document.getElementById('cardbody');
  if (i < 0) {
    card.classList.remove('on');
    [...ct.children].forEach(tr => tr.classList.remove('sel'));
    document.querySelectorAll('.card.hot').forEach(el => el.classList.remove('hot'));
    return;
  }
  const buckets = {};
  for (const ei of inc[i]) {
    const [a, b, es] = D.edges[ei];
    (buckets[es] = buckets[es] || new Set()).add(D.labels[a === i ? b : a]);
  }
  let html = `<div class="chip"><span id="cs"></span><b>${esc(D.labels[i])}</b></div>`;
  html += `<div class="nbr" style="margin-top:8px">${D.statusLabels[D.nodeStatus[i]]}</div>`;
  const note = D.changedNotes[D.labels[i]];
  if (note) html += `<div class="nbr">${esc(note)}</div>`;
  const mtc = D.matches[D.labels[i]];
  if (mtc) html += `<div class="nbr">${matchText(mtc)}</div>`;
  D.statuses.forEach((s, si) => {
    const set = buckets[si]; if (!set || !set.size) return;
    const names = [...set].slice(0, 8).map(esc).join(', ');
    html += `<div class="nbr"><b>${set.size}</b> ${D.statusLabels[si].toLowerCase()}` +
      ` <span class="list">— ${names}${set.size > 8 ? '…' : ''}</span></div>`;
  });
  body.innerHTML = html;
  body.querySelector('#cs').appendChild(glyph(D.nodeStatus[i], 12));
  card.classList.add('on');
  [...ct.children].forEach(tr => tr.classList.toggle('sel', tr.dataset.label === D.labels[i]));
}

function pickLabel(label) {
  const i = D.labels.indexOf(label);
  document.querySelectorAll('.card.hot').forEach(el => el.classList.remove('hot'));
  const el = document.querySelector(`.card[data-label="${CSS.escape(label)}"]`);
  if (el) {
    el.classList.add('hot');
    if (vw === 'cards') el.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  }
  if (i < 0) { return; }
  sel = i; showCard(i);
  if (vw === 'overview') {
    view.k = Math.max(view.k, 4.5);
    const w = cO.clientWidth, h = cO.clientHeight;
    view.x = w / 2 - D.x[i] * spanOf(w, h) * view.k - PAD;
    view.y = h / 2 - D.y[i] * spanOf(w, h) * view.k - PAD;
  }
  paintAll();
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
        d.addEventListener('click', () => pickLabel(D.labels[i]));
        hits.appendChild(d); shown++;
      }
    }
    if (!shown) hits.innerHTML = '<div class="sub">No match.</div>';
  }
  paintAll();
});

// ---- controls -----------------------------------------------------------
document.querySelectorAll('#mapbar .seg').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#mapbar .seg').forEach(b => b.classList.remove('on'));
    btn.classList.add('on'); showMode = btn.dataset.show; paintAll();
  });
});
const visibleNode = i => {
  if (showMode === 'all') return true;
  if (showMode === 'changes') return D.nodeStatus[i] !== S_SHARED || inc[i].some(ei => D.edges[ei][2] !== S_SHARED);
  return TOP_SIG.has(i) || D.nodeSig[i] >= SIG_MAX * .35;
};

const diffBtn = document.getElementById('diffonly');
diffBtn.addEventListener('click', () => {
  const next = !(diffBtn.getAttribute('aria-pressed') === 'true');
  on[S_SHARED] = !next;
  const lab = legend.children[S_SHARED];
  lab.querySelector('input').checked = !next;
  lab.classList.toggle('off', next);
  diffBtn.setAttribute('aria-pressed', String(next));
  paintAll();
});
document.getElementById('reset').addEventListener('click', () => {
  view = {k: 1, x: 0, y: 0}; cview = {k: 1, x: 0, y: 0}; q.value = ''; hits.innerHTML = '';
  sel = -1; showCard(-1); setDrill(-1); paintAll();
});
const themeBtn = document.getElementById('theme');
themeBtn.addEventListener('click', () => setTheme(mode === 'light' ? 'dark' : 'light'));
function setTheme(next) {
  mode = next; T = D.themes[next];
  document.documentElement.setAttribute('data-theme', next);
  themeBtn.textContent = next === 'light' ? 'Dark' : 'Light';
  [...legend.children].forEach((lab, i) => {
    const old = lab.querySelector('svg.glyph');
    if (old) lab.replaceChild(glyph(i), old);  // the match-ring row has no glyph
  });
  buildCards();
  if (sel >= 0) showCard(sel);
  paintAll();
}

document.querySelectorAll('#tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#tabs button').forEach(b => b.classList.remove('on'));
    btn.classList.add('on');
    vw = btn.dataset.v;
    document.querySelectorAll('.pane').forEach(p => p.classList.remove('on'));
    document.getElementById('pane-' + vw).classList.add('on');
    resize();
  });
});

const blendEl = document.getElementById('blend');
blendEl.addEventListener('input', () => { blend = blendEl.value / 100; paintAll(); });
const splitBtn = document.getElementById('splitBtn');
splitBtn.addEventListener('click', () => {
  split = !split;
  splitBtn.setAttribute('aria-pressed', String(split));
  document.getElementById('cmpwrap').classList.toggle('split', split);
  blendEl.disabled = split;
  resize();
});
let flicker = null;
const flickBtn = document.getElementById('flick');
flickBtn.addEventListener('click', () => {
  if (flicker) { clearInterval(flicker); flicker = null; flickBtn.setAttribute('aria-pressed', 'false'); return; }
  if (split) splitBtn.click();
  flickBtn.setAttribute('aria-pressed', 'true');
  flicker = setInterval(() => {
    blend = blend > .5 ? 0 : 1; blendEl.value = blend * 100; paintAll();
  }, 750);
});

// ---- canvas interaction (overview only) ---------------------------------
let drag = null, moved = false;
cO.addEventListener('mousedown', e => {
  drag = {x: e.offsetX - view.x, y: e.offsetY - view.y}; moved = false;
  cO.classList.add('dragging');
});
window.addEventListener('mouseup', () => { drag = null; cO.classList.remove('dragging'); });
cO.addEventListener('mousemove', e => {
  if (drag) { view.x = e.offsetX - drag.x; view.y = e.offsetY - drag.y; moved = true; paintAll(); return; }
  const i = pick(e.offsetX, e.offsetY);
  if (i !== hover) { hover = i; paintAll(); }
  if (i >= 0) showTip(i, e.offsetX, e.offsetY); else tip.style.opacity = 0;
});
cO.addEventListener('mouseleave', () => { tip.style.opacity = 0; hover = -1; paintAll(); });
cO.addEventListener('click', e => {
  if (moved) return;
  const i = pick(e.offsetX, e.offsetY);
  sel = (i === sel) ? -1 : i; showCard(sel); paintAll();
});
[cO, cA, cB].forEach(cv => cv.addEventListener('wheel', e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.16 : 1 / 1.16;
  const nk = Math.max(.3, Math.min(60, view.k * f)), ratio = nk / view.k;
  view.x = e.offsetX - (e.offsetX - view.x) * ratio;
  view.y = e.offsetY - (e.offsetY - view.y) * ratio;
  view.k = nk; paintAll();
}, {passive: false}));
[cA, cB].forEach(cv => {
  cv.addEventListener('mousedown', e => {
    drag = {x: e.offsetX - view.x, y: e.offsetY - view.y}; cv.classList.add('dragging');
  });
  cv.addEventListener('mousemove', e => {
    if (!drag) return;
    view.x = e.offsetX - drag.x; view.y = e.offsetY - drag.y; paintAll();
  });
  cv.addEventListener('mouseup', () => cv.classList.remove('dragging'));
});
window.addEventListener('keydown', e => {
  if (e.key === 'Escape') { sel = -1; showCard(-1); q.value = ''; hits.innerHTML = ''; paintAll(); }
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

function matchText(mt) {
  return mt[0] === 'fuzzy'
    ? `&asymp; matched to <b>${esc(mt[2])}</b> in ${esc(D.meta.nameB)} by similarity ` +
      `&middot; confidence ${mt[1].toFixed(2)}`
    : `&asymp; matched to <b>${esc(mt[2])}</b> in ${esc(D.meta.nameB)} (same label after normalization)`;
}

function showTip(i, px, py) {
  const note = D.changedNotes[D.labels[i]];
  const mtt = D.matches[D.labels[i]];
  tip.innerHTML = '<b>' + esc(D.labels[i]) + '</b><br>' +
    D.statusLabels[D.nodeStatus[i]] + ' · ' + inc[i].length + ' edges' +
    (note ? '<br>' + esc(note) : '') + (mtt ? '<br>' + matchText(mtt) : '');
  tip.style.left = Math.min(px + 16, cO.clientWidth - 330) + 'px';
  tip.style.top = Math.min(py + 16, cO.clientHeight - 60) + 'px';
  tip.style.opacity = 1;
}

// ---- boot ---------------------------------------------------------------
function fit(cv) {
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(cv.clientWidth * dpr);
  cv.height = Math.round(cv.clientHeight * dpr);
  cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
}
function resize() { [cO, cA, cB, cC, c3].forEach(fit); paintAll(); }

cC.addEventListener('mousemove', e => {
  const i = pickCluster(e.offsetX, e.offsetY);
  if (i !== hoverCl) { hoverCl = i; cC.style.cursor = i >= 0 ? 'pointer' : 'default'; paintClusters(); }
});
cC.addEventListener('mouseleave', () => { hoverCl = -1; paintClusters(); });
cC.addEventListener('click', e => {
  const i = pickCluster(e.offsetX, e.offsetY);
  if (i < 0) return;
  setDrill(i);
  sel = -1; showCard(-1);
  document.querySelector('#tabs button[data-v="overview"]').click();
  const members = [];
  for (let n = 0; n < N; n++) if (D.nodeCluster[n] === i) members.push(n);
  fitTo(members);
  paintAll();
});
cC.addEventListener('wheel', e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  const nk = Math.max(.5, Math.min(8, cview.k * f)), ratio = nk / cview.k;
  cview.x = e.offsetX - (e.offsetX - cview.x) * ratio;
  cview.y = e.offsetY - (e.offsetY - cview.y) * ratio;
  cview.k = nk; paintClusters();
}, {passive: false});

const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
setTheme(prefersDark ? 'dark' : 'light');
window.addEventListener('resize', resize);
resize();
if (D.meta.initialView && D.meta.initialView !== 'overview') {
  document.querySelector(`#tabs button[data-v="${D.meta.initialView}"]`).click();
}
// Embedded in a timeline page: the parent can select a node or set the theme.
// Only same-origin/srcdoc parents can reach this; no data leaves the page.
window.addEventListener('message', e => {
  const m = e.data || {};
  if (typeof m.pick === 'string') pickLabel(m.pick);
  if (m.theme === 'dark' || m.theme === 'light') setTheme(m.theme);
});
</script>
</body>
</html>
"""
