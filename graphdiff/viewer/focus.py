"""Choosing what to draw.

A million-node union graph cannot be usefully rendered, and drawing a uniform
sample of it mostly draws the parts that did not change. The viewer instead
shows a *focus subgraph*: everything that differs between A and B, plus a ring
of unchanged context so the differences sit in a recognizable neighborhood.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .._types import SOURCE, STATUS, TARGET, Status
from ..core.graph import isin_labels
from ..core.union import UnionDiffGraph

__all__ = ["FocusSelection", "select_focus"]

_DIFFERING = {Status.A_ONLY.value, Status.B_ONLY.value, Status.CHANGED.value}


@dataclass
class FocusSelection:
    """The subset of the union graph the viewer will draw.

    Attributes
    ----------
    labels:
        Node labels included, differing nodes first.
    edges:
        The union edge rows whose endpoints are both included.
    truncated:
        True when the cap forced elements to be dropped.
    n_nodes_total / n_edges_total:
        Sizes of the full union, for honest reporting in the UI.
    """

    labels: pd.Index
    edges: pd.DataFrame
    truncated: bool
    n_nodes_total: int
    n_edges_total: int
    context_hops: int


def select_focus(
    union: UnionDiffGraph, *, max_nodes: int = 2000, context_hops: int = 1
) -> FocusSelection:
    """Pick the nodes and edges worth drawing.

    Parameters
    ----------
    union:
        The union diff graph.
    max_nodes:
        Hard cap on drawn nodes. Differing nodes are kept ahead of context.
    context_hops:
        How many hops of unchanged neighborhood to include around the
        differences. ``0`` draws only what changed.

    Returns
    -------
    FocusSelection
        Labels and edge rows to render, plus whether anything was dropped.

    Notes
    -----
    When the two graphs are identical there is nothing to focus on, so the
    highest-degree nodes are shown instead — the picture is then a plain view of
    the shared structure, which is the honest rendering of "no differences".
    """
    nodes, edges = union.nodes, union.edges
    n_nodes_total, n_edges_total = len(nodes), len(edges)

    differing_nodes = nodes.index[nodes[STATUS].isin(_DIFFERING)]
    if n_edges_total:
        differing_edges = edges[edges[STATUS].isin(_DIFFERING)]
        endpoint_labels = pd.Index(
            pd.unique(
                pd.concat([differing_edges[SOURCE], differing_edges[TARGET]], ignore_index=True)
            )
        )
    else:
        endpoint_labels = pd.Index([], dtype=object)

    seeds = differing_nodes.union(endpoint_labels)
    seeds = seeds.intersection(nodes.index)

    if len(seeds) == 0:
        # Identical graphs: show the densest part of the shared structure.
        degree = _union_degree(union)
        keep = degree.nlargest(min(max_nodes, len(degree))).index
        return _build(union, keep, n_nodes_total, n_edges_total, len(degree) > max_nodes, 0)

    selected = pd.Index(seeds)
    if n_edges_total:
        for _ in range(max(context_hops, 0)):
            if len(selected) >= max_nodes:
                break
            touching = edges[
                isin_labels(edges[SOURCE], selected) | isin_labels(edges[TARGET], selected)
            ]
            grown = pd.Index(
                pd.unique(pd.concat([touching[SOURCE], touching[TARGET]], ignore_index=True))
            )
            selected = selected.union(grown.intersection(nodes.index))

    truncated = False
    if len(selected) > max_nodes:
        truncated = True
        priority_first = pd.Index(seeds)[:max_nodes]
        remaining = max_nodes - len(priority_first)
        if remaining > 0:
            context = selected.difference(priority_first)
            degree = _union_degree(union).reindex(context).fillna(0)
            priority_first = priority_first.union(
                degree.nlargest(min(remaining, len(degree))).index
            )
        selected = priority_first

    return _build(union, selected, n_nodes_total, n_edges_total, truncated, context_hops)


def _union_degree(union: UnionDiffGraph) -> pd.Series:
    """Degree of each union node counting every edge regardless of status."""
    if not len(union.edges):
        return pd.Series(0, index=union.nodes.index, dtype="int64")
    endpoints = pd.concat([union.edges[SOURCE], union.edges[TARGET]], ignore_index=True)
    return endpoints.value_counts().reindex(union.nodes.index, fill_value=0).astype("int64")


def _build(
    union: UnionDiffGraph,
    labels: pd.Index,
    n_nodes_total: int,
    n_edges_total: int,
    truncated: bool,
    context_hops: int,
) -> FocusSelection:
    labels = pd.Index(labels)
    if len(union.edges):
        mask = isin_labels(union.edges[SOURCE], labels) & isin_labels(union.edges[TARGET], labels)
        edges = union.edges.loc[mask].reset_index(drop=True)
    else:
        edges = union.edges
    truncated = truncated or len(edges) < n_edges_total or len(labels) < n_nodes_total
    return FocusSelection(
        labels=labels,
        edges=edges,
        truncated=bool(truncated),
        n_nodes_total=n_nodes_total,
        n_edges_total=n_edges_total,
        context_hops=context_hops,
    )
