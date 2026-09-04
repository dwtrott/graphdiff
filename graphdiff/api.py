"""The high-level entry points: :func:`compare` and :func:`inspect_graph`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .core.graph import PropertyGraph
from .core.union import AttributeComparison, UnionDiffGraph, build_union_diff_graph
from .io import read_graph
from .metrics.base import DualScore
from .metrics.ged import GEDCosts, graph_edit_distance
from .metrics.neighborhood import Direction, neighborhood_delta
from .metrics.settheoretic import set_theoretic
from .metrics.weights import weight_agreement
from .report.report import ComparisonReport

__all__ = ["GraphLike", "compare", "compare_files", "inspect_graph"]

#: Anything :func:`compare` will accept for a graph argument.
GraphLike = PropertyGraph | str | Path


def _coerce(graph: GraphLike, *, directed: bool = True) -> PropertyGraph:
    if isinstance(graph, PropertyGraph):
        return graph
    return read_graph(graph, directed=directed)


def compare(
    a: GraphLike,
    b: GraphLike,
    *,
    name_a: str | None = None,
    name_b: str | None = None,
    comparison: AttributeComparison | None = None,
    ged_costs: GEDCosts | None = None,
    weight_attribute: str = "weight",
    direction: Direction = "both",
    top_n: int = 25,
    keep_union: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ComparisonReport:
    """Compare two graphs and return a full :class:`ComparisonReport`.

    Parameters
    ----------
    a, b:
        :class:`~graphdiff.core.PropertyGraph` objects, or paths to graph files
        in any supported format (auto-detected).
    name_a, name_b:
        Display names; default to each graph's own name.
    comparison:
        Which attributes drive ``CHANGED`` detection, and with what tolerance.
    ged_costs:
        Per-operation edit costs, including per-edge-type weighting.
    weight_attribute:
        Edge attribute used for the weight-agreement metric.
    direction:
        Neighborhood direction for the per-node delta.
    top_n:
        Default size of the most-changed-nodes table.
    keep_union:
        Retain the union diff graph on the report. Set ``False`` in batch runs
        to release the memory once scores are computed.
    metadata:
        Arbitrary extra fields to record in the report.

    Returns
    -------
    ComparisonReport
        Carries every metric both raw and restricted to the shared-node induced
        subgraph, plus the ranked most-changed-nodes table.

    Examples
    --------
    >>> report = compare("v1.graphml", "v2.graphml")  # doctest: +SKIP
    >>> report.score("jaccard_edges")                 # doctest: +SKIP
    """
    graph_a = _coerce(a)
    graph_b = _coerce(b)
    union = build_union_diff_graph(
        graph_a, graph_b, name_a=name_a, name_b=name_b, comparison=comparison
    )
    return report_from_union(
        union,
        ged_costs=ged_costs,
        weight_attribute=weight_attribute,
        direction=direction,
        top_n=top_n,
        keep_union=keep_union,
        metadata=metadata,
    )


def report_from_union(
    union: UnionDiffGraph,
    *,
    ged_costs: GEDCosts | None = None,
    weight_attribute: str = "weight",
    direction: Direction = "both",
    top_n: int = 25,
    keep_union: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ComparisonReport:
    """Run the metric suite over an already-built union diff graph."""
    shared = union.induced_shared()

    return ComparisonReport(
        name_a=union.name_a,
        name_b=union.name_b,
        directed=union.directed,
        node_status_counts=union.node_status_counts(),
        edge_status_counts=union.edge_status_counts(),
        graph_summary={
            "a": inspect_graph(union.graph_a),
            "b": inspect_graph(union.graph_b),
            "union": {"n_nodes": union.n_nodes, "n_edges": union.n_edges},
            "shared_subgraph": {"n_nodes": shared.n_nodes, "n_edges": shared.n_edges},
        },
        set_theoretic=DualScore(raw=set_theoretic(union), shared=set_theoretic(shared)),
        ged=DualScore(
            raw=graph_edit_distance(union, ged_costs),
            shared=graph_edit_distance(shared, ged_costs),
        ),
        weight_agreement=DualScore(
            raw=weight_agreement(union, weight_attribute),
            shared=weight_agreement(shared, weight_attribute),
        ),
        neighborhood=neighborhood_delta(union, direction=direction, top_n=top_n),
        union=union if keep_union else None,
        metadata=metadata or {},
    )


def compare_files(path_a: str | Path, path_b: str | Path, **kwargs: Any) -> ComparisonReport:
    """Convenience wrapper reading both graphs from disk before comparing."""
    return compare(Path(path_a), Path(path_b), **kwargs)


def inspect_graph(graph: GraphLike) -> dict[str, Any]:
    """Summarize a graph: counts, edge types, density, and degree statistics."""
    g = _coerce(graph)
    return {
        "name": g.name,
        "directed": g.directed,
        "n_nodes": g.n_nodes,
        "n_edges": g.n_edges,
        "n_edge_types": len(g.edge_types),
        "edge_types": g.edge_types[:50],
        "density": g.density,
        "node_attributes": g.node_attributes,
        "edge_attributes": g.edge_attributes,
        "degree": g.degree_stats(),
    }
