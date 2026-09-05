"""The high-level entry points: :func:`compare` and :func:`inspect_graph`."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

from .core.align import AlignmentResult, AlignMethod, align_graphs
from .core.graph import PropertyGraph
from .core.union import AttributeComparison, UnionDiffGraph, build_union_diff_graph
from .io import read_graph
from .metrics.base import DualScore, to_jsonable
from .metrics.cluster import cluster_union
from .metrics.ged import GEDCosts, graph_edit_distance
from .metrics.neighborhood import Direction, neighborhood_delta
from .metrics.settheoretic import set_theoretic
from .metrics.significance import node_significance
from .metrics.weights import weight_agreement
from .report.findings import generate_findings
from .report.report import ComparisonReport

__all__ = ["GraphLike", "compare", "compare_files", "inspect_graph"]

#: Anything :func:`compare` will accept for a graph argument.
GraphLike = PropertyGraph | str | Path


def _coerce(graph: GraphLike, *, directed: bool = True) -> PropertyGraph:
    if isinstance(graph, PropertyGraph):
        return graph
    return read_graph(graph, directed=directed)


def _describe_input(graph: GraphLike, loaded: PropertyGraph) -> dict[str, Any]:
    """Provenance for one input: where it came from and a content hash if on disk."""
    info: dict[str, Any] = {
        "name": loaded.name,
        "n_nodes": loaded.n_nodes,
        "n_edges": loaded.n_edges,
        "directed": loaded.directed,
    }
    if isinstance(graph, (str, Path)):
        path = Path(graph)
        info["path"] = str(path.resolve())
        try:
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
                info["sha256"] = digest.hexdigest()
                info["bytes"] = path.stat().st_size
            elif path.is_dir():
                parts = sorted(p for p in path.iterdir() if p.suffix == ".parquet")
                digest = hashlib.sha256()
                for part in parts:
                    digest.update(part.name.encode())
                    digest.update(part.read_bytes())
                info["sha256"] = digest.hexdigest()
                info["bytes"] = sum(p.stat().st_size for p in parts)
        except OSError:  # pragma: no cover - unreadable input still gets a report
            pass
    else:
        info["path"] = None
        info["source"] = "in-memory PropertyGraph"
    return info


def _environment() -> dict[str, str]:
    import numpy
    import pandas

    from . import __version__

    return {
        "graphdiff": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "argv": " ".join(sys.argv[:1]),
    }


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
    cluster_by: str | None = None,
    findings: bool = True,
    align: AlignMethod = "exact",
    align_threshold: float = 0.6,
    align_options: dict[str, Any] | None = None,
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
    cluster_by:
        Node attribute to partition by for the cluster summary; ``None`` detects
        communities.
    findings:
        Generate the plain-language findings list (needs significance and
        clusters; set ``False`` in tight batch loops).
    align:
        How B's nodes are matched to A's. ``"exact"`` (default) joins on
        identical labels. ``"normalized"`` also matches labels that agree after
        case-folding and stripping punctuation/whitespace. ``"fuzzy"`` further
        matches the remainder by trigram similarity and shared neighbourhood,
        each match carrying a confidence — see
        :func:`~graphdiff.core.align.align_graphs`.
    align_threshold:
        Minimum score for a fuzzy match.
    align_options:
        Extra keyword arguments for :func:`~graphdiff.core.align.align_graphs`
        (``label_weight``, ``max_candidates``, ``rounds``, ...).

    Returns
    -------
    ComparisonReport
        Carries every metric both raw and restricted to the shared-node induced
        subgraph, the significance ranking, cluster summary, findings, and
        provenance.

    Examples
    --------
    >>> report = compare("v1.graphml", "v2.graphml")  # doctest: +SKIP
    >>> report.score("jaccard_edges")                 # doctest: +SKIP
    """
    graph_a = _coerce(a)
    graph_b = _coerce(b)
    alignment: AlignmentResult | None = None
    if align != "exact":
        alignment = align_graphs(
            graph_a, graph_b, method=align, threshold=align_threshold, **(align_options or {})
        )
    union = build_union_diff_graph(
        graph_a,
        graph_b,
        name_a=name_a,
        name_b=name_b,
        comparison=comparison,
        alignment=alignment,
    )
    provenance = {
        "input_a": _describe_input(a, graph_a),
        "input_b": _describe_input(b, graph_b),
        "parameters": {
            "weight_attribute": weight_attribute,
            "direction": direction,
            "cluster_by": cluster_by,
            "align": align,
            "align_threshold": align_threshold if align != "exact" else None,
            "ged_costs": to_jsonable(ged_costs) if ged_costs is not None else "unit",
            "attribute_comparison": to_jsonable(comparison) if comparison is not None else "all",
        },
        "environment": _environment(),
    }
    return report_from_union(
        union,
        ged_costs=ged_costs,
        weight_attribute=weight_attribute,
        direction=direction,
        top_n=top_n,
        keep_union=keep_union,
        metadata=metadata,
        cluster_by=cluster_by,
        findings=findings,
        provenance=provenance,
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
    cluster_by: str | None = None,
    findings: bool = True,
    provenance: dict[str, Any] | None = None,
) -> ComparisonReport:
    """Run the metric suite over an already-built union diff graph."""
    shared = union.induced_shared()
    neighborhood = neighborhood_delta(union, direction=direction, top_n=top_n)

    significance = node_significance(
        union, neighborhood_jaccard=neighborhood.table.set_index("label")["jaccard"]
    )
    clusters = cluster_union(union, attribute=cluster_by) if findings else None
    ged_raw = graph_edit_distance(union, ged_costs)
    weights_raw = weight_agreement(union, weight_attribute)
    found = (
        generate_findings(
            union,
            significance,
            clusters,
            similarity=ged_raw.similarity,
            weight_spearman=weights_raw.spearman,
        )
        if findings
        else []
    )

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
        ged=DualScore(raw=ged_raw, shared=graph_edit_distance(shared, ged_costs)),
        weight_agreement=DualScore(
            raw=weights_raw, shared=weight_agreement(shared, weight_attribute)
        ),
        neighborhood=neighborhood,
        significance=significance,
        clusters=clusters,
        findings=found,
        provenance=provenance or {},
        union=union if keep_union else None,
        alignment=union.alignment,
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
