"""Core data structures: the property graph container and the union diff graph."""

from __future__ import annotations

from .align import AlignmentResult, align_graphs, normalize_label
from .graph import DuplicateEdgeError, PropertyGraph
from .union import AttributeComparison, UnionDiffGraph, build_union_diff_graph

__all__ = [
    "AlignmentResult",
    "AttributeComparison",
    "DuplicateEdgeError",
    "PropertyGraph",
    "UnionDiffGraph",
    "align_graphs",
    "build_union_diff_graph",
    "normalize_label",
]
