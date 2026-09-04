"""Core data structures: the property graph container and the union diff graph."""

from __future__ import annotations

from .graph import DuplicateEdgeError, PropertyGraph
from .union import AttributeComparison, UnionDiffGraph, build_union_diff_graph

__all__ = [
    "AttributeComparison",
    "DuplicateEdgeError",
    "PropertyGraph",
    "UnionDiffGraph",
    "build_union_diff_graph",
]
