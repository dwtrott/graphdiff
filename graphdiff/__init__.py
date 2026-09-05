"""graphdiff — graph similarity, alignment, and structural diffing at scale.

The package is organized around one idea: align two property graphs by node
label, materialize the result as a single *union diff graph* where every node
and edge carries a ``status`` (``SHARED``, ``A_ONLY``, ``B_ONLY``, ``CHANGED``),
and derive every metric, report, and visualization from that one structure.

Quickstart
----------
>>> import graphdiff as gd                            # doctest: +SKIP
>>> report = gd.compare("v1.graphml", "v2.graphml")   # doctest: +SKIP
>>> report.score("jaccard_edges")                     # doctest: +SKIP
>>> report.neighborhood.top_changed(10)               # doctest: +SKIP
"""

from __future__ import annotations

from ._types import Status
from .api import compare, compare_files, inspect_graph, report_from_union
from .core import (
    AlignmentResult,
    AttributeComparison,
    DuplicateEdgeError,
    PropertyGraph,
    UnionDiffGraph,
    align_graphs,
    build_union_diff_graph,
)
from .io import detect_format, read_graph, write_graph
from .metrics import (
    GEDCosts,
    graph_edit_distance,
    neighborhood_delta,
    set_theoretic,
    weight_agreement,
)
from .report import SCALAR_METRICS, ComparisonReport

__version__ = "0.1.0"

__all__ = [
    "SCALAR_METRICS",
    "AlignmentResult",
    "AttributeComparison",
    "ComparisonReport",
    "DuplicateEdgeError",
    "GEDCosts",
    "PropertyGraph",
    "Status",
    "UnionDiffGraph",
    "__version__",
    "align_graphs",
    "build_union_diff_graph",
    "compare",
    "compare_files",
    "detect_format",
    "graph_edit_distance",
    "inspect_graph",
    "neighborhood_delta",
    "read_graph",
    "report_from_union",
    "set_theoretic",
    "weight_agreement",
    "write_graph",
]
