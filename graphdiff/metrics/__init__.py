"""Similarity metrics over a :class:`~graphdiff.core.UnionDiffGraph`.

Every metric is a pure function of the union diff graph and returns a typed,
JSON-serializable result. Metrics are reported both raw and restricted to the
subgraph induced on shared nodes (see :class:`~graphdiff.metrics.base.DualScore`).
"""

from __future__ import annotations

from .base import DualScore, flatten_scores, to_jsonable
from .cluster import Cluster, ClusterMap, cluster_union
from .ged import GEDCosts, GEDResult, graph_edit_distance
from .neighborhood import Direction, NeighborhoodDeltaResult, neighborhood_delta
from .settheoretic import SetScores, SetTheoreticResult, set_theoretic
from .significance import BackgroundRates, SignificanceResult, node_significance
from .weights import WeightAgreementResult, weight_agreement

__all__ = [
    "BackgroundRates",
    "Cluster",
    "ClusterMap",
    "Direction",
    "DualScore",
    "GEDCosts",
    "GEDResult",
    "NeighborhoodDeltaResult",
    "SetScores",
    "SetTheoreticResult",
    "SignificanceResult",
    "WeightAgreementResult",
    "cluster_union",
    "flatten_scores",
    "graph_edit_distance",
    "neighborhood_delta",
    "node_significance",
    "set_theoretic",
    "to_jsonable",
    "weight_agreement",
]
