"""Metric 2 — exact normalized graph edit distance.

Because alignment is fixed (nodes join on label, edges on ``(source, type,
target)``), the edit distance is not a search problem: the optimal edit script
is read directly off the union diff graph. Every ``A_ONLY`` element is a
deletion, every ``B_ONLY`` element an insertion, every ``CHANGED`` element an
attribute substitution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._types import ETYPE, STATUS, Status
from ..core.union import UnionDiffGraph

__all__ = ["GEDCosts", "GEDResult", "graph_edit_distance"]


@dataclass(frozen=True)
class GEDCosts:
    """Per-operation edit costs.

    ``edge_type_costs`` multiplies every edge operation by a per-type factor,
    so that (say) a missing ``owns`` edge can count for more than a missing
    ``mentions`` edge. Types absent from the mapping use ``default_edge_type_cost``.
    """

    node_insert: float = 1.0
    node_delete: float = 1.0
    node_substitute: float = 1.0
    edge_insert: float = 1.0
    edge_delete: float = 1.0
    edge_substitute: float = 1.0
    edge_type_costs: dict[str, float] = field(default_factory=dict)
    default_edge_type_cost: float = 1.0

    def type_multipliers(self, types: pd.Series) -> np.ndarray:
        """Vector of per-edge multipliers for a Series of edge types."""
        if not self.edge_type_costs:
            return np.full(len(types), self.default_edge_type_cost, dtype="float64")
        mapped = types.astype(str).map(self.edge_type_costs)
        return mapped.fillna(self.default_edge_type_cost).to_numpy(dtype="float64")


@dataclass
class GEDResult:
    """Edit distance between two aligned graphs.

    ``normalized_distance`` divides the edit cost by the cost of the worst case
    (delete all of A, insert all of B), placing it in ``[0, 1]``.
    ``similarity`` is ``1 - normalized_distance``.
    """

    cost: float
    normalized_distance: float
    similarity: float
    denominator: float
    node_insertions: int
    node_deletions: int
    node_substitutions: int
    edge_insertions: int
    edge_deletions: int
    edge_substitutions: int


def _edge_cost(edges: pd.DataFrame, status: Status, unit: float, costs: GEDCosts) -> float:
    mask = (edges[STATUS] == status.value).to_numpy()
    n = int(mask.sum())
    if n == 0:
        return 0.0
    if not costs.edge_type_costs:
        return float(unit * costs.default_edge_type_cost * n)
    return float(unit * costs.type_multipliers(edges.loc[mask, ETYPE]).sum())


def graph_edit_distance(union: UnionDiffGraph, costs: GEDCosts | None = None) -> GEDResult:
    """Compute the exact edit distance implied by the union diff graph.

    Parameters
    ----------
    union:
        Union diff graph. Pass ``union.induced_shared()`` for the
        density-normalized variant.
    costs:
        Operation costs; defaults to unit cost for every operation.

    Returns
    -------
    GEDResult
        Raw cost, the ``[0, 1]`` normalized distance, and the operation counts
        it was assembled from.

    Notes
    -----
    Identical graphs give ``cost == 0`` and ``similarity == 1``. Graphs sharing
    nothing give ``normalized_distance == 1`` under default costs.
    """
    costs = costs or GEDCosts()
    node_counts = union.node_status_counts()
    edge_counts = union.edge_status_counts()

    node_deletions = node_counts[Status.A_ONLY.value]
    node_insertions = node_counts[Status.B_ONLY.value]
    node_substitutions = node_counts[Status.CHANGED.value]

    cost = (
        costs.node_delete * node_deletions
        + costs.node_insert * node_insertions
        + costs.node_substitute * node_substitutions
        + _edge_cost(union.edges, Status.A_ONLY, costs.edge_delete, costs)
        + _edge_cost(union.edges, Status.B_ONLY, costs.edge_insert, costs)
        + _edge_cost(union.edges, Status.CHANGED, costs.edge_substitute, costs)
    )

    # Worst case: delete everything in A, then insert everything in B.
    n_a_nodes = (
        node_counts[Status.SHARED.value]
        + node_counts[Status.CHANGED.value]
        + node_counts[Status.A_ONLY.value]
    )
    n_b_nodes = (
        node_counts[Status.SHARED.value]
        + node_counts[Status.CHANGED.value]
        + node_counts[Status.B_ONLY.value]
    )
    in_a_mask = union.edges[STATUS].isin(
        {Status.SHARED.value, Status.CHANGED.value, Status.A_ONLY.value}
    )
    in_b_mask = union.edges[STATUS].isin(
        {Status.SHARED.value, Status.CHANGED.value, Status.B_ONLY.value}
    )
    if not costs.edge_type_costs:
        unit = costs.default_edge_type_cost
        edge_delete_all = float(costs.edge_delete * unit * int(in_a_mask.sum()))
        edge_insert_all = float(costs.edge_insert * unit * int(in_b_mask.sum()))
    else:
        edge_delete_all = float(
            costs.edge_delete * costs.type_multipliers(union.edges.loc[in_a_mask, ETYPE]).sum()
        )
        edge_insert_all = float(
            costs.edge_insert * costs.type_multipliers(union.edges.loc[in_b_mask, ETYPE]).sum()
        )
    denominator = (
        costs.node_delete * n_a_nodes
        + costs.node_insert * n_b_nodes
        + edge_delete_all
        + edge_insert_all
    )

    normalized = 0.0 if denominator == 0 else min(1.0, cost / denominator)
    return GEDResult(
        cost=float(cost),
        normalized_distance=float(normalized),
        similarity=float(1.0 - normalized),
        denominator=float(denominator),
        node_insertions=int(node_insertions),
        node_deletions=int(node_deletions),
        node_substitutions=int(node_substitutions),
        edge_insertions=int(edge_counts[Status.B_ONLY.value]),
        edge_deletions=int(edge_counts[Status.A_ONLY.value]),
        edge_substitutions=int(edge_counts[Status.CHANGED.value]),
    )
