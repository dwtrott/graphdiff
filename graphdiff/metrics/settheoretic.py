"""Metric 1 — set-theoretic overlap on node sets, edge sets, and typed-edge sets.

Scores are reported separately per element kind and are never blended into a
single number: a graph pair can share every node while sharing no edges, and
collapsing that into one figure hides exactly the thing the user is looking for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .._types import ETYPE, SOURCE, STATUS, TARGET, Status
from ..core.union import UnionDiffGraph

__all__ = ["SetScores", "SetTheoreticResult", "set_theoretic"]

_IN_A = {Status.SHARED.value, Status.CHANGED.value, Status.A_ONLY.value}
_IN_B = {Status.SHARED.value, Status.CHANGED.value, Status.B_ONLY.value}


@dataclass
class SetScores:
    """Overlap of two sets, with the counts the scores were computed from.

    ``jaccard`` = intersection / union, ``overlap`` = intersection / min(|A|, |B|),
    ``dice`` = 2 * intersection / (|A| + |B|). All three are 1.0 when both sets
    are empty.
    """

    jaccard: float
    overlap: float
    dice: float
    n_a: int
    n_b: int
    n_intersection: int
    n_union: int

    @classmethod
    def from_counts(cls, n_a: int, n_b: int, n_intersection: int) -> SetScores:
        """Derive all three coefficients from raw cardinalities."""
        n_union = n_a + n_b - n_intersection
        if n_union == 0:
            return cls(1.0, 1.0, 1.0, 0, 0, 0, 0)
        smaller = min(n_a, n_b)
        return cls(
            jaccard=n_intersection / n_union,
            overlap=(n_intersection / smaller) if smaller else 0.0,
            dice=(2 * n_intersection / (n_a + n_b)) if (n_a + n_b) else 1.0,
            n_a=n_a,
            n_b=n_b,
            n_intersection=n_intersection,
            n_union=n_union,
        )


@dataclass
class SetTheoreticResult:
    """Set overlap reported independently for each element kind."""

    nodes: SetScores
    edges: SetScores
    typed_edges: SetScores
    per_edge_type: dict[str, SetScores] = field(default_factory=dict)


def _counts_from_status(status: pd.Series) -> tuple[int, int, int]:
    counts = status.value_counts()
    shared = int(counts.get(Status.SHARED.value, 0))
    changed = int(counts.get(Status.CHANGED.value, 0))
    a_only = int(counts.get(Status.A_ONLY.value, 0))
    b_only = int(counts.get(Status.B_ONLY.value, 0))
    both = shared + changed
    return both + a_only, both + b_only, both


def _untyped_edge_counts(edges: pd.DataFrame) -> tuple[int, int, int]:
    """Collapse parallel typed edges onto ``(source, target)`` before counting."""
    if edges.empty:
        return 0, 0, 0
    # isin() on the categorical avoids materializing 10^6 Python strings.
    status = edges[STATUS]
    frame = edges[[SOURCE, TARGET]].assign(
        in_a=status.isin(_IN_A).to_numpy(),
        in_b=status.isin(_IN_B).to_numpy(),
    )
    grouped = frame.groupby([SOURCE, TARGET], sort=False)[["in_a", "in_b"]].any()
    n_a = int(grouped["in_a"].sum())
    n_b = int(grouped["in_b"].sum())
    both = int((grouped["in_a"] & grouped["in_b"]).sum())
    return n_a, n_b, both


def set_theoretic(union: UnionDiffGraph) -> SetTheoreticResult:
    """Compute Jaccard, overlap coefficient, and Dice for nodes and edges.

    Parameters
    ----------
    union:
        The union diff graph. Pass ``union.induced_shared()`` for the
        density-normalized variant.

    Returns
    -------
    SetTheoreticResult
        Separate :class:`SetScores` for node labels, untyped edges
        ``(source, target)``, typed edges ``(source, type, target)``, and one
        entry per edge type.
    """
    nodes = SetScores.from_counts(*_counts_from_status(union.nodes[STATUS]))
    typed_edges = SetScores.from_counts(*_counts_from_status(union.edges[STATUS]))
    edges = SetScores.from_counts(*_untyped_edge_counts(union.edges))

    per_type: dict[str, SetScores] = {}
    if not union.edges.empty:
        for etype, group in union.edges.groupby(ETYPE, sort=True, observed=True):
            per_type[str(etype)] = SetScores.from_counts(*_counts_from_status(group[STATUS]))

    return SetTheoreticResult(
        nodes=nodes, edges=edges, typed_edges=typed_edges, per_edge_type=per_type
    )
