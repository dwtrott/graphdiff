"""Ego networks for the ranked small-multiples view.

The overview draws one big picture and leaves you to hunt through it. This
module backs the view that does the opposite: one small card per most-changed
node, in rank order, so the analysis does the searching and your eye only has to
read a card.

Each card is a node's immediate neighbourhood with every incident edge carrying
its diff status, so "what happened to Nakamura" is answerable at a glance
instead of by tracing lines across a hairball.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .._types import SOURCE, STATUS, TARGET, Status
from ..core.graph import isin_labels
from ..core.union import UnionDiffGraph

__all__ = ["EgoNeighbor", "EgoNetwork", "ego_networks"]

#: Precedence when several edges join the same pair: a difference outranks a
#: shared edge, so a card never hides a change behind an unchanged parallel edge.
_RANK = {
    Status.SHARED.value: 0,
    Status.CHANGED.value: 1,
    Status.A_ONLY.value: 2,
    Status.B_ONLY.value: 3,
}


@dataclass(frozen=True)
class EgoNeighbor:
    """One neighbour of an ego node, and how the connection changed."""

    label: str
    status: str
    outgoing: bool


@dataclass
class EgoNetwork:
    """One node's immediate neighbourhood, grouped by what happened to it."""

    label: str
    status: str
    neighbors: list[EgoNeighbor] = field(default_factory=list)
    truncated: int = 0

    def counts(self) -> dict[str, int]:
        """Neighbour count per edge status, always including all four keys."""
        out = dict.fromkeys(_RANK, 0)
        for n in self.neighbors:
            out[n.status] = out.get(n.status, 0) + 1
        return out


def ego_networks(
    union: UnionDiffGraph,
    labels: list[str] | pd.Index,
    *,
    max_neighbors: int = 24,
) -> list[EgoNetwork]:
    """Extract the ego network of each label from the full union graph.

    Parameters
    ----------
    union:
        The union diff graph. The *whole* graph is used, not the drawn subset,
        so a card is never missing a neighbour that the overview happened to cap.
    labels:
        Ego nodes, in the order the cards should appear (normally the
        most-changed ranking).
    max_neighbors:
        Cap per card. Differences are kept ahead of unchanged neighbours, and
        the overflow count is reported so the card can say what it omitted.

    Returns
    -------
    list[EgoNetwork]
        One entry per requested label, in the order given.
    """
    wanted = pd.Index(pd.Series(list(labels), dtype=object).astype(str))
    if not len(wanted) or not len(union.edges):
        return [EgoNetwork(label=str(x), status=_node_status(union, str(x))) for x in wanted]

    edges = union.edges
    touching = edges[isin_labels(edges[SOURCE], wanted) | isin_labels(edges[TARGET], wanted)]

    # One row per (ego, neighbour, status, direction), both orientations.
    frames = []
    for ego_col, other_col, outgoing in ((SOURCE, TARGET, True), (TARGET, SOURCE, False)):
        mask = isin_labels(touching[ego_col], wanted)
        part = touching.loc[mask, [ego_col, other_col, STATUS]].copy()
        part.columns = ["ego", "neighbor", "status"]
        part["outgoing"] = outgoing
        frames.append(part)
    pairs = pd.concat(frames, ignore_index=True)
    pairs = pairs[pairs["ego"] != pairs["neighbor"]]
    pairs["status"] = pairs["status"].astype(str)
    pairs["rank"] = pairs["status"].map(_RANK).fillna(0).astype(int)

    # Collapse parallel edges, keeping the most interesting status.
    pairs = (
        pairs.sort_values("rank", ascending=False, kind="stable")
        .drop_duplicates(subset=["ego", "neighbor"], keep="first")
        .reset_index(drop=True)
    )

    # Iterating a pandas GroupBy, not a dict: dict() on a DataFrameGroupBy
    # resolves as column access and raises, so the comprehension is required.
    grouped = {ego: frame for ego, frame in pairs.groupby("ego", sort=False)}
    out: list[EgoNetwork] = []
    for label in wanted:
        label = str(label)
        frame = grouped.get(label)
        network = EgoNetwork(label=label, status=_node_status(union, label))
        if frame is not None and len(frame):
            ordered = frame.sort_values(
                ["rank", "neighbor"], ascending=[False, True], kind="stable"
            )
            kept = ordered.head(max_neighbors)
            network.truncated = int(len(ordered) - len(kept))
            network.neighbors = [
                EgoNeighbor(str(r["neighbor"]), str(r["status"]), bool(r["outgoing"]))
                for r in kept.to_dict("records")
            ]
        out.append(network)
    return out


def _node_status(union: UnionDiffGraph, label: str) -> str:
    try:
        return str(union.nodes.loc[label, STATUS])
    except KeyError:  # pragma: no cover - defensive
        return Status.SHARED.value
