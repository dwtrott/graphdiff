"""Metric 4 — per-node neighborhood delta.

For every node present in both graphs, compare its neighbor set in A against
its neighbor set in B. The ranked table this produces is the primary
user-facing output: it answers "which entities changed the most?" rather than
"how similar are these graphs overall?".

The computation is a vectorized join on ``(node, neighbor)`` pairs, not a
per-node Python set intersection, so it stays usable at 10^6 edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .._types import LABEL, SOURCE, TARGET
from ..core.graph import PropertyGraph, isin_labels
from ..core.union import UnionDiffGraph

__all__ = ["Direction", "NeighborhoodDeltaResult", "neighborhood_delta"]

Direction = Literal["both", "out", "in"]


@dataclass
class NeighborhoodDeltaResult:
    """Neighborhood Jaccard per shared node, plus the ranked change table.

    ``table`` columns: ``label``, ``jaccard``, ``n_neighbors_a``,
    ``n_neighbors_b``, ``n_shared``, ``n_added``, ``n_removed``. Rows are sorted
    by ``jaccard`` ascending — most-changed nodes first.
    """

    table: pd.DataFrame
    mean_jaccard: float
    median_jaccard: float
    n_nodes_compared: int
    n_unchanged: int
    direction: Direction = "both"
    _top_n: int = field(default=25, repr=False)

    def top_changed(self, n: int | None = None) -> pd.DataFrame:
        """The ``n`` most-changed shared nodes (lowest neighborhood Jaccard)."""
        return self.table.head(n if n is not None else self._top_n)

    def to_dict(self) -> dict[str, object]:
        """Summary statistics plus the top rows; the full table lives in ``table``."""
        return {
            "mean_jaccard": self.mean_jaccard,
            "median_jaccard": self.median_jaccard,
            "n_nodes_compared": self.n_nodes_compared,
            "n_unchanged": self.n_unchanged,
            "direction": self.direction,
            "top_changed": self.top_changed().to_dict("records"),
        }


def _adjacency_pairs(graph: PropertyGraph, direction: Direction) -> pd.DataFrame:
    """Deduplicated ``(node, neighbor)`` pairs implied by the edge table."""
    if not graph.n_edges:
        return pd.DataFrame({"node": pd.Series(dtype=object), "neighbor": pd.Series(dtype=object)})

    src, tgt = graph.edges[SOURCE], graph.edges[TARGET]

    forward = direction in ("both", "out") or not graph.directed
    backward = direction in ("both", "in") or not graph.directed

    frames: list[pd.DataFrame] = []
    if forward:
        frames.append(pd.DataFrame({"node": src, "neighbor": tgt}))
    if backward:
        frames.append(pd.DataFrame({"node": tgt, "neighbor": src}))

    pairs = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return pairs.drop_duplicates(ignore_index=True)


def neighborhood_delta(
    union: UnionDiffGraph,
    *,
    direction: Direction = "both",
    top_n: int = 25,
) -> NeighborhoodDeltaResult:
    """Jaccard similarity of each shared node's neighborhood between A and B.

    Parameters
    ----------
    union:
        Union diff graph.
    direction:
        Which incidences count as neighbors. ``"both"`` unions successors and
        predecessors; ignored for undirected graphs.
    top_n:
        Default row count returned by :meth:`NeighborhoodDeltaResult.top_changed`.

    Returns
    -------
    NeighborhoodDeltaResult
        A per-node table sorted most-changed first, with summary statistics.

    Notes
    -----
    A node isolated in both graphs has an undefined Jaccard; it is reported as
    ``1.0`` (nothing changed) and counted in ``n_unchanged``.
    """
    shared_labels = union.shared_node_labels
    n_shared = len(shared_labels)
    columns = [
        LABEL,
        "jaccard",
        "n_neighbors_a",
        "n_neighbors_b",
        "n_shared",
        "n_added",
        "n_removed",
    ]
    if n_shared == 0:
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
        empty[LABEL] = pd.Series(dtype=object)
        return NeighborhoodDeltaResult(
            table=empty,
            mean_jaccard=float("nan"),
            median_jaccard=float("nan"),
            n_nodes_compared=0,
            n_unchanged=0,
            direction=direction,
            _top_n=top_n,
        )

    pairs_a = _adjacency_pairs(union.graph_a, direction)
    pairs_b = _adjacency_pairs(union.graph_b, direction)
    pairs_a = pairs_a[isin_labels(pairs_a["node"], shared_labels)]
    pairs_b = pairs_b[isin_labels(pairs_b["node"], shared_labels)]

    merged = pairs_a.merge(pairs_b, on=["node", "neighbor"], how="outer", indicator=True)
    ind = merged["_merge"].astype(str)
    merged = merged.assign(
        both=(ind == "both").to_numpy(),
        only_a=(ind == "left_only").to_numpy(),
        only_b=(ind == "right_only").to_numpy(),
    )
    grouped = merged.groupby("node", sort=False)[["both", "only_a", "only_b"]].sum()
    grouped = grouped.reindex(shared_labels, fill_value=0)

    n_both = grouped["both"].to_numpy(dtype="int64")
    n_removed = grouped["only_a"].to_numpy(dtype="int64")
    n_added = grouped["only_b"].to_numpy(dtype="int64")
    union_size = n_both + n_removed + n_added

    with np.errstate(invalid="ignore", divide="ignore"):
        jaccard = np.where(union_size == 0, 1.0, n_both / np.maximum(union_size, 1))

    table = pd.DataFrame(
        {
            LABEL: pd.Series(shared_labels).reset_index(drop=True),
            "jaccard": jaccard.astype("float64"),
            "n_neighbors_a": n_both + n_removed,
            "n_neighbors_b": n_both + n_added,
            "n_shared": n_both,
            "n_added": n_added,
            "n_removed": n_removed,
        }
    )
    table = table.sort_values(
        ["jaccard", "n_added", "n_removed", LABEL], ascending=[True, False, False, True]
    ).reset_index(drop=True)

    return NeighborhoodDeltaResult(
        table=table,
        mean_jaccard=float(np.mean(jaccard)),
        median_jaccard=float(np.median(jaccard)),
        n_nodes_compared=n_shared,
        n_unchanged=int((jaccard >= 1.0).sum()),
        direction=direction,
        _top_n=top_n,
    )
