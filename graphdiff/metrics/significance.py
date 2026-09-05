"""Which changes actually matter.

Neighbourhood Jaccard is a *proportion*, and proportions rank a two-neighbour
node that lost one edge (0.5) above a 130-edge hub that lost thirty (0.77). For
"where are the significant changes" that is exactly backwards.

This module ranks by **statistical surprise** instead: given the graph-wide
background rate of removals, additions and reweightings, how unlikely is the
change this particular node experienced? A leaf losing its one edge when 7% of
edges were removed is unremarkable. A hub losing thirty of 130 under the same
rate has a binomial tail probability around 1e-12 — that is the event to show
someone first. Degree is folded in naturally: the same number of removals is
far more surprising on a small node than a large one, and a removed hub (every
edge gone) scores as the extreme event it is.

Two refinements make this robust on small graphs and dominant nodes:

* **Leave-one-out background.** The rate each node is tested against excludes
  that node's own edges. Otherwise a hub that lost everything in an otherwise
  static graph *is* the background rate, and scores as unremarkable.
* **Magnitude.** ``significance = (1 + surprise) * log2(2 + changes)``. Pure
  surprise ties a hub that lost ten edges with a leaf that lost one when the
  graph is small enough that neither is statistically unusual; the magnitude
  term breaks that tie the way a reader would. ``surprise`` is kept as its own
  column because it is the interpretable one (``-log10 p``).

Two complements are reported alongside: **impact** (raw edge changes weighted
by the node's PageRank in the union, so "big change on an important node"
is visible even when the graph-wide rate is high), and **centrality shift**
(PageRank in B minus PageRank in A, for nodes present in both) which catches a
node that quietly became — or stopped being — a hub.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .._types import LABEL, SOURCE, STATUS, TARGET, Status
from ..core.union import UnionDiffGraph

__all__ = ["BackgroundRates", "SignificanceResult", "node_significance"]

_CAP = 300.0  # -log10(p) ceiling; p underflows to 0 well before this


@dataclass(frozen=True)
class BackgroundRates:
    """Graph-wide change rates the per-node test is measured against."""

    removal: float  # A_ONLY edges / edges in A
    addition: float  # B_ONLY edges / edges in B
    reweight: float  # CHANGED edges / edges present in both


@dataclass
class SignificanceResult:
    """Per-node significance, ranked most-significant first.

    ``table`` columns: ``label``, ``status``, ``significance`` (the ranking
    score, ``(1 + surprise) * log2(2 + changes)``), ``surprise`` (summed
    ``-log10 p`` across the three tests), ``magnitude``, ``surprise_removed``,
    ``surprise_added``, ``surprise_changed``, ``n_removed``, ``n_added``,
    ``n_changed``, ``degree_a``, ``degree_b``, ``pagerank``, ``impact``,
    ``centrality_shift``, ``jaccard`` (the old neighbourhood proportion, kept
    for reference).
    """

    table: pd.DataFrame
    rates: BackgroundRates
    n_nodes: int

    def top(self, n: int = 25) -> pd.DataFrame:
        """The ``n`` most significant nodes."""
        return self.table.head(n)

    def to_dict(self) -> dict[str, object]:
        """Rates and the top rows; the full table lives in ``table``."""
        return {
            "rates": {
                "removal": self.rates.removal,
                "addition": self.rates.addition,
                "reweight": self.rates.reweight,
            },
            "n_nodes": self.n_nodes,
            "top": self.top(25).to_dict("records"),
        }


def _incident_counts(union: UnionDiffGraph) -> pd.DataFrame:
    """Per node: incident edge counts by status (both endpoints count)."""
    labels = union.nodes.index
    zero = pd.DataFrame(0, index=labels, columns=[s.value for s in Status], dtype="int64")
    if not len(union.edges):
        return zero
    long = pd.concat(
        [
            union.edges[[SOURCE, STATUS]].rename(columns={SOURCE: LABEL}),
            union.edges[[TARGET, STATUS]].rename(columns={TARGET: LABEL}),
        ],
        ignore_index=True,
    )
    long[STATUS] = long[STATUS].astype(str)
    counts = long.groupby([LABEL, STATUS], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(index=labels, columns=zero.columns, fill_value=0)
    return counts.astype("int64")


def _pagerank(labels: pd.Index, edges: pd.DataFrame) -> np.ndarray:
    """PageRank over an undirected simple graph on ``labels``."""
    import igraph as ig

    if not len(edges) or not len(labels):
        return np.full(len(labels), 1.0 / max(len(labels), 1))
    position = pd.Series(np.arange(len(labels)), index=labels)
    src = position.reindex(edges[SOURCE]).to_numpy()
    dst = position.reindex(edges[TARGET]).to_numpy()
    ok = ~(pd.isna(src) | pd.isna(dst))
    pairs = list(zip(src[ok].astype(int), dst[ok].astype(int), strict=True))
    graph = ig.Graph(n=len(labels), edges=pairs, directed=False)
    graph.simplify(multiple=True, loops=True)
    return np.asarray(graph.pagerank(), dtype="float64")


def _surprise(k: np.ndarray, n: np.ndarray, total_k: int, total_n: int) -> np.ndarray:
    """``-log10 P(X >= k)`` under a leave-one-out binomial background.

    The rate for node *i* is ``(total_k - k_i + 0.5) / (total_n - n_i + 1)``:
    everyone else's events over everyone else's trials, with a half-count of
    smoothing so a graph where nothing else changed still yields a finite,
    large surprise rather than a division by zero.
    """
    out = np.zeros(len(k), dtype="float64")
    mask = (k > 0) & (n > 0)
    if not mask.any():
        return out
    rest_k = np.maximum(total_k - k[mask], 0).astype("float64")
    rest_n = np.maximum(total_n - n[mask], 0).astype("float64")
    p = np.clip((rest_k + 0.5) / (rest_n + 1.0), 1e-9, 1 - 1e-9)
    tail = stats.binom.sf(k[mask] - 1, n[mask], p)  # P(X >= k)
    with np.errstate(divide="ignore"):
        out[mask] = np.minimum(_CAP, -np.log10(np.maximum(tail, 1e-300)))
    return out


def node_significance(
    union: UnionDiffGraph, *, neighborhood_jaccard: pd.Series | None = None
) -> SignificanceResult:
    """Rank every node in the union by how surprising its change is.

    Parameters
    ----------
    union:
        The union diff graph.
    neighborhood_jaccard:
        Optional ``label -> jaccard`` from the neighbourhood-delta metric, kept
        as a reference column so the two rankings can be compared.

    Returns
    -------
    SignificanceResult
        Table sorted by ``significance`` descending, ties broken by impact.
    """
    counts = _incident_counts(union)
    labels = union.nodes.index
    n_rem = counts[Status.A_ONLY.value].to_numpy()
    n_add = counts[Status.B_ONLY.value].to_numpy()
    n_chg = counts[Status.CHANGED.value].to_numpy()
    n_shr = counts[Status.SHARED.value].to_numpy()
    deg_a = n_shr + n_chg + n_rem
    deg_b = n_shr + n_chg + n_add
    both = n_shr + n_chg

    edge_counts = union.edge_status_counts()
    e_a = edge_counts["SHARED"] + edge_counts["CHANGED"] + edge_counts["A_ONLY"]
    e_b = edge_counts["SHARED"] + edge_counts["CHANGED"] + edge_counts["B_ONLY"]
    e_both = edge_counts["SHARED"] + edge_counts["CHANGED"]
    rates = BackgroundRates(
        removal=edge_counts["A_ONLY"] / e_a if e_a else 0.0,
        addition=edge_counts["B_ONLY"] / e_b if e_b else 0.0,
        reweight=edge_counts["CHANGED"] / e_both if e_both else 0.0,
    )

    # Leave-one-out in *edge* units: a node's incident edges are whole edges,
    # so removing them from the graph-wide totals removes them entirely (the
    # far endpoints do not keep a copy). A hub that lost all fifteen of the
    # graph's fifteen removed edges is then tested against a background of
    # zero-out-of-one, not against itself.
    s_rem = _surprise(n_rem, deg_a, edge_counts["A_ONLY"], e_a)
    s_add = _surprise(n_add, deg_b, edge_counts["B_ONLY"], e_b)
    s_chg = _surprise(n_chg, both, edge_counts["CHANGED"], e_both)
    surprise = s_rem + s_add + s_chg
    magnitude = n_rem + n_add + 0.5 * n_chg
    significance = np.where(magnitude > 0, (1.0 + surprise) * np.log2(2.0 + magnitude), 0.0)

    # Importance and impact.
    pagerank = _pagerank(labels, union.edges)
    pr_norm = pagerank / pagerank.max() if pagerank.max() > 0 else pagerank
    impact = (n_rem + n_add + 0.5 * n_chg) * pr_norm

    # Centrality shift for nodes present in both graphs.
    pr_a = _pagerank(union.graph_a.nodes.index, union.graph_a.edges)
    pr_b = _pagerank(union.graph_b.nodes.index, union.graph_b.edges)
    shift = (
        pd.Series(pr_b, index=union.graph_b.nodes.index).reindex(labels)
        - pd.Series(pr_a, index=union.graph_a.nodes.index).reindex(labels)
    ).to_numpy()

    table = pd.DataFrame(
        {
            LABEL: np.asarray(labels, dtype=object),
            "status": union.nodes[STATUS].astype(str).to_numpy(),
            "significance": significance,
            "surprise": surprise,
            "magnitude": magnitude,
            "surprise_removed": s_rem,
            "surprise_added": s_add,
            "surprise_changed": s_chg,
            "n_removed": n_rem,
            "n_added": n_add,
            "n_changed": n_chg,
            "degree_a": deg_a,
            "degree_b": deg_b,
            "pagerank": pagerank,
            "impact": impact,
            "centrality_shift": shift,
        }
    )
    if neighborhood_jaccard is not None:
        table["jaccard"] = neighborhood_jaccard.reindex(labels).to_numpy()
    else:
        table["jaccard"] = np.nan

    table = table.sort_values(
        ["significance", "impact", LABEL], ascending=[False, False, True]
    ).reset_index(drop=True)
    return SignificanceResult(table=table, rates=rates, n_nodes=len(labels))
