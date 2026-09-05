"""Construction of the :class:`UnionDiffGraph` — the single structure every
metric, report, and visualization in :mod:`graphdiff` is derived from.

Alignment is an exact join: nodes on label, edges on ``(source, type, target)``.
That makes construction ``O(n + m)`` with no heuristic matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._types import (
    A_PREFIX,
    B_PREFIX,
    CHANGED_ATTRS,
    ETYPE,
    LABEL,
    SOURCE,
    STATUS,
    STATUS_ORDER,
    TARGET,
    Status,
)
from .align import AlignmentResult
from .graph import PropertyGraph, isin_labels

if TYPE_CHECKING:  # pragma: no cover
    import igraph
    import rustworkx

__all__ = ["AttributeComparison", "UnionDiffGraph", "build_union_diff_graph"]

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 0.0


@dataclass(frozen=True)
class AttributeComparison:
    """Which attributes participate in ``CHANGED`` detection, and how closely.

    Parameters
    ----------
    node_attributes / edge_attributes:
        Explicit attribute names to compare. ``None`` means "every attribute
        present in both graphs". An empty sequence disables change detection
        for that element kind.
    rtol / atol:
        Tolerances applied to numeric columns via :func:`numpy.isclose`.
    """

    node_attributes: tuple[str, ...] | None = None
    edge_attributes: tuple[str, ...] | None = None
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL


def _differs(a: pd.Series, b: pd.Series, rtol: float, atol: float) -> np.ndarray:
    """Element-wise inequality with NaN-equals-NaN and numeric tolerance."""
    a_null = a.isna().to_numpy()
    b_null = b.isna().to_numpy()
    both_null = a_null & b_null
    exactly_one_null = a_null ^ b_null

    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        a_num = pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        b_num = pd.to_numeric(b, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        differ = ~np.isclose(a_num, b_num, rtol=rtol, atol=atol, equal_nan=True)
    else:
        differ = a.to_numpy(dtype=object) != b.to_numpy(dtype=object)

    differ = np.asarray(differ, dtype=bool)
    differ[both_null] = False
    differ[exactly_one_null] = True
    return differ


def _compare_attributes(
    frame: pd.DataFrame,
    columns: list[str],
    rtol: float,
    atol: float,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare ``a_<col>`` against ``b_<col>`` for each column.

    Parameters
    ----------
    eligible:
        Boolean mask of rows present in *both* graphs. Only these can be
        ``CHANGED``; for one-sided rows the missing side is absent rather than
        different, so they are excluded from both the mask and the detail dicts.

    Returns
    -------
    (changed_mask, changed_attrs)
        ``changed_attrs`` is an object array holding ``{col: {"a": old, "b": new}}``
        for changed rows and ``None`` elsewhere.
    """
    n = len(frame)
    changed = np.zeros(n, dtype=bool)
    per_column: dict[str, np.ndarray] = {}

    for col in columns:
        a_col, b_col = f"{A_PREFIX}{col}", f"{B_PREFIX}{col}"
        if a_col not in frame.columns or b_col not in frame.columns:
            continue
        d = _differs(frame[a_col], frame[b_col], rtol, atol) & eligible
        per_column[col] = d
        changed |= d

    changed &= eligible
    details = np.full(n, None, dtype=object)
    if changed.any() and per_column:
        idx_changed = np.flatnonzero(changed)
        a_values = {c: frame[f"{A_PREFIX}{c}"].to_numpy(dtype=object) for c in per_column}
        b_values = {c: frame[f"{B_PREFIX}{c}"].to_numpy(dtype=object) for c in per_column}
        for i in idx_changed:
            details[i] = {
                col: {"a": a_values[col][i], "b": b_values[col][i]}
                for col, mask in per_column.items()
                if mask[i]
            }
    return changed, details


def _status_from_masks(in_a: np.ndarray, in_b: np.ndarray, changed: np.ndarray) -> pd.Categorical:
    values = np.empty(len(in_a), dtype=object)
    both = in_a & in_b
    values[in_a & ~in_b] = Status.A_ONLY.value
    values[~in_a & in_b] = Status.B_ONLY.value
    values[both & ~changed] = Status.SHARED.value
    values[both & changed] = Status.CHANGED.value
    return pd.Categorical(values, categories=STATUS_ORDER, ordered=False)


@dataclass
class UnionDiffGraph:
    """Graphs A and B materialized as one graph with per-element ``status``.

    Attributes
    ----------
    nodes:
        Indexed by label. Columns: ``status``, ``changed_attrs``, and prefixed
        attribute columns (``a_*`` / ``b_*``).
    edges:
        Columns: ``source``, ``type``, ``target``, ``status``, ``changed_attrs``,
        and prefixed attribute columns.
    graph_a / graph_b:
        The inputs, retained so metrics needing an original view (weight
        correlation, neighborhood deltas, spectral fallbacks) need not rebuild them.
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame
    directed: bool
    graph_a: PropertyGraph
    graph_b: PropertyGraph
    name_a: str = "A"
    name_b: str = "B"
    comparison: AttributeComparison = AttributeComparison()
    #: How B's nodes were matched to A's when the join was not exact. ``None``
    #: for a plain label join. When set, ``nodes`` carries ``match_method``,
    #: ``match_confidence`` and ``label_b`` (the node's original label in B).
    alignment: AlignmentResult | None = None

    # ------------------------------------------------------------ status views

    def node_status_counts(self) -> dict[str, int]:
        """Count of nodes per status, always including all four keys."""
        counts = self.nodes[STATUS].value_counts()
        return {s: int(counts.get(s, 0)) for s in STATUS_ORDER}

    def edge_status_counts(self) -> dict[str, int]:
        """Count of edges per status, always including all four keys."""
        counts = self.edges[STATUS].value_counts()
        return {s: int(counts.get(s, 0)) for s in STATUS_ORDER}

    def nodes_with_status(self, *statuses: Status | str) -> pd.DataFrame:
        """Rows of :attr:`nodes` whose status is any of ``statuses``."""
        wanted = {str(s) for s in statuses}
        return self.nodes[self.nodes[STATUS].isin(wanted)]

    def edges_with_status(self, *statuses: Status | str) -> pd.DataFrame:
        """Rows of :attr:`edges` whose status is any of ``statuses``."""
        wanted = {str(s) for s in statuses}
        return self.edges[self.edges[STATUS].isin(wanted)]

    @property
    def shared_node_labels(self) -> pd.Index:
        """Labels present in both graphs (status ``SHARED`` or ``CHANGED``)."""
        mask = self.nodes[STATUS].isin({Status.SHARED.value, Status.CHANGED.value})
        return self.nodes.index[mask]

    @property
    def n_nodes(self) -> int:
        """Node count of the union."""
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Edge count of the union."""
        return len(self.edges)

    # -------------------------------------------------------------- subgraphs

    def induced_shared(self) -> UnionDiffGraph:
        """The union graph restricted to nodes present in both inputs.

        This is the density-normalized view: it removes the size asymmetry that
        otherwise dominates set-theoretic scores when A and B differ in scale.
        """
        shared = self.shared_node_labels
        nodes = self.nodes.loc[isin_labels(self.nodes.index, shared)]
        if self.n_edges:
            mask = isin_labels(self.edges[SOURCE], shared) & isin_labels(self.edges[TARGET], shared)
            edges = self.edges.loc[mask].reset_index(drop=True)
        else:
            edges = self.edges
        return UnionDiffGraph(
            nodes=nodes,
            edges=edges,
            directed=self.directed,
            graph_a=self.graph_a.induced_subgraph(shared),
            graph_b=self.graph_b.induced_subgraph(shared),
            name_a=self.name_a,
            name_b=self.name_b,
            comparison=self.comparison,
            alignment=self.alignment,
        )

    # ------------------------------------------------------------- interop

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(nodes, edges)`` copies for downstream user analysis."""
        return self.nodes.copy(), self.edges.copy()

    def to_property_graph(self) -> PropertyGraph:
        """The union as a :class:`PropertyGraph` (status kept as an attribute)."""
        nodes = self.nodes.copy()
        edges = self.edges.copy()
        nodes[STATUS] = nodes[STATUS].astype(str)
        edges = edges.rename(columns={STATUS: "diff_status"})
        edges["diff_status"] = edges["diff_status"].astype(str)
        nodes = nodes.rename(columns={STATUS: "diff_status"})
        for frame in (nodes, edges):
            if CHANGED_ATTRS in frame.columns:
                frame[CHANGED_ATTRS] = frame[CHANGED_ATTRS].map(
                    lambda v: None if v is None else str(v)
                )
        return PropertyGraph(nodes=nodes, edges=edges, directed=self.directed, name="union")

    def to_rustworkx(self) -> rustworkx.PyGraph | rustworkx.PyDiGraph:
        """Export the union graph to :mod:`rustworkx`."""
        return self.to_property_graph().to_rustworkx()

    def to_igraph(self) -> igraph.Graph:
        """Export the union graph to :mod:`igraph`."""
        return self.to_property_graph().to_igraph()

    def __repr__(self) -> str:  # pragma: no cover - display only
        n = self.node_status_counts()
        e = self.edge_status_counts()
        return (
            f"<UnionDiffGraph {self.name_a}|{self.name_b} "
            f"nodes={self.n_nodes} (shared={n['SHARED']} changed={n['CHANGED']} "
            f"a_only={n['A_ONLY']} b_only={n['B_ONLY']}) "
            f"edges={self.n_edges} (shared={e['SHARED']} changed={e['CHANGED']} "
            f"a_only={e['A_ONLY']} b_only={e['B_ONLY']})>"
        )


def _prefixed(frame: pd.DataFrame, prefix: str, columns: list[str]) -> pd.DataFrame:
    sub = frame[columns] if columns else frame.iloc[:, :0]
    return sub.add_prefix(prefix)


def _resolve_columns(
    requested: tuple[str, ...] | None, a_cols: list[str], b_cols: list[str]
) -> list[str]:
    if requested is None:
        return sorted(set(a_cols) & set(b_cols))
    return [c for c in requested if c in a_cols and c in b_cols]


def build_union_diff_graph(
    a: PropertyGraph,
    b: PropertyGraph,
    *,
    name_a: str | None = None,
    name_b: str | None = None,
    comparison: AttributeComparison | None = None,
    alignment: AlignmentResult | None = None,
) -> UnionDiffGraph:
    """Align ``a`` and ``b`` by label and materialize the union diff graph.

    Parameters
    ----------
    a, b:
        Graphs to compare. They must agree on directedness.
    name_a, name_b:
        Display names; default to the graphs' own ``name`` or ``"A"``/``"B"``.
    comparison:
        Controls which attributes drive ``CHANGED`` detection.
    alignment:
        A :class:`~graphdiff.core.align.AlignmentResult` from
        :func:`~graphdiff.core.align.align_graphs`. B's matched nodes are
        renamed onto A's labels before the join, and every union node records
        how it was matched (``match_method``, ``match_confidence``) and its
        original label in B (``label_b``).

    Raises
    ------
    ValueError
        If directedness differs between the two graphs.
    """
    if a.directed != b.directed:
        raise ValueError(
            "cannot compare a directed graph with an undirected one "
            f"(A directed={a.directed}, B directed={b.directed})"
        )
    cmp = comparison or AttributeComparison()
    original_b_name = b.name
    if alignment is not None:
        b = alignment.relabel_b(b)

    # ---- nodes -------------------------------------------------------------
    union_index = a.nodes.index.union(b.nodes.index)
    union_index.name = LABEL
    node_cols = _resolve_columns(cmp.node_attributes, a.node_attributes, b.node_attributes)

    a_nodes = _prefixed(a.nodes.reindex(union_index), A_PREFIX, a.node_attributes)
    b_nodes = _prefixed(b.nodes.reindex(union_index), B_PREFIX, b.node_attributes)
    nodes = pd.concat([a_nodes, b_nodes], axis=1)
    nodes.index = union_index

    in_a = isin_labels(union_index, a.nodes.index)
    in_b = isin_labels(union_index, b.nodes.index)
    node_changed, node_details = _compare_attributes(
        nodes, node_cols, cmp.rtol, cmp.atol, in_a & in_b
    )

    nodes.insert(0, CHANGED_ATTRS, node_details)
    nodes.insert(0, STATUS, _status_from_masks(in_a, in_b, node_changed))
    if alignment is not None:
        table = alignment.table.set_index("label_a")
        method = table["method"].reindex(union_index)
        method = method.where(~(in_a & in_b) | method.notna(), "exact")
        nodes.insert(2, "match_method", method.fillna("").astype(str).to_numpy())
        confidence = table["confidence"].reindex(union_index).astype("float64")
        confidence = confidence.where(~(in_a & in_b) | confidence.notna(), 1.0)
        nodes.insert(3, "match_confidence", confidence.to_numpy())
        label_b = table["label_b"].reindex(union_index)
        label_b = label_b.where(~in_b | label_b.notna(), pd.Series(union_index, index=union_index))
        nodes.insert(4, "label_b", label_b.to_numpy())

    # ---- edges -------------------------------------------------------------
    key = [SOURCE, ETYPE, TARGET]
    edge_cols = _resolve_columns(cmp.edge_attributes, a.edge_attributes, b.edge_attributes)

    a_edges = a.edges.rename(columns={c: f"{A_PREFIX}{c}" for c in a.edge_attributes})
    b_edges = b.edges.rename(columns={c: f"{B_PREFIX}{c}" for c in b.edge_attributes})
    merged = a_edges.merge(b_edges, on=key, how="outer", indicator=True, sort=False)

    ind = merged["_merge"].to_numpy()
    e_in_a = (ind == "left_only") | (ind == "both")
    e_in_b = (ind == "right_only") | (ind == "both")
    merged = merged.drop(columns=["_merge"])

    edge_changed, edge_details = _compare_attributes(
        merged, edge_cols, cmp.rtol, cmp.atol, e_in_a & e_in_b
    )

    merged.insert(3, CHANGED_ATTRS, edge_details)
    merged.insert(3, STATUS, _status_from_masks(e_in_a, e_in_b, edge_changed))
    merged = merged.sort_values(key, kind="stable").reset_index(drop=True)

    return UnionDiffGraph(
        nodes=nodes,
        edges=merged,
        directed=a.directed,
        graph_a=a,
        graph_b=b,
        name_a=name_a or a.name or "A",
        name_b=name_b or original_b_name or "B",
        comparison=cmp,
        alignment=alignment,
    )
