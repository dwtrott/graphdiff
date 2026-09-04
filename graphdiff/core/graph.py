"""The :class:`PropertyGraph` container: a DataFrame-backed property graph.

Nodes are uniquely labeled; edges are identified by the triple
``(source, type, target)``.  Both tables are plain :class:`pandas.DataFrame`
objects so that everything downstream (alignment, metrics, serialization)
is a vectorized join rather than a Python-level graph traversal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .._types import ETYPE, LABEL, RESERVED_EDGE_COLUMNS, SOURCE, TARGET

if TYPE_CHECKING:  # pragma: no cover
    import igraph
    import rustworkx

__all__ = ["DuplicateEdgeError", "PropertyGraph", "isin_labels"]

DuplicatePolicy = Literal["error", "first", "last"]


class DuplicateEdgeError(ValueError):
    """Raised when an edge table contains repeated ``(source, type, target)`` triples."""


def isin_labels(values: pd.Series | pd.Index, allowed: pd.Index) -> np.ndarray:
    """Boolean mask of ``values`` present in the unique index ``allowed``.

    Deliberately not ``Series.isin``: on pandas 3 that falls back to a Python
    listcomp over an Arrow string array, which dominates every label-membership
    test in this package at 10^6 rows. ``Index.get_indexer`` uses the hash table
    instead and is roughly an order of magnitude faster.
    """
    index = values if isinstance(values, pd.Index) else pd.Index(values)
    return allowed.get_indexer(index) >= 0


def _is_string_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) and series.dtype != object


def _as_label_column(values: Any, *, index: pd.Index | None = None) -> pd.Series:
    """Coerce label input to whatever string dtype the installed pandas prefers.

    Labels stay in pandas' native string dtype rather than being forced to numpy
    object. On pandas 3 that dtype is Arrow-backed, and round-tripping a 10^6-row
    label column out to object (and back on assignment, which re-infers) costs
    more than every join in this package put together. Downstream code therefore
    keeps label columns as Series and never calls ``.to_numpy()`` on them in a
    hot path.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values, index=index)
    if index is not None and not series.index.equals(index):
        series = pd.Series(series.to_numpy(), index=index)
    if _is_string_dtype(series):
        return series
    if series.dtype == object and pd.api.types.infer_dtype(series, skipna=True) in (
        "string",
        "empty",
    ):
        return series.astype(str)
    return series.astype(str)


@dataclass
class PropertyGraph:
    """A property graph with uniquely labeled nodes and typed edges.

    Parameters
    ----------
    nodes:
        Node table indexed by label. Any remaining columns are node attributes.
    edges:
        Edge table with at least ``source`` and ``target`` columns. A ``type``
        column is added (filled with ``""``) when absent. Remaining columns are
        edge attributes; ``weight`` is conventional but not required.
    directed:
        When ``False``, endpoint pairs are canonicalized so that ``source <= target``.
    name:
        Optional human-readable name used in reports.
    on_duplicate_edge:
        What to do when the same ``(source, type, target)`` appears more than once.
        ``"error"`` (default) refuses the graph, ``"first"``/``"last"`` keep one row.
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame
    directed: bool = True
    name: str | None = None
    on_duplicate_edge: DuplicatePolicy = "error"
    _normalized: bool = field(default=False, repr=False, compare=False)
    _neighbor_cache: dict[str, dict[str, set[str]]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # ------------------------------------------------------------------ setup

    def __post_init__(self) -> None:
        if self._normalized:
            # Frames produced by this class (e.g. induced_subgraph) are already
            # canonical; re-running normalization on them is pure overhead.
            return
        self.nodes = self._normalize_nodes(self.nodes)
        self.edges = self._normalize_edges(self.edges)
        self._validate()

    @staticmethod
    def _normalize_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
        out = nodes.copy()
        if LABEL in out.columns:
            out = out.set_index(LABEL)
        out.index = pd.Index(_as_label_column(pd.Series(out.index)), name=LABEL)
        if not out.index.is_unique:
            dupes = out.index[out.index.duplicated()].unique().tolist()
            raise ValueError(
                f"node labels must be unique; {len(dupes)} repeated (first few: {dupes[:5]})"
            )
        return out.sort_index()

    def _normalize_edges(self, edges: pd.DataFrame) -> pd.DataFrame:
        out = edges.copy()
        missing = {SOURCE, TARGET} - set(out.columns)
        if missing:
            raise ValueError(f"edge table is missing required column(s): {sorted(missing)}")
        if ETYPE not in out.columns:
            out[ETYPE] = ""
        out[SOURCE] = _as_label_column(out[SOURCE], index=out.index)
        out[TARGET] = _as_label_column(out[TARGET], index=out.index)
        out[ETYPE] = _as_label_column(out[ETYPE].fillna(""), index=out.index)

        if not self.directed and len(out):
            src, tgt = out[SOURCE], out[TARGET]
            swap = (src > tgt).to_numpy()
            out[SOURCE] = src.where(~swap, tgt)
            out[TARGET] = tgt.where(~swap, src)

        ordered = [SOURCE, ETYPE, TARGET]
        rest = [c for c in out.columns if c not in ordered]
        out = out[ordered + rest]

        dupe_mask = out.duplicated(subset=list(ordered), keep=False)
        if dupe_mask.any():
            if self.on_duplicate_edge == "error":
                n = int(dupe_mask.sum())
                sample = out.loc[dupe_mask, ordered].head(3).to_dict("records")
                raise DuplicateEdgeError(
                    f"{n} duplicate (source, type, target) edge rows; graphdiff assumes at "
                    f"most one edge per triple. Pass on_duplicate_edge='first' to drop "
                    f"extras. First few: {sample}"
                )
            keep = "first" if self.on_duplicate_edge == "first" else "last"
            out = out.drop_duplicates(subset=list(ordered), keep=keep)

        return out.reset_index(drop=True)

    def _validate(self) -> None:
        bad = RESERVED_EDGE_COLUMNS.intersection(self.edges.columns) - {SOURCE, TARGET, ETYPE}
        if bad:
            raise ValueError(f"edge attribute columns use reserved names: {sorted(bad)}")

    # ------------------------------------------------------------- properties

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self.edges)

    @property
    def node_attributes(self) -> list[str]:
        """Names of node attribute columns."""
        return list(self.nodes.columns)

    @property
    def edge_attributes(self) -> list[str]:
        """Names of edge attribute columns (excluding the identity triple)."""
        return [c for c in self.edges.columns if c not in (SOURCE, ETYPE, TARGET)]

    @property
    def edge_types(self) -> list[str]:
        """Sorted distinct edge type values."""
        if not self.n_edges:
            return []
        return sorted(self.edges[ETYPE].unique().tolist())

    @property
    def density(self) -> float:
        """Edge density relative to a simple graph on the same node set."""
        n = self.n_nodes
        if n < 2:
            return 0.0
        possible = n * (n - 1) if self.directed else n * (n - 1) / 2
        return float(self.n_edges / possible)

    def degrees(self) -> pd.Series:
        """Total degree per node label, including isolated nodes (as 0)."""
        if not self.n_edges:
            return pd.Series(0, index=self.nodes.index, dtype="int64", name="degree")
        endpoints = pd.concat([self.edges[SOURCE], self.edges[TARGET]], ignore_index=True)
        counts = endpoints.value_counts()
        out = counts.reindex(self.nodes.index, fill_value=0).astype("int64")
        out.name = "degree"
        out.index.name = LABEL
        return out

    def degree_stats(self) -> dict[str, float]:
        """Summary statistics of the degree distribution."""
        deg = self.degrees().to_numpy()
        if deg.size == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
        return {
            "min": float(deg.min()),
            "max": float(deg.max()),
            "mean": float(deg.mean()),
            "median": float(np.median(deg)),
            "std": float(deg.std()),
        }

    # ---------------------------------------------------------- neighborhoods

    def neighborhoods(
        self, direction: Literal["both", "out", "in"] = "both"
    ) -> dict[str, set[str]]:
        """Map each node label to the set of labels it is adjacent to.

        Parameters
        ----------
        direction:
            ``"out"`` follows ``source -> target``, ``"in"`` follows
            ``target -> source``, ``"both"`` unions the two. For undirected
            graphs all three are equivalent.
        """
        if direction in self._neighbor_cache:
            return self._neighbor_cache[direction]

        adj: dict[str, set[str]] = {label: set() for label in self.nodes.index}
        if self.n_edges:
            src = self.edges[SOURCE].to_numpy(dtype=object)
            tgt = self.edges[TARGET].to_numpy(dtype=object)
            forward = direction in ("both", "out") or not self.directed
            backward = direction in ("both", "in") or not self.directed
            if forward:
                for s, t in zip(src, tgt, strict=True):
                    adj.setdefault(s, set()).add(t)
            if backward:
                for s, t in zip(src, tgt, strict=True):
                    adj.setdefault(t, set()).add(s)
        self._neighbor_cache[direction] = adj
        return adj

    # -------------------------------------------------------------- subgraphs

    def induced_subgraph(self, labels: pd.Index | set[str] | list[str]) -> PropertyGraph:
        """Return the subgraph induced on ``labels`` (edges with both endpoints kept)."""
        keep = pd.Index(_as_label_column(pd.Series(list(labels))))
        nodes = self.nodes.reindex(self.nodes.index.intersection(keep))
        if self.n_edges:
            kept = nodes.index
            mask = isin_labels(self.edges[SOURCE], kept) & isin_labels(self.edges[TARGET], kept)
            edges = self.edges.loc[mask].reset_index(drop=True)
        else:
            edges = self.edges
        return PropertyGraph(
            nodes=nodes,
            edges=edges,
            directed=self.directed,
            name=self.name,
            on_duplicate_edge=self.on_duplicate_edge,
            _normalized=True,
        )

    # -------------------------------------------------------------- interop

    def to_rustworkx(self) -> rustworkx.PyGraph | rustworkx.PyDiGraph:
        """Export to a :mod:`rustworkx` graph carrying label/attribute payloads."""
        import rustworkx as rx

        graph: Any = rx.PyDiGraph() if self.directed else rx.PyGraph()
        attrs = self.nodes.to_dict("index")
        index_of: dict[str, int] = {}
        for label in self.nodes.index:
            payload = {LABEL: label, **attrs.get(label, {})}
            index_of[label] = graph.add_node(payload)
        for row in self.edges.to_dict("records"):
            s, t = row[SOURCE], row[TARGET]
            if s not in index_of or t not in index_of:
                continue
            graph.add_edge(index_of[s], index_of[t], row)
        return graph

    def to_igraph(self) -> igraph.Graph:
        """Export to an :mod:`igraph` graph with ``name`` set to the node label."""
        import igraph as ig

        labels = list(self.nodes.index)
        index_of = {label: i for i, label in enumerate(labels)}
        edge_pairs = [
            (index_of[s], index_of[t])
            for s, t in zip(self.edges[SOURCE], self.edges[TARGET], strict=True)
            if s in index_of and t in index_of
        ]
        graph = ig.Graph(n=len(labels), edges=edge_pairs, directed=self.directed)
        graph.vs["name"] = labels
        for col in self.node_attributes:
            graph.vs[col] = self.nodes[col].tolist()
        for col in self.edge_attributes:
            graph.es[col] = self.edges[col].tolist()
        if self.n_edges:
            graph.es[ETYPE] = self.edges[ETYPE].tolist()
        return graph

    # -------------------------------------------------------------- factories

    @classmethod
    def from_edges(
        cls,
        edges: pd.DataFrame,
        *,
        nodes: pd.DataFrame | None = None,
        directed: bool = True,
        name: str | None = None,
        on_duplicate_edge: DuplicatePolicy = "error",
    ) -> PropertyGraph:
        """Build a graph from an edge table, inferring the node set when needed."""
        if nodes is None:
            labels = pd.unique(
                np.concatenate([edges[SOURCE].to_numpy(), edges[TARGET].to_numpy()])
                if len(edges)
                else np.array([], dtype=object)
            )
            nodes = pd.DataFrame(index=pd.Index(labels, name=LABEL))
        return cls(
            nodes=nodes,
            edges=edges,
            directed=directed,
            name=name,
            on_duplicate_edge=on_duplicate_edge,
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        kind = "directed" if self.directed else "undirected"
        label = f" {self.name!r}" if self.name else ""
        return (
            f"<PropertyGraph{label} {kind} nodes={self.n_nodes} edges={self.n_edges} "
            f"types={len(self.edge_types)}>"
        )
