"""Aggregating the union graph into clusters, so 10^6 nodes become ~20 blobs.

No node-link drawing survives a million nodes, in two dimensions or three. The
only way to answer "where did it change" at that scale is to stop drawing nodes:
partition the graph, and draw one mark per partition sized by membership and
shaded by how much of it changed. The differences then read as a handful of dark
blobs rather than a needle-in-haystack search, and drilling into one is a
bounded problem the overview can already handle.

Community detection is delegated to :mod:`igraph` (already a core dependency,
and C-implemented, so Louvain on 10^6 edges is seconds rather than minutes).
A caller who already knows the right partition — an entity type, an owning
organization, a source system — can supply it instead and skip the detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._types import SOURCE, STATUS, STATUS_ORDER, TARGET, Status
from ..core.union import UnionDiffGraph

__all__ = ["Cluster", "ClusterMap", "cluster_union", "membership_from_attribute"]

_DIFFERING = (Status.A_ONLY.value, Status.B_ONLY.value, Status.CHANGED.value)


@dataclass
class Cluster:
    """One partition of the union graph, summarized.

    Attributes
    ----------
    key:
        Stable identifier (the community index, or the attribute value).
    name:
        Human-readable label — the highest-degree member, which in practice is
        the hub the cluster forms around.
    size:
        Node count.
    node_counts / edge_counts:
        Membership per status; ``edge_counts`` covers edges internal to the
        cluster only.
    change_density:
        Share of the cluster's own elements (nodes plus internal edges) that
        differ between the two graphs, in ``[0, 1]``. This is what the view
        shades by, and it is a rate rather than a count so a small cluster that
        changed completely is as visible as a large one that changed a little.
    """

    key: str
    name: str
    size: int
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    change_density: float = 0.0

    @property
    def n_changed_nodes(self) -> int:
        """Nodes in this cluster that are not ``SHARED``."""
        return sum(self.node_counts.get(s, 0) for s in _DIFFERING)

    @property
    def n_changed_edges(self) -> int:
        """Internal edges that are not ``SHARED``."""
        return sum(self.edge_counts.get(s, 0) for s in _DIFFERING)


@dataclass
class ClusterMap:
    """The full partition: clusters, the links between them, and the membership."""

    clusters: list[Cluster]
    #: ``(i, j, counts)`` per ordered cluster pair with at least one edge between them.
    links: list[tuple[int, int, dict[str, int]]]
    #: Node label -> cluster index, for drill-down.
    membership: pd.Series
    method: str

    def index_of(self, key: str) -> int:
        """Position of a cluster in :attr:`clusters` by its key."""
        for i, c in enumerate(self.clusters):
            if c.key == key:
                return i
        raise KeyError(key)


def membership_from_attribute(union: UnionDiffGraph, attribute: str) -> pd.Series:
    """Partition by a node attribute instead of by detected community.

    Looks for ``a_<attribute>`` then ``b_<attribute>`` in the union node table,
    so a node present in only one graph still lands in a cluster. Missing values
    become ``"(none)"`` rather than being dropped.
    """
    nodes = union.nodes
    for col in (f"a_{attribute}", f"b_{attribute}", attribute):
        if col in nodes.columns:
            primary = nodes[col]
            other = f"b_{attribute}" if col == f"a_{attribute}" else None
            if other and other in nodes.columns:
                primary = primary.fillna(nodes[other])
            return primary.fillna("(none)").astype(str)
    raise KeyError(
        f"no node attribute {attribute!r} in the union "
        f"(looked for a_{attribute}, b_{attribute}, {attribute})"
    )


def _detect_communities(union: UnionDiffGraph, resolution: float) -> pd.Series:
    """Louvain communities over the undirected union structure."""
    import igraph as ig

    labels = union.nodes.index
    position = pd.Series(np.arange(len(labels)), index=labels)
    if not len(union.edges):
        return pd.Series(np.arange(len(labels)), index=labels).astype(str)

    src = position.reindex(union.edges[SOURCE]).to_numpy()
    dst = position.reindex(union.edges[TARGET]).to_numpy()
    ok = ~(pd.isna(src) | pd.isna(dst))
    pairs = list(zip(src[ok].astype(int), dst[ok].astype(int), strict=True))

    graph = ig.Graph(n=len(labels), edges=pairs, directed=False)
    graph.simplify(multiple=True, loops=True)
    communities = graph.community_multilevel(resolution=resolution)
    return pd.Series([str(m) for m in communities.membership], index=labels)


def cluster_union(
    union: UnionDiffGraph,
    *,
    membership: pd.Series | None = None,
    attribute: str | None = None,
    resolution: float = 1.0,
    max_clusters: int = 60,
) -> ClusterMap:
    """Partition the union graph and summarize how much each part changed.

    Parameters
    ----------
    union:
        The union diff graph.
    membership:
        A precomputed ``label -> cluster`` Series. Overrides everything else.
    attribute:
        Partition by this node attribute instead of detecting communities.
    resolution:
        Louvain resolution; higher splits into more, smaller communities.
    max_clusters:
        Keep the largest ``max_clusters`` partitions and fold the remainder into
        a single ``"(other)"`` cluster, so the view stays readable.

    Returns
    -------
    ClusterMap
        Clusters ordered by change density (most-changed first), the links
        between them, and the membership needed for drill-down.
    """
    if membership is not None:
        assign = membership.reindex(union.nodes.index).fillna("(none)").astype(str)
        method = "supplied"
    elif attribute is not None:
        assign = membership_from_attribute(union, attribute)
        method = f"attribute:{attribute}"
    else:
        assign = _detect_communities(union, resolution)
        method = "louvain"

    # Fold the long tail so the picture stays a handful of marks.
    sizes = assign.value_counts()
    if len(sizes) > max_clusters:
        keep = set(sizes.head(max_clusters).index)
        assign = assign.where(assign.isin(keep), "(other)")

    nodes = union.nodes.assign(_cluster=assign)
    node_status = nodes[STATUS].astype(str)

    degree = _degree(union)
    clusters: list[Cluster] = []
    for key, frame in nodes.groupby("_cluster", sort=False):
        statuses = node_status.reindex(frame.index)
        counts = {s: int((statuses == s).sum()) for s in STATUS_ORDER}
        hub = degree.reindex(frame.index).idxmax() if len(frame) else key
        clusters.append(
            Cluster(
                key=str(key),
                name=str(hub) if pd.notna(hub) else str(key),
                size=len(frame),
                node_counts=counts,
            )
        )
    order = {c.key: i for i, c in enumerate(clusters)}

    # ---- edges, split into internal and between-cluster --------------------
    links: dict[tuple[int, int], dict[str, int]] = {}
    if len(union.edges):
        edge_cluster_a = assign.reindex(union.edges[SOURCE]).to_numpy()
        edge_cluster_b = assign.reindex(union.edges[TARGET]).to_numpy()
        frame = pd.DataFrame(
            {
                "a": edge_cluster_a,
                "b": edge_cluster_b,
                "status": union.edges[STATUS].astype(str).to_numpy(),
            }
        ).dropna()

        internal = frame[frame["a"] == frame["b"]]
        for key, group in internal.groupby("a", sort=False):
            i = order.get(str(key))
            if i is None:  # pragma: no cover - defensive
                continue
            vc = group["status"].value_counts()
            clusters[i].edge_counts = {s: int(vc.get(s, 0)) for s in STATUS_ORDER}

        crossing = frame[frame["a"] != frame["b"]]
        for (ka, kb), group in crossing.groupby(["a", "b"], sort=False):
            i, j = order.get(str(ka)), order.get(str(kb))
            if i is None or j is None:  # pragma: no cover - defensive
                continue
            pair = (min(i, j), max(i, j))
            bucket = links.setdefault(pair, dict.fromkeys(STATUS_ORDER, 0))
            for s, n in group["status"].value_counts().items():
                bucket[str(s)] += int(n)

    for c in clusters:
        if not c.edge_counts:
            c.edge_counts = dict.fromkeys(STATUS_ORDER, 0)
        total = c.size + sum(c.edge_counts.values())
        changed = c.n_changed_nodes + c.n_changed_edges
        c.change_density = float(changed / total) if total else 0.0

    ranked = sorted(clusters, key=lambda c: (-c.change_density, -c.size, c.key))
    remap = {order[c.key]: i for i, c in enumerate(ranked)}
    relinked = [
        (min(remap[i], remap[j]), max(remap[i], remap[j]), counts)
        for (i, j), counts in links.items()
    ]
    return ClusterMap(
        clusters=ranked,
        links=relinked,
        membership=assign,
        method=method,
    )


def _degree(union: UnionDiffGraph) -> pd.Series:
    if not len(union.edges):
        return pd.Series(0, index=union.nodes.index, dtype="int64")
    endpoints = pd.concat([union.edges[SOURCE], union.edges[TARGET]], ignore_index=True)
    return endpoints.value_counts().reindex(union.nodes.index, fill_value=0).astype("int64")
