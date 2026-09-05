"""Label-free structural similarity: how alike are the *shapes* of two graphs?

Everything else in graphdiff aligns nodes by label. These metrics deliberately
do not. They answer a different question — "are these two graphs the same
kind of object?" — and they are the fallback when labels cannot be trusted at
all: a synthetic graph versus a real one, an anonymized export, two systems
that name things differently and defeat even fuzzy alignment. They also make a
useful sanity check next to the aligned scores: a pair that is 95% similar by
edit distance but far apart spectrally has had its structure reorganized
under stable names.

Four signatures, each a similarity in ``[0, 1]``:

* **Degree distribution** — Jensen-Shannon divergence between the two degree
  distributions (in/out/total for directed graphs, averaged), on log-spaced
  bins so a heavy tail is compared as a shape rather than as a handful of
  exact counts.
* **Spectral** — the ``k`` largest eigenvalues of the normalized adjacency
  ``D^-1/2 A D^-1/2`` (equivalently the smallest of the normalized
  Laplacian), compared by Euclidean distance. Captures community structure and
  connectivity at every scale; blind to size.
* **NetSimile** (Berlingerio et al., 2012) — seven local features per node
  (degree, clustering, mean neighbour degree, ego-net edges, ego-net boundary
  edges, two-hop reach, mean neighbour clustering), each summarized by five
  moments into a 35-number signature, compared by Canberra distance.
* **Weisfeiler-Lehman** subtree kernel — iterated neighbourhood relabelling
  from degree; the cosine of the label histograms at each depth, averaged
  over depths. The most sensitive of the four to local wiring patterns.

All four are size-independent in the sense that two graphs of very different
node counts can still score 1.0 if they have the same shape; that is the point.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import kurtosis, skew

from .._types import SOURCE, TARGET
from ..core.graph import PropertyGraph

__all__ = [
    "StructuralResult",
    "degree_distribution_similarity",
    "netsimile_similarity",
    "spectral_similarity",
    "structural_similarity",
    "wl_similarity",
]

_NETSIMILE_FEATURES = (
    "degree",
    "clustering",
    "neighbor_degree",
    "ego_edges",
    "ego_boundary",
    "two_hop",
    "neighbor_clustering",
)


@dataclass
class StructuralResult:
    """The four label-free similarities plus the numbers behind them."""

    degree_js_similarity: float | None
    spectral_similarity: float | None
    netsimile_similarity: float | None
    wl_similarity: float | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        from .base import to_jsonable

        return {
            "degree_js_similarity": to_jsonable(self.degree_js_similarity),
            "spectral_similarity": to_jsonable(self.spectral_similarity),
            "netsimile_similarity": to_jsonable(self.netsimile_similarity),
            "wl_similarity": to_jsonable(self.wl_similarity),
            "details": to_jsonable(self.details),
        }


# ------------------------------------------------------------------ helpers


def _adjacency(graph: PropertyGraph) -> sparse.csr_matrix:
    """Symmetric binary adjacency (self-loops dropped, multi-edges collapsed)."""
    n = graph.n_nodes
    if not n or not graph.n_edges:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    pos = pd.Series(np.arange(n), index=graph.nodes.index)
    s = pos.reindex(graph.edges[SOURCE]).to_numpy()
    t = pos.reindex(graph.edges[TARGET]).to_numpy()
    ok = ~(pd.isna(s) | pd.isna(t))
    s, t = s[ok].astype(np.int64), t[ok].astype(np.int64)
    keep = s != t
    s, t = s[keep], t[keep]
    a = sparse.coo_matrix((np.ones(2 * len(s)), (np.r_[s, t], np.r_[t, s])), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    return a


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits; ``0`` identical, ``1`` disjoint."""
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def kl(x: np.ndarray) -> float:
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / m[nz])))

    return 0.5 * kl(p) + 0.5 * kl(q)


def _log_bins(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    top = max(1, int(values_a.max(initial=0)), int(values_b.max(initial=0)))
    # 0, 1, 2, 3, 4, 6, 8, 12, 16, ... : exact at the low end, geometric above.
    edges = [0, 1, 2, 3, 4, 5]
    while edges[-1] <= top:
        edges.append(int(np.ceil(edges[-1] * 1.5)))
    return np.asarray(edges, dtype=float)


# ------------------------------------------------------------------ metrics


def degree_distribution_similarity(
    a: PropertyGraph, b: PropertyGraph
) -> tuple[float, dict[str, float]]:
    """``1 - JS(degree_a, degree_b)`` on log-spaced bins, averaged over directions."""
    parts: dict[str, float] = {}
    directions: list[tuple[str, pd.Series, pd.Series]] = [("total", a.degrees(), b.degrees())]
    if a.directed and b.directed and a.n_edges and b.n_edges:
        directions.append(
            (
                "out",
                a.edges[SOURCE].value_counts().reindex(a.nodes.index, fill_value=0),
                b.edges[SOURCE].value_counts().reindex(b.nodes.index, fill_value=0),
            )
        )
        directions.append(
            (
                "in",
                a.edges[TARGET].value_counts().reindex(a.nodes.index, fill_value=0),
                b.edges[TARGET].value_counts().reindex(b.nodes.index, fill_value=0),
            )
        )
    for name, da, db in directions:
        va, vb = da.to_numpy(dtype=float), db.to_numpy(dtype=float)
        if not len(va) or not len(vb):
            parts[name] = 0.0 if len(va) == len(vb) else 1.0  # JS: 0 = identical
            continue
        bins = _log_bins(va, vb)
        ha, _ = np.histogram(va, bins=bins)
        hb, _ = np.histogram(vb, bins=bins)
        parts[name] = _js_divergence(ha.astype(float), hb.astype(float))
    js = float(np.mean(list(parts.values())))
    return 1.0 - js, {f"js_{k}": v for k, v in parts.items()}


def spectral_similarity(
    a: PropertyGraph, b: PropertyGraph, *, k: int = 30
) -> tuple[float | None, dict[str, Any]]:
    """Distance between the top-``k`` normalized-adjacency spectra, as a similarity.

    Eigenvalues of ``D^-1/2 A D^-1/2`` lie in ``[-1, 1]``; the ``k`` largest
    are compared (padded with ``0`` when a graph has fewer), and the RMS gap is
    mapped to ``1 - rms / 2``. Returns ``None`` when either graph is too small
    to have a spectrum worth comparing.
    """

    def spectrum(graph: PropertyGraph) -> np.ndarray | None:
        adj = _adjacency(graph)
        n = adj.shape[0]
        if n < 3 or adj.nnz == 0:
            return None
        deg = np.asarray(adj.sum(axis=1)).ravel()
        inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
        norm = sparse.diags(inv) @ adj @ sparse.diags(inv)
        kk = min(k, n - 2)
        if kk < 1:
            return None
        if n <= 400:
            vals = np.linalg.eigvalsh(norm.toarray())[::-1][:kk]
        else:
            vals = eigsh(norm, k=kk, which="LA", return_eigenvectors=False, tol=1e-3)
            vals = np.sort(vals)[::-1]
        out = np.zeros(k)
        out[: len(vals)] = vals
        return out

    sa, sb = spectrum(a), spectrum(b)
    if sa is None or sb is None:
        return None, {"k": k, "note": "graph too small or empty"}
    rms = float(np.sqrt(np.mean((sa - sb) ** 2)))
    return max(0.0, 1.0 - rms / 2.0), {
        "k": k,
        "rms_gap": rms,
        "top_a": sa[:5].tolist(),
        "top_b": sb[:5].tolist(),
    }


def _netsimile_features(graph: PropertyGraph) -> np.ndarray:
    """Node x 7 feature matrix (see module docstring)."""
    adj = _adjacency(graph)
    n = adj.shape[0]
    if n == 0:
        return np.zeros((0, len(_NETSIMILE_FEATURES)))
    deg = np.asarray(adj.sum(axis=1)).ravel()
    # Triangles through each node: diag(A^3)/2, done as row-wise dot of A with A^2.
    a2 = adj @ adj
    tri = np.asarray(adj.multiply(a2).sum(axis=1)).ravel() / 2.0
    pairs = deg * (deg - 1) / 2.0
    clustering = np.where(pairs > 0, tri / np.maximum(pairs, 1), 0.0)
    nb_deg_sum = adj @ deg
    neighbor_degree = np.where(deg > 0, nb_deg_sum / np.maximum(deg, 1), 0.0)
    ego_edges = deg + tri  # spokes plus edges among neighbours
    ego_boundary = np.maximum(nb_deg_sum - deg - 2 * tri, 0.0)
    # Two-hop reach: distinct nodes at distance exactly 2.
    reach2 = a2.copy()
    reach2.data[:] = 1.0
    reach2 = reach2 - reach2.multiply(adj)  # drop direct neighbours
    reach2.setdiag(0)
    reach2.eliminate_zeros()
    two_hop = np.asarray((reach2 > 0).sum(axis=1)).ravel().astype(float)
    neighbor_clustering = np.where(deg > 0, (adj @ clustering) / np.maximum(deg, 1), 0.0)
    return np.column_stack(
        [deg, clustering, neighbor_degree, ego_edges, ego_boundary, two_hop, neighbor_clustering]
    )


def _signature(features: np.ndarray) -> np.ndarray:
    """Five moments per feature column → 35 numbers."""
    if not len(features):
        return np.zeros(5 * features.shape[1] if features.ndim == 2 else 35)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # constant columns: moments are 0
        moments = [
            features.mean(axis=0),
            np.median(features, axis=0),
            features.std(axis=0),
            np.nan_to_num(skew(features, axis=0)),
            np.nan_to_num(kurtosis(features, axis=0)),
        ]
    return np.concatenate(moments)


def netsimile_similarity(a: PropertyGraph, b: PropertyGraph) -> tuple[float, dict[str, Any]]:
    """``1 - Canberra(sig_a, sig_b) / 35`` over the NetSimile signatures."""
    sa, sb = _signature(_netsimile_features(a)), _signature(_netsimile_features(b))
    denom = np.abs(sa) + np.abs(sb)
    terms = np.where(denom > 0, np.abs(sa - sb) / np.maximum(denom, 1e-12), 0.0)
    canberra = float(terms.sum())
    per_feature = {
        name: float(terms[i :: len(_NETSIMILE_FEATURES)].sum() / 5.0)
        for i, name in enumerate(_NETSIMILE_FEATURES)
    }
    return 1.0 - canberra / len(sa), {"canberra": canberra, "per_feature_distance": per_feature}


def _wl_histograms(graph: PropertyGraph, iterations: int) -> list[dict[int, int]]:
    """Label histograms per WL iteration, labels starting from degree."""
    adj = _adjacency(graph)
    n = adj.shape[0]
    if n == 0:
        return [{} for _ in range(iterations + 1)]
    # Start from log-binned degree, not raw degree: with exact degrees the
    # depth-2 labels are near-unique fingerprints and any two non-identical
    # graphs score ~0. Bins 0,1,2,3,4,5-6,7-9,10-14,... keep the shape and lose
    # the noise.
    deg = np.asarray(adj.sum(axis=1)).ravel()
    labels = np.where(
        deg <= 4, deg, 4 + np.floor(np.log(np.maximum(deg, 1) / 4) / np.log(1.5)) + 1
    ).astype(np.int64)
    hists: list[dict[int, int]] = []
    indptr, indices = adj.indptr, adj.indices
    for _ in range(iterations + 1):
        uniq, counts = np.unique(labels, return_counts=True)
        hists.append(dict(zip(uniq.tolist(), counts.tolist(), strict=True)))
        # New label = hash of (own label, sorted multiset of neighbour labels).
        # Hash strings across both graphs deterministically (no PYTHONHASHSEED).
        new = np.empty(n, dtype=np.int64)
        for v in range(n):
            nb = labels[indices[indptr[v] : indptr[v + 1]]]
            nb.sort()
            key = f"{labels[v]}|{','.join(map(str, nb.tolist()))}"
            new[v] = _stable_hash(key)
        labels = new
    return hists


def _stable_hash(key: str) -> int:
    import zlib

    return zlib.crc32(key.encode()) & 0x7FFFFFFF


def wl_similarity(
    a: PropertyGraph, b: PropertyGraph, *, iterations: int = 2
) -> tuple[float, dict[str, Any]]:
    """Normalized Weisfeiler-Lehman subtree kernel (cosine of the label histograms)."""
    ha, hb = _wl_histograms(a, iterations), _wl_histograms(b, iterations)

    def cosine(hx: dict[int, int], hy: dict[int, int]) -> float:
        keys = list(hx.keys() | hy.keys())
        x = np.array([hx.get(k, 0) for k in keys], dtype=np.float64)
        y = np.array([hy.get(k, 0) for k in keys], dtype=np.float64)
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        if nx == 0 or ny == 0:
            return 1.0 if nx == ny else 0.0
        return float(x @ y / (nx * ny))

    per_iter = [cosine(hx, hy) for hx, hy in zip(ha, hb, strict=True)]
    if not per_iter:
        return 1.0, {"iterations": iterations}
    return float(np.mean(per_iter)), {"iterations": iterations, "per_iteration": per_iter}


def structural_similarity(
    a: PropertyGraph,
    b: PropertyGraph,
    *,
    spectral_k: int = 30,
    wl_iterations: int = 2,
    max_nodes_spectral: int = 250_000,
    max_nodes_wl: int = 300_000,
) -> StructuralResult:
    """All four label-free similarities at once.

    The spectral and WL signatures are skipped (``None``) above the given node
    counts; the other two are linear in the graph size.
    """
    degree_sim, degree_details = degree_distribution_similarity(a, b)
    big = max(a.n_nodes, b.n_nodes)
    spectral: float | None
    spectral_details: dict[str, Any]
    if big <= max_nodes_spectral:
        spectral, spectral_details = spectral_similarity(a, b, k=spectral_k)
    else:
        spectral, spectral_details = None, {"skipped": f"more than {max_nodes_spectral:,} nodes"}
    netsim, netsim_details = netsimile_similarity(a, b)
    wl: float | None
    wl_details: dict[str, Any]
    if big <= max_nodes_wl:
        wl, wl_details = wl_similarity(a, b, iterations=wl_iterations)
    else:
        wl, wl_details = None, {"skipped": f"more than {max_nodes_wl:,} nodes"}
    return StructuralResult(
        degree_js_similarity=degree_sim,
        spectral_similarity=spectral,
        netsimile_similarity=netsim,
        wl_similarity=wl,
        details={
            "degree": degree_details,
            "spectral": spectral_details,
            "netsimile": netsim_details,
            "wl": wl_details,
        },
    )
