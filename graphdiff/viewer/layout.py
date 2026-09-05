"""Force-directed layout, computed in Python so the browser never has to.

The viewer ships coordinates, not a physics engine: layout runs once here and
the resulting HTML is a static, self-contained file with no layout library to
bundle. This is what keeps the air-gapped constraint cheap to satisfy.

The implementation is Fruchterman-Reingold with grid-approximated repulsion:
nodes are binned into a coarse grid each iteration and repelled from cell
centroids rather than from every other node, which turns the O(n^2) inner loop
into O(n * cells) and keeps a few tens of thousands of nodes practical.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LayoutParams", "force_directed_layout"]

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutParams:
    """Tuning for :func:`force_directed_layout`.

    Parameters
    ----------
    iterations:
        Number of simulation steps. 500 is comfortable up to ~10k nodes.
    grid:
        Repulsion grid resolution per axis. Higher is more accurate and slower;
        the cost is ``O(n * grid^2)`` per iteration in the worst case.
    seed:
        Makes the layout reproducible — the same graph always draws the same way,
        which matters when comparing two renderings by eye.
    gravity:
        Pull toward the origin, keeping disconnected components from drifting off.
    """

    iterations: int = 500
    grid: int = 24
    seed: int = 0
    gravity: float = 0.015
    initial_temperature: float = 0.10


def _grid_repulsion(pos: np.ndarray, k: float, grid: int, span: float) -> np.ndarray:
    """Repulsive displacement, approximating distant nodes by cell centroids.

    Works in any number of dimensions: cells are the product of per-axis bins,
    so a 3D layout with ``grid=12`` uses 1728 cells where 2D with ``grid=24``
    uses 576 — comparable cost, and the approximation quality is the same.
    """
    dims = pos.shape[1]
    lo = pos.min(axis=0)
    extent = np.maximum(pos.max(axis=0) - lo, 1e-9)
    cell = np.clip(((pos - lo) / extent * (grid - 1)).astype(np.int64), 0, grid - 1)
    flat = np.zeros(pos.shape[0], dtype=np.int64)
    for axis in range(dims):
        flat = flat * grid + cell[:, axis]
    n_cells = grid**dims

    counts = np.bincount(flat, minlength=n_cells).astype(np.float64)
    sums = np.stack(
        [np.bincount(flat, weights=pos[:, axis], minlength=n_cells) for axis in range(dims)],
        axis=1,
    )

    occupied = counts > 0
    if not occupied.any():  # pragma: no cover - defensive
        return np.zeros_like(pos)
    weights = counts[occupied]
    centroids = sums[occupied] / weights[:, None]

    disp = np.zeros_like(pos)
    for i in range(centroids.shape[0]):
        delta = pos - centroids[i]
        dist_sq = np.einsum("ij,ij->i", delta, delta) + 1e-9
        disp += delta * (weights[i] * k * k / dist_sq)[:, None]

    # Remove each node's self-repulsion, which the centroid sum wrongly included.
    own = centroids[np.searchsorted(np.flatnonzero(occupied), flat)]
    delta_self = pos - own
    dist_self = np.einsum("ij,ij->i", delta_self, delta_self) + 1e-9
    disp -= delta_self * (k * k / dist_self)[:, None]

    np.clip(disp, -span, span, out=disp)
    return disp


def force_directed_layout(
    n_nodes: int,
    edges: np.ndarray,
    *,
    params: LayoutParams | None = None,
    dims: int = 2,
) -> np.ndarray:
    """Lay out a graph in 2D or 3D and return coordinates normalized to ``[0, 1]``.

    Parameters
    ----------
    n_nodes:
        Number of nodes; coordinates are returned in this index order.
    edges:
        Integer array of shape ``(m, 2)`` holding node index pairs. Direction is
        irrelevant to the layout.
    params:
        Tuning; see :class:`LayoutParams`.
    dims:
        ``2`` for the flat views, ``3`` for the rotating view. In 3D the
        repulsion grid is coarsened to ``grid // 2`` per axis so the cell count
        stays in the same range.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_nodes, dims)``, each coordinate in ``[0, 1]``.

    Notes
    -----
    Deterministic for a given ``seed``: the same graph lays out identically
    every run, so two renderings can be compared visually without the picture
    reshuffling underneath you.
    """
    cfg = params or LayoutParams()
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3, got {dims}")
    if n_nodes == 0:
        return np.zeros((0, dims), dtype=np.float64)
    if n_nodes == 1:
        return np.full((1, dims), 0.5, dtype=np.float64)

    rng = np.random.default_rng(cfg.seed)
    pos = rng.uniform(-1.0, 1.0, size=(n_nodes, dims))
    grid = cfg.grid if dims == 2 else max(6, cfg.grid // 2)

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges):
        valid = (
            (edges[:, 0] >= 0)
            & (edges[:, 0] < n_nodes)
            & (edges[:, 1] >= 0)
            & (edges[:, 1] < n_nodes)
            & (edges[:, 0] != edges[:, 1])
        )
        edges = edges[valid]

    k = (1.0 / n_nodes) ** (1.0 / dims)
    temperature = cfg.initial_temperature

    for _ in range(cfg.iterations):
        disp = _grid_repulsion(pos, k, grid, span=10.0)

        if len(edges):
            src, dst = edges[:, 0], edges[:, 1]
            delta = pos[src] - pos[dst]
            dist = np.sqrt(np.einsum("ij,ij->i", delta, delta)) + 1e-9
            # Fruchterman-Reingold attraction is d^2 / k along the unit vector,
            # which reduces to delta * d / k. A linear spring here leaves the
            # graph as an under-contracted hairball.
            force = delta * (dist / k)[:, None]
            np.subtract.at(disp, src, force)
            np.add.at(disp, dst, force)

        disp -= pos * cfg.gravity

        length = np.sqrt(np.einsum("ij,ij->i", disp, disp)) + 1e-9
        step = np.minimum(length, temperature) / length
        pos += disp * step[:, None]

        temperature *= 0.97  # cool

    lo = pos.min(axis=0)
    extent = np.maximum(pos.max(axis=0) - lo, 1e-9)
    return (pos - lo) / extent
