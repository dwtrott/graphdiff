"""Performance smoke test at the documented scale target.

Run with ``pytest -m slow``. This is a regression guard, not a benchmark: the
budget is deliberately loose so it does not flap on a loaded or slower machine,
but tight enough to catch an accidental O(n^2) or a per-row Python loop
sneaking into the hot path.

Reference: ~11 s for this fixture on the development machine (2 x 500k edges,
200k shared labels, 20% of shared edges reweighted).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import graphdiff as gd
from graphdiff import PropertyGraph

from helpers import random_graph

N_NODES = 200_000
N_EDGES = 500_000

#: Wall-clock budget for a single ``compare`` at the scale above.
COMPARE_BUDGET_SECONDS = 90.0


def _overlapping_pair(seed: int = 7) -> tuple[PropertyGraph, PropertyGraph]:
    """Two ~500k-edge graphs over a shared label space, B a perturbation of A."""
    a = random_graph(N_NODES, N_EDGES, seed=seed, name="A")
    rng = np.random.default_rng(seed + 4)

    edges = a.edges
    kept = edges.loc[rng.random(len(edges)) > 0.10].copy()  # drop 10%
    bump = rng.random(len(kept)) < 0.20  # reweight 20%
    kept.loc[bump, "weight"] = kept.loc[bump, "weight"] + 0.5
    added = random_graph(N_NODES, N_EDGES // 10, seed=seed + 92).edges
    combined = pd.concat([kept, added], ignore_index=True).drop_duplicates(
        subset=["source", "type", "target"], ignore_index=True
    )
    b = PropertyGraph.from_edges(combined, nodes=a.nodes, name="B")
    return a, b


@pytest.mark.slow
def test_compare_at_500k_edges_within_budget() -> None:
    a, b = _overlapping_pair()
    assert a.n_edges > 450_000
    assert b.n_edges > 450_000

    start = time.perf_counter()
    report = gd.compare(a, b)
    elapsed = time.perf_counter() - start

    # Sanity: the perturbation is visible in the scores, so we timed real work.
    assert 0.5 < report.score("jaccard_typed_edges") < 1.0
    assert report.edge_status_counts["CHANGED"] > 50_000
    assert report.edge_status_counts["A_ONLY"] > 10_000
    assert report.edge_status_counts["B_ONLY"] > 10_000
    assert len(report.neighborhood.table) == N_NODES

    assert elapsed < COMPARE_BUDGET_SECONDS, (
        f"compare() took {elapsed:.1f}s for {a.n_edges:,} vs {b.n_edges:,} edges, "
        f"budget is {COMPARE_BUDGET_SECONDS:.0f}s"
    )


@pytest.mark.slow
def test_union_construction_scales_linearly() -> None:
    """Doubling the edge count must not more than triple build time."""

    def build_time(n_edges: int) -> float:
        a = random_graph(N_NODES, n_edges, seed=3)
        b = random_graph(N_NODES, n_edges, seed=4)
        start = time.perf_counter()
        gd.build_union_diff_graph(a, b)
        return time.perf_counter() - start

    small = build_time(125_000)
    large = build_time(250_000)
    # Generous: an accidental quadratic blows way past 3x, jitter does not.
    assert large < max(small * 3.0, 5.0), f"{small:.2f}s -> {large:.2f}s is superlinear"
