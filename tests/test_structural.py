"""Label-free structural similarity."""

from __future__ import annotations

import pandas as pd
import pytest

import graphdiff as gd
from graphdiff import PropertyGraph
from graphdiff.data import example_pair, large_example_pair
from graphdiff.metrics.structural import (
    degree_distribution_similarity,
    netsimile_similarity,
    spectral_similarity,
    structural_similarity,
    wl_similarity,
)

from helpers import make_graph, random_graph


def _relabel(graph: PropertyGraph, prefix: str = "x") -> PropertyGraph:
    mapping = {lab: f"{prefix}{i}" for i, lab in enumerate(graph.nodes.index)}
    nodes = graph.nodes.copy()
    nodes.index = pd.Index([mapping[x] for x in nodes.index], name="label")
    edges = graph.edges.copy()
    edges["source"] = edges["source"].map(mapping)
    edges["target"] = edges["target"].map(mapping)
    return PropertyGraph(nodes=nodes, edges=edges, directed=graph.directed, name="relabelled")


def _star(n: int) -> PropertyGraph:
    return make_graph([("hub", "t", f"l{i}", 1.0) for i in range(n)])


def _path(n: int) -> PropertyGraph:
    return make_graph([(f"p{i}", "t", f"p{i + 1}", 1.0) for i in range(n)])


class TestInvariants:
    @pytest.mark.parametrize(
        "fn",
        [
            degree_distribution_similarity,
            spectral_similarity,
            netsimile_similarity,
            wl_similarity,
        ],
    )
    def test_identity_and_relabelling(self, fn) -> None:  # type: ignore[no-untyped-def]
        a, _ = example_pair()
        assert fn(a, a)[0] == pytest.approx(1.0)
        # Labels carry no information for these metrics.
        assert fn(a, _relabel(a))[0] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "fn",
        [
            degree_distribution_similarity,
            spectral_similarity,
            netsimile_similarity,
            wl_similarity,
        ],
    )
    def test_symmetric_and_bounded(self, fn) -> None:  # type: ignore[no-untyped-def]
        a, b = example_pair()
        ab, ba = fn(a, b)[0], fn(b, a)[0]
        assert ab == pytest.approx(ba, abs=1e-9)
        assert 0.0 <= ab <= 1.0

    def test_different_shapes_score_low(self) -> None:
        star, path = _star(60), _path(60)
        assert degree_distribution_similarity(star, path)[0] < 0.5
        assert wl_similarity(star, path)[0] < 0.1
        assert netsimile_similarity(star, path)[0] < 0.6
        s = spectral_similarity(star, path)[0]
        assert s is not None and s < 0.95

    def test_drift_beats_random(self) -> None:
        a, b = large_example_pair(n_communities=6, community_size=60)
        r = random_graph(a.n_nodes, a.n_edges, seed=3)
        for fn in (
            degree_distribution_similarity,
            spectral_similarity,
            netsimile_similarity,
            wl_similarity,
        ):
            assert fn(a, b)[0] > fn(a, r)[0], fn.__name__


class TestEdgeCases:
    def test_tiny_and_empty(self) -> None:
        empty = PropertyGraph(
            nodes=pd.DataFrame(index=pd.Index([], name="label")),
            edges=pd.DataFrame({"source": [], "target": []}),
        )
        one = make_graph([("a", "t", "b", 1.0)])
        res = structural_similarity(empty, one)
        assert res.spectral_similarity is None
        assert 0.0 <= res.degree_js_similarity <= 1.0
        assert 0.0 <= (res.wl_similarity or 0.0) <= 1.0
        same = structural_similarity(empty, empty)
        assert same.degree_js_similarity == 1.0 and same.wl_similarity == 1.0

    def test_size_caps(self) -> None:
        a, b = example_pair()
        res = structural_similarity(a, b, max_nodes_spectral=10, max_nodes_wl=10)
        assert res.spectral_similarity is None and res.wl_similarity is None
        assert "skipped" in res.details["spectral"]

    def test_spectral_dense_and_sparse_paths_agree(self) -> None:
        a, b = large_example_pair(n_communities=8, community_size=60)  # > 400 nodes → eigsh
        small_a, small_b = example_pair()  # ≤ 400 nodes → dense
        for x, y in ((a, b), (small_a, small_b)):
            s, details = spectral_similarity(x, y, k=10)
            assert s is not None and 0 <= s <= 1
            assert len(details["top_a"]) == 5
            assert details["top_a"][0] == pytest.approx(1.0, abs=1e-3)

    def test_directed_degree_uses_in_and_out(self) -> None:
        a, b = example_pair()
        _, details = degree_distribution_similarity(a, b)
        assert set(details) == {"js_total", "js_out", "js_in"}


class TestReportIntegration:
    def test_scores_present_and_dual(self) -> None:
        a, b = example_pair()
        report = gd.compare(a, b)
        scores = report.scalar_scores()
        for name in (
            "degree_js_similarity",
            "spectral_similarity",
            "netsimile_similarity",
            "wl_similarity",
        ):
            assert name in gd.SCALAR_METRICS
            assert scores[name]["raw"] is not None and scores[name]["shared"] is not None
        assert report.structural is not None
        assert report.to_dict()["metrics"]["structural"]["raw"]["wl_similarity"] is not None

    def test_can_be_disabled(self) -> None:
        a, b = example_pair()
        report = gd.compare(a, b, structural=False)
        assert report.structural is None
        assert report.score("wl_similarity") is None
        assert report.to_dict()["metrics"]["structural"] is None
        assert "wl_similarity" in report.to_markdown()  # row still listed, as —

    def test_matrix_accepts_structural_metric(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from graphdiff import write_graph
        from graphdiff.batch import all_pairs

        a, b = example_pair()
        pa, pb = write_graph(a, tmp_path / "a.graphml"), write_graph(b, tmp_path / "b.graphml")
        frame = all_pairs([pa, pb], metrics=["spectral_similarity"], workers=1)
        assert frame.iloc[0]["raw_score"] == pytest.approx(
            gd.compare(a, b).score("spectral_similarity")
        )
