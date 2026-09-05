"""Clustering: partition correctness, change density, and the aggregate payload."""

from __future__ import annotations

import pandas as pd
import pytest

import graphdiff as gd
from graphdiff import build_union_diff_graph
from graphdiff.data import example_pair
from graphdiff.metrics.cluster import cluster_union, membership_from_attribute

from helpers import make_graph, random_graph


@pytest.fixture(scope="module")
def union():  # type: ignore[no-untyped-def]
    a, b = example_pair()
    return build_union_diff_graph(a, b)


class TestPartition:
    def test_every_node_lands_in_exactly_one_cluster(self, union) -> None:  # type: ignore[no-untyped-def]
        cmap = cluster_union(union)
        assert set(cmap.membership.index) == set(union.nodes.index)
        assert sum(c.size for c in cmap.clusters) == union.n_nodes

    def test_two_components_separate(self) -> None:
        a = make_graph(
            [
                ("a1", "t", "a2", 1.0),
                ("a2", "t", "a3", 1.0),
                ("a3", "t", "a1", 1.0),
                ("b1", "t", "b2", 1.0),
                ("b2", "t", "b3", 1.0),
                ("b3", "t", "b1", 1.0),
            ]
        )
        cmap = cluster_union(build_union_diff_graph(a, a))
        groups = cmap.membership.to_dict()
        assert groups["a1"] == groups["a2"] == groups["a3"]
        assert groups["b1"] == groups["b2"] == groups["b3"]
        assert groups["a1"] != groups["b1"]

    def test_supplied_membership_wins(self, union) -> None:  # type: ignore[no-untyped-def]
        forced = pd.Series("everything", index=union.nodes.index)
        cmap = cluster_union(union, membership=forced)
        assert cmap.method == "supplied"
        assert len(cmap.clusters) == 1
        assert cmap.clusters[0].size == union.n_nodes

    def test_cluster_cap_folds_the_tail(self) -> None:
        a = random_graph(200, 260, seed=31)
        cmap = cluster_union(build_union_diff_graph(a, a), max_clusters=3)
        assert len(cmap.clusters) <= 4  # 3 kept plus "(other)"
        assert any(c.key == "(other)" for c in cmap.clusters)

    def test_attribute_clustering(self) -> None:
        a = make_graph(
            [("x", "t", "y", 1.0), ("y", "t", "z", 1.0)],
            node_attrs={"x": {"kind": "p"}, "y": {"kind": "p"}, "z": {"kind": "q"}},
        )
        cmap = cluster_union(build_union_diff_graph(a, a), attribute="kind")
        assert cmap.method == "attribute:kind"
        assert cmap.membership["x"] == cmap.membership["y"] != cmap.membership["z"]

    def test_missing_attribute_raises(self, union) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(KeyError, match="no node attribute"):
            membership_from_attribute(union, "not_a_column")

    def test_empty_graph(self) -> None:
        empty = make_graph([("solo", "t", "solo2", 1.0)])
        cmap = cluster_union(build_union_diff_graph(empty, empty))
        assert sum(c.size for c in cmap.clusters) == 2


class TestChangeDensity:
    def test_identical_graphs_have_zero_density(self, union) -> None:  # type: ignore[no-untyped-def]
        a, _ = example_pair()
        cmap = cluster_union(build_union_diff_graph(a, a))
        assert all(c.change_density == 0.0 for c in cmap.clusters)

    def test_density_is_a_rate_in_the_unit_interval(self, union) -> None:  # type: ignore[no-untyped-def]
        for c in cluster_union(union).clusters:
            assert 0.0 <= c.change_density <= 1.0

    def test_fully_changed_cluster_scores_one(self) -> None:
        a = make_graph([("x", "t", "y", 1.0)])
        b = make_graph([("p", "t", "r", 1.0)])
        cmap = cluster_union(build_union_diff_graph(a, b))
        assert all(c.change_density == pytest.approx(1.0) for c in cmap.clusters)

    def test_clusters_are_ranked_most_changed_first(self, union) -> None:  # type: ignore[no-untyped-def]
        densities = [c.change_density for c in cluster_union(union).clusters]
        assert densities == sorted(densities, reverse=True)

    def test_counts_agree_with_the_union(self, union) -> None:  # type: ignore[no-untyped-def]
        cmap = cluster_union(union)
        totals = {}
        for c in cmap.clusters:
            for status, n in c.node_counts.items():
                totals[status] = totals.get(status, 0) + n
        assert totals == union.node_status_counts()

    def test_internal_plus_crossing_edges_equal_the_union(self, union) -> None:  # type: ignore[no-untyped-def]
        cmap = cluster_union(union)
        internal = sum(sum(c.edge_counts.values()) for c in cmap.clusters)
        crossing = sum(sum(counts.values()) for _, _, counts in cmap.links)
        assert internal + crossing == union.n_edges


class TestClusterPayload:
    def test_clusters_reach_the_page(self) -> None:
        from graphdiff.viewer import render_html

        a, b = example_pair()
        html = render_html(gd.compare(a, b))
        assert '"clusters":' in html
        assert '"nodeCluster":' in html
        assert 'data-v="clusters"' in html

    def test_ramp_is_monotonic_in_lightness(self) -> None:
        """A sequential ramp encodes magnitude, so its lightness must be ordered."""
        from graphdiff.viewer.theme import CHANGE_RAMP

        def luminance(hex_: str) -> float:
            channels = [int(hex_[i : i + 2], 16) / 255 for i in (1, 3, 5)]
            lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
            return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

        for name, ramp in CHANGE_RAMP.items():
            values = [luminance(c) for c in ramp]
            ascending = all(values[i] < values[i + 1] for i in range(len(values) - 1))
            descending = all(values[i] > values[i + 1] for i in range(len(values) - 1))
            assert ascending or descending, f"{name} ramp is not monotonic: {values}"

    def test_ramp_hue_is_distinct_from_the_status_colors(self) -> None:
        from graphdiff.viewer.theme import CHANGE_RAMP, DARK, LIGHT

        for theme, key in ((LIGHT, "light"), (DARK, "dark")):
            status = {theme[s] for s in ("SHARED", "CHANGED", "A_ONLY", "B_ONLY")}
            assert not status.intersection(CHANGE_RAMP[key])
