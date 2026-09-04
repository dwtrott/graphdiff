"""Alignment correctness, CHANGED detection, and PropertyGraph invariants."""

from __future__ import annotations

import pandas as pd
import pytest

from graphdiff import AttributeComparison, PropertyGraph, Status, build_union_diff_graph
from graphdiff.core.graph import DuplicateEdgeError

from helpers import make_graph


class TestPropertyGraph:
    def test_infers_node_set_from_edges(self) -> None:
        g = make_graph([("a", "t", "b", 1.0), ("b", "t", "c", 1.0)])
        assert g.n_nodes == 3
        assert set(g.nodes.index) == {"a", "b", "c"}

    def test_rejects_duplicate_node_labels(self) -> None:
        nodes = pd.DataFrame({"label": ["a", "a"], "x": [1, 2]})
        with pytest.raises(ValueError, match="unique"):
            PropertyGraph(nodes=nodes, edges=pd.DataFrame({"source": [], "target": []}))

    def test_rejects_duplicate_edges_by_default(self) -> None:
        edges = pd.DataFrame(
            {"source": ["a", "a"], "target": ["b", "b"], "type": ["t", "t"], "weight": [1, 2]}
        )
        with pytest.raises(DuplicateEdgeError):
            PropertyGraph.from_edges(edges)

    def test_duplicate_policy_first_keeps_one(self) -> None:
        edges = pd.DataFrame(
            {"source": ["a", "a"], "target": ["b", "b"], "type": ["t", "t"], "weight": [1, 2]}
        )
        g = PropertyGraph.from_edges(edges, on_duplicate_edge="first")
        assert g.n_edges == 1
        assert g.edges["weight"].iloc[0] == 1

    def test_undirected_canonicalizes_endpoints(self) -> None:
        g = make_graph([("z", "t", "a", 1.0)], directed=False)
        assert g.edges.iloc[0]["source"] == "a"
        assert g.edges.iloc[0]["target"] == "z"

    def test_undirected_dedupes_reversed_duplicates(self) -> None:
        edges = pd.DataFrame(
            {"source": ["a", "b"], "target": ["b", "a"], "type": ["t", "t"], "weight": [1, 1]}
        )
        with pytest.raises(DuplicateEdgeError):
            PropertyGraph.from_edges(edges, directed=False)

    def test_degrees_include_isolated_nodes(self) -> None:
        g = make_graph([("a", "t", "b", 1.0)], nodes=["a", "b", "isolated"])
        degrees = g.degrees()
        assert degrees["isolated"] == 0
        assert degrees["a"] == 1

    def test_density_directed(self) -> None:
        g = make_graph([("a", "t", "b", 1.0)], nodes=["a", "b"])
        assert g.density == pytest.approx(1 / 2)

    def test_induced_subgraph_drops_dangling_edges(self) -> None:
        g = make_graph([("a", "t", "b", 1.0), ("b", "t", "c", 1.0)])
        sub = g.induced_subgraph({"a", "b"})
        assert sub.n_nodes == 2
        assert sub.n_edges == 1

    def test_rustworkx_roundtrip_shape(self) -> None:
        g = make_graph([("a", "t", "b", 1.0), ("b", "t", "c", 1.0)])
        rx_graph = g.to_rustworkx()
        assert rx_graph.num_nodes() == 3
        assert rx_graph.num_edges() == 2

    def test_igraph_roundtrip_shape(self) -> None:
        g = make_graph([("a", "t", "b", 1.0), ("b", "t", "c", 1.0)])
        ig_graph = g.to_igraph()
        assert ig_graph.vcount() == 3
        assert ig_graph.ecount() == 2
        assert set(ig_graph.vs["name"]) == {"a", "b", "c"}


class TestAlignment:
    def test_status_assignment(self, graph_a: PropertyGraph, graph_b: PropertyGraph) -> None:
        union = build_union_diff_graph(graph_a, graph_b)

        nodes = union.nodes["status"].astype(str).to_dict()
        assert nodes == {
            "a": Status.SHARED.value,
            "b": Status.SHARED.value,
            "c": Status.SHARED.value,
            "d": Status.A_ONLY.value,
            "e": Status.B_ONLY.value,
        }

        edges = {
            (r["source"], r["type"], r["target"]): str(r["status"])
            for r in union.edges.to_dict("records")
        }
        assert edges == {
            ("a", "knows", "b"): Status.SHARED.value,
            ("b", "knows", "c"): Status.CHANGED.value,
            ("c", "owns", "d"): Status.A_ONLY.value,
            ("c", "owns", "e"): Status.B_ONLY.value,
        }

    def test_changed_records_old_and_new(
        self, graph_a: PropertyGraph, graph_b: PropertyGraph
    ) -> None:
        union = build_union_diff_graph(graph_a, graph_b)
        row = union.edges.query("source == 'b' and target == 'c'").iloc[0]
        assert row["changed_attrs"] == {"weight": {"a": 2.0, "b": 5.0}}

    def test_one_sided_edges_have_no_changed_attrs(
        self, graph_a: PropertyGraph, graph_b: PropertyGraph
    ) -> None:
        union = build_union_diff_graph(graph_a, graph_b)
        one_sided = union.edges[union.edges["status"].astype(str).isin({"A_ONLY", "B_ONLY"})]
        assert one_sided["changed_attrs"].isna().all()

    def test_edge_type_participates_in_identity(self) -> None:
        a = make_graph([("x", "knows", "y", 1.0)])
        b = make_graph([("x", "owns", "y", 1.0)])
        union = build_union_diff_graph(a, b)
        statuses = set(union.edges["status"].astype(str))
        assert statuses == {"A_ONLY", "B_ONLY"}

    def test_node_attribute_change_detected(self) -> None:
        a = make_graph([("x", "t", "y", 1.0)], node_attrs={"x": {"kind": "person"}, "y": {}})
        b = make_graph([("x", "t", "y", 1.0)], node_attrs={"x": {"kind": "org"}, "y": {}})
        union = build_union_diff_graph(a, b)
        assert str(union.nodes.loc["x", "status"]) == Status.CHANGED.value
        assert union.nodes.loc["x", "changed_attrs"] == {"kind": {"a": "person", "b": "org"}}

    def test_float_tolerance_suppresses_spurious_changes(self) -> None:
        a = make_graph([("x", "t", "y", 1.0)])
        b = make_graph([("x", "t", "y", 1.0 + 1e-12)])
        union = build_union_diff_graph(a, b)
        assert str(union.edges.iloc[0]["status"]) == Status.SHARED.value

    def test_comparison_can_ignore_attributes(
        self, graph_a: PropertyGraph, graph_b: PropertyGraph
    ) -> None:
        union = build_union_diff_graph(
            graph_a, graph_b, comparison=AttributeComparison(edge_attributes=())
        )
        assert union.edge_status_counts()[Status.CHANGED.value] == 0
        assert union.edge_status_counts()[Status.SHARED.value] == 2

    def test_alignment_is_order_independent(self) -> None:
        a = make_graph([("a", "t", "b", 1.0), ("b", "t", "c", 2.0)])
        shuffled = make_graph([("b", "t", "c", 2.0), ("a", "t", "b", 1.0)])
        union = build_union_diff_graph(a, shuffled)
        assert union.edge_status_counts()[Status.SHARED.value] == 2

    def test_directedness_mismatch_rejected(self) -> None:
        a = make_graph([("a", "t", "b", 1.0)], directed=True)
        b = make_graph([("a", "t", "b", 1.0)], directed=False)
        with pytest.raises(ValueError, match="directed"):
            build_union_diff_graph(a, b)

    def test_induced_shared_keeps_only_shared_nodes(
        self, graph_a: PropertyGraph, graph_b: PropertyGraph
    ) -> None:
        shared = build_union_diff_graph(graph_a, graph_b).induced_shared()
        assert set(shared.nodes.index) == {"a", "b", "c"}
        assert shared.n_edges == 2
        assert not shared.nodes["status"].astype(str).isin({"A_ONLY", "B_ONLY"}).any()
