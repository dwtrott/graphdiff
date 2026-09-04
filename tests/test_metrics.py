"""Metrics 1-4 checked against values computed by hand from the fixture pair.

Fixture recap (directed)::

    A: a-knows->b (w=1), b-knows->c (w=2), c-owns->d (w=3)
    B: a-knows->b (w=1), b-knows->c (w=5), c-owns->e (w=3)

Union: nodes a,b,c SHARED; d A_ONLY; e B_ONLY.
Edges: (a,knows,b) SHARED; (b,knows,c) CHANGED; (c,owns,d) A_ONLY; (c,owns,e) B_ONLY.
"""

from __future__ import annotations

import pytest

from graphdiff import GEDCosts, PropertyGraph, build_union_diff_graph
from graphdiff.metrics import (
    graph_edit_distance,
    neighborhood_delta,
    set_theoretic,
    weight_agreement,
)

from helpers import make_graph, random_graph


@pytest.fixture
def union(graph_a: PropertyGraph, graph_b: PropertyGraph):  # type: ignore[no-untyped-def]
    return build_union_diff_graph(graph_a, graph_b)


class TestSetTheoretic:
    def test_node_scores(self, union) -> None:  # type: ignore[no-untyped-def]
        # |A|=4, |B|=4, intersection=3, union=5
        nodes = set_theoretic(union).nodes
        assert (nodes.n_a, nodes.n_b, nodes.n_intersection, nodes.n_union) == (4, 4, 3, 5)
        assert nodes.jaccard == pytest.approx(3 / 5)
        assert nodes.overlap == pytest.approx(3 / 4)
        assert nodes.dice == pytest.approx(2 * 3 / 8)

    def test_typed_edge_scores(self, union) -> None:  # type: ignore[no-untyped-def]
        # |A|=3, |B|=3, intersection=2 (SHARED + CHANGED), union=4
        edges = set_theoretic(union).typed_edges
        assert (edges.n_a, edges.n_b, edges.n_intersection, edges.n_union) == (3, 3, 2, 4)
        assert edges.jaccard == pytest.approx(0.5)
        assert edges.overlap == pytest.approx(2 / 3)
        assert edges.dice == pytest.approx(2 / 3)

    def test_untyped_edges_collapse_parallel_types(self) -> None:
        # A has two typed edges between the same pair; B has one of them.
        a = make_graph([("x", "knows", "y", 1.0), ("x", "owns", "y", 1.0)])
        b = make_graph([("x", "knows", "y", 1.0)])
        result = set_theoretic(build_union_diff_graph(a, b))
        assert result.typed_edges.jaccard == pytest.approx(0.5)  # 1 of 2 typed edges
        assert result.edges.jaccard == pytest.approx(1.0)  # one (x,y) pair, in both

    def test_per_edge_type_breakdown(self, union) -> None:  # type: ignore[no-untyped-def]
        per_type = set_theoretic(union).per_edge_type
        assert set(per_type) == {"knows", "owns"}
        assert per_type["knows"].jaccard == pytest.approx(1.0)  # both knows edges align
        assert per_type["owns"].jaccard == pytest.approx(0.0)  # c->d vs c->e

    def test_shared_subgraph_removes_size_penalty(self, union) -> None:  # type: ignore[no-untyped-def]
        shared = set_theoretic(union.induced_shared())
        assert shared.nodes.jaccard == pytest.approx(1.0)
        assert shared.typed_edges.jaccard == pytest.approx(1.0)

    def test_identical_graphs_score_one(self, graph_a: PropertyGraph) -> None:
        result = set_theoretic(build_union_diff_graph(graph_a, graph_a))
        for scores in (result.nodes, result.edges, result.typed_edges):
            assert scores.jaccard == pytest.approx(1.0)
            assert scores.overlap == pytest.approx(1.0)
            assert scores.dice == pytest.approx(1.0)

    def test_disjoint_graphs_score_zero(self) -> None:
        a = make_graph([("a1", "t", "a2", 1.0)])
        b = make_graph([("b1", "t", "b2", 1.0)])
        result = set_theoretic(build_union_diff_graph(a, b))
        assert result.nodes.jaccard == 0.0
        assert result.typed_edges.jaccard == 0.0
        assert result.nodes.overlap == 0.0

    def test_empty_graphs_are_identical(self, empty_graph: PropertyGraph) -> None:
        result = set_theoretic(build_union_diff_graph(empty_graph, empty_graph))
        assert result.nodes.jaccard == 1.0
        assert result.typed_edges.jaccard == 1.0

    def test_empty_versus_nonempty(
        self, empty_graph: PropertyGraph, graph_a: PropertyGraph
    ) -> None:
        result = set_theoretic(build_union_diff_graph(empty_graph, graph_a))
        assert result.nodes.jaccard == 0.0
        assert result.nodes.n_a == 0
        assert result.nodes.n_b == 4


class TestGraphEditDistance:
    def test_hand_computed_cost(self, union) -> None:  # type: ignore[no-untyped-def]
        # 1 node deletion (d) + 1 node insertion (e)
        # + 1 edge deletion + 1 edge insertion + 1 edge substitution = 5
        result = graph_edit_distance(union)
        assert result.node_deletions == 1
        assert result.node_insertions == 1
        assert result.node_substitutions == 0
        assert result.edge_deletions == 1
        assert result.edge_insertions == 1
        assert result.edge_substitutions == 1
        assert result.cost == pytest.approx(5.0)

    def test_hand_computed_normalization(self, union) -> None:  # type: ignore[no-untyped-def]
        # denominator = |A nodes| + |B nodes| + |A edges| + |B edges| = 4+4+3+3 = 14
        result = graph_edit_distance(union)
        assert result.denominator == pytest.approx(14.0)
        assert result.normalized_distance == pytest.approx(5 / 14)
        assert result.similarity == pytest.approx(9 / 14)

    def test_shared_subgraph_cost(self, union) -> None:  # type: ignore[no-untyped-def]
        # Induced on {a,b,c}: 2 edges, one substituted. denominator = 3+3+2+2 = 10
        result = graph_edit_distance(union.induced_shared())
        assert result.cost == pytest.approx(1.0)
        assert result.denominator == pytest.approx(10.0)
        assert result.normalized_distance == pytest.approx(0.1)

    def test_identical_graphs_have_zero_distance(self, graph_a: PropertyGraph) -> None:
        result = graph_edit_distance(build_union_diff_graph(graph_a, graph_a))
        assert result.cost == 0.0
        assert result.normalized_distance == 0.0
        assert result.similarity == 1.0

    def test_disjoint_graphs_have_maximal_distance(self) -> None:
        a = make_graph([("a1", "t", "a2", 1.0)])
        b = make_graph([("b1", "t", "b2", 1.0)])
        result = graph_edit_distance(build_union_diff_graph(a, b))
        assert result.normalized_distance == pytest.approx(1.0)
        assert result.similarity == pytest.approx(0.0)

    def test_per_edge_type_costs(self, union) -> None:  # type: ignore[no-untyped-def]
        # Weight 'owns' operations 10x. The A_ONLY and B_ONLY edges are both 'owns'.
        costs = GEDCosts(edge_type_costs={"owns": 10.0})
        result = graph_edit_distance(union, costs)
        # nodes 2 + owns delete 10 + owns insert 10 + knows substitute 1 = 23
        assert result.cost == pytest.approx(23.0)
        # denominator: nodes 8 + A edges (knows 1 + knows 1 + owns 10) = 12
        #            + B edges (knows 1 + knows 1 + owns 10) = 12  -> 32
        assert result.denominator == pytest.approx(32.0)

    def test_empty_graphs_are_distance_zero(self, empty_graph: PropertyGraph) -> None:
        result = graph_edit_distance(build_union_diff_graph(empty_graph, empty_graph))
        assert result.cost == 0.0
        assert result.normalized_distance == 0.0
        assert result.similarity == 1.0


class TestWeightAgreement:
    def test_hand_computed(self, union) -> None:  # type: ignore[no-untyped-def]
        # Shared edges: (a,knows,b) 1 vs 1, (b,knows,c) 2 vs 5
        result = weight_agreement(union)
        assert result.n_shared_edges == 2
        assert result.n_compared == 2
        assert result.spearman == pytest.approx(1.0)
        assert result.mean_abs_diff == pytest.approx(1.5)

    def test_perfect_negative_correlation(self) -> None:
        a = make_graph([("x", "t", "y", 1.0), ("y", "t", "z", 2.0), ("z", "t", "w", 3.0)])
        b = make_graph([("x", "t", "y", 3.0), ("y", "t", "z", 2.0), ("z", "t", "w", 1.0)])
        result = weight_agreement(build_union_diff_graph(a, b))
        assert result.spearman == pytest.approx(-1.0)

    def test_identical_constant_weights_report_agreement(self) -> None:
        a = make_graph([("x", "t", "y", 1.0), ("y", "t", "z", 1.0)])
        result = weight_agreement(build_union_diff_graph(a, a))
        assert result.spearman == pytest.approx(1.0)

    def test_no_shared_edges_returns_none(self) -> None:
        a = make_graph([("a1", "t", "a2", 1.0)])
        b = make_graph([("b1", "t", "b2", 1.0)])
        result = weight_agreement(build_union_diff_graph(a, b))
        assert result.spearman is None
        assert result.n_shared_edges == 0

    def test_missing_weight_attribute_returns_none(self, union) -> None:  # type: ignore[no-untyped-def]
        result = weight_agreement(union, weight_attribute="confidence")
        assert result.spearman is None
        assert result.n_compared == 0


class TestNeighborhoodDelta:
    def test_hand_computed_per_node_jaccard(self, union) -> None:  # type: ignore[no-untyped-def]
        # A: a{b} b{a,c} c{b,d} | B: a{b} b{a,c} c{b,e}
        result = neighborhood_delta(union)
        by_label = result.table.set_index("label")["jaccard"].to_dict()
        assert by_label["a"] == pytest.approx(1.0)
        assert by_label["b"] == pytest.approx(1.0)
        assert by_label["c"] == pytest.approx(1 / 3)
        assert result.mean_jaccard == pytest.approx((1 + 1 + 1 / 3) / 3)
        assert result.median_jaccard == pytest.approx(1.0)
        assert result.n_nodes_compared == 3
        assert result.n_unchanged == 2

    def test_table_is_ranked_most_changed_first(self, union) -> None:  # type: ignore[no-untyped-def]
        result = neighborhood_delta(union)
        assert result.table.iloc[0]["label"] == "c"
        assert list(result.table["jaccard"]) == sorted(result.table["jaccard"])

    def test_added_and_removed_counts(self, union) -> None:  # type: ignore[no-untyped-def]
        row = neighborhood_delta(union).table.set_index("label").loc["c"]
        assert row["n_shared"] == 1  # b
        assert row["n_removed"] == 1  # d
        assert row["n_added"] == 1  # e

    def test_direction_out_only(self) -> None:
        a = make_graph([("x", "t", "y", 1.0)])
        b = make_graph([("y", "t", "x", 1.0)])
        both = neighborhood_delta(build_union_diff_graph(a, b), direction="both")
        out = neighborhood_delta(build_union_diff_graph(a, b), direction="out")
        assert both.mean_jaccard == pytest.approx(1.0)  # undirected view: same pair
        assert out.mean_jaccard == pytest.approx(0.0)  # x->y vs y->x disagree

    def test_isolated_in_both_counts_as_unchanged(self) -> None:
        a = make_graph([("x", "t", "y", 1.0)], nodes=["x", "y", "lonely"])
        b = make_graph([("x", "t", "y", 1.0)], nodes=["x", "y", "lonely"])
        result = neighborhood_delta(build_union_diff_graph(a, b))
        assert result.table.set_index("label").loc["lonely", "jaccard"] == pytest.approx(1.0)

    def test_no_shared_nodes_gives_empty_table(self) -> None:
        a = make_graph([("a1", "t", "a2", 1.0)])
        b = make_graph([("b1", "t", "b2", 1.0)])
        result = neighborhood_delta(build_union_diff_graph(a, b))
        assert result.n_nodes_compared == 0
        assert result.table.empty

    def test_identical_graphs_all_ones(self, graph_a: PropertyGraph) -> None:
        result = neighborhood_delta(build_union_diff_graph(graph_a, graph_a))
        assert result.mean_jaccard == pytest.approx(1.0)
        assert result.n_unchanged == result.n_nodes_compared


class TestIdentityProperty:
    """A graph compared with itself must score perfectly, whatever its shape."""

    @pytest.mark.parametrize("seed", range(12))
    def test_self_comparison_is_perfect(self, seed: int) -> None:
        rng_nodes = 5 + (seed * 7) % 60
        rng_edges = (seed * 13) % 120
        graph = random_graph(rng_nodes, rng_edges, seed=seed)
        union = build_union_diff_graph(graph, graph)

        sets = set_theoretic(union)
        assert sets.nodes.jaccard == pytest.approx(1.0)
        assert sets.edges.jaccard == pytest.approx(1.0)
        assert sets.typed_edges.jaccard == pytest.approx(1.0)

        ged = graph_edit_distance(union)
        assert ged.cost == 0.0
        assert ged.normalized_distance == 0.0
        assert ged.similarity == 1.0

        neighborhood = neighborhood_delta(union)
        assert neighborhood.mean_jaccard == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", range(6))
    def test_scores_are_symmetric(self, seed: int) -> None:
        a = random_graph(40, 90, seed=seed, name="A")
        b = random_graph(40, 90, seed=seed + 500, name="B")
        forward = set_theoretic(build_union_diff_graph(a, b))
        backward = set_theoretic(build_union_diff_graph(b, a))
        assert forward.nodes.jaccard == pytest.approx(backward.nodes.jaccard)
        assert forward.typed_edges.jaccard == pytest.approx(backward.typed_edges.jaccard)
        assert graph_edit_distance(
            build_union_diff_graph(a, b)
        ).normalized_distance == pytest.approx(
            graph_edit_distance(build_union_diff_graph(b, a)).normalized_distance
        )

    @pytest.mark.parametrize("seed", range(6))
    def test_scores_stay_in_unit_interval(self, seed: int) -> None:
        a = random_graph(30, 60, seed=seed)
        b = random_graph(30, 60, seed=seed + 900)
        union = build_union_diff_graph(a, b)
        sets = set_theoretic(union)
        for scores in (sets.nodes, sets.edges, sets.typed_edges):
            assert 0.0 <= scores.jaccard <= 1.0
            assert 0.0 <= scores.overlap <= 1.0
            assert 0.0 <= scores.dice <= 1.0
        ged = graph_edit_distance(union)
        assert 0.0 <= ged.normalized_distance <= 1.0
