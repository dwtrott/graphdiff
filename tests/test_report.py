"""ComparisonReport: scalar registry, serialization, and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from graphdiff import SCALAR_METRICS, PropertyGraph, compare, inspect_graph

from helpers import make_graph


@pytest.fixture
def report(graph_a: PropertyGraph, graph_b: PropertyGraph):  # type: ignore[no-untyped-def]
    return compare(graph_a, graph_b)


class TestScalarScores:
    def test_registry_is_complete(self, report) -> None:  # type: ignore[no-untyped-def]
        scores = report.scalar_scores()
        assert set(scores) == set(SCALAR_METRICS)
        for entry in scores.values():
            assert set(entry) == {"raw", "shared"}

    def test_score_lookup_matches_metric(self, report) -> None:  # type: ignore[no-untyped-def]
        assert report.score("jaccard_nodes") == pytest.approx(3 / 5)
        assert report.score("jaccard_nodes", shared_subgraph=True) == pytest.approx(1.0)
        assert report.score("ged_similarity") == pytest.approx(9 / 14)

    def test_unknown_metric_raises(self, report) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(KeyError, match="unknown metric"):
            report.score("not_a_metric")

    def test_identical_graphs_score_perfectly(self, graph_a: PropertyGraph) -> None:
        result = compare(graph_a, graph_a)
        assert result.score("jaccard_nodes") == 1.0
        assert result.score("jaccard_edges") == 1.0
        assert result.score("jaccard_typed_edges") == 1.0
        assert result.score("ged_normalized_distance") == 0.0
        assert result.score("ged_similarity") == 1.0
        assert result.score("neighborhood_mean_jaccard") == pytest.approx(1.0)


class TestSerialization:
    def test_json_is_valid_and_complete(self, report, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "report.json"
        report.to_json(path)
        payload = json.loads(path.read_text())
        assert payload["graph_a"] == "A"
        assert payload["graph_b"] == "B"
        assert payload["node_status_counts"]["A_ONLY"] == 1
        assert set(payload["metrics"]) == {
            "set_theoretic",
            "graph_edit_distance",
            "weight_agreement",
            "neighborhood_delta",
        }
        assert payload["scalar_scores"]["jaccard_nodes"]["raw"] == pytest.approx(0.6)

    def test_json_has_no_nan_literals(self, tmp_path: Path) -> None:
        """NaN is not valid JSON; undefined metrics must serialize as null."""
        a = make_graph([("a1", "t", "a2", 1.0)])
        b = make_graph([("b1", "t", "b2", 1.0)])
        text = compare(a, b).to_json()
        assert "NaN" not in text
        json.loads(text)  # would raise if NaN leaked through

    def test_parquet_is_tidy_long_format(self, report, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = report.to_parquet(tmp_path / "scores.parquet")
        frame = pd.read_parquet(path)
        assert list(frame.columns) == [
            "graph_a",
            "graph_b",
            "metric",
            "raw_score",
            "shared_subgraph_score",
        ]
        assert len(frame) == len(SCALAR_METRICS)

    def test_union_parquet_written(self, report, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        directory = report.write_union_parquet(tmp_path / "union")
        nodes = pd.read_parquet(directory / "union_nodes.parquet")
        edges = pd.read_parquet(directory / "union_edges.parquet")
        assert len(nodes) == 5
        assert len(edges) == 4
        assert set(edges["status"]) == {"SHARED", "CHANGED", "A_ONLY", "B_ONLY"}

    def test_union_parquet_requires_union(self, graph_a: PropertyGraph, tmp_path: Path) -> None:
        result = compare(graph_a, graph_a, keep_union=False)
        with pytest.raises(ValueError, match="no union"):
            result.write_union_parquet(tmp_path / "union")


class TestRendering:
    def test_markdown_contains_key_sections(self, report) -> None:  # type: ignore[no-untyped-def]
        text = report.to_markdown()
        assert "# graphdiff: A vs B" in text
        assert "## Composition" in text
        assert "## Scores" in text
        assert "Most significant nodes" in text
        assert "## Findings" in text
        assert "jaccard_edges" in text

    def test_markdown_file_written(self, report, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = report.write_markdown(tmp_path / "summary.md")
        assert path.read_text().startswith("# graphdiff")

    def test_repr_html_is_html(self, report) -> None:  # type: ignore[no-untyped-def]
        html = report._repr_html_()
        assert html.strip().startswith("<div")
        assert "graphdiff: A vs B" in html
        assert "<table" in html


class TestInspect:
    def test_summary_fields(self, graph_a: PropertyGraph) -> None:
        summary = inspect_graph(graph_a)
        assert summary["n_nodes"] == 4
        assert summary["n_edges"] == 3
        assert summary["edge_types"] == ["knows", "owns"]
        assert summary["directed"] is True
        assert summary["degree"]["max"] == 2

    def test_accepts_a_path(self, graph_a: PropertyGraph, tmp_path: Path) -> None:
        from graphdiff import write_graph

        path = write_graph(graph_a, tmp_path / "a.graphml")
        assert inspect_graph(path)["n_nodes"] == 4


def test_compare_accepts_paths(
    graph_a: PropertyGraph, graph_b: PropertyGraph, tmp_path: Path
) -> None:
    from graphdiff import write_graph

    path_a = write_graph(graph_a, tmp_path / "a.graphml")
    path_b = write_graph(graph_b, tmp_path / "b.graphml")
    result = compare(path_a, path_b)
    assert result.score("jaccard_nodes") == pytest.approx(0.6)
    assert result.name_a == "a"
