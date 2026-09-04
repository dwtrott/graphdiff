"""Loader round-trips and format auto-detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from graphdiff import PropertyGraph, detect_format, read_graph, write_graph
from graphdiff.io import read_edgelist, read_graphml, read_node_link_json, write_graphml
from graphdiff.io.tabular import write_parquet

from helpers import make_graph


@pytest.fixture
def sample() -> PropertyGraph:
    return make_graph(
        [("a", "knows", "b", 1.5), ("b", "owns", "c", 2.5)],
        node_attrs={"a": {"kind": "person"}, "b": {"kind": "person"}, "c": {"kind": "asset"}},
        name="sample",
    )


class TestRoundTrips:
    @pytest.mark.parametrize("suffix", [".graphml", ".json"])
    def test_full_round_trip(self, sample: PropertyGraph, tmp_path: Path, suffix: str) -> None:
        path = tmp_path / f"g{suffix}"
        write_graph(sample, path)
        loaded = read_graph(path)

        assert loaded.n_nodes == sample.n_nodes
        assert loaded.n_edges == sample.n_edges
        assert loaded.directed == sample.directed
        assert set(loaded.nodes.index) == set(sample.nodes.index)
        assert sorted(loaded.edge_types) == sorted(sample.edge_types)
        pd.testing.assert_series_equal(
            loaded.edges["weight"].astype(float).sort_values().reset_index(drop=True),
            sample.edges["weight"].astype(float).sort_values().reset_index(drop=True),
            check_names=False,
        )

    def test_parquet_round_trip(self, sample: PropertyGraph, tmp_path: Path) -> None:
        directory = write_parquet(sample, tmp_path / "pq")
        loaded = read_graph(directory)
        assert loaded.n_nodes == sample.n_nodes
        assert loaded.n_edges == sample.n_edges
        assert loaded.nodes["kind"].tolist() == sample.nodes["kind"].tolist()

    def test_edgelist_round_trip(self, sample: PropertyGraph, tmp_path: Path) -> None:
        path = tmp_path / "edges.csv"
        write_graph(sample, path)
        loaded = read_graph(path)
        assert loaded.n_edges == sample.n_edges
        assert sorted(loaded.edge_types) == sorted(sample.edge_types)

    def test_graphml_preserves_node_attributes(self, sample: PropertyGraph, tmp_path: Path) -> None:
        path = write_graphml(sample, tmp_path / "g.graphml")
        loaded = read_graphml(path)
        assert loaded.nodes.loc["a", "kind"] == "person"
        assert loaded.nodes.loc["c", "kind"] == "asset"

    def test_graphml_preserves_undirected_flag(self, tmp_path: Path) -> None:
        graph = make_graph([("a", "t", "b", 1.0)], directed=False)
        loaded = read_graphml(write_graphml(graph, tmp_path / "u.graphml"))
        assert loaded.directed is False

    def test_graphml_types_are_cast(self, sample: PropertyGraph, tmp_path: Path) -> None:
        loaded = read_graphml(write_graphml(sample, tmp_path / "g.graphml"))
        assert loaded.edges["weight"].dtype.kind == "f"


class TestFormatDetection:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("g.graphml", "graphml"),
            ("g.json", "json"),
            ("g.csv", "edgelist"),
            ("g.tsv", "edgelist"),
            ("g.parquet", "parquet"),
        ],
    )
    def test_by_extension(self, tmp_path: Path, filename: str, expected: str) -> None:
        path = tmp_path / filename
        path.write_text("placeholder")
        assert detect_format(path) == expected

    def test_by_content_when_extension_unknown(self, sample: PropertyGraph, tmp_path: Path) -> None:
        path = tmp_path / "graph.dat"
        write_graphml(sample, path)
        assert detect_format(path) == "graphml"

    def test_json_by_content(self, sample: PropertyGraph, tmp_path: Path) -> None:
        path = tmp_path / "graph.dat"
        write_graph(sample, path, fmt="json")
        assert detect_format(path) == "json"

    def test_parquet_directory_detected(self, sample: PropertyGraph, tmp_path: Path) -> None:
        directory = write_parquet(sample, tmp_path / "pq")
        assert detect_format(directory) == "parquet"


class TestEdgelistParsing:
    def test_header_aliases(self, tmp_path: Path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("src,dst,relation,weight\na,b,knows,1.0\nb,c,owns,2.0\n")
        graph = read_edgelist(path)
        assert graph.n_edges == 2
        assert sorted(graph.edge_types) == ["knows", "owns"]
        assert graph.edges["weight"].tolist() == [1.0, 2.0]

    def test_headerless(self, tmp_path: Path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("a,b,knows,1.0\nb,c,owns,2.0\n")
        graph = read_edgelist(path)
        assert graph.n_edges == 2
        assert set(graph.nodes.index) == {"a", "b", "c"}

    def test_tab_separated(self, tmp_path: Path) -> None:
        path = tmp_path / "e.tsv"
        path.write_text("source\ttarget\ttype\na\tb\tknows\n")
        graph = read_edgelist(path)
        assert graph.n_edges == 1

    def test_companion_node_table(self, tmp_path: Path) -> None:
        (tmp_path / "e.csv").write_text("source,target,type\na,b,knows\n")
        (tmp_path / "n.csv").write_text("label,kind\na,person\nb,org\n")
        graph = read_edgelist(tmp_path / "e.csv", nodes_path=tmp_path / "n.csv")
        assert graph.nodes.loc["a", "kind"] == "person"


class TestNodeLinkJson:
    def test_reads_id_key(self, tmp_path: Path) -> None:
        path = tmp_path / "g.json"
        path.write_text(
            '{"directed": true, "nodes": [{"id": "a"}, {"id": "b"}],'
            ' "links": [{"source": "a", "target": "b", "type": "knows"}]}'
        )
        graph = read_node_link_json(path)
        assert set(graph.nodes.index) == {"a", "b"}
        assert graph.edge_types == ["knows"]

    def test_accepts_edges_key(self, tmp_path: Path) -> None:
        path = tmp_path / "g.json"
        path.write_text(
            '{"directed": false, "nodes": [{"id": "a"}, {"id": "b"}],'
            ' "edges": [{"source": "a", "target": "b"}]}'
        )
        graph = read_node_link_json(path)
        assert graph.directed is False
        assert graph.n_edges == 1


def test_edges_can_reference_undeclared_nodes(tmp_path: Path) -> None:
    """A node mentioned only by an edge still joins the node set."""
    path = tmp_path / "g.graphml"
    path.write_text(
        '<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
        '<graph id="G" edgedefault="directed"><node id="a"/>'
        '<edge source="a" target="ghost"/></graph></graphml>'
    )
    graph = read_graphml(path)
    assert set(graph.nodes.index) == {"a", "ghost"}
