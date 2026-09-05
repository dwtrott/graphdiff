"""CLI and batch matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import graphdiff as gd
from graphdiff import write_graph
from graphdiff.batch import all_pairs, discover_graphs
from graphdiff.cli import app
from graphdiff.data import example_pair

runner = CliRunner()


@pytest.fixture(scope="module")
def graph_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("graphs")
    a, b = example_pair()
    write_graph(a, directory / "snap_2024.graphml")
    write_graph(b, directory / "snap_2025.graphml")
    c = gd.PropertyGraph.from_edges(a.edges.sample(frac=0.7, random_state=1), name="snap_2023")
    write_graph(c, directory / "snap_2023.json")
    (directory / "notes.txt").write_text("not a graph\n")
    return directory


class TestCompare:
    def test_prints_markdown_when_no_outputs(self, graph_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["compare", str(graph_dir / "snap_2024.graphml"), str(graph_dir / "snap_2025.graphml")],
        )
        assert result.exit_code == 0, result.output
        assert "# graphdiff: snap_2024 vs snap_2025" in result.output
        assert "jaccard_typed_edges" in result.output

    def test_writes_every_output(self, graph_dir: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "compare",
                str(graph_dir / "snap_2024.graphml"),
                str(graph_dir / "snap_2025.graphml"),
                "--out",
                str(tmp_path / "r.json"),
                "--markdown",
                str(tmp_path / "s.md"),
                "--parquet",
                str(tmp_path / "sc.parquet"),
                "--html",
                str(tmp_path / "d.html"),
                "--union-dir",
                str(tmp_path / "u"),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads((tmp_path / "r.json").read_text())
        assert payload["graph_a"] == "snap_2024"
        assert (tmp_path / "s.md").read_text().startswith("# graphdiff")
        assert len(pd.read_parquet(tmp_path / "sc.parquet")) == len(gd.SCALAR_METRICS)
        assert "<canvas" in (tmp_path / "d.html").read_text()
        assert (tmp_path / "u" / "union_edges.parquet").exists()

    def test_missing_file_is_an_error(self) -> None:
        result = runner.invoke(app, ["compare", "nope.graphml", "nope2.graphml"])
        assert result.exit_code != 0


class TestInspect:
    def test_text_output(self, graph_dir: Path) -> None:
        result = runner.invoke(app, ["inspect", str(graph_dir / "snap_2024.graphml")])
        assert result.exit_code == 0, result.output
        assert "nodes       52" in result.output
        assert "edge types  5" in result.output

    def test_json_output(self, graph_dir: Path) -> None:
        result = runner.invoke(app, ["inspect", str(graph_dir / "snap_2024.graphml"), "--json"])
        assert result.exit_code == 0
        assert json.loads(result.output)["n_nodes"] == 52


class TestMatrix:
    def test_discover_skips_non_graphs(self, graph_dir: Path) -> None:
        names = [p.name for p in discover_graphs(graph_dir)]
        assert names == ["snap_2023.json", "snap_2024.graphml", "snap_2025.graphml"]

    def test_all_pairs_tidy_shape(self, graph_dir: Path) -> None:
        frame = all_pairs(
            discover_graphs(graph_dir), metrics=["jaccard_typed_edges", "ged_similarity"], workers=1
        )
        assert list(frame.columns) == [
            "graph_a",
            "graph_b",
            "metric",
            "raw_score",
            "shared_subgraph_score",
        ]
        assert len(frame) == 3 * 2  # 3 pairs x 2 metrics
        assert frame["raw_score"].between(0, 1).all()

    def test_self_pairs_score_perfectly(self, graph_dir: Path) -> None:
        frame = all_pairs(
            discover_graphs(graph_dir),
            metrics=["jaccard_typed_edges"],
            workers=1,
            include_self=True,
        )
        selfs = frame[frame["graph_a"] == frame["graph_b"]]
        assert len(selfs) == 3
        assert (selfs["raw_score"] == 1.0).all()

    def test_unknown_metric_raises(self, graph_dir: Path) -> None:
        with pytest.raises(KeyError, match="unknown metric"):
            all_pairs(discover_graphs(graph_dir), metrics=["nope"], workers=1)

    def test_parallel_matches_serial(self, graph_dir: Path) -> None:
        paths = discover_graphs(graph_dir)
        serial = all_pairs(paths, metrics=["ged_similarity"], workers=1)
        parallel = all_pairs(paths, metrics=["ged_similarity"], workers=2)
        pd.testing.assert_frame_equal(serial, parallel)

    def test_cli_writes_parquet_and_csv(self, graph_dir: Path, tmp_path: Path) -> None:
        for name in ("m.parquet", "m.csv"):
            result = runner.invoke(
                app, ["matrix", str(graph_dir), "--out", str(tmp_path / name), "--workers", "1"]
            )
            assert result.exit_code == 0, result.output
            assert (tmp_path / name).exists()

    def test_cli_rejects_unknown_metric(self, graph_dir: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["matrix", str(graph_dir), "--out", str(tmp_path / "m.parquet"), "--metric", "nope"],
        )
        assert result.exit_code == 2
        assert "unknown metric" in result.output


class TestRender:
    def test_writes_offline_html(self, graph_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "diff.html"
        result = runner.invoke(
            app,
            [
                "render",
                str(graph_dir / "snap_2024.graphml"),
                str(graph_dir / "snap_2025.graphml"),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        text = out.read_text()
        assert "<canvas" in text
        assert "https://" not in text and "http://" not in text


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "graphdiff 0.1.0" in result.output


class TestPackagingScripts:
    """The offline-bundle scripts must at least parse; building needs network."""

    @pytest.mark.parametrize("name", ["build_offline_bundle.sh", "install_offline.sh"])
    def test_scripts_parse(self, name: str) -> None:
        import subprocess

        path = Path(__file__).resolve().parents[1] / "scripts" / name
        assert path.exists()
        subprocess.run(["bash", "-n", str(path)], check=True)

    def test_dockerfile_installs_from_bundle_only(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
        assert "--no-index" in text and "--find-links wheelhouse" in text
        assert "pip install graphdiff" not in text  # never from the index
