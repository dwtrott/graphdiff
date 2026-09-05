"""Regression checks (`graphdiff check`) and static figures (`graphdiff.plot`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import graphdiff as gd
from graphdiff import write_graph
from graphdiff.cli import app
from graphdiff.data import example_pair
from graphdiff.report.check import Check, load_baseline, parse_check, run_checks

runner = CliRunner()


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("check")
    a, b = example_pair()
    return write_graph(a, d / "a.graphml"), write_graph(b, d / "b.graphml")


@pytest.fixture(scope="module")
def report() -> gd.ComparisonReport:
    a, b = example_pair()
    return gd.compare(a, b)


class TestParse:
    @pytest.mark.parametrize(
        "rule, expected",
        [
            ("jaccard_typed_edges>=0.95", Check("jaccard_typed_edges", ">=", 0.95)),
            (" nodes.A_ONLY <= 0 ", Check("nodes.A_ONLY", "<=", 0.0)),
            ("similarity>0.5", Check("ged_similarity", ">", 0.5)),
            ("significance.max<1e2", Check("significance.max", "<", 100.0)),
            ("edges.CHANGED!=0", Check("edges.CHANGED", "!=", 0.0)),
        ],
    )
    def test_good(self, rule: str, expected: Check) -> None:
        assert parse_check(rule) == expected

    @pytest.mark.parametrize("rule", ["jaccard", ">=0.5", "a=>1", "x >= high", ""])
    def test_bad(self, rule: str) -> None:
        with pytest.raises(ValueError):
            parse_check(rule)


class TestRun:
    def test_pass_and_fail(self, report: gd.ComparisonReport) -> None:
        sim = report.score("ged_similarity")
        assert sim is not None
        out = run_checks(report, [f"similarity>={sim - 0.01:.4f}", f"similarity>={sim + 0.01:.4f}"])
        assert [r.ok for r in out.results] == [True, False]
        assert not out.ok
        assert len(out.failures) == 1
        assert "FAIL" in out.summary() and "PASS" in out.summary()

    def test_status_counts(self, report: gd.ComparisonReport) -> None:
        out = run_checks(report, ["nodes.A_ONLY<=0", "edges.B_ONLY>0", "nodes.changed_any>0"])
        assert [r.ok for r in out.results] == [False, True, True]
        assert out.results[0].value == report.node_status_counts["A_ONLY"]

    def test_significance_quantities(self, report: gd.ComparisonReport) -> None:
        assert report.significance is not None
        top = float(report.significance.table["significance"].max())
        out = run_checks(
            report,
            [f"significance.max<={top + 1:.3f}", "significance.n_over_5>=1", "significance.mean>0"],
        )
        assert out.ok, out.summary()

    def test_shared_variant(self, report: gd.ComparisonReport) -> None:
        out = run_checks(report, ["jaccard_edges.shared>=0"])
        assert out.ok
        assert out.results[0].value == report.score("jaccard_edges", shared_subgraph=True)

    def test_unknown_quantity(self, report: gd.ComparisonReport) -> None:
        with pytest.raises(KeyError):
            run_checks(report, ["nonsense>=1"])
        with pytest.raises(KeyError):
            run_checks(report, ["nodes.WHATEVER>=1"])

    def test_unavailable_is_a_failure_not_a_crash(self) -> None:
        a, b = example_pair()
        rep = gd.compare(a, b, weight_attribute="no_such_attribute")
        out = run_checks(rep, ["weight_spearman>=0.5"])
        assert not out.ok
        assert out.results[0].value is None

    def test_drift(self, report: gd.ComparisonReport, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        report.to_json(path)
        base = load_baseline(path)
        same = run_checks(
            report, [], baseline=base, tolerance=0.0, drift=["ged_similarity", "nodes.A_ONLY"]
        )
        assert same.ok
        base["ged_similarity"] = (base["ged_similarity"] or 0) - 0.2
        moved = run_checks(report, [], baseline=base, tolerance=0.05, drift=["ged_similarity"])
        assert not moved.ok
        assert moved.results[0].value == pytest.approx(0.2)
        with pytest.raises(ValueError):
            run_checks(report, [], drift=["ged_similarity"])

    def test_to_dict_is_json(self, report: gd.ComparisonReport) -> None:
        out = run_checks(report, ["similarity>=0"])
        payload = json.loads(json.dumps(out.to_dict()))
        assert payload["ok"] is True
        assert payload["checks"][0]["quantity"] == "ged_similarity"


class TestCheckCommand:
    def test_exit_zero_on_pass(self, pair: tuple[Path, Path]) -> None:
        r = runner.invoke(app, ["check", str(pair[0]), str(pair[1]), "-r", "similarity>=0.1"])
        assert r.exit_code == 0, r.output
        assert "all checks passed" in r.output

    def test_exit_one_on_fail(self, pair: tuple[Path, Path]) -> None:
        r = runner.invoke(app, ["check", str(pair[0]), str(pair[1]), "-r", "nodes.A_ONLY<=0"])
        assert r.exit_code == 1
        assert "FAIL" in r.output

    def test_json_output(self, pair: tuple[Path, Path]) -> None:
        r = runner.invoke(
            app, ["check", str(pair[0]), str(pair[1]), "-r", "similarity>=0.1", "--json"]
        )
        assert r.exit_code == 0
        assert json.loads(r.output)["ok"] is True

    def test_usage_errors_exit_two(self, pair: tuple[Path, Path]) -> None:
        assert runner.invoke(app, ["check", str(pair[0]), str(pair[1])]).exit_code == 2
        assert (
            runner.invoke(app, ["check", str(pair[0]), str(pair[1]), "-r", "bogus"]).exit_code == 2
        )
        assert (
            runner.invoke(app, ["check", str(pair[0]), str(pair[1]), "-r", "nope>=1"]).exit_code
            == 2
        )
        assert (
            runner.invoke(
                app, ["check", str(pair[0]), str(pair[1]), "--drift", "similarity"]
            ).exit_code
            == 2
        )

    def test_drift_against_baseline(self, pair: tuple[Path, Path], tmp_path: Path) -> None:
        base = tmp_path / "base.json"
        r = runner.invoke(
            app, ["check", str(pair[0]), str(pair[1]), "-r", "similarity>=0", "--out", str(base)]
        )
        assert r.exit_code == 0
        r = runner.invoke(
            app,
            [
                "check",
                str(pair[0]),
                str(pair[1]),
                "--baseline",
                str(base),
                "--drift",
                "all",
                "--tolerance",
                "0.001",
            ],
        )
        assert r.exit_code == 0, r.output
        # Self-comparison drifts far from the stored a-vs-b baseline.
        r = runner.invoke(
            app,
            ["check", str(pair[0]), str(pair[0]), "--baseline", str(base), "--drift", "similarity"],
        )
        assert r.exit_code == 1


matplotlib = pytest.importorskip("matplotlib")


class TestPlot:
    def test_every_figure_renders(self, report: gd.ComparisonReport, tmp_path: Path) -> None:
        from graphdiff import plot

        for name, make in (
            ("overview", lambda: plot.plot_overview(report)),
            ("clusters", lambda: plot.plot_clusters(report)),
            ("top", lambda: plot.plot_top_changed(report, n=10)),
            ("degrees", lambda: plot.plot_degree_distributions(report)),
            ("dashboard", lambda: plot.plot_dashboard(report, top=8)),
        ):
            for suffix in (".png", ".svg"):
                out = plot.save(make(), tmp_path / f"{name}{suffix}")
                assert out.exists() and out.stat().st_size > 1000

    def test_svg_has_no_external_references(
        self, report: gd.ComparisonReport, tmp_path: Path
    ) -> None:
        from graphdiff import plot

        out = plot.save(plot.plot_overview(report), tmp_path / "o.svg")
        text = out.read_text(encoding="utf-8")
        # XML namespace URIs are identifiers, never fetched; what must not appear
        # is anything a renderer would load: linked images, imports, fonts.
        assert 'href="http' not in text
        assert "@import" not in text and "url(http" not in text

    def test_similarity_matrix(self, tmp_path: Path) -> None:
        import pandas as pd

        from graphdiff import plot

        frame = pd.DataFrame(
            {
                "graph_a": ["x", "x", "y"],
                "graph_b": ["y", "z", "z"],
                "metric": ["jaccard_edges"] * 3,
                "raw_score": [0.9, 0.4, 0.6],
            }
        )
        out = plot.save(plot.plot_similarity_matrix(frame), tmp_path / "m.png")
        assert out.exists()
        frame2 = pd.concat([frame, frame.assign(metric="dice_edges")])
        with pytest.raises(ValueError):
            plot.plot_similarity_matrix(frame2)
        plot.save(plot.plot_similarity_matrix(frame2, metric="dice_edges"), tmp_path / "m2.svg")

    def test_plot_command(self, pair: tuple[Path, Path], tmp_path: Path) -> None:
        out = tmp_path / "dash.png"
        r = runner.invoke(
            app, ["plot", str(pair[0]), str(pair[1]), "--out", str(out), "--kind", "dashboard"]
        )
        assert r.exit_code == 0, r.output
        assert out.exists()
        r = runner.invoke(app, ["plot", str(pair[0]), str(pair[1]), "--kind", "nope"])
        assert r.exit_code == 2
