"""Timeline: comparing a sequence of snapshots."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import graphdiff as gd
from graphdiff import write_graph
from graphdiff.batch import TimelineReport, compare_sequence
from graphdiff.cli import app
from graphdiff.data import example_pair, example_sequence, perturb_labels
from graphdiff.viewer import render_timeline_html, write_timeline_html

EXTERNAL_URL = re.compile(r"""(?:https?:)?//[A-Za-z0-9._-]+\.[A-Za-z]{2,}""")
REMOTE_TAG = re.compile(r"<(?:script|link|img|source|embed|object)\b[^>]*\b(?:src|href)=")


@pytest.fixture(scope="module")
def series() -> list[gd.PropertyGraph]:
    return example_sequence()


@pytest.fixture(scope="module")
def timeline(series: list[gd.PropertyGraph]) -> TimelineReport:
    return compare_sequence(series)


class TestShape:
    def test_steps_and_names(self, series, timeline) -> None:  # type: ignore[no-untyped-def]
        assert timeline.n_steps == len(series) - 1
        assert timeline.names == [g.name for g in series]
        assert timeline.step_labels()[0] == "t0 → t1"
        assert all(isinstance(s, gd.ComparisonReport) for s in timeline.steps)

    def test_metrics_long_format(self, timeline) -> None:  # type: ignore[no-untyped-def]
        m = timeline.metrics
        assert set(m.columns) == {
            "step",
            "from",
            "to",
            "metric",
            "raw_score",
            "shared_subgraph_score",
        }
        assert m.groupby("step").size().nunique() == 1  # every step has every metric
        sim = timeline.metric("ged_similarity")
        assert len(sim) == timeline.n_steps and (sim > 0).all()

    def test_churn(self, timeline) -> None:  # type: ignore[no-untyped-def]
        churn = timeline.churn()
        assert len(churn) == timeline.n_steps
        for k, rep in enumerate(timeline.steps):
            assert churn.loc[k, "edges_removed"] == rep.edge_status_counts["A_ONLY"]
            assert churn.loc[k, "similarity"] == rep.score("ged_similarity")

    def test_node_history_is_ranked_and_bounded(self, series) -> None:  # type: ignore[no-untyped-def]
        tl = compare_sequence(series, top_nodes=25)
        h = tl.node_history
        assert len(h) <= 25 and list(h.columns) == list(range(tl.n_steps))
        assert (h >= 0).all().all()
        assert h.sum(axis=1).is_monotonic_decreasing

    def test_presence(self, series, timeline) -> None:  # type: ignore[no-untyped-def]
        p = timeline.presence
        assert list(p.columns) == timeline.names
        for i, g in enumerate(series):
            assert set(p.index[p.iloc[:, i]]) == set(g.nodes.index)


class TestNarrative:
    def test_event_step_is_found(self, timeline) -> None:  # type: ignore[no-untyped-def]
        step = next(f for f in timeline.findings if f.kind == "step")
        assert step.target == "2"  # t2 → t3 is the engineered event
        assert "dominates" in step.text

    def test_hotspot_is_found(self, timeline) -> None:  # type: ignore[no-untyped-def]
        rec = next(f for f in timeline.findings if f.kind == "recurrent")
        assert rec.target is not None and rec.target.startswith("c01-")
        table = timeline.recurrent_nodes(min_steps=3)
        # The hotspot community dominates; the rest are neighbours of flickers.
        assert table["label"].str.startswith("c01-").mean() > 0.6

    def test_flickers_are_found(self, timeline) -> None:  # type: ignore[no-untyped-def]
        fl = timeline.flickers()
        assert len(fl) >= 4
        assert set(fl["label"].str[:3]) >= {"c02"}
        assert fl.iloc[0]["pattern"].count("●") >= 2
        finding = next(f for f in timeline.findings if f.kind == "flicker")
        assert "came back" in finding.text
        # Flickers are not counted as the hotspot.
        rec = next(f for f in timeline.findings if f.kind == "recurrent")
        assert not any(lab in rec.text for lab in fl["label"].head(4))

    def test_quiet_series(self) -> None:
        a, _ = example_pair()
        tl = compare_sequence([a, a, a])
        assert tl.node_history.empty
        assert tl.recurrent_nodes().empty and tl.flickers().empty
        assert tl.findings[0].kind == "step"

    def test_alignment_passes_through(self) -> None:
        a, b = example_pair()
        b2, _ = perturb_labels(b, fraction=0.25, seed=1)
        tl = compare_sequence([a, b2, a], align="fuzzy")
        assert all(s.alignment is not None for s in tl.steps)
        assert tl.parameters["align"] == "fuzzy"
        # A rename is not a leave-and-arrive: only the genuinely one-sided
        # nodes flicker across a → b' → a, never the renamed ones.
        truth = gd.compare(a, b)
        one_sided = set(truth.union.nodes_with_status("A_ONLY", "B_ONLY").index)
        assert set(tl.flickers()["label"]) == one_sided


class TestExport:
    def test_json_roundtrip(self, timeline, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        payload = json.loads(timeline.to_json(tmp_path / "tl.json"))
        assert payload["names"] == timeline.names
        assert len(payload["steps"]) == timeline.n_steps
        assert payload["churn"][2]["edges_removed"] > payload["churn"][1]["edges_removed"]
        assert payload["node_history"]["labels"]

    def test_markdown(self, timeline) -> None:  # type: ignore[no-untyped-def]
        md = timeline.to_markdown()
        for section in (
            "## Findings",
            "## Churn per step",
            "## Recurrently changing nodes",
            "## Flickering nodes",
        ):
            assert section in md
        assert "t2 → t3" in md

    def test_page_is_offline_and_embeds_every_step(self, timeline) -> None:  # type: ignore[no-untyped-def]
        html = render_timeline_html(timeline, max_nodes=300)
        assert not EXTERNAL_URL.findall(html)
        assert not REMOTE_TAG.search(html)
        for needle in ("fetch(", "XMLHttpRequest", "WebSocket", "@import"):
            assert needle not in html
        # One embedded viewer per step (the inner documents are JSON-escaped).
        assert html.count('\\"initialView\\"') == timeline.n_steps
        assert "Findings across the series" in html

    def test_write(self, timeline, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = write_timeline_html(timeline, tmp_path / "tl.html", max_nodes=200)
        assert path.stat().st_size > 50_000

    @pytest.mark.skipif(pytest.importorskip("matplotlib") is None, reason="no matplotlib")
    def test_plot(self, timeline, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        from graphdiff import plot

        out = plot.save(plot.plot_timeline(timeline), tmp_path / "tl.png")
        assert out.exists() and out.stat().st_size > 5_000

    def test_errors(self) -> None:
        a, _ = example_pair()
        with pytest.raises(ValueError):
            compare_sequence([a])
        with pytest.raises(ValueError):
            compare_sequence([a, a], names=["only-one"])


class TestCLI:
    def test_timeline_command(self, series, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        paths = [str(write_graph(g, tmp_path / f"{g.name}.graphml")) for g in series[:4]]
        out, md, page = tmp_path / "tl.json", tmp_path / "tl.md", tmp_path / "tl.html"
        r = CliRunner().invoke(
            app,
            [
                "timeline",
                *paths,
                "--out",
                str(out),
                "--markdown",
                str(md),
                "--html",
                str(page),
                "--max-nodes",
                "200",
            ],
        )
        assert r.exit_code == 0, r.output
        assert json.loads(out.read_text())["names"] == ["t0", "t1", "t2", "t3"]
        assert "## Churn per step" in md.read_text()
        assert page.stat().st_size > 50_000
        r = CliRunner().invoke(app, ["timeline", paths[0]])
        assert r.exit_code == 2
