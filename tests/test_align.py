"""Fuzzy node alignment and its integration with compare()."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import graphdiff as gd
from graphdiff import PropertyGraph, align_graphs, build_union_diff_graph
from graphdiff.core.align import normalize_label
from graphdiff.data import example_pair, large_example_pair, perturb_labels

from helpers import make_graph


class TestNormalize:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Acme Corp.", "acme corp"),
            ("  ACME   corp ", "acme corp"),
            ("Café-Noir", "cafe noir"),
            ("c24_n004", "c24 n004"),
            ("", ""),
        ],
    )
    def test_forms(self, raw: str, expected: str) -> None:
        assert normalize_label(raw) == expected


class TestAlign:
    def test_exact_method_is_the_plain_join(self) -> None:
        a, b = example_pair()
        res = align_graphs(a, b, method="exact")
        assert res.counts["fuzzy"] == 0 and res.counts["normalized"] == 0
        assert res.counts["exact"] == len(a.nodes.index.intersection(b.nodes.index))
        assert res.relabel_b(b) is b

    def test_normalized_recovers_case_and_punctuation(self) -> None:
        a, b = example_pair()
        b2, truth = perturb_labels(b, fraction=0.2, seed=3)
        res = align_graphs(a, b2, method="normalized")
        got = res.table[res.table["method"] == "normalized"]
        assert len(got) > 0
        assert all(truth[r.label_b] == r.label_a for r in got.itertuples())
        assert (got["confidence"] == 1.0).all()

    def test_fuzzy_recovers_every_perturbation_on_the_example(self) -> None:
        a, b = example_pair()
        b2, truth = perturb_labels(b, fraction=0.25, seed=1)
        res = align_graphs(a, b2)
        inexact = res.table[res.table["method"] != "exact"]
        # Every renamed node that exists in A should be found, and found correctly.
        renamed_in_a = {new for new, old in truth.items() if old in a.nodes.index}
        assert set(inexact["label_b"]) == renamed_in_a
        assert all(truth[r.label_b] == r.label_a for r in inexact.itertuples())
        assert (inexact["confidence"] > 0).all() and (inexact["confidence"] <= 1).all()

    def test_fuzzy_at_scale_is_accurate_and_fast(self) -> None:
        import time

        a, b = large_example_pair()
        b2, truth = perturb_labels(b, fraction=0.1, seed=2)
        t = time.perf_counter()
        res = align_graphs(a, b2)
        assert time.perf_counter() - t < 5
        inexact = res.table[res.table["method"] != "exact"]
        correct = sum(truth.get(r.label_b) == r.label_a for r in inexact.itertuples())
        assert correct / len(inexact) > 0.98
        assert len(inexact) > 0.9 * sum(1 for old in truth.values() if old in a.nodes.index)

    def test_structure_breaks_label_ties(self) -> None:
        # Two B candidates look equally like A's "node-1"; only one shares its
        # neighbours. Structure must pick it, and the confidence must reflect
        # that the labels alone were ambiguous.
        a = make_graph([("node-1", "t", "p", 1.0), ("node-1", "t", "q", 1.0), ("z", "t", "p", 1.0)])
        b = make_graph(
            [("node-1a", "t", "p", 1.0), ("node-1a", "t", "q", 1.0), ("node-1b", "t", "z", 1.0)]
        )
        res = align_graphs(a, b, threshold=0.3)
        fuzzy = res.table[res.table["method"] == "fuzzy"].set_index("label_a")
        assert fuzzy.loc["node-1", "label_b"] == "node-1a"
        assert fuzzy.loc["node-1", "structural_similarity"] == 1.0
        assert not np.isnan(fuzzy.loc["node-1", "runner_up"])
        assert fuzzy.loc["node-1", "confidence"] < fuzzy.loc["node-1", "score"]

    def test_one_to_one(self) -> None:
        a = make_graph([("alpha", "t", "x", 1.0), ("alphb", "t", "x", 1.0)])
        b = make_graph([("alphx", "t", "x", 1.0)])
        res = align_graphs(a, b, threshold=0.3)
        assert res.table["label_b"].is_unique and res.table["label_a"].is_unique
        assert res.counts["fuzzy"] == 1
        assert res.counts["unmatched_a"] == 1

    def test_threshold_respected(self) -> None:
        a = make_graph([("completely", "t", "x", 1.0)])
        b = make_graph([("different", "t", "x", 1.0)])
        res = align_graphs(a, b, threshold=0.6)
        assert res.counts["fuzzy"] == 0
        assert list(res.unmatched_a) == ["completely"]

    def test_empty_sides(self) -> None:
        a, _ = example_pair()
        empty = PropertyGraph(
            nodes=pd.DataFrame(index=pd.Index([], name="label")),
            edges=pd.DataFrame({"source": [], "target": []}),
            directed=True,
        )
        res = align_graphs(a, empty)
        assert res.counts["exact"] == 0 and res.counts["unmatched_a"] == a.n_nodes

    def test_relabel_keeps_everything_else(self) -> None:
        a, b = example_pair()
        b2, _ = perturb_labels(b, fraction=0.2, seed=5)
        res = align_graphs(a, b2)
        b3 = res.relabel_b(b2)
        assert b3.n_nodes == b2.n_nodes and b3.n_edges == b2.n_edges
        assert b3.directed == b2.directed and b3.name == b2.name
        assert set(b3.nodes.index) >= set(res.table["label_a"])


class TestCompareIntegration:
    def test_similarity_recovers(self) -> None:
        a, b = example_pair()
        b2, _ = perturb_labels(b, fraction=0.25, seed=1)
        exact = gd.compare(a, b2)
        fuzzy = gd.compare(a, b2, align="fuzzy")
        truth = gd.compare(a, b)
        assert fuzzy.score("ged_similarity") > exact.score("ged_similarity")
        assert fuzzy.node_status_counts == truth.node_status_counts
        assert fuzzy.edge_status_counts == truth.edge_status_counts

    def test_union_carries_match_columns(self) -> None:
        a, b = example_pair()
        b2, truth = perturb_labels(b, fraction=0.2, seed=1)
        report = gd.compare(a, b2, align="fuzzy")
        nodes = report.union.nodes
        for col in ("match_method", "match_confidence", "label_b"):
            assert col in nodes.columns
        shared = nodes[nodes["status"] == "SHARED"]
        assert set(shared["match_method"]) <= {"exact", "normalized", "fuzzy"}
        assert (shared["match_confidence"] > 0).all()
        a_only = nodes[nodes["status"] == "A_ONLY"]
        assert (a_only["match_method"] == "").all() and a_only["match_confidence"].isna().all()
        inexact = shared[shared["match_method"] != "exact"]
        assert all(truth[row.label_b] == label for label, row in inexact.iterrows())

    def test_exact_default_has_no_alignment(self) -> None:
        a, b = example_pair()
        report = gd.compare(a, b)
        assert report.alignment is None
        assert "match_method" not in report.union.nodes.columns
        assert json.loads(report.to_json())["alignment"] is None

    def test_report_surfaces(self) -> None:
        a, b = example_pair()
        b2, _ = perturb_labels(b, fraction=0.25, seed=1)
        report = gd.compare(a, b2, align="fuzzy")
        assert report.alignment is not None
        payload = json.loads(report.to_json())
        assert payload["alignment"]["counts"]["fuzzy"] > 0
        assert payload["provenance"]["parameters"]["align"] == "fuzzy"
        md = report.to_markdown()
        assert "## Alignment" in md
        kinds = [f.kind for f in report.findings]
        assert "alignment" in kinds
        assert kinds.index("alignment") == 1  # right after the headline

    def test_induced_shared_keeps_alignment(self) -> None:
        a, b = example_pair()
        b2, _ = perturb_labels(b, fraction=0.2, seed=1)
        res = align_graphs(a, b2)
        union = build_union_diff_graph(a, b2, alignment=res)
        assert union.induced_shared().alignment is res
        assert union.name_b == b2.name

    def test_viewer_marks_matches(self) -> None:
        from graphdiff.viewer import render_html

        a, b = example_pair()
        b2, _ = perturb_labels(b, fraction=0.25, seed=1)
        html = render_html(gd.compare(a, b2, align="fuzzy"))
        assert '"matches":{' in html and '"alignCounts":{' in html
        assert "Matched despite differing labels" in html
        plain = render_html(gd.compare(a, b))
        assert '"matches":{}' in plain and '"alignCounts":null' in plain


class TestCLI:
    def test_align_flag(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from typer.testing import CliRunner

        from graphdiff import write_graph
        from graphdiff.cli import app

        a, b = example_pair()
        b2, _ = perturb_labels(b, fraction=0.25, seed=1)
        pa = write_graph(a, tmp_path / "a.graphml")
        pb = write_graph(b2, tmp_path / "b.graphml")
        runner = CliRunner()
        out = tmp_path / "r.json"
        r = runner.invoke(app, ["compare", str(pa), str(pb), "--align", "fuzzy", "--out", str(out)])
        assert r.exit_code == 0, r.output
        assert json.loads(out.read_text())["alignment"]["counts"]["fuzzy"] > 0
        r = runner.invoke(app, ["compare", str(pa), str(pb), "--align", "nope"])
        assert r.exit_code == 2
