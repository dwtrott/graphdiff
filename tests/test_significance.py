"""Significance ranking and the findings it drives."""

from __future__ import annotations

import pytest

import graphdiff as gd
from graphdiff import build_union_diff_graph
from graphdiff.data import example_pair, large_example_pair
from graphdiff.metrics import node_significance
from graphdiff.report.findings import generate_findings

from helpers import make_graph


def _star(center: str, n: int, prefix: str) -> list[tuple[str, str, str, float]]:
    return [(center, "t", f"{prefix}{i}", 1.0) for i in range(n)]


class TestRanking:
    def test_hub_losing_many_outranks_leaf_losing_one(self) -> None:
        """The whole point: a proportion ranks these backwards, surprise does not."""
        # hub: 30 edges, loses 10 (jaccard 0.67). leaf: 2 edges, loses 1 (jaccard 0.5).
        a = make_graph([*_star("hub", 30, "h"), ("leaf", "t", "x1", 1.0), ("leaf", "t", "x2", 1.0)])
        b_edges = [*_star("hub", 20, "h"), ("leaf", "t", "x1", 1.0)]
        b = make_graph(b_edges, nodes=[*a.nodes.index])
        sig = node_significance(build_union_diff_graph(a, b))
        ranking = sig.table.set_index("label")
        assert ranking.loc["hub", "significance"] > ranking.loc["leaf", "significance"]
        assert sig.table.iloc[0]["label"] == "hub"

    def test_removed_hub_is_scored_and_ranks_first(self) -> None:
        """A removed node has no neighbourhood Jaccard; surprise still scores it."""
        a = make_graph([*_star("gone", 15, "g"), ("p", "t", "q", 1.0)])
        b = make_graph(
            [("p", "t", "q", 1.0)] + [(f"g{i}", "t", f"g{i + 1}", 1.0) for i in range(14)]
        )
        sig = node_significance(build_union_diff_graph(a, b))
        top = sig.table.iloc[0]
        assert top["label"] == "gone"
        assert top["status"] == "A_ONLY"
        assert top["n_removed"] == 15
        assert top["surprise"] > 5

    def test_identical_graphs_score_zero(self) -> None:
        a, _ = example_pair()
        sig = node_significance(build_union_diff_graph(a, a))
        assert (sig.table["significance"] == 0).all()
        assert sig.rates.removal == 0.0 and sig.rates.addition == 0.0

    def test_every_node_is_present_once(self) -> None:
        a, b = example_pair()
        union = build_union_diff_graph(a, b)
        sig = node_significance(union)
        assert sorted(sig.table["label"]) == sorted(union.nodes.index)
        assert sig.n_nodes == union.n_nodes

    def test_background_rates_match_the_union(self) -> None:
        a, b = example_pair()
        union = build_union_diff_graph(a, b)
        sig = node_significance(union)
        ec = union.edge_status_counts()
        assert sig.rates.removal == pytest.approx(
            ec["A_ONLY"] / (ec["SHARED"] + ec["CHANGED"] + ec["A_ONLY"])
        )

    def test_degrees_are_consistent(self) -> None:
        a, b = example_pair()
        union = build_union_diff_graph(a, b)
        sig = node_significance(union)
        t = sig.table
        assert ((t["degree_a"] - t["n_removed"]) == (t["degree_b"] - t["n_added"])).all()

    def test_surprise_components_are_non_negative_and_bounded(self) -> None:
        a, b = large_example_pair(n_communities=4, community_size=60)
        sig = node_significance(build_union_diff_graph(a, b))
        for col in ("surprise_removed", "surprise_added", "surprise_changed", "significance"):
            assert (sig.table[col] >= 0).all()
            assert sig.table[col].max() <= 900

    def test_jaccard_reference_column_is_carried(self) -> None:
        a, b = example_pair()
        report = gd.compare(a, b)
        t = report.significance.table.set_index("label")
        nb = report.neighborhood.table.set_index("label")["jaccard"]
        common = t.index.intersection(nb.index)
        assert (t.loc[common, "jaccard"] == nb.loc[common]).all()

    def test_report_top_changed_uses_significance(self) -> None:
        a, b = example_pair()
        report = gd.compare(a, b)
        top = report.top_changed(3)
        assert "significance" in top.columns
        assert top["significance"].is_monotonic_decreasing


class TestFindings:
    def test_identical_graphs_say_so(self) -> None:
        a, _ = example_pair()
        report = gd.compare(a, a)
        assert len(report.findings) == 1
        assert "identical" in report.findings[0].text

    def test_concentrated_change_is_called_out(self) -> None:
        a, b = large_example_pair()
        report = gd.compare(a, b)
        kinds = [f.kind for f in report.findings]
        assert kinds[0] == "headline"
        conc = next(f for f in report.findings if f.kind == "concentration")
        assert "concentrated" in conc.text
        assert conc.evidence["k80"] == 3
        assert sum(1 for f in report.findings if f.kind == "cluster") == 3

    def test_diffuse_change_is_called_out(self) -> None:
        a, b = example_pair()
        report = gd.compare(a, b)
        conc = next(f for f in report.findings if f.kind == "concentration")
        assert "diffuse" in conc.text

    def test_removed_hub_named(self) -> None:
        a, b = example_pair()
        report = gd.compare(a, b)
        hub = next(f for f in report.findings if f.kind == "removed_hub")
        assert "Quintero" in hub.text
        assert hub.target == "Quintero"

    def test_findings_carry_navigation(self) -> None:
        a, b = large_example_pair()
        report = gd.compare(a, b)
        for f in report.findings:
            if f.kind in ("cluster", "node", "removed_hub", "added_hub"):
                assert f.view is not None and f.target is not None

    def test_background_is_last_and_low(self) -> None:
        a, b = example_pair()
        report = gd.compare(a, b)
        assert report.findings[-1].kind == "background"
        assert report.findings[-1].severity == "low"

    def test_serializes(self) -> None:
        import json

        a, b = example_pair()
        report = gd.compare(a, b)
        text = report.to_json()
        payload = json.loads(text)
        assert len(payload["findings"]) == len(report.findings)
        assert payload["findings"][0]["kind"] == "headline"
        assert payload["significance"]["rates"]["removal"] > 0
        assert payload["clusters"]["method"] == "leiden"

    def test_findings_can_be_disabled(self) -> None:
        a, b = example_pair()
        report = gd.compare(a, b, findings=False)
        assert report.findings == []
        assert report.clusters is None
        assert report.significance is not None  # cheap; always computed

    def test_direct_call_without_clusters(self) -> None:
        a, b = example_pair()
        union = build_union_diff_graph(a, b)
        found = generate_findings(
            union, node_significance(union), None, similarity=0.8, weight_spearman=0.9
        )
        assert not any(f.kind in ("concentration", "cluster") for f in found)
        assert found[0].kind == "headline"


class TestProvenance:
    def test_file_inputs_are_hashed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from graphdiff import write_graph

        a, b = example_pair()
        pa = write_graph(a, tmp_path / "a.graphml")
        pb = write_graph(b, tmp_path / "b.graphml")
        report = gd.compare(pa, pb)
        prov = report.provenance
        assert len(prov["input_a"]["sha256"]) == 64
        assert prov["input_a"]["bytes"] == pa.stat().st_size
        assert prov["input_b"]["path"].endswith("b.graphml")
        assert prov["environment"]["graphdiff"] == gd.__version__

    def test_same_file_same_hash(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from graphdiff import write_graph

        a, b = example_pair()
        pa = write_graph(a, tmp_path / "a.graphml")
        pb = write_graph(b, tmp_path / "b.graphml")
        h1 = gd.compare(pa, pb).provenance["input_a"]["sha256"]
        h2 = gd.compare(pa, pb).provenance["input_a"]["sha256"]
        assert h1 == h2

    def test_in_memory_inputs_are_labelled(self) -> None:
        a, b = example_pair()
        prov = gd.compare(a, b).provenance
        assert prov["input_a"]["path"] is None
        assert "in-memory" in prov["input_a"]["source"]

    def test_parameters_recorded(self) -> None:
        a, b = example_pair()
        prov = gd.compare(a, b, weight_attribute="weight", cluster_by=None).provenance
        assert prov["parameters"]["weight_attribute"] == "weight"
        assert prov["parameters"]["ged_costs"] == "unit"

    def test_provenance_in_markdown(self) -> None:
        a, b = example_pair()
        md = gd.compare(a, b).to_markdown()
        assert "## Provenance" in md
        assert "## Findings" in md
        assert md.index("## Findings") < md.index("## Composition")
