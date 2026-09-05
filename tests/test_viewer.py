"""Viewer: layout behaviour, focus selection, and the offline guarantee.

The offline test is the one that matters most — it is the automated form of the
air-gapped requirement, and it fails the build if any external reference ever
creeps into the rendered page.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import graphdiff as gd
from graphdiff import PropertyGraph, build_union_diff_graph
from graphdiff._types import STATUS_ORDER
from graphdiff.data import example_pair
from graphdiff.viewer import (
    LayoutParams,
    force_directed_layout,
    render_html,
    select_focus,
    write_html,
)

from helpers import make_graph, random_graph

#: Any scheme-ful or protocol-relative URL pointing off-box.
EXTERNAL_URL = re.compile(r"""(?:https?:)?//[A-Za-z0-9._-]+\.[A-Za-z]{2,}""")
#: Tags that would pull a remote resource at render time.
REMOTE_TAG = re.compile(r"<(?:script|link|img|iframe|source|embed|object)\b[^>]*\b(?:src|href)=")


@pytest.fixture(scope="module")
def demo_report():  # type: ignore[no-untyped-def]
    a, b = example_pair()
    return gd.compare(a, b)


class TestOfflineGuarantee:
    """The rendered page must never reach the network."""

    def test_no_external_urls(self, demo_report) -> None:  # type: ignore[no-untyped-def]
        html = render_html(demo_report)
        found = EXTERNAL_URL.findall(html)
        assert not found, f"rendered page references external hosts: {sorted(set(found))}"

    def test_no_remote_resource_tags(self, demo_report) -> None:  # type: ignore[no-untyped-def]
        html = render_html(demo_report)
        assert not REMOTE_TAG.search(html), "page loads an external script/style/image"

    def test_no_fetch_or_websocket(self, demo_report) -> None:  # type: ignore[no-untyped-def]
        html = render_html(demo_report)
        for needle in ("fetch(", "XMLHttpRequest", "WebSocket", "importScripts", "@import"):
            assert needle not in html, f"page contains network primitive {needle!r}"

    def test_is_a_complete_document(self, demo_report) -> None:  # type: ignore[no-untyped-def]
        html = render_html(demo_report)
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert "<canvas" in html and "</html>" in html

    def test_written_file_is_standalone(self, demo_report, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        path = write_html(demo_report, tmp_path / "diff.html")
        text = path.read_text(encoding="utf-8")
        assert not EXTERNAL_URL.findall(text)
        assert path.stat().st_size > 5_000


class TestLayout:
    def test_shape_and_range(self) -> None:
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        pos = force_directed_layout(4, edges)
        assert pos.shape == (4, 2)
        assert pos.min() >= 0.0 and pos.max() <= 1.0

    def test_deterministic_for_a_seed(self) -> None:
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        first = force_directed_layout(4, edges, params=LayoutParams(seed=7, iterations=60))
        second = force_directed_layout(4, edges, params=LayoutParams(seed=7, iterations=60))
        np.testing.assert_allclose(first, second)

    def test_different_seeds_differ(self) -> None:
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        a = force_directed_layout(4, edges, params=LayoutParams(seed=1, iterations=60))
        b = force_directed_layout(4, edges, params=LayoutParams(seed=2, iterations=60))
        assert not np.allclose(a, b)

    def test_empty_and_single_node(self) -> None:
        assert force_directed_layout(0, np.zeros((0, 2))).shape == (0, 2)
        np.testing.assert_allclose(force_directed_layout(1, np.zeros((0, 2))), [[0.5, 0.5]])

    def test_no_nans_on_disconnected_graph(self) -> None:
        pos = force_directed_layout(30, np.zeros((0, 2)), params=LayoutParams(iterations=80))
        assert np.isfinite(pos).all()

    def test_ignores_out_of_range_and_self_edges(self) -> None:
        edges = np.array([[0, 0], [0, 99], [-1, 1], [0, 1]])
        pos = force_directed_layout(3, edges, params=LayoutParams(iterations=40))
        assert np.isfinite(pos).all()

    def test_connected_nodes_end_closer_than_unconnected(self) -> None:
        # Two triangles joined by nothing: within-triangle distance should be
        # smaller than the distance between the two components.
        edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]])
        pos = force_directed_layout(6, edges, params=LayoutParams(seed=3))
        within = np.linalg.norm(pos[0] - pos[1])
        across = np.linalg.norm(pos[:3].mean(axis=0) - pos[3:].mean(axis=0))
        assert within < across


class TestFocusSelection:
    def test_keeps_all_differing_nodes(self) -> None:
        a, b = example_pair()
        union = build_union_diff_graph(a, b)
        focus = select_focus(union, max_nodes=2000)
        differing = union.nodes.index[union.nodes["status"].isin({"A_ONLY", "B_ONLY", "CHANGED"})]
        assert set(differing).issubset(set(focus.labels))

    def test_respects_the_cap(self) -> None:
        a = random_graph(400, 900, seed=1)
        b = random_graph(400, 900, seed=2)
        focus = select_focus(build_union_diff_graph(a, b), max_nodes=50)
        assert len(focus.labels) <= 50
        assert focus.truncated

    def test_context_hops_zero_draws_only_differences(self) -> None:
        a = make_graph([("x", "t", "y", 1.0), ("y", "t", "z", 1.0), ("z", "t", "w", 1.0)])
        b = make_graph([("x", "t", "y", 1.0), ("y", "t", "z", 1.0)])
        union = build_union_diff_graph(a, b)
        focus = select_focus(union, context_hops=0)
        assert set(focus.labels) == {"z", "w"}  # endpoints of the dropped edge

    def test_identical_graphs_still_render_something(self) -> None:
        a, _ = example_pair()
        focus = select_focus(build_union_diff_graph(a, a), max_nodes=20)
        assert len(focus.labels) > 0

    def test_edges_kept_only_when_both_endpoints_drawn(self) -> None:
        a = random_graph(200, 500, seed=5)
        b = random_graph(200, 500, seed=6)
        focus = select_focus(build_union_diff_graph(a, b), max_nodes=40)
        drawn = set(focus.labels)
        assert focus.edges["source"].isin(drawn).all()
        assert focus.edges["target"].isin(drawn).all()


class TestRenderedContent:
    def test_reports_honest_drawn_counts(self, demo_report) -> None:  # type: ignore[no-untyped-def]
        html = render_html(demo_report, max_nodes=20)
        assert '"drawnNodes":' in html
        assert '"totalNodes":' in html

    def test_status_colors_present(self, demo_report) -> None:  # type: ignore[no-untyped-def]
        from graphdiff.viewer import STATUS_COLORS

        html = render_html(demo_report)
        for color in STATUS_COLORS.values():
            assert color in html

    def test_accepts_a_bare_union_graph(self) -> None:
        a, b = example_pair()
        html = render_html(build_union_diff_graph(a, b))
        assert "<canvas" in html
        assert '"metrics":[]' in html  # no report, so no scores panel data

    def test_rejects_report_without_union(self) -> None:
        a, b = example_pair()
        report = gd.compare(a, b, keep_union=False)
        with pytest.raises(ValueError, match="no union"):
            render_html(report)

    def test_closing_script_tag_is_escaped(self) -> None:
        """A label containing '</script>' must not break out of the data block."""
        a = make_graph([("</script><b>x", "t", "y", 1.0)])
        b = make_graph([("</script><b>x", "t", "z", 1.0)])
        html = render_html(build_union_diff_graph(a, b))
        assert "</script><b>x" not in html.split("<script>")[1].split("</script>")[0]

    def test_empty_graphs_render(self, empty_graph: PropertyGraph) -> None:
        html = render_html(build_union_diff_graph(empty_graph, empty_graph))
        assert "<canvas" in html


class TestEncoding:
    """Status is encoded twice — hue and shape — so colour is never load-bearing."""

    def test_every_status_has_a_distinct_shape(self) -> None:
        from graphdiff.viewer import SHAPES

        assert set(SHAPES) == set(STATUS_ORDER)
        assert len(set(SHAPES.values())) == len(SHAPES), "two statuses share a silhouette"

    def test_shapes_reach_the_page(self, demo_report) -> None:  # type: ignore[no-untyped-def]
        from graphdiff.viewer import SHAPES

        html = render_html(demo_report)
        for shape in SHAPES.values():
            assert f"'{shape}'" in html or f'"{shape}"' in html

    def test_legend_names_the_two_graphs(self, demo_report) -> None:  # type: ignore[no-untyped-def]
        html = render_html(demo_report)
        assert "Only in snapshot_2024" in html
        assert "Only in snapshot_2025" in html

    def test_both_themes_are_embedded(self, demo_report) -> None:  # type: ignore[no-untyped-def]
        from graphdiff.viewer import DARK, LIGHT

        html = render_html(demo_report)
        for theme in (LIGHT, DARK):
            for status in STATUS_ORDER:
                assert theme[status] in html
        assert LIGHT["surface"] in html and DARK["surface"] in html

    def test_dark_is_not_a_flip_of_light(self) -> None:
        from graphdiff.viewer import DARK, LIGHT

        # The neutral and the two graph hues are re-stepped for the dark surface.
        assert DARK["SHARED"] != LIGHT["SHARED"]
        assert DARK["A_ONLY"] != LIGHT["A_ONLY"]
        assert DARK["B_ONLY"] != LIGHT["B_ONLY"]

    def test_status_labels_fall_back_to_graph_names(self) -> None:
        from graphdiff.viewer import status_labels

        labels = status_labels("v1", "v2")
        assert labels["A_ONLY"] == "Only in v1"
        assert labels["B_ONLY"] == "Only in v2"
        assert labels["SHARED"] == "In both"
