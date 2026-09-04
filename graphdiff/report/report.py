"""The :class:`ComparisonReport` — everything one graph-pair comparison produced.

A report serializes to JSON (full detail) and to a tidy long-format Parquet
table (one row per scalar score), and renders as markdown or as an HTML summary
inside Jupyter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .._types import STATUS_ORDER
from ..core.union import UnionDiffGraph
from ..metrics.base import DualScore, to_jsonable
from ..metrics.ged import GEDResult
from ..metrics.neighborhood import NeighborhoodDeltaResult
from ..metrics.settheoretic import SetTheoreticResult
from ..metrics.weights import WeightAgreementResult

__all__ = ["SCALAR_METRICS", "ComparisonReport"]

#: Canonical scalar metric names, in report order. The CLI's ``--metric`` flag
#: and the all-pairs matrix accept exactly these.
SCALAR_METRICS: tuple[str, ...] = (
    "jaccard_nodes",
    "overlap_nodes",
    "dice_nodes",
    "jaccard_edges",
    "overlap_edges",
    "dice_edges",
    "jaccard_typed_edges",
    "overlap_typed_edges",
    "dice_typed_edges",
    "ged_cost",
    "ged_normalized_distance",
    "ged_similarity",
    "weight_spearman",
    "weight_pearson",
    "weight_mean_abs_diff",
    "neighborhood_mean_jaccard",
    "neighborhood_median_jaccard",
)


def _set_scalars(result: SetTheoreticResult) -> dict[str, float]:
    return {
        "jaccard_nodes": result.nodes.jaccard,
        "overlap_nodes": result.nodes.overlap,
        "dice_nodes": result.nodes.dice,
        "jaccard_edges": result.edges.jaccard,
        "overlap_edges": result.edges.overlap,
        "dice_edges": result.edges.dice,
        "jaccard_typed_edges": result.typed_edges.jaccard,
        "overlap_typed_edges": result.typed_edges.overlap,
        "dice_typed_edges": result.typed_edges.dice,
    }


def _ged_scalars(result: GEDResult) -> dict[str, float]:
    return {
        "ged_cost": result.cost,
        "ged_normalized_distance": result.normalized_distance,
        "ged_similarity": result.similarity,
    }


def _weight_scalars(result: WeightAgreementResult) -> dict[str, float | None]:
    return {
        "weight_spearman": result.spearman,
        "weight_pearson": result.pearson,
        "weight_mean_abs_diff": result.mean_abs_diff,
    }


@dataclass
class ComparisonReport:
    """Result of comparing one pair of graphs.

    Attributes
    ----------
    set_theoretic, ged, weight_agreement:
        Each a :class:`~graphdiff.metrics.base.DualScore` holding the metric
        computed over the full union (``raw``) and over the shared-node induced
        subgraph (``shared``).
    neighborhood:
        Per-node neighborhood delta; inherently restricted to shared nodes, so
        it has no dual form.
    union:
        The union diff graph the report was derived from. Not serialized to JSON
        by default — write it separately with
        :meth:`~graphdiff.report.ComparisonReport.write_union_parquet`.
    """

    name_a: str
    name_b: str
    directed: bool
    node_status_counts: dict[str, int]
    edge_status_counts: dict[str, int]
    graph_summary: dict[str, Any]
    set_theoretic: DualScore[SetTheoreticResult]
    ged: DualScore[GEDResult]
    weight_agreement: DualScore[WeightAgreementResult]
    neighborhood: NeighborhoodDeltaResult
    union: UnionDiffGraph | None = field(default=None, repr=False, compare=False)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    graphdiff_version: str = "0.1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- scalar view

    def scalar_scores(self) -> dict[str, dict[str, float | None]]:
        """Canonical scalar metrics as ``{name: {"raw": ..., "shared": ...}}``."""
        raw: dict[str, float | None] = {
            **_set_scalars(self.set_theoretic.raw),
            **_ged_scalars(self.ged.raw),
            **_weight_scalars(self.weight_agreement.raw),
            "neighborhood_mean_jaccard": self.neighborhood.mean_jaccard,
            "neighborhood_median_jaccard": self.neighborhood.median_jaccard,
        }
        shared: dict[str, float | None] = {
            **_set_scalars(self.set_theoretic.shared),
            **_ged_scalars(self.ged.shared),
            **_weight_scalars(self.weight_agreement.shared),
            "neighborhood_mean_jaccard": self.neighborhood.mean_jaccard,
            "neighborhood_median_jaccard": self.neighborhood.median_jaccard,
        }
        return {name: {"raw": raw.get(name), "shared": shared.get(name)} for name in SCALAR_METRICS}

    def score(self, metric: str, *, shared_subgraph: bool = False) -> float | None:
        """Look up one canonical scalar score by name."""
        if metric not in SCALAR_METRICS:
            raise KeyError(
                f"unknown metric {metric!r}; expected one of {', '.join(SCALAR_METRICS)}"
            )
        entry = self.scalar_scores()[metric]
        return entry["shared" if shared_subgraph else "raw"]

    def to_frame(self) -> pd.DataFrame:
        """Tidy long-format table: one row per scalar metric."""
        rows = [
            {
                "graph_a": self.name_a,
                "graph_b": self.name_b,
                "metric": name,
                "raw_score": values["raw"],
                "shared_subgraph_score": values["shared"],
            }
            for name, values in self.scalar_scores().items()
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ serialization

    def to_dict(self, *, include_top_changed: int = 25) -> dict[str, Any]:
        """Full report as JSON-safe plain Python containers."""
        return {
            "graphdiff_version": self.graphdiff_version,
            "created_at": self.created_at,
            "graph_a": self.name_a,
            "graph_b": self.name_b,
            "directed": self.directed,
            "graph_summary": to_jsonable(self.graph_summary),
            "node_status_counts": self.node_status_counts,
            "edge_status_counts": self.edge_status_counts,
            "metrics": {
                "set_theoretic": self.set_theoretic.to_dict(),
                "graph_edit_distance": self.ged.to_dict(),
                "weight_agreement": self.weight_agreement.to_dict(),
                "neighborhood_delta": {
                    "mean_jaccard": to_jsonable(self.neighborhood.mean_jaccard),
                    "median_jaccard": to_jsonable(self.neighborhood.median_jaccard),
                    "n_nodes_compared": self.neighborhood.n_nodes_compared,
                    "n_unchanged": self.neighborhood.n_unchanged,
                    "direction": self.neighborhood.direction,
                    "top_changed": to_jsonable(self.neighborhood.top_changed(include_top_changed)),
                },
            },
            # to_jsonable turns NaN into null; bare NaN is not valid JSON.
            "scalar_scores": to_jsonable(self.scalar_scores()),
            "metadata": to_jsonable(self.metadata),
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize to JSON, optionally writing to ``path``."""
        text = json.dumps(self.to_dict(), indent=indent, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_parquet(self, path: str | Path) -> Path:
        """Write the tidy scalar-score table to Parquet."""
        path = Path(path)
        self.to_frame().to_parquet(path, index=False)
        return path

    def write_union_parquet(self, directory: str | Path) -> Path:
        """Write the union diff graph's node and edge tables to Parquet."""
        if self.union is None:
            raise ValueError("report carries no union diff graph")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        nodes = self.union.nodes.copy()
        edges = self.union.edges.copy()
        for frame in (nodes, edges):
            if "changed_attrs" in frame.columns:
                frame["changed_attrs"] = frame["changed_attrs"].map(
                    lambda v: None if v is None else json.dumps(v, default=str)
                )
        nodes.assign(status=nodes["status"].astype(str)).reset_index().to_parquet(
            directory / "union_nodes.parquet", index=False
        )
        edges.assign(status=edges["status"].astype(str)).to_parquet(
            directory / "union_edges.parquet", index=False
        )
        return directory

    # ---------------------------------------------------------------- rendering

    def to_markdown(self, *, top_n: int = 10) -> str:
        """A compact markdown summary suitable for a PR comment or a ticket."""
        n, e = self.node_status_counts, self.edge_status_counts
        scores = self.scalar_scores()

        def fmt(value: float | None) -> str:
            return "—" if value is None else f"{value:.4f}"

        lines = [
            f"# graphdiff: {self.name_a} vs {self.name_b}",
            "",
            f"*{'Directed' if self.directed else 'Undirected'} graphs, compared {self.created_at}*",
            "",
            "## Composition",
            "",
            "| Status | Nodes | Edges |",
            "| --- | ---: | ---: |",
        ]
        lines += [f"| {s} | {n[s]:,} | {e[s]:,} |" for s in STATUS_ORDER]
        lines += [
            f"| **total** | **{sum(n.values()):,}** | **{sum(e.values()):,}** |",
            "",
            "## Scores",
            "",
            "| Metric | Raw | Shared-subgraph |",
            "| --- | ---: | ---: |",
        ]
        lines += [
            f"| {name} | {fmt(v['raw'])} | {fmt(v['shared'])} |" for name, v in scores.items()
        ]

        top = self.neighborhood.top_changed(top_n)
        if len(top):
            lines += [
                "",
                f"## Most-changed nodes (top {len(top)})",
                "",
                "| Node | Neighborhood Jaccard | Neighbors A | Neighbors B | Added | Removed |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            lines += [
                f"| {row['label']} | {row['jaccard']:.4f} | {row['n_neighbors_a']} | "
                f"{row['n_neighbors_b']} | {row['n_added']} | {row['n_removed']} |"
                for row in top.to_dict("records")
            ]
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path, *, top_n: int = 10) -> Path:
        """Write :meth:`to_markdown` to ``path``."""
        path = Path(path)
        path.write_text(self.to_markdown(top_n=top_n), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        """Compact HTML summary for notebook display."""
        scores = self.scalar_scores()

        def fmt(value: float | None) -> str:
            return "&mdash;" if value is None else f"{value:.4f}"

        status_rows = "".join(
            f"<tr><td>{s}</td><td style='text-align:right'>{self.node_status_counts[s]:,}</td>"
            f"<td style='text-align:right'>{self.edge_status_counts[s]:,}</td></tr>"
            for s in STATUS_ORDER
        )
        score_rows = "".join(
            f"<tr><td>{name}</td><td style='text-align:right'>{fmt(v['raw'])}</td>"
            f"<td style='text-align:right'>{fmt(v['shared'])}</td></tr>"
            for name, v in scores.items()
        )
        top = self.neighborhood.top_changed(10)
        top_rows = "".join(
            f"<tr><td>{r['label']}</td><td style='text-align:right'>{r['jaccard']:.4f}</td>"
            f"<td style='text-align:right'>{r['n_added']}</td>"
            f"<td style='text-align:right'>{r['n_removed']}</td></tr>"
            for r in top.to_dict("records")
        )
        style = "border-collapse:collapse;font-size:12px;margin-right:24px;vertical-align:top"
        cell = "padding:2px 8px;border-bottom:1px solid #eee"
        return f"""
<div style="font-family:system-ui,sans-serif">
  <h3 style="margin:0 0 4px">graphdiff: {self.name_a} vs {self.name_b}</h3>
  <div style="color:#666;font-size:11px;margin-bottom:8px">
    {"directed" if self.directed else "undirected"} &middot; {self.created_at}
  </div>
  <div style="display:flex;flex-wrap:wrap">
    <table style="{style}">
      <thead><tr><th style="{cell}">Status</th><th style="{cell}">Nodes</th>
      <th style="{cell}">Edges</th></tr></thead><tbody>{status_rows}</tbody>
    </table>
    <table style="{style}">
      <thead><tr><th style="{cell}">Metric</th><th style="{cell}">Raw</th>
      <th style="{cell}">Shared</th></tr></thead><tbody>{score_rows}</tbody>
    </table>
    <table style="{style}">
      <thead><tr><th style="{cell}">Most-changed node</th><th style="{cell}">Jaccard</th>
      <th style="{cell}">+</th><th style="{cell}">&minus;</th></tr></thead>
      <tbody>{top_rows}</tbody>
    </table>
  </div>
</div>
"""
