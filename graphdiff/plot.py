"""Static figures — the viewer's encoding, exported for slides, papers and CI.

Everything here needs matplotlib (``pip install 'graphdiff[plot]'``) and
nothing else: the same layout, palette and shapes as the HTML viewer, drawn
with the Agg backend so it works headless and offline.

The figures:

* :func:`plot_overview` — the diff map. Nodes shaped and coloured by status,
  sized by significance, the most significant labelled; unchanged context in
  grey underneath.
* :func:`plot_clusters` — one mark per cluster, area = size, shade = change
  density; the "where" at a glance.
* :func:`plot_top_changed` — horizontal bar chart of the most significant
  nodes, bars split into removed / added / reweighted.
* :func:`plot_degree_distributions` — log-log degree distributions of A and B
  overlaid; a global-shape check that needs no alignment at all.
* :func:`plot_similarity_matrix` — heatmap from an all-pairs
  :func:`~graphdiff.batch.all_pairs` table.
* :func:`plot_dashboard` — overview, clusters, top-changed and the findings in
  one page.
* :func:`plot_timeline` — churn and similarity per step over a node x step
  significance heatmap, for a :func:`~graphdiff.batch.compare_sequence` result.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._types import SOURCE, STATUS, STATUS_ORDER, TARGET, Status
from .metrics.cluster import ClusterMap, cluster_union
from .metrics.significance import node_significance
from .report.report import ComparisonReport
from .viewer.focus import select_focus
from .viewer.layout import LayoutParams, force_directed_layout
from .viewer.theme import CHANGE_RAMP, LIGHT, SHAPES, status_labels

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_clusters",
    "plot_dashboard",
    "plot_degree_distributions",
    "plot_overview",
    "plot_similarity_matrix",
    "plot_timeline",
    "plot_top_changed",
    "save",
]

_MARKERS = {"circle": "o", "square": "s", "triangle": "^", "diamond": "D"}
_STATUS_MARKER = {s: _MARKERS[SHAPES[s]] for s in STATUS_ORDER}
_CHANGE_STATUSES = (Status.CHANGED.value, Status.A_ONLY.value, Status.B_ONLY.value)


def _mpl() -> Any:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("graphdiff.plot needs matplotlib: pip install 'graphdiff[plot]'") from exc
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _style(ax: Axes) -> None:
    ax.set_facecolor(LIGHT["surface"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _union(report: ComparisonReport) -> Any:
    if report.union is None:
        raise ValueError(
            "report carries no union diff graph; re-run compare() with keep_union=True"
        )
    return report.union


# --------------------------------------------------------------------------- overview


def _geometry(
    report: ComparisonReport, *, max_nodes: int, context_hops: int, layout: LayoutParams | None
) -> tuple[pd.Index, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    """Focus selection, 2D layout and per-node significance for drawing."""
    union = _union(report)
    focus = select_focus(union, max_nodes=max_nodes, context_hops=context_hops)
    labels = focus.labels
    pos = pd.Series(np.arange(len(labels)), index=labels)
    if len(focus.edges):
        src = pos.reindex(focus.edges[SOURCE]).to_numpy()
        dst = pos.reindex(focus.edges[TARGET]).to_numpy()
        keep = ~(pd.isna(src) | pd.isna(dst))
        pairs = np.stack([src[keep].astype(np.int64), dst[keep].astype(np.int64)], axis=1)
        edges = focus.edges.loc[keep]
    else:
        pairs = np.zeros((0, 2), dtype=np.int64)
        edges = focus.edges
    coords = force_directed_layout(len(labels), pairs, params=layout or LayoutParams())
    sig = report.significance if report.significance is not None else node_significance(union)
    sig_series = sig.table.set_index("label")["significance"].reindex(labels).fillna(0.0)
    status = union.nodes.reindex(labels)[STATUS].astype(str)
    return labels, coords, edges, pairs, sig_series.to_numpy(), status


def plot_overview(
    report: ComparisonReport,
    *,
    ax: Axes | None = None,
    max_nodes: int = 2000,
    context_hops: int = 1,
    layout: LayoutParams | None = None,
    label_top: int = 8,
    title: str | None = None,
) -> Figure:
    """The diff map: status as shape+colour, significance as size and label."""
    plt = _mpl()
    fig = ax.figure if ax is not None else plt.figure(figsize=(10, 8), facecolor=LIGHT["surface"])
    ax = ax or fig.add_subplot(111)
    _style(ax)
    labels, coords, edges, pairs, sig, status = _geometry(
        report, max_nodes=max_nodes, context_hops=context_hops, layout=layout
    )
    union = _union(report)

    if len(pairs):
        e_status = edges[STATUS].astype(str).to_numpy()
        for s, z in (("SHARED", 1), ("CHANGED", 2), ("A_ONLY", 3), ("B_ONLY", 3)):
            sel = e_status == s
            if not sel.any():
                continue
            segs = coords[pairs[sel]]
            from matplotlib.collections import LineCollection

            ax.add_collection(
                LineCollection(
                    segs,
                    colors=LIGHT[s],
                    linewidths=0.5 if s == "SHARED" else 1.1,
                    alpha=0.35 if s == "SHARED" else 0.85,
                    linestyles="--" if s == "CHANGED" else "-",
                    zorder=z,
                )
            )

    smax = float(sig.max()) if len(sig) and sig.max() > 0 else 1.0
    size = 12 + 220 * np.sqrt(np.clip(sig / smax, 0, 1))
    names = status_labels(union.name_a, union.name_b)
    for s in STATUS_ORDER:
        sel = (status == s).to_numpy()
        if not sel.any():
            continue
        ax.scatter(
            coords[sel, 0],
            coords[sel, 1],
            s=size[sel] if s != "SHARED" else 10,
            c=LIGHT[s],
            marker=_STATUS_MARKER[s],
            edgecolors=LIGHT["surface"],
            linewidths=0.6,
            alpha=0.55 if s == "SHARED" else 0.95,
            zorder=4 if s == "SHARED" else 6,
            label=f"{names[s]} ({int(sel.sum()):,})",
        )
    # Halo the most significant, then label them.
    order = np.argsort(-sig)
    top = [i for i in order[:label_top] if sig[i] > 0]
    if top:
        ax.scatter(
            coords[top, 0],
            coords[top, 1],
            s=size[top] * 2.2,
            facecolors="none",
            edgecolors=LIGHT["ink"],
            linewidths=0.9,
            alpha=0.6,
            zorder=5,
        )
        for i in top:
            ax.annotate(
                str(labels[i]),
                (coords[i, 0], coords[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=LIGHT["ink"],
                zorder=8,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": LIGHT["surface"],
                    "ec": "none",
                    "alpha": 0.8,
                },
            )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=0.9)
    ax.set_title(
        title or f"{union.name_a} → {union.name_b}: differences (size = significance)",
        fontsize=11,
        loc="left",
        color=LIGHT["ink"],
    )
    return fig


# --------------------------------------------------------------------------- clusters


def _ramp_color(density: float) -> str:
    ramp = CHANGE_RAMP["light"]
    return ramp[min(len(ramp) - 1, int(density * len(ramp)))]


def plot_clusters(
    report: ComparisonReport,
    *,
    ax: Axes | None = None,
    cluster_by: str | None = None,
    max_clusters: int = 60,
    max_nodes: int = 2000,
    layout: LayoutParams | None = None,
    label_top: int = 8,
) -> Figure:
    """One mark per cluster: area = size, shade = share of it that changed."""
    plt = _mpl()
    fig = ax.figure if ax is not None else plt.figure(figsize=(9, 7), facecolor=LIGHT["surface"])
    ax = ax or fig.add_subplot(111)
    _style(ax)
    union = _union(report)
    cmap: ClusterMap = (
        report.clusters
        if (report.clusters is not None and cluster_by is None)
        else cluster_union(union, attribute=cluster_by, max_clusters=max_clusters)
    )
    n = len(cmap.clusters)
    links = (
        np.array([[i, j] for i, j, _ in cmap.links], dtype=np.int64)
        if cmap.links
        else np.zeros((0, 2), dtype=np.int64)
    )
    coords = force_directed_layout(n, links, params=layout or LayoutParams())
    if len(links):
        from matplotlib.collections import LineCollection

        weights = np.array([sum(c.values()) for _, _, c in cmap.links], dtype=float)
        lw = 0.4 + 2.5 * weights / max(weights.max(), 1.0)
        ax.add_collection(
            LineCollection(coords[links], colors=LIGHT["line"], linewidths=lw, zorder=1)
        )
    sizes = np.array([c.size for c in cmap.clusters], dtype=float)
    area = 60 + 1400 * sizes / max(sizes.max(), 1.0)
    colors = [_ramp_color(c.change_density) for c in cmap.clusters]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=area,
        c=colors,
        edgecolors=LIGHT["inkSecondary"],
        linewidths=0.6,
        zorder=3,
    )
    order = np.argsort([-c.change_density for c in cmap.clusters])
    for i in order[:label_top]:
        c = cmap.clusters[i]
        if c.change_density < 0.05:  # the quiet ones are the point of the picture, not its labels
            break
        ax.annotate(
            f"{c.name}\n{c.change_density * 100:.0f}% changed · {c.size:,} nodes",
            (coords[i, 0], coords[i, 1]),
            xytext=(0, -np.sqrt(area[i]) / 2 - 4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.5,
            color=LIGHT["ink"],
            zorder=5,
        )
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.12, 1.06)
    ax.set_aspect("equal")
    # Legend for the ramp.
    ramp = CHANGE_RAMP["light"]
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=ramp[k], mec=LIGHT["inkSecondary"], mew=0.5)
        for k in range(len(ramp))
    ]
    steps = [f"{k * 100 // len(ramp)}-{(k + 1) * 100 // len(ramp)}%" for k in range(len(ramp))]
    ax.legend(
        handles,
        steps,
        title="share changed",
        loc="lower left",
        fontsize=7.5,
        title_fontsize=8,
        frameon=False,
        ncol=2,
    )
    ax.set_title(
        f"Where the change is: {n} clusters ({cmap.method}); area = size",
        fontsize=11,
        loc="left",
        color=LIGHT["ink"],
    )
    return fig


# --------------------------------------------------------------------------- top changed


def plot_top_changed(report: ComparisonReport, *, ax: Axes | None = None, n: int = 20) -> Figure:
    """Horizontal bars of the most significant nodes, split by what happened."""
    plt = _mpl()
    fig = (
        ax.figure
        if ax is not None
        else plt.figure(figsize=(8, 0.35 * n + 1.6), facecolor=LIGHT["surface"])
    )
    ax = ax or fig.add_subplot(111)
    ax.set_facecolor(LIGHT["surface"])
    top = report.top_changed(n)
    if "significance" not in top.columns or not len(top):
        ax.text(0.5, 0.5, "no significance table", ha="center", va="center")
        _style(ax)
        return fig
    top = top.iloc[::-1]
    y = np.arange(len(top))
    total = (top["n_removed"] + top["n_added"] + 0.5 * top["n_changed"]).to_numpy().astype(float)
    scale = top["significance"].to_numpy() / np.maximum(total, 1e-9)
    left = np.zeros(len(top))
    union = _union(report)
    names = status_labels(union.name_a, union.name_b)
    for col, s, legend in (
        ("n_removed", "A_ONLY", f"edges {names['A_ONLY'].lower()}"),
        ("n_added", "B_ONLY", f"edges {names['B_ONLY'].lower()}"),
        ("n_changed", "CHANGED", "edges reweighted"),
    ):
        w = top[col].to_numpy().astype(float) * (0.5 if col == "n_changed" else 1.0) * scale
        ax.barh(y, w, left=left, color=LIGHT[s], label=legend, height=0.7)
        left = left + w
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r.label}  {_glyph(str(r.status))}" for r in top.itertuples()], fontsize=8
    )
    ax.set_xlabel(
        "significance  (-log10 p · log2 magnitude)", fontsize=8, color=LIGHT["inkSecondary"]
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=LIGHT["inkSecondary"], labelsize=8)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.set_title("Most significant changes", fontsize=11, loc="left", color=LIGHT["ink"])
    return fig


def _glyph(status: str) -> str:
    return {"A_ONLY": "■", "B_ONLY": "▲", "CHANGED": "◆", "SHARED": "●"}.get(status, "")


# --------------------------------------------------------------------------- degrees


def plot_degree_distributions(report: ComparisonReport, *, ax: Axes | None = None) -> Figure:
    """Log-log degree distributions of A and B, overlaid."""
    plt = _mpl()
    fig = ax.figure if ax is not None else plt.figure(figsize=(6, 4.5), facecolor=LIGHT["surface"])
    ax = ax or fig.add_subplot(111)
    ax.set_facecolor(LIGHT["surface"])
    union = _union(report)
    for graph, s in ((union.graph_a, "A_ONLY"), (union.graph_b, "B_ONLY")):
        deg = graph.degrees().to_numpy()
        deg = deg[deg > 0]
        if not len(deg):
            continue
        values, counts = np.unique(deg, return_counts=True)
        ax.plot(
            values,
            counts / counts.sum(),
            marker=_STATUS_MARKER[s],
            ms=4,
            lw=1,
            color=LIGHT[s],
            alpha=0.85,
            label=f"{graph.name} (n={graph.n_nodes:,}, m={graph.n_edges:,})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree", fontsize=8, color=LIGHT["inkSecondary"])
    ax.set_ylabel("fraction of nodes", fontsize=8, color=LIGHT["inkSecondary"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=LIGHT["inkSecondary"], labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Degree distributions", fontsize=11, loc="left", color=LIGHT["ink"])
    return fig


# --------------------------------------------------------------------------- matrix


def plot_similarity_matrix(
    frame: pd.DataFrame,
    *,
    metric: str | None = None,
    value: str = "raw_score",
    ax: Axes | None = None,
    order: Sequence[str] | None = None,
    annotate: bool = True,
) -> Figure:
    """Heatmap of an all-pairs score table (from :func:`graphdiff.batch.all_pairs`).

    ``frame`` needs ``graph_a``, ``graph_b``, ``metric`` and the ``value`` column;
    the matrix is symmetrized and the diagonal filled with 1.
    """
    plt = _mpl()
    if metric is None:
        metrics = frame["metric"].unique()
        if len(metrics) != 1:
            raise ValueError(f"table has several metrics {list(metrics)}; pass metric=")
        metric = str(metrics[0])
    sub = frame[frame["metric"] == metric]
    names = list(order) if order is not None else sorted(set(sub["graph_a"]) | set(sub["graph_b"]))
    idx = {n: i for i, n in enumerate(names)}
    mat = np.full((len(names), len(names)), np.nan)
    np.fill_diagonal(mat, 1.0)
    for r in sub.itertuples():
        i, j = idx[r.graph_a], idx[r.graph_b]
        v = getattr(r, value)
        mat[i, j] = mat[j, i] = v
    fig = (
        ax.figure
        if ax is not None
        else plt.figure(
            figsize=(1.1 + 0.55 * len(names), 0.9 + 0.55 * len(names)), facecolor=LIGHT["surface"]
        )
    )
    ax = ax or fig.add_subplot(111)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("gd_violet", CHANGE_RAMP["light"][::-1])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    if annotate and len(names) <= 20:
        for i in range(len(names)):
            for j in range(len(names)):
                if not np.isnan(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=LIGHT["ink"] if mat[i, j] < 0.6 else "#ffffff",
                    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(metric, fontsize=8)
    ax.set_title(f"All-pairs {metric}", fontsize=11, loc="left", color=LIGHT["ink"])
    return fig


# --------------------------------------------------------------------------- dashboard


def plot_dashboard(
    report: ComparisonReport, *, max_nodes: int = 2000, top: int = 15, max_findings: int = 6
) -> Figure:
    """One page: overview map, clusters, top-changed bars, and the findings."""
    plt = _mpl()
    fig = plt.figure(figsize=(16, 11), facecolor=LIGHT["surface"])
    grid = fig.add_gridspec(
        2, 2, height_ratios=[3.2, 2.2], width_ratios=[1.35, 1], hspace=0.22, wspace=0.12
    )
    plot_overview(report, ax=fig.add_subplot(grid[0, 0]), max_nodes=max_nodes)
    plot_clusters(report, ax=fig.add_subplot(grid[0, 1]), max_nodes=max_nodes)
    plot_top_changed(report, ax=fig.add_subplot(grid[1, 0]), n=top)
    ax = fig.add_subplot(grid[1, 1])
    _style(ax)
    ax.set_title("Findings", fontsize=11, loc="left", color=LIGHT["ink"])
    import textwrap

    y = 0.97
    for f in report.findings[:max_findings]:
        marker = {"high": "●", "medium": "●", "low": "○"}[f.severity]
        color = {"high": LIGHT["B_ONLY"], "medium": LIGHT["A_ONLY"], "low": LIGHT["inkMuted"]}[
            f.severity
        ]
        lines = textwrap.wrap(f.text, 78)
        ax.text(0.0, y, marker, transform=ax.transAxes, fontsize=9, color=color, va="top")
        ax.text(
            0.04,
            y,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=8.2,
            va="top",
            color=LIGHT["ink"],
            linespacing=1.35,
        )
        y -= 0.052 * len(lines) + 0.035
        if y < 0.02:
            break
    fig.suptitle(
        f"graphdiff · {report.name_a} vs {report.name_b}",
        x=0.02,
        ha="left",
        fontsize=13,
        color=LIGHT["ink"],
        y=0.985,
    )
    return fig


def save(fig: Figure, path: str | Path, *, dpi: int = 160) -> Path:
    """Write a figure as PNG / SVG / PDF (by suffix) and close it."""
    import matplotlib.pyplot as plt

    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- timeline


def plot_timeline(
    timeline: Any, *, max_nodes: int = 30, figsize: tuple[float, float] = (12, 8)
) -> Figure:
    """Churn per step with similarity, above a node x step significance heatmap.

    ``timeline`` is a :class:`~graphdiff.batch.timeline.TimelineReport`.
    """
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap

    churn = timeline.churn()
    n = len(churn)
    hist = timeline.node_history.head(max_nodes)
    fig = plt.figure(figsize=figsize, facecolor=LIGHT["surface"])
    grid = fig.add_gridspec(2, 1, height_ratios=[1, max(1.2, 0.09 * len(hist))], hspace=0.35)

    ax = fig.add_subplot(grid[0])
    ax.set_facecolor(LIGHT["surface"])
    x = np.arange(n)
    ax.bar(x, churn["edges_removed"], 0.6, color=LIGHT["A_ONLY"], label="edges removed")
    ax.bar(
        x,
        churn["edges_added"],
        0.6,
        bottom=churn["edges_removed"],
        color=LIGHT["B_ONLY"],
        label="edges added",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(timeline.step_labels(), fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("edges changed", fontsize=8, color=LIGHT["inkSecondary"])
    peak = float((churn["edges_removed"] + churn["edges_added"]).max() or 1)
    ax.set_ylim(0, peak * 1.45)  # headroom for the legend and the similarity labels
    ax2 = ax.twinx()
    sims = churn["similarity"].astype(float)
    ax2.plot(x, sims, color=LIGHT["ink"], marker="o", ms=4, lw=1.4, label="similarity")
    for xi, s in zip(x, sims, strict=True):
        if not np.isnan(s):
            ax2.annotate(
                f"{s:.2f}",
                (xi, s),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
            )
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("similarity", fontsize=8, color=LIGHT["inkSecondary"])
    for a in (ax, ax2):
        for spine in ("top",):
            a.spines[spine].set_visible(False)
        a.tick_params(colors=LIGHT["inkSecondary"], labelsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper left", ncol=3)
    ax.set_title("Churn per step", fontsize=11, loc="left", color=LIGHT["ink"])

    axh = fig.add_subplot(grid[1])
    if len(hist):
        cmap = LinearSegmentedColormap.from_list("gd_violet", ["#ffffff", *CHANGE_RAMP["light"]])
        im = axh.imshow(hist.to_numpy(), aspect="auto", cmap=cmap, vmin=0)
        axh.set_yticks(range(len(hist)))
        axh.set_yticklabels([str(i) for i in hist.index], fontsize=7.5)
        axh.set_xticks(range(n))
        axh.set_xticklabels([str(k + 1) for k in range(n)], fontsize=8)
        axh.set_xlabel("step", fontsize=8, color=LIGHT["inkSecondary"])
        fig.colorbar(im, ax=axh, fraction=0.03, pad=0.01).set_label("significance", fontsize=8)
        for spine in axh.spines.values():
            spine.set_visible(False)
        axh.tick_params(colors=LIGHT["inkSecondary"], length=0)
    else:
        _style(axh)
        axh.text(0.5, 0.5, "no notable nodes", ha="center", va="center")
    axh.set_title(
        f"Where and when: the {len(hist)} most significant nodes across the series",
        fontsize=11,
        loc="left",
        color=LIGHT["ink"],
    )
    return fig
