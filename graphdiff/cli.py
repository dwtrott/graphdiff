"""Command-line interface.

::

    graphdiff compare A.graphml B.graphml --out report.json [--markdown s.md] [--html d.html]
    graphdiff inspect A.graphml
    graphdiff matrix ./graphs/ --metric jaccard_edges --out scores.parquet --workers 8
    graphdiff render A.graphml B.graphml --out diff.html
    graphdiff timeline t0.graphml t1.graphml t2.graphml --html timeline.html
    graphdiff check A.graphml B.graphml -r "jaccard_typed_edges>=0.95"
    graphdiff serve diff.html --port 8080

Every command is offline: ``serve`` binds a stdlib HTTP server to localhost and
serves a file that was already rendered with no external references.
"""

from __future__ import annotations

import http.server
import json
import sys
import webbrowser
from pathlib import Path
from typing import Annotated, Any

import typer

from . import __version__
from .api import compare as _compare
from .api import inspect_graph
from .report.report import SCALAR_METRICS

_ALIGN_HELP = "Node matching: exact | normalized | fuzzy (see align_graphs)."
AlignOpt = Annotated[str, typer.Option("--align", help=_ALIGN_HELP)]
ThresholdOpt = Annotated[float, typer.Option("--align-threshold", help="Min fuzzy match score.")]


def _align_kwargs(align: str, threshold: float) -> dict[str, Any]:
    if align not in ("exact", "normalized", "fuzzy"):
        typer.echo(f"--align must be exact, normalized or fuzzy (got {align!r})", err=True)
        raise typer.Exit(code=2)
    return {"align": align, "align_threshold": threshold}


app = typer.Typer(
    name="graphdiff",
    help="Graph similarity, alignment, and structural diffing.",
    no_args_is_help=True,
    add_completion=False,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"graphdiff {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """Graph similarity, alignment, and structural diffing."""


@app.command()
def compare(
    graph_a: Annotated[Path, typer.Argument(exists=True, help="First graph (any format).")],
    graph_b: Annotated[Path, typer.Argument(exists=True, help="Second graph (any format).")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Report JSON path.")] = None,
    markdown: Annotated[Path | None, typer.Option(help="Also write a markdown summary.")] = None,
    parquet: Annotated[Path | None, typer.Option(help="Also write tidy scores as Parquet.")] = None,
    html: Annotated[Path | None, typer.Option(help="Also write the visual diff.")] = None,
    union_dir: Annotated[
        Path | None, typer.Option(help="Also write the union diff graph as Parquet here.")
    ] = None,
    undirected: Annotated[
        bool, typer.Option(help="Treat edge lists / Parquet as undirected.")
    ] = False,
    weight: Annotated[str, typer.Option(help="Edge attribute for weight agreement.")] = "weight",
    top: Annotated[int, typer.Option(help="Rows in the most-changed-nodes table.")] = 25,
    align: AlignOpt = "exact",
    align_threshold: ThresholdOpt = 0.6,
) -> None:
    """Compare two graphs and write a report."""
    from .io import read_graph

    a = read_graph(graph_a, directed=not undirected)
    b = read_graph(graph_b, directed=not undirected)
    report = _compare(
        a, b, weight_attribute=weight, top_n=top, **_align_kwargs(align, align_threshold)
    )

    if out is not None:
        report.to_json(out)
        typer.echo(f"report  → {out}", err=True)
    if markdown is not None:
        report.write_markdown(markdown, top_n=top)
        typer.echo(f"summary → {markdown}", err=True)
    if parquet is not None:
        report.to_parquet(parquet)
        typer.echo(f"scores  → {parquet}", err=True)
    if union_dir is not None:
        report.write_union_parquet(union_dir)
        typer.echo(f"union   → {union_dir}/", err=True)
    if html is not None:
        from .viewer import write_html

        write_html(report, html)
        typer.echo(f"viewer  → {html}", err=True)

    if out is None and markdown is None and parquet is None and html is None:
        typer.echo(report.to_markdown(top_n=top))


@app.command()
def inspect(
    graph: Annotated[Path, typer.Argument(exists=True, help="Graph file (any format).")],
    undirected: Annotated[
        bool, typer.Option(help="Treat edge lists / Parquet as undirected.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of text.")] = False,
) -> None:
    """Node/edge counts, edge types, density, and degree statistics."""
    from .io import read_graph

    summary = inspect_graph(read_graph(graph, directed=not undirected))
    if as_json:
        typer.echo(json.dumps(summary, indent=2, default=str))
        return
    deg = summary["degree"]
    typer.echo(f"{summary['name']}  ({'directed' if summary['directed'] else 'undirected'})")
    typer.echo(f"  nodes       {summary['n_nodes']:,}")
    typer.echo(f"  edges       {summary['n_edges']:,}")
    typer.echo(f"  edge types  {summary['n_edge_types']}  {', '.join(summary['edge_types'][:8])}")
    typer.echo(f"  density     {summary['density']:.6f}")
    typer.echo(
        f"  degree      min {deg['min']:.0f} · median {deg['median']:.0f} · "
        f"mean {deg['mean']:.2f} · max {deg['max']:.0f}"
    )
    if summary["node_attributes"]:
        typer.echo(f"  node attrs  {', '.join(summary['node_attributes'])}")
    if summary["edge_attributes"]:
        typer.echo(f"  edge attrs  {', '.join(summary['edge_attributes'])}")


@app.command()
def matrix(
    directory: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Folder of graphs.")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output Parquet (or .csv).")],
    metric: Annotated[
        list[str], typer.Option("--metric", "-m", help="Metric name; repeatable.")
    ] = ["jaccard_typed_edges"],  # noqa: B006 - typer needs a concrete default
    workers: Annotated[int | None, typer.Option(help="Pool size; default = all CPUs.")] = None,
    include_self: Annotated[bool, typer.Option(help="Also score each graph with itself.")] = False,
) -> None:
    """Score every pair of graphs in a directory into a tidy long-format table."""
    from .batch.matrix import _stderr_progress, all_pairs, discover_graphs

    for m in metric:
        if m not in SCALAR_METRICS:
            typer.echo(
                f"unknown metric {m!r}. Choose from:\n  " + "\n  ".join(SCALAR_METRICS), err=True
            )
            raise typer.Exit(code=2)
    paths = discover_graphs(directory)
    if len(paths) < 2:
        typer.echo(f"need at least two graphs in {directory}, found {len(paths)}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"{len(paths)} graphs → {len(paths) * (len(paths) - 1) // 2} pairs", err=True)

    frame = all_pairs(
        paths, metrics=metric, workers=workers, progress=_stderr_progress, include_self=include_self
    )
    if out.suffix.lower() == ".csv":
        frame.to_csv(out, index=False)
    else:
        frame.to_parquet(out, index=False)
    typer.echo(f"scores → {out}  ({len(frame)} rows)", err=True)


@app.command()
def check(
    graph_a: Annotated[Path, typer.Argument(exists=True, help="First graph (any format).")],
    graph_b: Annotated[Path, typer.Argument(exists=True, help="Second graph (any format).")],
    rule: Annotated[
        list[str] | None,
        typer.Option(
            "--rule",
            "-r",
            help="Threshold, repeatable: jaccard_typed_edges>=0.95, nodes.A_ONLY<=0, "
            "significance.max<=20, ged_similarity.shared>=0.9",
        ),
    ] = None,
    baseline: Annotated[
        Path | None, typer.Option(exists=True, help="A previous report.json to guard against.")
    ] = None,
    drift: Annotated[
        list[str] | None,
        typer.Option(
            help="Quantity that may move at most --tolerance from the baseline; repeatable."
        ),
    ] = None,
    tolerance: Annotated[float, typer.Option(help="Allowed absolute drift per quantity.")] = 0.01,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Also write report JSON.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the verdict as JSON.")] = False,
    undirected: Annotated[bool, typer.Option()] = False,
    weight: Annotated[str, typer.Option(help="Edge attribute for weight agreement.")] = "weight",
    align: AlignOpt = "exact",
    align_threshold: ThresholdOpt = 0.6,
) -> None:
    """Regression gate: exit 1 when any rule fails, 0 when all pass.

    Rules compare a quantity with a number. Quantities are any scalar metric
    (optionally ``.shared``), ``nodes.<STATUS>`` / ``edges.<STATUS>`` counts,
    or ``significance.max`` / ``.mean`` / ``.n_over_X``. With ``--baseline``,
    ``--drift`` names quantities that must stay within ``--tolerance`` of the
    stored report; ``--drift all`` guards every scalar metric.
    """
    from .io import read_graph
    from .report.check import load_baseline, parse_check, run_checks

    rules = rule or []
    drifts = drift or []
    if not rules and not drifts:
        typer.echo("nothing to check: pass at least one --rule or --drift", err=True)
        raise typer.Exit(code=2)
    try:
        checks = [parse_check(r) for r in rules]
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    if drifts and baseline is None:
        typer.echo("--drift needs --baseline", err=True)
        raise typer.Exit(code=2)
    if drifts == ["all"]:
        drifts = list(SCALAR_METRICS)

    a = read_graph(graph_a, directed=not undirected)
    b = read_graph(graph_b, directed=not undirected)
    report = _compare(a, b, weight_attribute=weight, **_align_kwargs(align, align_threshold))
    if out is not None:
        report.to_json(out)
    try:
        verdict = run_checks(
            report,
            checks,
            baseline=load_baseline(baseline) if baseline is not None else None,
            tolerance=tolerance,
            drift=drifts,
        )
    except KeyError as exc:
        typer.echo(str(exc.args[0]), err=True)
        raise typer.Exit(code=2) from None

    if as_json:
        typer.echo(json.dumps(verdict.to_dict(), indent=2))
    else:
        typer.echo(verdict.summary())
    raise typer.Exit(code=0 if verdict.ok else 1)


@app.command()
def plot(
    graph_a: Annotated[Path, typer.Argument(exists=True)],
    graph_b: Annotated[Path, typer.Argument(exists=True)],
    out: Annotated[Path, typer.Option("--out", "-o", help="PNG/SVG/PDF path.")] = Path("diff.png"),
    kind: Annotated[
        str,
        typer.Option(help="overview | clusters | top | degrees | dashboard"),
    ] = "dashboard",
    max_nodes: Annotated[int, typer.Option(help="Cap on drawn nodes.")] = 2000,
    top: Annotated[int, typer.Option(help="Bars in the top-changed chart.")] = 20,
    cluster_by: Annotated[str | None, typer.Option(help="Node attribute to cluster by.")] = None,
    undirected: Annotated[bool, typer.Option()] = False,
    align: AlignOpt = "exact",
    align_threshold: ThresholdOpt = 0.6,
) -> None:
    """Static figure of the diff (needs the [plot] extra: matplotlib)."""
    from .io import read_graph

    try:
        from . import plot as _plot
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        typer.echo(f"{exc}\ninstall with: pip install 'graphdiff[plot]'", err=True)
        raise typer.Exit(code=2) from None

    a = read_graph(graph_a, directed=not undirected)
    b = read_graph(graph_b, directed=not undirected)
    report = _compare(a, b, cluster_by=cluster_by, **_align_kwargs(align, align_threshold))
    kinds = {
        "overview": lambda: _plot.plot_overview(report, max_nodes=max_nodes),
        "clusters": lambda: _plot.plot_clusters(report, max_nodes=max_nodes),
        "top": lambda: _plot.plot_top_changed(report, n=top),
        "degrees": lambda: _plot.plot_degree_distributions(report),
        "dashboard": lambda: _plot.plot_dashboard(report, max_nodes=max_nodes, top=top),
    }
    if kind not in kinds:
        typer.echo(f"unknown --kind {kind!r}; choose from {', '.join(kinds)}", err=True)
        raise typer.Exit(code=2)
    fig = kinds[kind]()
    _plot.save(fig, out)
    typer.echo(f"figure → {out}", err=True)


@app.command()
def timeline(
    graphs: Annotated[list[Path], typer.Argument(exists=True, help="Snapshots, in order.")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Timeline JSON path.")] = None,
    markdown: Annotated[Path | None, typer.Option(help="Also write a markdown summary.")] = None,
    html: Annotated[Path | None, typer.Option(help="Also write the timeline page.")] = None,
    figure: Annotated[
        Path | None, typer.Option(help="Also write a PNG/SVG (needs [plot]).")
    ] = None,
    max_nodes: Annotated[int, typer.Option(help="Cap on drawn nodes per step page.")] = 2000,
    undirected: Annotated[bool, typer.Option()] = False,
    align: AlignOpt = "exact",
    align_threshold: ThresholdOpt = 0.6,
) -> None:
    """Compare a sequence of snapshots: when it changed, where it keeps changing."""
    from .batch.timeline import compare_sequence
    from .io import read_graph

    if len(graphs) < 2:
        typer.echo("timeline needs at least two graphs", err=True)
        raise typer.Exit(code=2)
    loaded = [read_graph(p, directed=not undirected) for p in graphs]
    names = [p.stem for p in graphs]
    tl = compare_sequence(loaded, names=names, **_align_kwargs(align, align_threshold))
    if out is not None:
        tl.to_json(out)
        typer.echo(f"timeline → {out}", err=True)
    if markdown is not None:
        tl.write_markdown(markdown)
        typer.echo(f"summary  → {markdown}", err=True)
    if html is not None:
        from .viewer import write_timeline_html

        write_timeline_html(tl, html, max_nodes=max_nodes)
        typer.echo(f"page     → {html}", err=True)
    if figure is not None:
        try:
            from . import plot as _plot
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            typer.echo(f"{exc}\ninstall with: pip install 'graphdiff[plot]'", err=True)
            raise typer.Exit(code=2) from None
        _plot.save(_plot.plot_timeline(tl), figure)
        typer.echo(f"figure   → {figure}", err=True)
    if out is None and markdown is None and html is None and figure is None:
        typer.echo(tl.to_markdown())


@app.command()
def render(
    graph_a: Annotated[Path, typer.Argument(exists=True)],
    graph_b: Annotated[Path, typer.Argument(exists=True)],
    out: Annotated[Path, typer.Option("--out", "-o", help="HTML output path.")] = Path("diff.html"),
    max_nodes: Annotated[int, typer.Option(help="Cap on drawn nodes.")] = 2000,
    context_hops: Annotated[int, typer.Option(help="Unchanged context around differences.")] = 1,
    cluster_by: Annotated[str | None, typer.Option(help="Node attribute to cluster by.")] = None,
    undirected: Annotated[bool, typer.Option()] = False,
    align: AlignOpt = "exact",
    align_threshold: ThresholdOpt = 0.6,
) -> None:
    """Render the self-contained visual diff for two graphs."""
    from .io import read_graph
    from .viewer import write_html

    a = read_graph(graph_a, directed=not undirected)
    b = read_graph(graph_b, directed=not undirected)
    report = _compare(a, b, **_align_kwargs(align, align_threshold))
    write_html(report, out, max_nodes=max_nodes, context_hops=context_hops, cluster_by=cluster_by)
    typer.echo(f"viewer → {out}", err=True)


@app.command(name="app")
def app_command(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Folder of graphs; results go under .graphdiff/."),
    ] = Path("."),
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1.")] = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
    workers: Annotated[int, typer.Option(help="Comparisons that may run at once.")] = 2,
) -> None:
    """Start the local web app: upload or pick graphs, compare, browse the results.

    Bound to 127.0.0.1, serves one inlined page and a JSON API, makes no
    outbound requests. Needs the [viewer] extra (fastapi, uvicorn).
    """
    try:
        import uvicorn

        from .server import create_app
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        typer.echo(f"{exc}\ninstall with: pip install 'graphdiff[viewer]'", err=True)
        raise typer.Exit(code=2) from None

    application = create_app(workspace, workers=workers)
    url = f"http://127.0.0.1:{port}/"
    typer.echo(
        f"graphdiff app on {url}  (workspace {workspace.resolve()}; Ctrl+C to stop)", err=True
    )
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(application, host="127.0.0.1", port=port, log_level="warning")


@app.command()
def serve(
    html: Annotated[Path, typer.Argument(exists=True, help="A rendered diff.html.")],
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1.")] = 8080,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
) -> None:
    """Serve a rendered viewer on localhost. Bound to 127.0.0.1; makes no outbound requests."""
    directory = html.resolve().parent
    name = html.name

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, fmt: str, *args) -> None:  # type: ignore[no-untyped-def]
            sys.stderr.write(f"  {fmt % args}\n")

    with http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        url = f"http://127.0.0.1:{port}/{name}"
        typer.echo(f"serving {url}  (Ctrl+C to stop)", err=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            typer.echo("\nstopped", err=True)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
