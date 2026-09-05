"""Command-line interface.

::

    graphdiff compare A.graphml B.graphml --out report.json [--markdown s.md] [--html d.html]
    graphdiff inspect A.graphml
    graphdiff matrix ./graphs/ --metric jaccard_edges --out scores.parquet --workers 8
    graphdiff render A.graphml B.graphml --out diff.html
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
from typing import Annotated

import typer

from . import __version__
from .api import compare as _compare
from .api import inspect_graph
from .report.report import SCALAR_METRICS

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
) -> None:
    """Compare two graphs and write a report."""
    from .io import read_graph

    a = read_graph(graph_a, directed=not undirected)
    b = read_graph(graph_b, directed=not undirected)
    report = _compare(a, b, weight_attribute=weight, top_n=top)

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
def render(
    graph_a: Annotated[Path, typer.Argument(exists=True)],
    graph_b: Annotated[Path, typer.Argument(exists=True)],
    out: Annotated[Path, typer.Option("--out", "-o", help="HTML output path.")] = Path("diff.html"),
    max_nodes: Annotated[int, typer.Option(help="Cap on drawn nodes.")] = 2000,
    context_hops: Annotated[int, typer.Option(help="Unchanged context around differences.")] = 1,
    cluster_by: Annotated[str | None, typer.Option(help="Node attribute to cluster by.")] = None,
    undirected: Annotated[bool, typer.Option()] = False,
) -> None:
    """Render the self-contained visual diff for two graphs."""
    from .io import read_graph
    from .viewer import write_html

    a = read_graph(graph_a, directed=not undirected)
    b = read_graph(graph_b, directed=not undirected)
    report = _compare(a, b)
    write_html(report, out, max_nodes=max_nodes, context_hops=context_hops, cluster_by=cluster_by)
    typer.echo(f"viewer → {out}", err=True)


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
