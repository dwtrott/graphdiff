"""Graph loading with format auto-detection.

Supported formats: GraphML, node-link JSON, edge-list CSV/TSV, and Parquet.
:func:`read_graph` dispatches on file extension, falling back to content
sniffing for unknown or missing extensions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..core.graph import DuplicatePolicy, PropertyGraph
from .graphml import read_graphml, write_graphml
from .tabular import (
    read_edgelist,
    read_node_link_json,
    read_parquet,
    write_edgelist,
    write_node_link_json,
    write_parquet,
)

__all__ = [
    "Format",
    "detect_format",
    "read_edgelist",
    "read_graph",
    "read_graphml",
    "read_node_link_json",
    "read_parquet",
    "write_edgelist",
    "write_graph",
    "write_graphml",
    "write_node_link_json",
    "write_parquet",
]

Format = Literal["graphml", "json", "edgelist", "parquet"]

_EXTENSIONS: dict[str, Format] = {
    ".graphml": "graphml",
    ".xml": "graphml",
    ".gml": "graphml",
    ".json": "json",
    ".jsonl": "json",
    ".csv": "edgelist",
    ".tsv": "edgelist",
    ".txt": "edgelist",
    ".parquet": "parquet",
    ".pq": "parquet",
}


class UnknownFormatError(ValueError):
    """Raised when a file's graph format cannot be determined."""


def detect_format(path: str | Path) -> Format:
    """Infer the graph format of ``path`` from its extension, then its content.

    Raises
    ------
    UnknownFormatError
        If neither the extension nor a short content probe is conclusive.
    """
    path = Path(path)
    if path.is_dir():
        if (path / "edges.parquet").exists():
            return "parquet"
        raise UnknownFormatError(f"{path} is a directory without edges.parquet")

    suffix = path.suffix.lower()
    if suffix in _EXTENSIONS:
        return _EXTENSIONS[suffix]

    with path.open("rb") as handle:
        head = handle.read(2048)
    if head.startswith(b"PAR1"):
        return "parquet"
    text = head.decode("utf-8", errors="replace").lstrip()
    if text.startswith("<"):
        return "graphml"
    if text.startswith(("{", "[")):
        return "json"
    lines = text.splitlines()
    if lines and any(sep in lines[0] for sep in (",", "\t", ";", "|")):
        return "edgelist"
    raise UnknownFormatError(f"cannot determine graph format for {path}")


def read_graph(
    path: str | Path,
    *,
    fmt: Format | None = None,
    directed: bool = True,
    name: str | None = None,
    edge_type_attr: str | None = None,
    nodes_path: str | Path | None = None,
    on_duplicate_edge: DuplicatePolicy = "error",
) -> PropertyGraph:
    """Load a graph, auto-detecting the format unless ``fmt`` is given.

    Notes
    -----
    ``directed`` is honored for formats that do not encode directedness
    (edge lists, Parquet). GraphML and node-link JSON carry it in the file and
    the file wins.
    """
    path = Path(path)
    resolved = fmt or detect_format(path)
    if resolved == "graphml":
        return read_graphml(
            path, edge_type_attr=edge_type_attr, name=name, on_duplicate_edge=on_duplicate_edge
        )
    if resolved == "json":
        return read_node_link_json(path, name=name, on_duplicate_edge=on_duplicate_edge)
    if resolved == "parquet":
        return read_parquet(path, directed=directed, name=name, on_duplicate_edge=on_duplicate_edge)
    return read_edgelist(
        path,
        directed=directed,
        name=name,
        nodes_path=nodes_path,
        on_duplicate_edge=on_duplicate_edge,
    )


def write_graph(graph: PropertyGraph, path: str | Path, *, fmt: Format | None = None) -> Path:
    """Write ``graph``, choosing the writer by extension unless ``fmt`` is given."""
    path = Path(path)
    resolved = fmt or _EXTENSIONS.get(path.suffix.lower(), "parquet")
    if resolved == "graphml":
        return write_graphml(graph, path)
    if resolved == "json":
        return write_node_link_json(graph, path)
    if resolved == "parquet":
        return write_parquet(graph, path)
    return write_edgelist(graph, path, delimiter="\t" if path.suffix == ".tsv" else ",")
