"""Loaders for tabular and JSON graph formats: edge lists, Parquet, node-link JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .._types import ETYPE, LABEL, SOURCE, TARGET
from ..core.graph import DuplicatePolicy, PropertyGraph

__all__ = [
    "read_edgelist",
    "read_node_link_json",
    "read_parquet",
    "write_edgelist",
    "write_node_link_json",
    "write_parquet",
]

_SOURCE_ALIASES = ("source", "src", "from", "u", "node1", "start")
_TARGET_ALIASES = ("target", "dst", "to", "v", "node2", "end")
_TYPE_ALIASES = ("type", "edge_type", "label", "relation", "relationship")


def _sniff_delimiter(path: Path, default: str = ",") -> str:
    sample = path.read_text(encoding="utf-8", errors="replace")[:8192]
    if not sample.strip():
        return default
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;| ").delimiter
    except csv.Error:
        return "\t" if path.suffix.lower() == ".tsv" else default


def _rename_endpoint_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lowered = {str(c).lower(): c for c in frame.columns}
    mapping: dict[str, str] = {}
    for aliases, canonical in (
        (_SOURCE_ALIASES, SOURCE),
        (_TARGET_ALIASES, TARGET),
        (_TYPE_ALIASES, ETYPE),
    ):
        for alias in aliases:
            if alias in lowered and lowered[alias] not in mapping:
                mapping[lowered[alias]] = canonical
                break
    return frame.rename(columns=mapping)


def read_edgelist(
    path: str | Path,
    *,
    delimiter: str | None = None,
    directed: bool = True,
    nodes_path: str | Path | None = None,
    name: str | None = None,
    on_duplicate_edge: DuplicatePolicy = "error",
) -> PropertyGraph:
    """Read a CSV/TSV edge list.

    The first two columns are taken as ``source`` and ``target`` when no
    recognizable header is present; otherwise common aliases (``src``/``dst``,
    ``from``/``to``, ...) are mapped onto the canonical names. A ``type`` column
    (or alias) becomes the edge type; every other column is an edge attribute.

    Parameters
    ----------
    nodes_path:
        Optional companion table of node attributes; must contain a ``label``
        column (or an ``id``/``node`` alias).
    """
    path = Path(path)
    sep = delimiter or _sniff_delimiter(path)
    frame = pd.read_csv(path, sep=sep, dtype=object)
    frame = _rename_endpoint_columns(frame)

    if SOURCE not in frame.columns or TARGET not in frame.columns:
        # Headerless file: re-read treating the first row as data.
        frame = pd.read_csv(path, sep=sep, dtype=object, header=None)
        if frame.shape[1] < 2:
            raise ValueError(f"{path} does not look like an edge list (need >= 2 columns)")
        rename = {0: SOURCE, 1: TARGET}
        if frame.shape[1] >= 3:
            rename[2] = ETYPE
        if frame.shape[1] >= 4:
            rename[3] = "weight"
        frame = frame.rename(columns=rename)
        frame.columns = [str(c) for c in frame.columns]

    if "weight" in frame.columns:
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")

    nodes = None
    if nodes_path is not None:
        nodes = _read_node_table(Path(nodes_path))

    return PropertyGraph.from_edges(
        frame,
        nodes=nodes,
        directed=directed,
        name=name or path.stem,
        on_duplicate_edge=on_duplicate_edge,
    )


def _read_node_table(path: Path) -> pd.DataFrame:
    sep = _sniff_delimiter(path)
    frame = (
        pd.read_csv(path, sep=sep, dtype=object)
        if path.suffix != ".parquet"
        else pd.read_parquet(path)
    )
    lowered = {str(c).lower(): c for c in frame.columns}
    for alias in (LABEL, "id", "node", "name"):
        if alias in lowered:
            return frame.rename(columns={lowered[alias]: LABEL}).set_index(LABEL)
    raise ValueError(f"{path} has no label/id column for node attributes")


def write_edgelist(graph: PropertyGraph, path: str | Path, *, delimiter: str = ",") -> Path:
    """Write the edge table to CSV/TSV. Node-only attributes are not preserved."""
    path = Path(path)
    graph.edges.to_csv(path, sep=delimiter, index=False)
    return path


def read_node_link_json(
    path: str | Path,
    *,
    name: str | None = None,
    on_duplicate_edge: DuplicatePolicy = "error",
) -> PropertyGraph:
    """Read node-link JSON (``{"directed":..., "nodes":[...], "links":[...]}``).

    ``links`` may also be spelled ``edges``; node identity may be ``id``,
    ``label``, or ``name``.
    """
    path = Path(path)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    directed = bool(payload.get("directed", True))

    raw_nodes = payload.get("nodes", [])
    node_records = []
    for entry in raw_nodes:
        record = dict(entry)
        for key in ("id", LABEL, "name"):
            if key in record:
                record[LABEL] = record.pop(key)
                break
        node_records.append(record)
    nodes = pd.DataFrame(node_records)
    nodes = (
        nodes.set_index(LABEL)
        if LABEL in nodes.columns
        else pd.DataFrame(index=pd.Index([], name=LABEL))
    )

    raw_edges = payload.get("links", payload.get("edges", []))
    edges = pd.DataFrame([dict(e) for e in raw_edges])
    if edges.empty:
        edges = pd.DataFrame({SOURCE: pd.Series(dtype="object"), TARGET: pd.Series(dtype="object")})
    edges = _rename_endpoint_columns(edges)

    return PropertyGraph.from_edges(
        edges,
        nodes=nodes if len(nodes) else None,
        directed=directed,
        name=name or path.stem,
        on_duplicate_edge=on_duplicate_edge,
    )


def write_node_link_json(graph: PropertyGraph, path: str | Path) -> Path:
    """Write node-link JSON compatible with :func:`read_node_link_json`."""
    path = Path(path)
    nodes = [
        {LABEL: label, **{k: v for k, v in row.items() if pd.notna(v)}}
        for label, row in graph.nodes.to_dict("index").items()
    ]
    links = [
        {k: v for k, v in row.items() if pd.notna(v)} for row in graph.edges.to_dict("records")
    ]
    payload = {"directed": graph.directed, "nodes": nodes, "links": links}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_parquet(
    path: str | Path,
    *,
    directed: bool = True,
    name: str | None = None,
    on_duplicate_edge: DuplicatePolicy = "error",
) -> PropertyGraph:
    """Read Parquet graph data.

    ``path`` may be a single edges Parquet file, or a directory holding
    ``edges.parquet`` and optionally ``nodes.parquet``.
    """
    path = Path(path)
    if path.is_dir():
        edges = pd.read_parquet(path / "edges.parquet")
        node_file = path / "nodes.parquet"
        nodes = _read_node_table(node_file) if node_file.exists() else None
    else:
        edges = pd.read_parquet(path)
        nodes = None
    edges = _rename_endpoint_columns(edges)
    return PropertyGraph.from_edges(
        edges,
        nodes=nodes,
        directed=directed,
        name=name or path.stem,
        on_duplicate_edge=on_duplicate_edge,
    )


def write_parquet(graph: PropertyGraph, path: str | Path) -> Path:
    """Write ``nodes.parquet`` and ``edges.parquet`` into the directory ``path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    graph.nodes.reset_index().to_parquet(path / "nodes.parquet", index=False)
    graph.edges.to_parquet(path / "edges.parquet", index=False)
    return path
