"""GraphML reading and writing without a networkx dependency.

Parsing uses :func:`xml.etree.ElementTree.iterparse` and clears elements as it
goes, so memory stays proportional to the resulting tables rather than to the
document tree.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd

from .._types import ETYPE, LABEL, SOURCE, TARGET
from ..core.graph import DuplicatePolicy, PropertyGraph

__all__ = ["EDGE_TYPE_CANDIDATES", "read_graphml", "write_graphml"]

_NS = "{http://graphml.graphdrawing.org/xmlns}"

#: Attribute names treated as the edge type when the caller does not name one.
EDGE_TYPE_CANDIDATES: tuple[str, ...] = ("type", "edge_type", "label", "relation", "relationship")

_CASTS: dict[str, Any] = {
    "int": int,
    "long": int,
    "float": float,
    "double": float,
    "boolean": lambda v: str(v).strip().lower() in ("true", "1", "yes"),
    "string": str,
}


def _cast(value: str | None, attr_type: str) -> Any:
    if value is None:
        return None
    caster = _CASTS.get(attr_type, str)
    try:
        return caster(value)
    except (TypeError, ValueError):
        return value


def _strip(tag: str) -> str:
    return tag[len(_NS) :] if tag.startswith(_NS) else tag


def read_graphml(
    path: str | Path,
    *,
    edge_type_attr: str | None = None,
    name: str | None = None,
    on_duplicate_edge: DuplicatePolicy = "error",
) -> PropertyGraph:
    """Read a GraphML file into a :class:`~graphdiff.core.PropertyGraph`.

    Parameters
    ----------
    path:
        GraphML file to read.
    edge_type_attr:
        Name of the edge attribute holding the edge type. When ``None``, the
        first of :data:`EDGE_TYPE_CANDIDATES` present in the file is used, and
        edges fall back to the empty type if none is found.
    name:
        Graph name for reports; defaults to the file stem.
    on_duplicate_edge:
        Passed through to :class:`~graphdiff.core.PropertyGraph`.
    """
    path = Path(path)
    keys: dict[str, dict[str, Any]] = {}
    node_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    directed = True
    saw_graph = False

    context = ET.iterparse(str(path), events=("start", "end"))
    for event, elem in context:
        tag = _strip(elem.tag)
        if event == "start" and tag == "graph" and not saw_graph:
            saw_graph = True
            directed = elem.get("edgedefault", "directed") != "undirected"
        elif event == "end":
            if tag == "key":
                default_el = elem.find(f"{_NS}default")
                attr_type = elem.get("attr.type", "string")
                keys[elem.get("id", "")] = {
                    "name": elem.get("attr.name", elem.get("id", "")),
                    "type": attr_type,
                    "for": elem.get("for", "all"),
                    "default": _cast(default_el.text, attr_type)
                    if default_el is not None
                    else None,
                }
                elem.clear()
            elif tag == "node":
                record: dict[str, Any] = {LABEL: elem.get("id")}
                for data in elem.findall(f"{_NS}data"):
                    key_id = data.get("key", "")
                    spec = keys.get(key_id, {})
                    record[str(spec.get("name", key_id))] = _cast(
                        data.text, spec.get("type", "string")
                    )
                node_records.append(record)
                elem.clear()
            elif tag == "edge":
                edge: dict[str, Any] = {
                    SOURCE: elem.get("source"),
                    TARGET: elem.get("target"),
                }
                for data in elem.findall(f"{_NS}data"):
                    key_id = data.get("key", "")
                    spec = keys.get(key_id, {})
                    edge[str(spec.get("name", key_id))] = _cast(
                        data.text, spec.get("type", "string")
                    )
                edge_records.append(edge)
                elem.clear()

    nodes = pd.DataFrame(node_records)
    if nodes.empty:
        nodes = pd.DataFrame({LABEL: pd.Series(dtype="object")})
    nodes = nodes.set_index(LABEL)

    # Apply declared defaults for node attributes that some nodes omitted.
    for spec in keys.values():
        if spec["for"] in ("node", "all") and spec["default"] is not None:
            col = spec["name"]
            if col in nodes.columns:
                nodes[col] = nodes[col].fillna(spec["default"])

    edges = pd.DataFrame(edge_records)
    if edges.empty:
        edges = pd.DataFrame({SOURCE: pd.Series(dtype="object"), TARGET: pd.Series(dtype="object")})

    for spec in keys.values():
        if spec["for"] in ("edge", "all") and spec["default"] is not None:
            col = spec["name"]
            if col in edges.columns:
                edges[col] = edges[col].fillna(spec["default"])

    type_col = edge_type_attr
    if type_col is None:
        type_col = next((c for c in EDGE_TYPE_CANDIDATES if c in edges.columns), None)
    if type_col and type_col in edges.columns:
        edges = edges.rename(columns={type_col: ETYPE})
    if ETYPE not in edges.columns:
        edges[ETYPE] = ""

    # Nodes referenced only by edges still belong to the node set.
    if len(edges):
        referenced = pd.Index(
            pd.unique(pd.concat([edges[SOURCE], edges[TARGET]], ignore_index=True))
        )
        missing = referenced.difference(nodes.index)
        if len(missing):
            nodes = pd.concat([nodes, pd.DataFrame(index=pd.Index(missing, name=LABEL))])

    return PropertyGraph(
        nodes=nodes,
        edges=edges,
        directed=directed,
        name=name or path.stem,
        on_duplicate_edge=on_duplicate_edge,
    )


_XML_TYPES: dict[str, str] = {
    "i": "long",
    "u": "long",
    "f": "double",
    "b": "boolean",
}


def _xml_type(series: pd.Series) -> str:
    return _XML_TYPES.get(series.dtype.kind, "string")


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return escape(str(value))


def write_graphml(graph: PropertyGraph, path: str | Path) -> Path:
    """Write ``graph`` to GraphML. Round-trips with :func:`read_graphml`."""
    path = Path(path)
    node_attrs = list(graph.nodes.columns)
    edge_attrs = [ETYPE, *[c for c in graph.edges.columns if c not in (SOURCE, TARGET, ETYPE)]]

    key_ids: dict[tuple[str, str], str] = {}
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
    ]
    for i, col in enumerate(node_attrs):
        kid = f"nd{i}"
        key_ids[("node", col)] = kid
        lines.append(
            f'  <key id="{kid}" for="node" attr.name="{escape(col)}" '
            f'attr.type="{_xml_type(graph.nodes[col])}"/>'
        )
    for i, col in enumerate(edge_attrs):
        kid = f"ed{i}"
        key_ids[("edge", col)] = kid
        lines.append(
            f'  <key id="{kid}" for="edge" attr.name="{escape(col)}" '
            f'attr.type="{_xml_type(graph.edges[col])}"/>'
        )

    default = "directed" if graph.directed else "undirected"
    lines.append(f'  <graph id="G" edgedefault="{default}">')

    for label, row in graph.nodes.iterrows():
        data = "".join(
            f'<data key="{key_ids[("node", c)]}">{_fmt(row[c])}</data>'
            for c in node_attrs
            if pd.notna(row[c])
        )
        lines.append(f'    <node id="{escape(str(label))}">{data}</node>')

    for row in graph.edges.to_dict("records"):
        data = "".join(
            f'<data key="{key_ids[("edge", c)]}">{_fmt(row[c])}</data>'
            for c in edge_attrs
            if pd.notna(row.get(c))
        )
        lines.append(
            f'    <edge source="{escape(str(row[SOURCE]))}" '
            f'target="{escape(str(row[TARGET]))}">{data}</edge>'
        )

    lines += ["  </graph>", "</graphml>", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
