"""Graph builders shared by the test modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from graphdiff.core.graph import PropertyGraph


def make_graph(
    edges: list[tuple[str, str, str, float]],
    *,
    nodes: list[str] | None = None,
    node_attrs: dict[str, dict[str, object]] | None = None,
    directed: bool = True,
    name: str | None = None,
) -> PropertyGraph:
    """Build a PropertyGraph from ``(source, type, target, weight)`` tuples."""
    frame = pd.DataFrame(edges, columns=["source", "type", "target", "weight"])
    node_frame = None
    if nodes is not None or node_attrs is not None:
        labels = list(nodes) if nodes is not None else sorted(node_attrs or {})
        node_frame = pd.DataFrame(
            [dict({"label": label}, **(node_attrs or {}).get(label, {})) for label in labels]
        ).set_index("label")
    return PropertyGraph.from_edges(frame, nodes=node_frame, directed=directed, name=name)


def random_graph(
    n_nodes: int,
    n_edges: int,
    *,
    seed: int,
    prefix: str = "n",
    n_types: int = 3,
    directed: bool = True,
    name: str | None = None,
) -> PropertyGraph:
    """A reproducible random graph with uniquely labeled nodes and typed edges."""
    rng = np.random.default_rng(seed)
    labels = np.array([f"{prefix}{i}" for i in range(n_nodes)], dtype=object)
    src = rng.integers(0, n_nodes, size=n_edges)
    dst = rng.integers(0, n_nodes, size=n_edges)
    types = rng.integers(0, n_types, size=n_edges)
    frame = pd.DataFrame(
        {
            "source": labels[src],
            "target": labels[dst],
            "type": [f"t{t}" for t in types],
            "weight": rng.random(n_edges).round(6),
        }
    ).drop_duplicates(subset=["source", "type", "target"], ignore_index=True)
    nodes = pd.DataFrame(index=pd.Index(labels, name="label"))
    return PropertyGraph.from_edges(frame, nodes=nodes, directed=directed, name=name)
