"""Fixture graphs whose metric values are computed by hand in the tests."""

from __future__ import annotations

import pandas as pd
import pytest

from graphdiff.core.graph import PropertyGraph

from helpers import make_graph


@pytest.fixture
def graph_a() -> PropertyGraph:
    """A: a->b (knows,1), b->c (knows,2), c->d (owns,3)."""
    return make_graph(
        [("a", "knows", "b", 1.0), ("b", "knows", "c", 2.0), ("c", "owns", "d", 3.0)],
        name="A",
    )


@pytest.fixture
def graph_b() -> PropertyGraph:
    """B: a->b (knows,1) shared, b->c (knows,5) changed, c->e (owns,3) b-only."""
    return make_graph(
        [("a", "knows", "b", 1.0), ("b", "knows", "c", 5.0), ("c", "owns", "e", 3.0)],
        name="B",
    )


@pytest.fixture
def empty_graph() -> PropertyGraph:
    """A graph with no nodes and no edges."""
    return PropertyGraph(
        nodes=pd.DataFrame(index=pd.Index([], name="label")),
        edges=pd.DataFrame(
            {
                "source": pd.Series(dtype=object),
                "target": pd.Series(dtype=object),
                "type": pd.Series(dtype=object),
            }
        ),
        name="empty",
    )
