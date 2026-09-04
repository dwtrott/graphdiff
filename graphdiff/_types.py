"""Shared type definitions and column-name constants for :mod:`graphdiff`."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = [
    "A_PREFIX",
    "B_PREFIX",
    "CHANGED_ATTRS",
    "EDGE_KEY",
    "ETYPE",
    "LABEL",
    "RESERVED_EDGE_COLUMNS",
    "SOURCE",
    "STATUS",
    "STATUS_ORDER",
    "TARGET",
    "Status",
]


class Status(StrEnum):
    """Provenance of a node or edge inside a :class:`~graphdiff.core.UnionDiffGraph`.

    ``SHARED``
        Present in both graphs with equal compared attributes. For edges this means
        an identical ``(source, type, target)`` triple.
    ``A_ONLY`` / ``B_ONLY``
        Present in exactly one of the two graphs.
    ``CHANGED``
        Same identity in both graphs, but at least one compared attribute differs.
        The old/new values are recorded in the ``changed_attrs`` column.
    """

    SHARED = "SHARED"
    A_ONLY = "A_ONLY"
    B_ONLY = "B_ONLY"
    CHANGED = "CHANGED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Canonical ordering used for categorical dtypes and report tables.
STATUS_ORDER: Final[list[str]] = [
    Status.SHARED.value,
    Status.CHANGED.value,
    Status.A_ONLY.value,
    Status.B_ONLY.value,
]

# Reserved column / index names.
LABEL: Final[str] = "label"
SOURCE: Final[str] = "source"
TARGET: Final[str] = "target"
ETYPE: Final[str] = "type"
STATUS: Final[str] = "status"
CHANGED_ATTRS: Final[str] = "changed_attrs"

#: Columns that jointly identify an edge.
EDGE_KEY: Final[tuple[str, str, str]] = (SOURCE, ETYPE, TARGET)

#: Columns a caller may not use for user attributes on an edge table.
RESERVED_EDGE_COLUMNS: Final[frozenset[str]] = frozenset(
    {SOURCE, TARGET, ETYPE, STATUS, CHANGED_ATTRS}
)

A_PREFIX: Final[str] = "a_"
B_PREFIX: Final[str] = "b_"
