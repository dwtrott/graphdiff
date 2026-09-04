"""Shared plumbing for metrics: serializable results and the raw/shared pairing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
import pandas as pd

__all__ = ["DualScore", "MetricResult", "flatten_scores", "to_jsonable"]


class MetricResult(Protocol):
    """Any dataclass returned by a metric function."""

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


T = TypeVar("T")


@dataclass
class DualScore(Generic[T]):
    """A metric evaluated twice: over the full union, and over shared nodes only.

    ``shared`` is the density-normalized view — the same metric restricted to the
    subgraph induced on nodes present in both graphs. Comparing ``raw`` against
    ``shared`` separates "the graphs differ in size" from "the overlapping part
    differs in structure".
    """

    raw: T
    shared: T

    def to_dict(self) -> dict[str, Any]:
        """Serialize both views to plain Python containers."""
        return {"raw": to_jsonable(self.raw), "shared_subgraph": to_jsonable(self.shared)}


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, numpy scalars, and frames to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return None if isinstance(value, float) and np.isnan(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return str(value)


def flatten_scores(
    prefix: str, payload: Any, out: dict[str, float] | None = None
) -> dict[str, float]:
    """Flatten a nested result into ``{"dotted.path": float}`` for tidy tables."""
    out = {} if out is None else out
    payload = to_jsonable(payload)
    if isinstance(payload, dict):
        for key, value in payload.items():
            flatten_scores(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        out[prefix] = float(payload)
    return out
