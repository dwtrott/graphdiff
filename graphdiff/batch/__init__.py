"""Batch comparison: all-pairs scoring over a directory of graphs."""

from __future__ import annotations

from .matrix import all_pairs, discover_graphs
from .timeline import TimelineReport, compare_sequence

__all__ = ["TimelineReport", "all_pairs", "compare_sequence", "discover_graphs"]
