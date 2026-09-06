"""Bundled example data."""

from __future__ import annotations

from .demo import write_demo_datasets
from .examples import example_pair, example_sequence, large_example_pair, perturb_labels

__all__ = [
    "example_pair",
    "example_sequence",
    "large_example_pair",
    "perturb_labels",
    "write_demo_datasets",
]
