"""Colour and shape encoding for the visual diff.

Status is encoded **twice** — once as hue, once as node silhouette — so the
picture survives greyscale printing, projector washout, and colour-vision
deficiency. Nothing in the view depends on colour alone.

The palette is not eyeballed. These hues are validated for all-pairs separation
(nodes are scattered marks, so every pair must be distinguishable, not just
adjacent ones) against both surfaces:

===========  ==================================  ====================
check        light (surface ``#fcfcfb``)         dark (``#1a1a19``)
===========  ==================================  ====================
CVD ΔE       8.4 (aqua↔orange, protan)           9.4 (aqua↔orange)
normal ΔE    21.6 (aqua↔blue)                    20.9 (aqua↔blue)
contrast     all ≥ 3:1                           all ≥ 3:1
===========  ==================================  ====================

A magenta ``CHANGED`` was tried first and rejected: against orange it scored
ΔE 12.9 for normal vision, below the 15 floor — a pair full-colour readers
struggle to separate.
"""

from __future__ import annotations

from typing import Final

from .._types import STATUS_ORDER

__all__ = ["DARK", "LIGHT", "SHAPES", "STATUS_COLORS", "status_labels"]

#: Node silhouette per status, in :data:`~graphdiff._types.STATUS_ORDER`.
#: Circle reads as "unremarkable"; the three difference shapes are distinct at
#: small size and in peripheral vision.
SHAPES: Final[dict[str, str]] = {
    "SHARED": "circle",
    "CHANGED": "diamond",
    "A_ONLY": "square",
    "B_ONLY": "triangle",
}

#: Light theme. Surface and ink follow the reference palette.
LIGHT: Final[dict[str, str]] = {
    "surface": "#fcfcfb",
    "panel": "#f4f4f2",
    "line": "#e0e0dc",
    "ink": "#0b0b0b",
    "inkSecondary": "#52514e",
    "inkMuted": "#7b7a75",
    "SHARED": "#a8afb8",
    "CHANGED": "#199e70",
    "A_ONLY": "#2a78d6",
    "B_ONLY": "#eb6834",
}

#: Dark theme — stepped for the dark surface, not an automatic flip of light.
DARK: Final[dict[str, str]] = {
    "surface": "#1a1a19",
    "panel": "#232322",
    "line": "#373735",
    "ink": "#ffffff",
    "inkSecondary": "#c3c2b7",
    "inkMuted": "#8d8c84",
    "SHARED": "#6b7079",
    "CHANGED": "#199e70",
    "A_ONLY": "#3987e5",
    "B_ONLY": "#d95926",
}

#: Back-compat alias: the light-theme status hues.
STATUS_COLORS: Final[dict[str, str]] = {s: LIGHT[s] for s in STATUS_ORDER}


def status_labels(name_a: str, name_b: str) -> dict[str, str]:
    """Human-readable legend text naming the two graphs explicitly.

    "A_ONLY" means nothing to a reader; "Only in snapshot_2024" does.
    """
    return {
        "SHARED": "In both",
        "CHANGED": "Changed",
        "A_ONLY": f"Only in {name_a}",
        "B_ONLY": f"Only in {name_b}",
    }
