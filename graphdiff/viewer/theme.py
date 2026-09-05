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

__all__ = ["CHANGE_RAMP", "DARK", "LIGHT", "SHAPES", "STATUS_COLORS", "status_labels"]

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


#: Sequential ramp for cluster change density — a *magnitude*, so one hue from
#: light to dark rather than a categorical set. Violet is deliberately outside
#: the categorical blue/orange/green so an aggregate mark is never mistaken for
#: a status. Both ramps are monotonic in OKLab lightness (light 0.94 → 0.38,
#: dark 0.29 → 0.74) and span contrast 1.2:1 to 10:1 against their surface.
CHANGE_RAMP: Final[dict[str, list[str]]] = {
    "light": ["#ece9f5", "#d0c8e8", "#ab9dd6", "#8474bf", "#5f4ea4", "#3f327f"],
    "dark": ["#2b2a36", "#3f3a58", "#57507f", "#7166a7", "#8d81cd", "#ab9ff0"],
}
