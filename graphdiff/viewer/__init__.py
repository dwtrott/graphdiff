"""Visual diff rendering.

The primary output is a single self-contained HTML file: layout is computed
here in Python, coordinates are baked into the page, and the result loads no
external resources at all. That satisfies the air-gapped constraint without a
frontend build step, and the file can be copied around on its own.

Status is encoded twice — hue *and* node shape — so the picture survives
greyscale printing and colour-vision deficiency. See :mod:`graphdiff.viewer.theme`.
"""

from __future__ import annotations

from .focus import FocusSelection, select_focus
from .html import render_html, write_html
from .layout import LayoutParams, force_directed_layout
from .theme import DARK, LIGHT, SHAPES, STATUS_COLORS, status_labels

__all__ = [
    "DARK",
    "LIGHT",
    "SHAPES",
    "STATUS_COLORS",
    "FocusSelection",
    "LayoutParams",
    "force_directed_layout",
    "render_html",
    "select_focus",
    "status_labels",
    "write_html",
]
