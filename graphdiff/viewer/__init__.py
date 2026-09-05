"""Visual diff rendering.

The primary output is a single self-contained HTML file: layout is computed
here in Python, coordinates are baked into the page, and the result loads no
external resources at all. That satisfies the air-gapped constraint without a
frontend build step, and the file can be copied around on its own.
"""

from __future__ import annotations

from .focus import FocusSelection, select_focus
from .html import STATUS_COLORS, render_html, write_html
from .layout import LayoutParams, force_directed_layout

__all__ = [
    "STATUS_COLORS",
    "FocusSelection",
    "LayoutParams",
    "force_directed_layout",
    "render_html",
    "select_focus",
    "write_html",
]
