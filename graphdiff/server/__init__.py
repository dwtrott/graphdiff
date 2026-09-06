"""Local web app (FastAPI + a single inlined page). Needs the ``[viewer]`` extra."""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
