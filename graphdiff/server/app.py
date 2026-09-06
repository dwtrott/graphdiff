"""The local web app: pick two graphs, compare, look.

``graphdiff app [--workspace DIR]`` starts it on ``127.0.0.1``. The front end
is one HTML file served from this package with everything inlined — no build
step, no CDN, no fonts — and it talks to a small JSON API:

============================  ==============================================
``GET  /api/graphs``           graph files in the workspace
``PUT  /api/graphs/{name}``    upload one (raw body; the name gives the format)
``GET  /api/graphs/{name}``    inspect: counts, types, density, degrees
``DELETE /api/graphs/{name}``  remove an uploaded file
``POST /api/compare``          ``{a, b, align, ...}`` → job
``POST /api/timeline``         ``{graphs: [...], ...}`` → job
``GET  /api/jobs``             history, newest first
``GET  /api/jobs/{id}``        status, parameters, summary
``GET  /api/jobs/{id}/{file}`` ``viewer.html`` · ``report.json`` · ``summary.md`` · ``figure.png``
``DELETE /api/jobs/{id}``      forget a job
============================  ==============================================

The workspace is a directory of graph files; uploads land there too, and job
outputs go under ``.graphdiff/jobs/``. Nothing outside it is readable through
the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..api import inspect_graph
from ..batch.matrix import discover_graphs
from ..io import _EXTENSIONS, detect_format, read_graph
from ..metrics.base import to_jsonable
from .jobs import JobStore

__all__ = ["create_app"]

_STATIC = Path(__file__).with_name("static")
_ALIGN = ("exact", "normalized", "fuzzy")
_MAX_UPLOAD = 2 * 1024**3  # 2 GiB; graphs at 10^6 edges are far smaller


class CompareRequest(BaseModel):
    a: str
    b: str
    align: str = "exact"
    align_threshold: float = Field(0.6, ge=0.0, le=1.0)
    undirected: bool = False
    weight: str = "weight"
    cluster_by: str | None = None
    structural: bool = True
    max_nodes: int = Field(2000, ge=50, le=20000)


class TimelineRequest(BaseModel):
    graphs: list[str] = Field(min_length=2)
    align: str = "exact"
    align_threshold: float = Field(0.6, ge=0.0, le=1.0)
    undirected: bool = False
    weight: str = "weight"
    cluster_by: str | None = None
    structural: bool = True
    max_nodes: int = Field(1500, ge=50, le=20000)


def _safe_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith(".") or name in (".", ".."):
        raise HTTPException(400, f"bad file name {name!r}")
    return name


def create_app(workspace: str | Path, *, workers: int = 2) -> FastAPI:
    """Build the FastAPI application for one workspace directory."""
    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    store = JobStore(ws, workers=workers)
    app = FastAPI(title="graphdiff", version=__version__, docs_url=None, redoc_url=None)
    app.state.workspace = ws
    app.state.store = store

    def graph_path(name: str) -> Path:
        path = ws / _safe_name(name)
        if not path.exists():
            raise HTTPException(404, f"no graph named {name!r}")
        return path

    # ------------------------------------------------------------ page

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {"version": __version__, "workspace": str(ws), "align_methods": list(_ALIGN)}

    # ------------------------------------------------------------ graphs

    @app.get("/api/graphs")
    def graphs() -> list[dict[str, Any]]:
        out = []
        for path in discover_graphs(ws):
            stat = path.stat()
            out.append(
                {
                    "name": path.name,
                    "format": detect_format(path),
                    "bytes": stat.st_size
                    if path.is_file()
                    else sum(p.stat().st_size for p in path.glob("*.parquet")),
                    "modified": stat.st_mtime,
                }
            )
        return out

    @app.get("/api/graphs/{name}")
    def inspect(name: str, undirected: bool = False) -> dict[str, Any]:
        path = graph_path(name)
        try:
            summary = inspect_graph(read_graph(path, directed=not undirected))
        except Exception as exc:
            raise HTTPException(422, f"could not read {name}: {exc}") from exc
        return dict(to_jsonable(summary))

    @app.put("/api/graphs/{name}")
    async def upload(name: str, request: Request) -> dict[str, Any]:
        name = _safe_name(name)
        target = ws / name
        if target.suffix.lower() not in _EXTENSIONS:
            raise HTTPException(
                415,
                f"unsupported file type {target.suffix!r}; use {', '.join(sorted(_EXTENSIONS))}",
            )
        size = 0
        tmp = target.with_suffix(target.suffix + ".part")
        with tmp.open("wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > _MAX_UPLOAD:
                    handle.close()
                    tmp.unlink(missing_ok=True)
                    raise HTTPException(413, "upload too large")
                handle.write(chunk)
        tmp.replace(target)
        try:
            summary = inspect_graph(read_graph(target))
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(422, f"{name} is not a readable graph: {exc}") from exc
        return {"name": name, "bytes": size, "summary": to_jsonable(summary)}

    @app.delete("/api/graphs/{name}")
    def remove_graph(name: str) -> dict[str, str]:
        path = graph_path(name)
        if path.is_dir():
            raise HTTPException(400, "refusing to delete a directory-backed graph")
        path.unlink()
        return {"deleted": name}

    # ------------------------------------------------------------ jobs

    def _validate_common(align: str, *names: str) -> None:
        if align not in _ALIGN:
            raise HTTPException(400, f"align must be one of {_ALIGN}")
        for n in names:
            graph_path(n)

    @app.post("/api/compare", status_code=202)
    def submit_compare(req: CompareRequest) -> dict[str, Any]:
        _validate_common(req.align, req.a, req.b)
        return store.submit("compare", req.model_dump())

    @app.post("/api/timeline", status_code=202)
    def submit_timeline(req: TimelineRequest) -> dict[str, Any]:
        _validate_common(req.align, *req.graphs)
        return store.submit("timeline", req.model_dump())

    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return store.list()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        try:
            return store.get(job_id)
        except KeyError:
            raise HTTPException(404, "no such job") from None

    @app.get("/api/jobs/{job_id}/{artifact}")
    def job_artifact(job_id: str, artifact: str) -> FileResponse:
        try:
            path, media = store.artifact(job_id, artifact)
        except KeyError:
            raise HTTPException(404, "no such artifact") from None
        return FileResponse(path, media_type=media, headers={"Cache-Control": "no-store"})

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, str]:
        try:
            store.delete(job_id)
        except KeyError:
            raise HTTPException(404, "no such job") from None
        return {"deleted": job_id}

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app
