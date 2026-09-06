"""Job store for the local web app: run comparisons off the request thread.

Every job lives in ``<workspace>/.graphdiff/jobs/<id>/`` as plain files —
``meta.json`` (status, parameters, timings, summary), ``report.json``,
``summary.md``, ``viewer.html``, and ``figure.png`` when matplotlib is around —
so the history survives a restart and every artifact is a file the user can
copy out. Nothing is held only in memory except the running future.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..api import compare
from ..batch.timeline import compare_sequence
from ..io import read_graph
from ..metrics.base import to_jsonable
from ..viewer import write_html, write_timeline_html

__all__ = ["JobStore"]

_ARTIFACTS = {
    "report.json": "application/json",
    "summary.md": "text/markdown",
    "viewer.html": "text/html",
    "figure.png": "image/png",
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobStore:
    """Persisted comparison jobs, run on a small thread pool."""

    def __init__(self, workspace: Path, *, workers: int = 2) -> None:
        self.workspace = workspace
        self.root = workspace / ".graphdiff" / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="graphdiff-job")
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ persistence

    def _dir(self, job_id: str) -> Path:
        if not job_id.replace("-", "").isalnum():
            raise KeyError(job_id)
        return self.root / job_id

    def _write_meta(self, job_id: str, meta: dict[str, Any]) -> None:
        path = self._dir(job_id) / "meta.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(to_jsonable(meta), indent=1, default=str), encoding="utf-8")
        tmp.replace(path)

    def get(self, job_id: str) -> dict[str, Any]:
        path = self._dir(job_id) / "meta.json"
        if not path.exists():
            raise KeyError(job_id)
        meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        meta["artifacts"] = [n for n in _ARTIFACTS if (self._dir(job_id) / n).exists()]
        return meta

    def list(self) -> list[dict[str, Any]]:
        out = []
        for d in self.root.iterdir():
            if (d / "meta.json").exists():
                try:
                    meta = self.get(d.name)
                except (KeyError, json.JSONDecodeError):  # pragma: no cover - corrupt dir
                    continue
                meta.pop("summary", None)
                out.append(meta)
        return sorted(out, key=lambda m: m.get("created_at", ""), reverse=True)

    def artifact(self, job_id: str, name: str) -> tuple[Path, str]:
        if name not in _ARTIFACTS:
            raise KeyError(name)
        path = self._dir(job_id) / name
        if not path.exists():
            raise KeyError(name)
        return path, _ARTIFACTS[name]

    def delete(self, job_id: str) -> None:
        d = self._dir(job_id)
        if not d.exists():
            raise KeyError(job_id)
        fut = self._futures.pop(job_id, None)
        if fut is not None:
            fut.cancel()
        shutil.rmtree(d)

    # ------------------------------------------------------------ submission

    def submit(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        """Create a job directory, queue the work, return the initial meta."""
        job_id = uuid.uuid4().hex[:12]
        self._dir(job_id).mkdir()
        meta = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "params": params,
            "error": None,
            "summary": None,
            "title": self._title(kind, params),
        }
        self._write_meta(job_id, meta)
        runner = self._run_compare if kind == "compare" else self._run_timeline
        with self._lock:
            self._futures[job_id] = self._pool.submit(self._guarded, job_id, runner, params)
        return meta

    @staticmethod
    def _title(kind: str, params: dict[str, Any]) -> str:
        if kind == "compare":
            return f"{Path(params['a']).stem} vs {Path(params['b']).stem}"
        names = [Path(p).stem for p in params["graphs"]]
        return (
            " → ".join(names) if len(names) <= 4 else f"{names[0]} → … → {names[-1]} ({len(names)})"
        )

    def _stage(self, job_id: str, text: str) -> None:
        meta = self.get(job_id)
        meta["stage"] = text
        self._write_meta(job_id, meta)

    def _guarded(
        self,
        job_id: str,
        runner: Callable[[str, dict[str, Any]], dict[str, Any]],
        params: dict[str, Any],
    ) -> None:
        meta = self.get(job_id)
        meta.update(status="running", started_at=_now(), stage="reading graphs")
        self._write_meta(job_id, meta)
        try:
            summary = runner(job_id, params)
            meta = self.get(job_id)
            meta.update(status="done", finished_at=_now(), summary=summary, stage=None)
        except Exception as exc:
            meta = self.get(job_id)
            meta.update(
                stage=None,
                status="failed",
                finished_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=6),
            )
        self._write_meta(job_id, meta)

    # ------------------------------------------------------------ the work

    def _resolve(self, name: str) -> Path:
        path = (self.workspace / name).resolve()
        if self.workspace.resolve() not in path.parents and path != self.workspace.resolve():
            raise ValueError(f"{name!r} is outside the workspace")
        if not path.exists():
            raise FileNotFoundError(name)
        return path

    def _common_kwargs(self, p: dict[str, Any]) -> dict[str, Any]:
        return {
            "align": p.get("align", "exact"),
            "align_threshold": float(p.get("align_threshold", 0.6)),
            "weight_attribute": p.get("weight", "weight"),
            "cluster_by": p.get("cluster_by") or None,
            "structural": bool(p.get("structural", True)),
        }

    def _run_compare(self, job_id: str, p: dict[str, Any]) -> dict[str, Any]:
        d = self._dir(job_id)
        directed = not bool(p.get("undirected", False))
        pa, pb = self._resolve(p["a"]), self._resolve(p["b"])
        a = read_graph(pa, directed=directed)
        b = read_graph(pb, directed=directed)
        self._stage(job_id, f"comparing {a.n_nodes:,} + {b.n_nodes:,} nodes")
        report = compare(a, b, **self._common_kwargs(p))
        # Provenance should point at the files, not the in-memory graphs.
        from ..api import _describe_input

        report.provenance["input_a"] = _describe_input(pa, a)
        report.provenance["input_b"] = _describe_input(pb, b)
        report.to_json(d / "report.json")
        report.write_markdown(d / "summary.md")
        self._stage(job_id, "laying out the viewer")
        write_html(report, d / "viewer.html", max_nodes=int(p.get("max_nodes", 2000)))
        self._stage(job_id, "drawing the figure")
        self._try_figure(d, lambda plot: plot.plot_dashboard(report))
        return {
            "findings": [f.to_dict() for f in report.findings],
            "node_status_counts": report.node_status_counts,
            "edge_status_counts": report.edge_status_counts,
            "scalar_scores": report.scalar_scores(),
            "top_changed": report.top_changed(15).to_dict("records"),
            "alignment": report.alignment.counts if report.alignment is not None else None,
            "graph_a": report.name_a,
            "graph_b": report.name_b,
        }

    def _run_timeline(self, job_id: str, p: dict[str, Any]) -> dict[str, Any]:
        d = self._dir(job_id)
        directed = not bool(p.get("undirected", False))
        paths = [self._resolve(n) for n in p["graphs"]]
        graphs = [read_graph(path, directed=directed) for path in paths]
        self._stage(job_id, f"comparing {len(graphs) - 1} steps")
        tl = compare_sequence(graphs, names=[path.stem for path in paths], **self._common_kwargs(p))
        tl.to_json(d / "report.json")
        tl.write_markdown(d / "summary.md")
        self._stage(job_id, f"laying out {len(graphs) - 1} step viewers")
        write_timeline_html(tl, d / "viewer.html", max_nodes=int(p.get("max_nodes", 1500)))
        self._stage(job_id, "drawing the figure")
        self._try_figure(d, lambda plot: plot.plot_timeline(tl))
        return {
            "findings": [f.to_dict() for f in tl.findings],
            "churn": tl.churn().to_dict("records"),
            "names": tl.names,
            "step_findings": [[f.to_dict() for f in s.findings[:4]] for s in tl.steps],
        }

    @staticmethod
    def _try_figure(d: Path, make: Callable[[Any], Any]) -> None:
        try:
            from .. import plot
        except ImportError:  # pragma: no cover - optional extra
            return
        with contextlib.suppress(Exception):  # a figure is a nicety, never a failure
            plot.save(make(plot), d / "figure.png")
