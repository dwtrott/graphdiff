"""The local web app: API, job store, and the inlined page."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from graphdiff import write_graph
from graphdiff.data import example_pair, example_sequence
from graphdiff.server import create_app

EXTERNAL_URL = re.compile(r"""(?:https?:)?//[A-Za-z0-9._-]+\.[A-Za-z]{2,}""")


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ws = tmp_path_factory.mktemp("ws")
    a, b = example_pair()
    write_graph(a, ws / "a.graphml")
    write_graph(b, ws / "b.json")
    for g in example_sequence(n_snapshots=3, n_communities=4, community_size=30):
        write_graph(g, ws / f"{g.name}.graphml")
    (ws / "notes.txt").write_text("not a graph\n")
    return ws


@pytest.fixture(scope="module")
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace, workers=1))


def _wait(client: TestClient, job_id: str, timeout: float = 120) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        meta: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
        if meta["status"] in ("done", "failed"):
            return meta
        time.sleep(0.2)
    raise AssertionError("job did not finish")


class TestPage:
    def test_index_is_inlined_and_offline(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        html = r.text
        assert "<script>" in html and "<style>" in html
        assert not EXTERNAL_URL.findall(html)
        assert 'src="http' not in html and 'href="http' not in html

    def test_meta(self, client: TestClient, workspace: Path) -> None:
        m = client.get("/api/meta").json()
        assert m["workspace"] == str(workspace.resolve())
        assert m["align_methods"] == ["exact", "normalized", "fuzzy"]


class TestGraphs:
    def test_list_and_inspect(self, client: TestClient) -> None:
        names = {g["name"]: g for g in client.get("/api/graphs").json()}
        assert {"a.graphml", "b.json"} <= set(names)
        assert "notes.txt" not in names  # not a graph
        assert names["b.json"]["format"] == "json"
        info = client.get("/api/graphs/a.graphml").json()
        assert info["n_nodes"] == 52 and info["n_edges"] > 0

    def test_upload_roundtrip(self, client: TestClient, workspace: Path) -> None:
        body = (workspace / "a.graphml").read_bytes()
        r = client.put("/api/graphs/uploaded.graphml", content=body)
        assert r.status_code == 200, r.text
        assert r.json()["summary"]["n_nodes"] == 52
        assert (workspace / "uploaded.graphml").exists()
        r = client.delete("/api/graphs/uploaded.graphml")
        assert r.status_code == 200 and not (workspace / "uploaded.graphml").exists()

    def test_upload_rejects_bad_names_and_formats(self, client: TestClient) -> None:
        assert client.put("/api/graphs/..%2Fescape.graphml", content=b"x").status_code in (400, 404)
        assert client.put("/api/graphs/.hidden.graphml", content=b"x").status_code == 400
        assert client.put("/api/graphs/thing.exe", content=b"x").status_code == 415
        r = client.put("/api/graphs/broken.graphml", content=b"<not graphml")
        assert r.status_code == 422
        assert "broken.graphml" not in {g["name"] for g in client.get("/api/graphs").json()}

    def test_missing(self, client: TestClient) -> None:
        assert client.get("/api/graphs/nope.graphml").status_code == 404
        assert client.get("/api/graphs/nope.graphml").json()["error"]


class TestJobs:
    def test_compare_job(self, client: TestClient, workspace: Path) -> None:
        r = client.post("/api/compare", json={"a": "a.graphml", "b": "b.json", "align": "fuzzy"})
        assert r.status_code == 202, r.text
        job = r.json()
        assert job["status"] == "queued" and job["title"] == "a vs b"
        meta = _wait(client, job["id"])
        assert meta["status"] == "done", meta.get("error")
        assert meta["stage"] is None
        assert set(meta["artifacts"]) >= {"report.json", "summary.md", "viewer.html"}
        s = meta["summary"]
        assert s["findings"][0]["kind"] == "headline"
        assert s["scalar_scores"]["ged_similarity"]["raw"] > 0
        assert s["alignment"]["exact"] > 0
        assert len(s["top_changed"]) > 0
        html = client.get(f"/api/jobs/{job['id']}/viewer.html")
        assert html.status_code == 200 and html.text.startswith("<!DOCTYPE html>")
        assert not EXTERNAL_URL.findall(html.text)
        rep = client.get(f"/api/jobs/{job['id']}/report.json").json()
        assert rep["provenance"]["input_a"]["path"].endswith("a.graphml")
        assert len(rep["provenance"]["input_a"]["sha256"]) == 64
        assert (workspace / ".graphdiff" / "jobs" / job["id"] / "meta.json").exists()

    def test_timeline_job(self, client: TestClient) -> None:
        r = client.post(
            "/api/timeline",
            json={"graphs": ["t0.graphml", "t1.graphml", "t2.graphml"], "max_nodes": 200},
        )
        assert r.status_code == 202, r.text
        meta = _wait(client, r.json()["id"])
        assert meta["status"] == "done", meta.get("error")
        assert meta["kind"] == "timeline" and meta["title"] == "t0 → t1 → t2"
        assert len(meta["summary"]["churn"]) == 2
        assert len(meta["summary"]["step_findings"]) == 2
        page = client.get(f"/api/jobs/{meta['id']}/viewer.html").text
        assert "Findings across the series" in page

    def test_history_and_delete(self, client: TestClient) -> None:
        jobs = client.get("/api/jobs").json()
        assert len(jobs) >= 2 and "summary" not in jobs[0]
        assert jobs == sorted(jobs, key=lambda j: j["created_at"], reverse=True)
        job_id = jobs[-1]["id"]
        assert client.delete(f"/api/jobs/{job_id}").status_code == 200
        assert client.get(f"/api/jobs/{job_id}").status_code == 404
        assert client.get(f"/api/jobs/{job_id}/viewer.html").status_code == 404

    def test_validation(self, client: TestClient) -> None:
        bad_b = {"a": "a.graphml", "b": "nope.graphml"}
        assert client.post("/api/compare", json=bad_b).status_code == 404
        bad_align = {"a": "a.graphml", "b": "b.json", "align": "magic"}
        assert client.post("/api/compare", json=bad_align).status_code == 400
        assert client.post("/api/timeline", json={"graphs": ["a.graphml"]}).status_code == 422
        assert client.get("/api/jobs/zzz").status_code == 404
        assert client.get("/api/jobs/../../etc").status_code in (404, 422)

    def test_failed_job_is_reported(self, client: TestClient, workspace: Path) -> None:
        (workspace / "worse.graphml").write_text(
            "<graphml><graph><node id='x'/></graph></graphml>\n"
        )
        (workspace / "bad.json").write_text('{"nodes": [], "links": [{"source": "x"}]}\n')
        r = client.post("/api/compare", json={"a": "a.graphml", "b": "bad.json"})
        assert r.status_code == 202
        meta = _wait(client, r.json()["id"])
        assert meta["status"] == "failed"
        assert meta["error"] and meta["traceback"]
        assert meta["stage"] is None


class TestCLI:
    def test_app_command_exists(self) -> None:
        from typer.testing import CliRunner

        from graphdiff.cli import app

        r = CliRunner().invoke(app, ["app", "--help"])
        assert r.exit_code == 0 and "--workspace" in r.output
