"""HTTP API tests through FastAPI's TestClient (runs the real lifespan and background jobs)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from log_analytics.api.app import create_app
from log_analytics.bootstrap import build_container
from log_analytics.config import Settings
from log_analytics.domain.errors import CapacityError
from log_analytics.domain.models import AnalysisRun, RunKind, RunStatus
from log_analytics.infrastructure.run_repositories import InMemoryRunRepository
from tests.conftest import JWT
from tests.fakes import FakeEmbedder, InMemoryVectorStore


def _client(settings: Settings, **overrides) -> TestClient:  # type: ignore[no-untyped-def]
    container = asyncio.run(build_container(settings, **overrides)) if overrides else None
    return TestClient(create_app(settings, container=container))


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with _client(settings) as c:
        yield c


def _wait(client: TestClient, url: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(url).json()
        if body["status"] in {"succeeded", "failed"}:
            return body
        time.sleep(0.1)
    raise AssertionError(f"job did not finish: {body}")


def _wait_ready(client: TestClient, timeout: float = 60.0):  # type: ignore[no-untyped-def]
    """Poll /ready while the dataset warms up (503 "warming" until the classification cache exists)."""
    deadline = time.monotonic() + timeout
    while True:
        response = client.get("/ready")
        body = response.json()
        if response.status_code == 200 or time.monotonic() > deadline:
            return response
        assert response.status_code == 503 and body["status"] == "warming", body
        time.sleep(0.05)


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    ready = _wait_ready(client)
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "degraded" and body["components"]["dataset"]["ok"] is True
    assert body["components"]["dataset"]["warming"] is False
    assert "x-request-id" in ready.headers
    assert ready.headers["x-content-type-options"] == "nosniff"


def test_request_id_is_propagated(client: TestClient) -> None:
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
    assert client.get("/health", headers={"X-Request-ID": "bad id\n"}).headers["x-request-id"] != "bad id\n"


def test_logs_filtering_and_redaction(client: TestClient) -> None:
    page = client.get("/api/v1/logs", params={"hostname": "web-01", "min_severity": "ERROR", "limit": 5}).json()
    assert page["total"] == 92 and len(page["items"]) == 5
    assert {i["severity"] for i in page["items"]} <= {"ERROR", "CRITICAL"}
    secret = client.get("/api/v1/logs", params={"keyword": "Authorization"}).json()["items"][0]["message"]
    assert JWT not in secret and "hunter2" not in secret and "<JWT>" in secret


def test_logs_validation(client: TestClient) -> None:
    response = client.get("/api/v1/logs", params={"start": "2026-10-02T00:00:00", "end": "2026-10-01T00:00:00"})
    assert response.status_code == 422
    assert client.get("/api/v1/logs", params={"limit": 0}).status_code == 422


def test_catalog_stats_and_insights(client: TestClient) -> None:
    catalog = client.get("/api/v1/logs/catalog").json()
    assert set(catalog["hostnames"]) == {"web-01", "db-01"}
    stats = client.get("/api/v1/logs/stats", params={"hostname": "db-01"}).json()
    assert stats["host_count"] == 1 and stats["severity_counts"]["CRITICAL"] == 1
    dashboard = client.get("/api/v1/insights/top-errors").json()
    assert dashboard["top_patterns"][0]["occurrences"] == 80 and dashboard["error_trend"]
    health = client.get("/api/v1/insights/host-health").json()
    assert {h["hostname"] for h in health["hosts"]} == {"web-01", "db-01"}
    kinds = {a["kind"] for a in client.get("/api/v1/insights/anomalies").json()["anomalies"]}
    assert {"error_spike", "silence_gap", "service_outage"} <= kinds


def test_synchronous_query(client: TestClient) -> None:
    body = client.post("/api/v1/query", json={"question": "Summarize errors by host."}).json()
    assert body["plan"]["intent"] == "errors_by_host" and "| web-01 |" in body["answer"]
    assert client.post("/api/v1/query", json={"question": "x"}).status_code == 422


def test_analysis_job_lifecycle_and_reports(client: TestClient) -> None:
    accepted = client.post("/api/v1/analyses", json={"filter": {"hostnames": ["web-01"]}})
    assert accepted.status_code == 202
    job = accepted.json()
    run = _wait(client, job["status_url"])
    assert run["status"] == "succeeded", run.get("error")
    run_id = job["run_id"]
    summary = client.get(f"/api/v1/analyses/{run_id}/reports/executive_summary").json()
    assert summary["total_logs"] > 0
    markdown = client.get(f"/api/v1/analyses/{run_id}/reports/incident_report", params={"format": "markdown"})
    assert markdown.text.startswith("## Incident Report")
    full = client.get(f"/api/v1/analyses/{run_id}/report.md")
    assert full.headers["content-type"].startswith("text/markdown") and "# Log Analysis Report" in full.text
    listed = client.get("/api/v1/analyses").json()
    assert listed[0]["id"] == run_id and listed[0]["status"] == "succeeded"
    assert client.get("/api/v1/analyses/does-not-exist").status_code == 404
    assert client.get(f"/api/v1/analyses/{run_id}/reports/nope").status_code == 422


def test_rag_unavailable_without_vector_store(client: TestClient) -> None:
    response = client.post("/api/v1/rag/search", json={"query": "connection refused"})
    assert response.status_code == 503 and "LA_DATABASE_URL" in response.json()["detail"]


def test_rag_index_and_search(settings: Settings) -> None:
    with _client(settings, embedder=FakeEmbedder(), vector_store=InMemoryVectorStore()) as client:
        job = client.post("/api/v1/rag/index").json()
        run = _wait(client, job["status_url"])
        assert run["status"] == "succeeded" and run["result"]["documents_indexed"] > 0
        assert client.get("/api/v1/rag/stats").json()["documents"] == run["result"]["documents_indexed"]
        hits = client.post("/api/v1/rag/search", json={"query": "connection refused", "k": 3}).json()["hits"]
        assert "connection refused" in hits[0]["template"]


def test_api_key_protects_v1_routes(settings: Settings) -> None:
    secured = settings.model_copy(update={"api_key": SecretStr("s3cret")})
    with _client(secured) as client:
        assert client.get("/health").status_code == 200  # probes stay open
        assert client.get("/api/v1/logs/catalog").status_code == 401
        assert client.get("/api/v1/logs/catalog", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/v1/logs/catalog", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_production_requires_an_api_key(settings: Settings) -> None:
    production = settings.model_copy(update={"environment": "production"})
    with pytest.raises(RuntimeError, match="LA_API_KEY is required"):
        create_app(production)
    assert create_app(production.model_copy(update={"api_key": SecretStr("k")}))
    assert create_app(production.model_copy(update={"allow_anonymous": True}))


class _SaturatedJobs:
    """Stands in for a JobRunner whose queue and slots are all taken."""

    async def submit(self, *args: object, **kwargs: object) -> None:
        raise CapacityError("2 jobs are already queued or running; retry later")

    async def run_now(self, work: object) -> None:
        raise CapacityError("All analysis slots stayed busy for 30s; retry later")


def test_capacity_errors_map_to_429_with_retry_after(settings: Settings) -> None:
    container = asyncio.run(build_container(settings))
    container.jobs = _SaturatedJobs()  # type: ignore[assignment]
    with TestClient(create_app(settings, container=container)) as client:
        responses = [
            client.post("/api/v1/analyses", json={}),
            client.post("/api/v1/query", json={"question": "Summarize errors by host."}),
            client.get("/api/v1/insights/top-errors"),
            client.get("/api/v1/logs/stats"),
        ]
        for response in responses:
            assert response.status_code == 429, response.text
            assert response.headers["retry-after"] == "30"
            assert "retry later" in response.json()["detail"]
        assert client.get("/api/v1/logs", params={"limit": 1}).status_code == 200  # cheap reads stay available


def test_unfinished_runs_are_failed_at_startup(settings: Settings) -> None:
    runs = InMemoryRunRepository()
    stale = AnalysisRun.new(RunKind.ANALYSIS, {"question": "left running"}).transition(RunStatus.RUNNING)
    asyncio.run(runs.save(stale))
    with _client(settings, runs=runs) as client:
        run = client.get(f"/api/v1/analyses/{stale.id}").json()
        assert run["status"] == "failed" and "interrupted" in run["error"]


def test_workflow_diagram(client: TestClient) -> None:
    text = client.get("/api/v1/workflow/diagram").text
    assert "graph TD" in text and "root_cause_analyst" in text
