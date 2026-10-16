"""Golden-master regression test: the full analysis output on the synthetic dataset must not drift.

Performance refactors of the detectors, the dataset cache or the workflow must leave every report
byte-for-byte identical. Run ids and timings are excluded because they change on every run.

After an *intended* output change, regenerate the snapshots and review the diff:

    LA_UPDATE_GOLDEN=1 pytest tests/test_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from log_analytics.bootstrap import build_container
from log_analytics.config import Settings
from log_analytics.domain.models import AnalysisRequest, Intent, LogFilter, Severity
from tests.fakes import FakeLLM

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.getenv("LA_UPDATE_GOLDEN") == "1"

CASES: dict[str, AnalysisRequest] = {
    "full_analysis": AnalysisRequest(),
    "host_error_floor": AnalysisRequest(filter=LogFilter(hostnames=("web-01",), min_severity=Severity.ERROR)),
    "keywords_any": AnalysisRequest(filter=LogFilter(keywords=("Bearer", "connection refused"))),
    "time_window": AnalysisRequest(
        filter=LogFilter(start="2026-09-30T01:30:00Z", end="2026-09-30T05:30:00Z"), intent=Intent.ANOMALIES
    ),
    "q_errors_by_host": AnalysisRequest(question="Summarize errors by host."),
    "q_failing_processes": AnalysisRequest(question="Which processes are failing most frequently?"),
    "q_recurring": AnalysisRequest(question="Show recurring error patterns."),
    "q_anomalies": AnalysisRequest(question="Were there any unusual spikes or outages?"),
    "q_host": AnalysisRequest(question="Analyze issues for hostname db-01"),
    "q_unknown_host": AnalysisRequest(question="Analyze issues for hostname XYZ."),
    "q_unmatched": AnalysisRequest(question="Tell me something interesting"),
}


def _snapshot(result: Any) -> dict[str, Any]:
    return result.model_dump(mode="json", exclude={"run_id", "timings_ms"})


async def _run_all(settings: Settings, **overrides: Any) -> dict[str, Any]:
    container = await build_container(settings, **overrides)
    try:
        return {name: _snapshot(await container.workflow.run(request)) for name, request in CASES.items()}
    finally:
        await container.aclose()


def _dump(snapshots: dict[str, Any]) -> str:
    """One compact JSON line per case: small files, and a diff shows which case changed."""
    lines = [
        f"{json.dumps(case)}: {json.dumps(snapshots[case], sort_keys=True, ensure_ascii=False, separators=(',', ':'))}"
        for case in sorted(snapshots)
    ]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def _compare(name: str, actual: dict[str, Any]) -> None:
    path = GOLDEN_DIR / f"{name}.json"
    if UPDATE or not path.exists():
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_text(_dump(actual), encoding="utf-8")
        if not UPDATE:
            pytest.fail(f"golden snapshot {path.name} did not exist and was created; review and commit it")
        return
    expected = json.loads(path.read_text(encoding="utf-8"))
    for case in sorted(set(expected) | set(actual)):
        assert actual.get(case) == expected.get(case), f"{name}/{case} drifted from the golden snapshot"


@pytest.mark.parametrize("cache", [True, False], ids=["cached", "streaming"])
async def test_golden_without_llm(settings: Settings, cache: bool) -> None:
    _compare("no_llm", await _run_all(settings.model_copy(update={"cache_dataset": cache})))


async def test_golden_with_fake_llm(settings: Settings) -> None:
    _compare("fake_llm", await _run_all(settings.model_copy(update={"llm_enabled": True}), llm_client=FakeLLM()))
