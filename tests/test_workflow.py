"""End-to-end LangGraph workflow tests (synthetic data, fake LLM / vector store)."""

from __future__ import annotations

import pytest

from log_analytics.bootstrap import build_container
from log_analytics.config import Settings
from log_analytics.domain.models import AnalysisRequest, Intent, LogFilter, Severity
from log_analytics.domain.reports import ReportType
from tests.fakes import FakeEmbedder, FakeLLM, InMemoryVectorStore


async def _container(settings: Settings, **overrides):  # type: ignore[no-untyped-def]
    return await build_container(settings, **overrides)


async def test_full_analysis_without_llm(settings: Settings) -> None:
    container = await _container(settings)
    result = await container.workflow.run(AnalysisRequest())
    reports = result.reports
    for report_type in ReportType:
        assert reports.get(report_type) is not None, report_type
    assert result.plan.intent is Intent.FULL_ANALYSIS and result.answer is None
    assert reports.executive_summary.narrative_source == "heuristic"
    assert reports.executive_summary.total_logs > 0
    assert all(c.source == "heuristic" for c in reports.root_cause_report.analyses)
    assert {a.source for a in reports.recommended_actions.actions} == {"heuristic"}
    assert "LLM disabled" in " ".join(result.plan.notes)
    assert {"planner", "data_loader", "classifier", "error_detector", "anomaly_detector", "reporter"} <= set(
        result.timings_ms
    )
    await container.aclose()


async def test_full_analysis_with_llm_uses_llm_outputs_and_rejects_hallucinated_ids(settings: Settings) -> None:
    llm = FakeLLM()
    container = await _container(settings.model_copy(update={"llm_enabled": True}), llm_client=llm)
    result = await container.workflow.run(AnalysisRequest(question="Give me an overview of system health"))
    reports = result.reports
    assert result.llm_model == "fake-llm" and result.answer.startswith("LLM answer")
    assert reports.executive_summary.headline == "LLM headline"
    llm_causes = [c for c in reports.root_cause_report.analyses if c.source == "llm"]
    assert llm_causes and all(c.probable_cause.startswith("LLM cause") for c in llm_causes)
    known_ids = {i.id for i in reports.incident_report.incidents}
    assert all(c.incident_id in known_ids for c in reports.root_cause_report.analyses)
    action = next(a for a in reports.recommended_actions.actions if a.source == "llm")
    assert "INC-HALLUCINATED" not in action.incident_ids and action.priority.value == "P1"
    patterns = reports.top_error_dashboard.top_patterns
    assert any(p.label and p.label.startswith("label-") for p in patterns)
    assert any(p.explanation for p in patterns)
    assert {"TemplateLabels", "PatternExplanations", "RootCauseBatch", "IncidentSummaryBatch", "ActionPlan"} <= set(
        llm.calls
    )
    await container.aclose()


async def test_statistical_intent_skips_deep_agents(settings: Settings) -> None:
    llm = FakeLLM()
    container = await _container(settings.model_copy(update={"llm_enabled": True}), llm_client=llm)
    result = await container.workflow.run(AnalysisRequest(question="Summarize errors by host."))
    assert result.plan.intent is Intent.ERRORS_BY_HOST and result.plan.source == "rules"
    assert result.reports.root_cause_report is None and result.reports.recommended_actions is None
    assert result.reports.host_health_report is not None
    assert "RootCauseBatch" not in llm.calls and "PlannedQuery" not in llm.calls
    assert "root_cause_analyst" not in result.timings_ms
    await container.aclose()


async def test_deterministic_answer_when_llm_absent(settings: Settings) -> None:
    container = await _container(settings)
    result = await container.workflow.run(AnalysisRequest(question="Which processes are failing most frequently?"))
    assert result.plan.intent is Intent.FAILING_PROCESSES
    assert "| python |" in result.answer and "restart" in result.answer
    await container.aclose()


async def test_llm_outage_degrades_gracefully(settings: Settings) -> None:
    llm = FakeLLM(fail=True)
    container = await _container(settings.model_copy(update={"llm_enabled": True}), llm_client=llm)
    result = await container.workflow.run(AnalysisRequest(question="What are today's critical issues?"))
    assert result.reports.incident_report.incidents  # analysis still complete
    assert all(c.source == "heuristic" for c in result.reports.root_cause_report.analyses)
    assert any("heuristic fallback" in w for w in result.warnings)
    assert len(llm.calls) == 2  # circuit breaker opened after two failed calls
    assert result.answer  # deterministic answer
    await container.aclose()


async def test_unknown_host_short_circuits_to_reporter(settings: Settings) -> None:
    container = await _container(settings)
    result = await container.workflow.run(AnalysisRequest(question="Analyze issues for hostname XYZ."))
    assert result.plan.intent is Intent.HOST_ANALYSIS and result.plan.filter.hostnames == ("XYZ",)
    assert result.reports.executive_summary.total_logs == 0
    assert "classifier" not in result.timings_ms
    assert "No log records matched" in result.answer
    assert any("does not appear" in n for n in result.plan.notes)
    await container.aclose()


async def test_explicit_filter_overrides_parsed_question(settings: Settings) -> None:
    container = await _container(settings)
    request = AnalysisRequest(
        question="Analyze issues for host web-01", filter=LogFilter(hostnames=("db-01",), min_severity=Severity.ERROR)
    )
    result = await container.workflow.run(request)
    assert result.plan.filter.hostnames == ("db-01",)
    hosts = {h.hostname for h in result.reports.host_health_report.hosts}
    assert hosts == {"db-01"}
    await container.aclose()


async def test_semantic_search_path(settings: Settings) -> None:
    container = await _container(settings)
    result = await container.workflow.run(AnalysisRequest(question="Find logs about database connection failures"))
    assert result.plan.intent is Intent.SEMANTIC_SEARCH and result.search_hits == []
    assert any("Semantic search unavailable" in w for w in result.warnings)
    await container.aclose()

    store = InMemoryVectorStore()
    container = await _container(settings, embedder=FakeEmbedder(), vector_store=store)
    stats = await container.indexer.index(LogFilter())
    assert stats.documents_indexed == stats.patterns_found > 0
    result = await container.workflow.run(AnalysisRequest(question="Find logs about connection refused"))
    assert result.search_hits and "connection refused" in result.search_hits[0].template
    assert "connection refused" in result.answer
    await container.aclose()


@pytest.mark.parametrize(
    "flt",
    [
        LogFilter(),
        LogFilter(hostnames=("web-01",), min_severity=Severity.ERROR),
        LogFilter(keywords=("Bearer", "connection refused")),
        LogFilter(keywords=("hunter2",)),  # secrets are redacted before keyword matching, on both paths
    ],
)
async def test_cached_and_streaming_paths_agree(settings: Settings, flt: LogFilter) -> None:
    results = []
    for cache in (True, False):
        container = await _container(settings.model_copy(update={"cache_dataset": cache}))
        frame = await container.dataset.load(flt)
        result = await container.workflow.run(AnalysisRequest(filter=flt))
        summary = result.reports.executive_summary
        results.append((frame.height, frame["message"].to_list(), summary.error_count, summary.incident_count))
        await container.aclose()
    assert results[0] == results[1]
    if flt.keywords == ("hunter2",):
        assert results[0][0] == 0


@pytest.mark.parametrize("node", ["planner", "classifier", "error_detector", "anomaly_detector", "correlator"])
async def test_mermaid_diagram_lists_nodes(settings: Settings, node: str) -> None:
    container = await _container(settings)
    assert node in container.workflow.mermaid()
    await container.aclose()
