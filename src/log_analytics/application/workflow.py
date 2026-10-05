"""The LangGraph analysis workflow.

    planner ─┬─► semantic_search ─────────────────────────────────────────────┐
             └─► data_loader ─┬─► classifier ─┬─► error_detector ──┐          │
                              │               └─► anomaly_detector ┴─► correlator
                              │                                         ├─► root_cause_analyst
                              │                                         │     └─► incident_summarizer
                              │                                         │           └─► recommender ─┐
                              └─(no rows)───────────────────────────────┴──────────────────────────► reporter ─► END

Error and anomaly detection run in parallel (fan-out/fan-in). Deep, LLM-heavy agents only run for
intents that need them, so "summarise errors by host" costs one LLM call instead of six.
"""

from __future__ import annotations

import time
import uuid

import structlog
from langgraph.graph import END, START, StateGraph

from log_analytics.application.agents import (
    AnomalyDetectionAgent,
    DataLoadingAgent,
    ErrorDetectionAgent,
    IncidentCorrelationAgent,
    IncidentSummaryAgent,
    LogClassificationAgent,
    QueryPlannerAgent,
    RecommendationAgent,
    ReportComposerAgent,
    RootCauseAnalysisAgent,
    SemanticSearchAgent,
)
from log_analytics.application.agents.base import Agent
from log_analytics.application.frame_store import FrameStore
from log_analytics.application.state import AnalysisState
from log_analytics.domain.models import AnalysisRequest, Intent
from log_analytics.domain.reports import AnalysisResult

log = structlog.get_logger(__name__)


def route_after_plan(state: AnalysisState) -> str:
    return SemanticSearchAgent.name if state["plan"].intent is Intent.SEMANTIC_SEARCH else DataLoadingAgent.name


def route_after_load(state: AnalysisState) -> str:
    return LogClassificationAgent.name if state["profile"].total_rows else ReportComposerAgent.name


def route_after_correlation(state: AnalysisState) -> str:
    if state["plan"].intent.needs_deep_analysis and state.get("incidents"):
        return RootCauseAnalysisAgent.name
    return ReportComposerAgent.name


class AnalysisWorkflow:
    def __init__(
        self,
        *,
        planner: QueryPlannerAgent,
        loader: DataLoadingAgent,
        classifier: LogClassificationAgent,
        error_detector: ErrorDetectionAgent,
        anomaly_detector: AnomalyDetectionAgent,
        correlator: IncidentCorrelationAgent,
        root_cause: RootCauseAnalysisAgent,
        summarizer: IncidentSummaryAgent,
        recommender: RecommendationAgent,
        searcher: SemanticSearchAgent,
        reporter: ReportComposerAgent,
        frames: FrameStore,
    ) -> None:
        self._frames = frames
        agents: list[Agent] = [
            planner,
            loader,
            classifier,
            error_detector,
            anomaly_detector,
            correlator,
            root_cause,
            summarizer,
            recommender,
            searcher,
            reporter,
        ]
        self._graph = self._compile(agents)

    @staticmethod
    def _compile(agents: list[Agent]):  # type: ignore[no-untyped-def]
        graph = StateGraph(AnalysisState)
        for agent in agents:
            graph.add_node(agent.name, agent.__call__)
        planner, loader, classifier = QueryPlannerAgent.name, DataLoadingAgent.name, LogClassificationAgent.name
        errors, anomalies, correlator = (
            ErrorDetectionAgent.name,
            AnomalyDetectionAgent.name,
            IncidentCorrelationAgent.name,
        )
        rca, summary, recommend = RootCauseAnalysisAgent.name, IncidentSummaryAgent.name, RecommendationAgent.name
        search, reporter = SemanticSearchAgent.name, ReportComposerAgent.name

        graph.add_edge(START, planner)
        graph.add_conditional_edges(planner, route_after_plan, {search: search, loader: loader})
        graph.add_conditional_edges(loader, route_after_load, {classifier: classifier, reporter: reporter})
        graph.add_edge(classifier, errors)  # fan-out: both detectors run concurrently
        graph.add_edge(classifier, anomalies)
        graph.add_edge([errors, anomalies], correlator)  # fan-in: waits for both
        graph.add_conditional_edges(correlator, route_after_correlation, {rca: rca, reporter: reporter})
        graph.add_edge(rca, summary)
        graph.add_edge(summary, recommend)
        graph.add_edge(recommend, reporter)
        graph.add_edge(search, reporter)
        graph.add_edge(reporter, END)
        return graph.compile()

    def mermaid(self) -> str:
        return self._graph.get_graph().draw_mermaid()

    async def run(self, request: AnalysisRequest, *, run_id: str | None = None) -> AnalysisResult:
        run_id = run_id or uuid.uuid4().hex
        initial: AnalysisState = {"run_id": run_id, "request": request, "warnings": [], "timings_ms": {}}
        started = time.perf_counter()
        with structlog.contextvars.bound_contextvars(run_id=run_id):
            log.info("analysis_started", question=request.question, filter=request.filter.describe())
            try:
                final = await self._graph.ainvoke(
                    initial,
                    config={"recursion_limit": 32, "run_name": "log-analysis", "metadata": {"run_id": run_id}},
                )
            finally:
                self._frames.drop(run_id)
            total_ms = round((time.perf_counter() - started) * 1000, 1)
            result: AnalysisResult = final["result"]
            result = result.model_copy(
                update={
                    "warnings": list(dict.fromkeys(final.get("warnings", []))),
                    "timings_ms": {**final.get("timings_ms", {}), "total": total_ms},
                }
            )
            log.info(
                "analysis_completed",
                intent=result.plan.intent.value,
                duration_ms=total_ms,
                warnings=len(result.warnings),
            )
            return result
