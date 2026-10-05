from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse

from log_analytics.api.deps import ContainerDep
from log_analytics.api.schemas import JobAccepted, QueryRequest, RunSummary
from log_analytics.application.reporting.markdown import render_report, render_result
from log_analytics.domain.errors import ConflictError, NotFoundError
from log_analytics.domain.models import AnalysisRequest, AnalysisRun, RunKind, RunStatus
from log_analytics.domain.reports import AnalysisResult, ReportType

router = APIRouter(tags=["analysis"])


@router.post("/query", response_model=AnalysisResult, summary="Ask a natural-language question (synchronous)")
async def query(body: QueryRequest, container: ContainerDep) -> AnalysisResult:
    request = AnalysisRequest(question=body.question, filter=body.filter, intent=body.intent)
    try:
        return await asyncio.wait_for(container.workflow.run(request), timeout=container.settings.query_timeout_s)
    except TimeoutError:
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "Analysis exceeded the synchronous timeout; submit it as a job via POST /api/v1/analyses.",
        ) from None


@router.post(
    "/analyses",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a full analysis in the background",
)
async def create_analysis(body: AnalysisRequest, container: ContainerDep, request: Request) -> JobAccepted:
    async def work(run_id: str) -> dict[str, Any]:
        result = await container.workflow.run(body, run_id=run_id)
        return result.model_dump(mode="json")

    run = await container.jobs.submit(RunKind.ANALYSIS, body.model_dump(mode="json"), work)
    return JobAccepted(
        run_id=run.id, kind=run.kind, status=run.status, status_url=str(request.url_for("get_analysis", run_id=run.id))
    )


@router.get("/analyses", response_model=list[RunSummary], summary="Recent runs")
async def list_analyses(container: ContainerDep, limit: Annotated[int, Query(ge=1, le=200)] = 20) -> list[RunSummary]:
    return [RunSummary.of(r) for r in await container.runs.list(limit=limit)]


@router.get("/analyses/{run_id}", response_model=AnalysisRun, name="get_analysis", summary="Run status and result")
async def get_analysis(run_id: str, container: ContainerDep) -> AnalysisRun:
    return await _get_run(container, run_id)


@router.get("/analyses/{run_id}/report.md", response_class=PlainTextResponse, summary="Full Markdown report")
async def full_report(run_id: str, container: ContainerDep) -> PlainTextResponse:
    result = await _get_result(container, run_id)
    return PlainTextResponse(render_result(result), media_type="text/markdown; charset=utf-8")


@router.get("/analyses/{run_id}/reports/{report_type}", summary="One report as JSON or Markdown")
async def get_report(
    run_id: str,
    report_type: ReportType,
    container: ContainerDep,
    format: Literal["json", "markdown"] = "json",
) -> Any:
    result = await _get_result(container, run_id)
    report = result.reports.get(report_type)
    if report is None:
        raise NotFoundError(f"Report '{report_type.value}' was not produced for intent '{result.plan.intent.value}'")
    if format == "markdown":
        return PlainTextResponse(render_report(report_type, result.reports) or "", media_type="text/markdown")
    return report.model_dump(mode="json")


async def _get_run(container: ContainerDep, run_id: str) -> AnalysisRun:
    run = await container.runs.get(run_id)
    if run is None:
        raise NotFoundError(f"Run {run_id} not found")
    return run


async def _get_result(container: ContainerDep, run_id: str) -> AnalysisResult:
    run = await _get_run(container, run_id)
    if run.kind is not RunKind.ANALYSIS:
        raise ConflictError(f"Run {run_id} is a {run.kind.value} job, not an analysis")
    if run.status is not RunStatus.SUCCEEDED or run.result is None:
        raise ConflictError(f"Run {run_id} is {run.status.value}; reports are available once it succeeds")
    return AnalysisResult.model_validate(run.result)
