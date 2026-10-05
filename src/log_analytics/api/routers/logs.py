from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query

from log_analytics.api.deps import ContainerDep, FilterDep
from log_analytics.domain.models import AnomalyFindings, DatasetCatalog, DatasetProfile, LogPage
from log_analytics.domain.reports import HostHealthReport, TopErrorDashboard

router = APIRouter(tags=["logs & insights"])


@router.get("/logs", response_model=LogPage, summary="Filtered, classified and redacted log lines")
async def list_logs(
    container: ContainerDep,
    flt: FilterDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0, le=10_000_000)] = 0,
    order: Literal["asc", "desc"] = "desc",
) -> LogPage:
    return await container.dataset.page(flt, limit=limit, offset=offset, descending=order == "desc")


@router.get("/logs/catalog", response_model=DatasetCatalog, summary="Known hosts, processes and time range")
async def catalog(container: ContainerDep) -> DatasetCatalog:
    return await container.dataset.catalog()


@router.get("/logs/stats", response_model=DatasetProfile, summary="Volume and severity/category distribution")
async def stats(container: ContainerDep, flt: FilterDep) -> DatasetProfile:
    return await container.insights.profile(flt)


@router.get("/insights/top-errors", response_model=TopErrorDashboard, summary="Top Error Dashboard (no LLM)")
async def top_errors(container: ContainerDep, flt: FilterDep) -> TopErrorDashboard:
    return await container.insights.top_errors(flt)


@router.get("/insights/host-health", response_model=HostHealthReport, summary="Host Health Report (no LLM)")
async def host_health(container: ContainerDep, flt: FilterDep) -> HostHealthReport:
    return await container.insights.host_health(flt)


@router.get("/insights/anomalies", response_model=AnomalyFindings, summary="Spikes, gaps and outages (no LLM)")
async def anomalies(container: ContainerDep, flt: FilterDep) -> AnomalyFindings:
    return await container.insights.anomalies(flt)
