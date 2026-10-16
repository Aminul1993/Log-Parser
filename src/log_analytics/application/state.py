"""LangGraph state. Small, typed and serialisable; DataFrames live in the FrameStore keyed by run_id."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from log_analytics.domain.models import (
    AnalysisRequest,
    AnomalyFindings,
    DatasetProfile,
    ErrorFindings,
    ExecutiveNarrative,
    Incident,
    QueryPlan,
    Recommendation,
    RootCause,
    SearchHit,
)
from log_analytics.domain.reports import AnalysisResult


def merge_dicts(left: dict[str, float] | None, right: dict[str, float] | None) -> dict[str, float]:
    return {**(left or {}), **(right or {})}


class AnalysisState(TypedDict, total=False):
    run_id: str
    request: AnalysisRequest
    plan: QueryPlan
    row_count: int
    profile: DatasetProfile
    template_labels: dict[str, str]
    errors: ErrorFindings
    anomalies: AnomalyFindings
    incidents: list[Incident]
    root_causes: list[RootCause]
    narrative: ExecutiveNarrative | None
    recommendations: list[Recommendation]
    search_hits: list[SearchHit]
    result: AnalysisResult
    # Reducers let parallel branches (error + anomaly detection) both contribute safely.
    warnings: Annotated[list[str], operator.add]
    timings_ms: Annotated[dict[str, float], merge_dicts]
