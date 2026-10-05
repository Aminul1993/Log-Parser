"""HTTP-specific request/response models (domain models are reused for everything else)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from log_analytics.domain.models import (
    AnalysisRun,
    Category,
    Intent,
    LogFilter,
    RunKind,
    RunStatus,
    SearchHit,
    Severity,
)


class JobAccepted(BaseModel):
    run_id: str
    kind: RunKind
    status: RunStatus
    status_url: str


class RunSummary(BaseModel):
    id: str
    kind: RunKind
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    question: str | None = None

    @classmethod
    def of(cls, run: AnalysisRun) -> RunSummary:
        return cls(
            id=run.id,
            kind=run.kind,
            status=run.status,
            created_at=run.created_at,
            updated_at=run.updated_at,
            error=run.error,
            question=run.request.get("question"),
        )


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["What are today's critical issues?"])
    filter: LogFilter = Field(default_factory=LogFilter, description="Explicit filters override parsed ones")
    intent: Intent | None = Field(default=None, description="Force an intent instead of inferring it")


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000, examples=["ssm agent cannot get credentials"])
    k: int = Field(default=8, ge=1, le=100)
    hostname: str | None = None
    process_name: str | None = None
    min_severity: Severity | None = None
    categories: list[Category] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class RagStats(BaseModel):
    documents: int
