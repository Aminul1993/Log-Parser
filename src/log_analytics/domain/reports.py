"""Report value objects produced at the end of every analysis run."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from log_analytics.domain.models import (
    AnomalyFindings,
    CountItem,
    DomainModel,
    ErrorPattern,
    GroupStat,
    Incident,
    QueryPlan,
    Recommendation,
    RootCause,
    SearchHit,
    TimeBucket,
)


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class ReportType(StrEnum):
    EXECUTIVE_SUMMARY = "executive_summary"
    INCIDENT_REPORT = "incident_report"
    ROOT_CAUSE_REPORT = "root_cause_report"
    TOP_ERROR_DASHBOARD = "top_error_dashboard"
    HOST_HEALTH_REPORT = "host_health_report"
    RECOMMENDED_ACTIONS = "recommended_actions"


class ExecutiveSummary(DomainModel):
    period_start: datetime | None
    period_end: datetime | None
    overall_status: HealthStatus
    headline: str
    narrative: str
    total_logs: int
    error_count: int
    critical_count: int
    error_rate: float
    incident_count: int
    anomaly_count: int
    top_incidents: list[str]
    narrative_source: str


class IncidentReport(DomainModel):
    incidents: list[Incident]


class RootCauseReport(DomainModel):
    analyses: list[RootCause]


class TopErrorDashboard(DomainModel):
    severity_distribution: dict[str, int]
    category_distribution: dict[str, int]
    top_patterns: list[ErrorPattern]
    recurring_patterns: list[ErrorPattern]
    errors_by_process: list[GroupStat]
    errors_by_host: list[GroupStat]
    top_processes_by_volume: list[CountItem]
    error_trend: list[TimeBucket]
    trend_bucket: str


class HostHealth(DomainModel):
    hostname: str
    status: HealthStatus
    health_score: float
    total_logs: int
    error_count: int
    critical_count: int
    warning_count: int
    error_rate: float
    anomaly_count: int
    restart_count: int
    top_issues: list[str]


class HostHealthReport(DomainModel):
    hosts: list[HostHealth]


class RecommendedActions(DomainModel):
    actions: list[Recommendation]


class AnalysisReports(DomainModel):
    executive_summary: ExecutiveSummary | None = None
    incident_report: IncidentReport | None = None
    root_cause_report: RootCauseReport | None = None
    top_error_dashboard: TopErrorDashboard | None = None
    host_health_report: HostHealthReport | None = None
    recommended_actions: RecommendedActions | None = None

    def get(self, report_type: ReportType) -> DomainModel | None:
        return getattr(self, report_type.value)


class AnalysisResult(DomainModel):
    run_id: str
    question: str | None
    plan: QueryPlan
    answer: str | None
    reports: AnalysisReports
    anomalies: AnomalyFindings | None = None
    search_hits: list[SearchHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    llm_model: str | None = None
