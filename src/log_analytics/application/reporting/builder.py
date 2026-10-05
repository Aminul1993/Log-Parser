"""Assemble the six analysis reports from findings. Pure functions over domain objects."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from log_analytics.application.analytics.health import HostHealthScorer
from log_analytics.domain.models import (
    Anomaly,
    AnomalyFindings,
    DatasetProfile,
    ErrorFindings,
    ExecutiveNarrative,
    FailureKind,
    Incident,
    Recommendation,
    RootCause,
    Severity,
)
from log_analytics.domain.reports import (
    AnalysisReports,
    ExecutiveSummary,
    HealthStatus,
    HostHealthReport,
    IncidentReport,
    RecommendedActions,
    RootCauseReport,
    TopErrorDashboard,
)

_STATUS_ORDER = {HealthStatus.HEALTHY: 0, HealthStatus.DEGRADED: 1, HealthStatus.CRITICAL: 2}


def overall_status(hosts: HostHealthReport | None, incidents: Sequence[Incident]) -> HealthStatus:
    status = (
        max((h.status for h in hosts.hosts), key=_STATUS_ORDER.__getitem__, default=HealthStatus.HEALTHY)
        if hosts
        else HealthStatus.HEALTHY
    )
    if any(i.severity is Severity.CRITICAL for i in incidents):
        return HealthStatus.CRITICAL
    if status is HealthStatus.HEALTHY and any(i.severity is Severity.ERROR for i in incidents):
        return HealthStatus.DEGRADED
    return status


class ReportBuilder:
    def __init__(self, scorer: HostHealthScorer | None = None) -> None:
        self._scorer = scorer or HostHealthScorer()

    def top_error_dashboard(
        self, profile: DatasetProfile, errors: ErrorFindings, anomalies: AnomalyFindings | None
    ) -> TopErrorDashboard:
        return TopErrorDashboard(
            severity_distribution=profile.severity_counts,
            category_distribution=profile.category_counts,
            top_patterns=errors.top_patterns,
            recurring_patterns=errors.recurring_patterns[:10],
            errors_by_process=errors.by_process[:15],
            errors_by_host=errors.by_host[:50],
            top_processes_by_volume=profile.top_processes,
            error_trend=anomalies.volume_series if anomalies else [],
            trend_bucket=anomalies.series_bucket if anomalies else "",
        )

    def host_health(
        self, errors: ErrorFindings, anomalies: AnomalyFindings | None, incidents: Sequence[Incident] | None
    ) -> HostHealthReport:
        return HostHealthReport(hosts=self._scorer.score(errors, anomalies, incidents))

    def executive_summary(
        self,
        profile: DatasetProfile,
        errors: ErrorFindings | None,
        anomalies: AnomalyFindings | None,
        incidents: Sequence[Incident],
        hosts: HostHealthReport | None,
        narrative: ExecutiveNarrative | None,
    ) -> ExecutiveSummary:
        status = overall_status(hosts, incidents)
        error_count = errors.total_errors if errors else profile.error_count
        critical_count = errors.total_criticals if errors else profile.severity_counts.get("CRITICAL", 0)
        rate = error_count / profile.total_rows if profile.total_rows else 0.0
        anomaly_list = anomalies.anomalies if anomalies else []
        headline = (
            narrative.headline
            if narrative
            else (
                f"{status.value.upper()}: {len(incidents)} incident(s) detected; error rate {rate:.2%} "
                f"across {profile.host_count} host(s)"
            )
        )
        return ExecutiveSummary(
            period_start=profile.start,
            period_end=profile.end,
            overall_status=status,
            headline=headline,
            narrative=narrative.narrative
            if narrative
            else self._fallback_narrative(profile, errors, anomaly_list, incidents, error_count, critical_count, rate),
            total_logs=profile.total_rows,
            error_count=error_count,
            critical_count=critical_count,
            error_rate=round(rate, 6),
            incident_count=len(incidents),
            anomaly_count=len(anomaly_list),
            top_incidents=[f"[{i.severity.value}] {i.title}" for i in incidents[:5]],
            narrative_source="llm" if narrative else "heuristic",
        )

    @staticmethod
    def _fallback_narrative(
        profile: DatasetProfile,
        errors: ErrorFindings | None,
        anomalies: Sequence[Anomaly],
        incidents: Sequence[Incident],
        error_count: int,
        critical_count: int,
        rate: float,
    ) -> str:
        if not profile.total_rows:
            return "No log records matched the selected filters."
        sentences = [
            f"Analysed {profile.total_rows:,} log lines from {profile.host_count} host(s) and "
            f"{profile.process_count} process(es) between {profile.start:%Y-%m-%d %H:%M} and "
            f"{profile.end:%Y-%m-%d %H:%M} UTC.",
            f"{error_count:,} lines were errors ({rate:.2%}), {critical_count:,} of them critical.",
        ]
        if anomalies:
            kinds = Counter(a.kind.value.replace("_", " ") for a in anomalies)
            sentences.append(
                f"{len(anomalies)} anomal{'y' if len(anomalies) == 1 else 'ies'} detected ("
                + ", ".join(f"{n} {k}" for k, n in kinds.most_common())
                + ")."
            )
        if errors:
            restarts = sum(f.count for f in errors.process_failures if f.kind is FailureKind.RESTART)
            if restarts:
                sentences.append(f"{restarts} process restart(s) were observed.")
        if incidents:
            sentences.append(f"Most significant incident: {incidents[0].title}.")
        return " ".join(sentences)

    def build(
        self,
        *,
        profile: DatasetProfile | None,
        errors: ErrorFindings | None,
        anomalies: AnomalyFindings | None,
        incidents: Sequence[Incident] | None,
        root_causes: Sequence[RootCause] | None,
        recommendations: Sequence[Recommendation] | None,
        narrative: ExecutiveNarrative | None,
    ) -> AnalysisReports:
        if profile is None:
            return AnalysisReports()
        incident_list = list(incidents or [])
        hosts = self.host_health(errors, anomalies, incident_list) if errors else None
        return AnalysisReports(
            executive_summary=self.executive_summary(profile, errors, anomalies, incident_list, hosts, narrative),
            incident_report=IncidentReport(incidents=incident_list) if incidents is not None else None,
            root_cause_report=RootCauseReport(analyses=list(root_causes)) if root_causes is not None else None,
            top_error_dashboard=self.top_error_dashboard(profile, errors, anomalies) if errors else None,
            host_health_report=hosts,
            recommended_actions=RecommendedActions(actions=list(recommendations))
            if recommendations is not None
            else None,
        )
