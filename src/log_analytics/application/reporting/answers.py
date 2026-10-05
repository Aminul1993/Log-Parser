"""Intent-specific answer context (for the LLM) and deterministic Markdown answers (fallback)."""

from __future__ import annotations

from typing import Any

from log_analytics.application.reporting import markdown as md
from log_analytics.domain.models import (
    AnomalyFindings,
    DatasetProfile,
    ErrorFindings,
    Incident,
    Intent,
    SearchHit,
)
from log_analytics.domain.reports import AnalysisReports

_PATTERN_FIELDS = {
    "process_name",
    "template",
    "label",
    "severity",
    "category",
    "occurrences",
    "active_hours",
    "first_seen",
    "last_seen",
    "hostnames",
    "explanation",
}


class AnswerMaterial:
    """Everything an answer may draw on, gathered from graph state."""

    def __init__(
        self,
        *,
        profile: DatasetProfile | None,
        errors: ErrorFindings | None,
        anomalies: AnomalyFindings | None,
        incidents: list[Incident] | None,
        hits: list[SearchHit],
        reports: AnalysisReports,
    ) -> None:
        self.profile = profile
        self.errors = errors
        self.anomalies = anomalies
        self.incidents = incidents or []
        self.hits = hits
        self.reports = reports

    @property
    def empty(self) -> bool:
        return (self.profile is None or self.profile.total_rows == 0) and not self.hits

    # ------------------------------------------------------------------ LLM context

    def context(self, intent: Intent) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        if self.profile:
            ctx["dataset"] = self.profile.model_dump(
                mode="json", include={"total_rows", "start", "end", "host_count", "process_count", "severity_counts"}
            )
        errors, anomalies = self.errors, self.anomalies
        if intent is Intent.SEMANTIC_SEARCH:
            ctx["search_hits"] = [h.model_dump(mode="json", exclude={"sample"}) for h in self.hits]
        elif intent is Intent.ERRORS_BY_HOST and errors:
            ctx["errors_by_host"] = [s.model_dump(mode="json") for s in errors.by_host[:25]]
        elif intent is Intent.FAILING_PROCESSES and errors:
            ctx["errors_by_process"] = [s.model_dump(mode="json") for s in errors.by_process if s.errors][:15]
            ctx["process_failures"] = [f.model_dump(mode="json") for f in errors.process_failures[:15]]
        elif intent is Intent.RECURRING_PATTERNS and errors:
            ctx["recurring_patterns"] = [
                p.model_dump(mode="json", include=_PATTERN_FIELDS) for p in errors.recurring_patterns[:15]
            ]
        elif intent is Intent.ANOMALIES and anomalies:
            ctx["anomalies"] = [a.model_dump(mode="json") for a in anomalies.anomalies[:20]]
        else:
            reports = self.reports
            if reports.executive_summary:
                ctx["executive_summary"] = reports.executive_summary.model_dump(
                    mode="json", include={"overall_status", "headline", "narrative", "error_rate"}
                )
            ctx["incidents"] = [
                i.model_dump(
                    mode="json",
                    include={
                        "id",
                        "title",
                        "severity",
                        "category",
                        "start",
                        "end",
                        "hostnames",
                        "processes",
                        "event_count",
                        "summary",
                        "impact",
                    },
                )
                for i in self.incidents[:8]
            ]
            if reports.root_cause_report:
                ctx["root_causes"] = [
                    c.model_dump(mode="json", include={"incident_id", "probable_cause", "confidence"})
                    for c in reports.root_cause_report.analyses
                ]
            if reports.recommended_actions:
                ctx["recommended_actions"] = [
                    {"priority": a.priority.value, "title": a.title, "first_steps": a.steps[:2]}
                    for a in reports.recommended_actions.actions[:6]
                ]
            if reports.host_health_report:
                ctx["host_health"] = [
                    h.model_dump(mode="json", include={"hostname", "status", "health_score", "error_count"})
                    for h in reports.host_health_report.hosts[:10]
                ]
        return ctx

    # ------------------------------------------------------------------ deterministic answer

    def fallback(self, intent: Intent) -> str:
        if self.empty and intent is not Intent.SEMANTIC_SEARCH:
            return "No log records matched the query filters, so there is nothing to report."
        errors, anomalies = self.errors, self.anomalies
        if intent is Intent.SEMANTIC_SEARCH:
            return (
                ("Most relevant log patterns:\n\n" + md.hits_table(self.hits))
                if self.hits
                else ("No matching log patterns were found in the vector index.")
            )
        if intent is Intent.ERRORS_BY_HOST and errors:
            return f"{errors.total_errors:,} error lines across {len(errors.by_host)} host(s):\n\n" + md.group_table(
                errors.by_host[:25], "Host"
            )
        if intent is Intent.FAILING_PROCESSES and errors:
            failing = [s for s in errors.by_process if s.errors][:15]
            text = "Processes ranked by error volume:\n\n" + md.group_table(failing, "Process")
            if errors.process_failures:
                text += "\n\nDetected failures and restarts:\n\n" + md.failures_table(errors.process_failures[:15])
            return text
        if intent is Intent.RECURRING_PATTERNS and errors:
            if not errors.recurring_patterns:
                return "No recurring error patterns (≥5 occurrences spread over ≥2 hours) were found."
            return f"{len(errors.recurring_patterns)} recurring error pattern(s):\n\n" + md.patterns_table(
                errors.recurring_patterns[:15]
            )
        if intent is Intent.ANOMALIES and anomalies:
            if not anomalies.anomalies:
                return "No volume spikes, error spikes, silence gaps or outage bursts were detected."
            return f"{len(anomalies.anomalies)} anomal(ies) detected:\n\n" + md.anomalies_table(
                anomalies.anomalies[:20]
            )
        summary = self.reports.executive_summary
        parts = [f"**{summary.overall_status.value.upper()}** — {summary.headline}"] if summary else []
        if self.incidents:
            parts += ["", md.incidents_table(self.incidents[:10])]
        if self.reports.recommended_actions and self.reports.recommended_actions.actions:
            parts += ["", "Top actions:"]
            parts += [f"- [{a.priority.value}] {a.title}" for a in self.reports.recommended_actions.actions[:5]]
        return "\n".join(parts) or "No significant issues were detected."
