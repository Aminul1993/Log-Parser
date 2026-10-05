"""Markdown rendering for reports and deterministic answers (used when no LLM is available)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from log_analytics.domain.models import (
    Anomaly,
    ErrorPattern,
    GroupStat,
    Incident,
    ProcessFailure,
    SearchHit,
    TimeBucket,
)
from log_analytics.domain.reports import (
    AnalysisReports,
    AnalysisResult,
    ExecutiveSummary,
    HostHealthReport,
    IncidentReport,
    RecommendedActions,
    ReportType,
    RootCauseReport,
    TopErrorDashboard,
)

_BARS = "▁▂▃▄▅▆▇█"


def _cell(value: Any, limit: int = 120) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    text = str(value).replace("|", "\\|").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) if len(lines) > 2 else "_No data._"


def sparkline(values: Sequence[int]) -> str:
    if not values:
        return ""
    high = max(values) or 1
    return "".join(_BARS[min(len(_BARS) - 1, int(v / high * (len(_BARS) - 1)))] for v in values)


def _period(start: datetime | None, end: datetime | None) -> str:
    return f"{_cell(start)} → {_cell(end)} UTC" if start and end else "n/a"


# --------------------------------------------------------------------------- building blocks


def patterns_table(patterns: Sequence[ErrorPattern]) -> str:
    return table(
        ["#", "Severity", "Process", "Occurrences", "Active hours", "Last seen", "Pattern"],
        [
            (i, p.severity.value, p.process_name, p.occurrences, p.active_hours, p.last_seen, p.label or p.template)
            for i, p in enumerate(patterns, 1)
        ],
    )


def group_table(stats: Sequence[GroupStat], key: str) -> str:
    return table(
        [key, "Total", "Errors", "Critical", "Warnings", "Error rate", "Top error"],
        [(s.key, s.total, s.errors, s.criticals, s.warnings, f"{s.error_rate:.2%}", s.top_pattern) for s in stats],
    )


def failures_table(failures: Sequence[ProcessFailure]) -> str:
    return table(
        ["Host", "Process / unit", "Kind", "Count", "First", "Last", "Evidence"],
        [
            (
                f.hostname,
                f.process_name,
                f.kind.value,
                f.count,
                f.first_seen,
                f.last_seen,
                f.evidence[0] if f.evidence else "",
            )
            for f in failures
        ],
    )


def anomalies_table(anomalies: Sequence[Anomaly]) -> str:
    return table(
        ["Severity", "Kind", "Host", "Process", "Start", "End", "Details"],
        [
            (a.severity.value, a.kind.value, a.hostname, a.process_name, a.start, a.end, a.description)
            for a in anomalies
        ],
    )


def incidents_table(incidents: Sequence[Incident]) -> str:
    return table(
        ["ID", "Severity", "Kind", "Title", "Hosts", "Events", "Start", "End"],
        [
            (i.id, i.severity.value, i.kind.value, i.title, ", ".join(i.hostnames[:3]), i.event_count, i.start, i.end)
            for i in incidents
        ],
    )


def hits_table(hits: Sequence[SearchHit]) -> str:
    return table(
        ["Score", "Similarity", "Severity", "Process", "Occurrences", "Last seen", "Pattern"],
        [
            (h.score, h.similarity, h.severity.value, h.process_name, h.occurrences, h.last_seen, h.template)
            for h in hits
        ],
    )


def trend(series: Sequence[TimeBucket], bucket: str) -> str:
    if not series:
        return "_No data._"
    return (
        f"Per {bucket}, {_cell(series[0].start)} → {_cell(series[-1].start)}\n\n"
        f"```\nvolume {sparkline([b.total for b in series])}\nerrors {sparkline([b.errors for b in series])}\n```"
    )


# --------------------------------------------------------------------------- reports


def render_executive_summary(s: ExecutiveSummary) -> str:
    return "\n".join(
        [
            "## Executive Summary",
            f"**Status: {s.overall_status.value.upper()}** — {s.headline}",
            "",
            s.narrative,
            "",
            table(
                ["Period", "Logs", "Errors", "Critical", "Error rate", "Incidents", "Anomalies"],
                [
                    (
                        _period(s.period_start, s.period_end),
                        s.total_logs,
                        s.error_count,
                        s.critical_count,
                        f"{s.error_rate:.2%}",
                        s.incident_count,
                        s.anomaly_count,
                    )
                ],
            ),
            "",
            *(f"- {t}" for t in s.top_incidents),
        ]
    )


def render_incident_report(r: IncidentReport) -> str:
    parts = ["## Incident Report", incidents_table(r.incidents)]
    for i in r.incidents:
        parts += [
            "",
            f"### {i.id} · {i.title}",
            f"*{i.severity.value} · {i.category.value} · {i.kind.value} · {_period(i.start, i.end)} · "
            f"{i.event_count:,} events*",
        ]
        if i.classification or i.impact:
            parts.append(f"**Classification:** {i.classification or '—'} · **Impact:** {i.impact or '—'}")
        if i.summary:
            parts.append(i.summary)
        if i.evidence:
            parts += ["", "Evidence:", *(f"- `{_cell(e, 240)}`" for e in i.evidence[:4])]
        if i.correlated_signals:
            parts += ["", "Correlated signals:", *(f"- {s}" for s in i.correlated_signals)]
    return "\n".join(parts)


def render_root_cause_report(r: RootCauseReport, titles: dict[str, str] | None = None) -> str:
    titles = titles or {}
    parts = ["## Root Cause Analysis"]
    if not r.analyses:
        parts.append("_No incidents required root-cause analysis._")
    for a in r.analyses:
        parts += [
            "",
            f"### {a.incident_id} · {titles.get(a.incident_id, '')}".rstrip(" ·"),
            f"**Probable cause** ({a.confidence.value} confidence, {a.source}): {a.probable_cause}",
            "",
            a.reasoning,
        ]
        if a.contributing_factors:
            parts += ["", "Contributing factors:", *(f"- {f}" for f in a.contributing_factors)]
        if a.similar_history:
            parts += ["", "Similar historical patterns:", *(f"- {h}" for h in a.similar_history)]
    return "\n".join(parts)


def render_top_error_dashboard(d: TopErrorDashboard) -> str:
    severities = table(["Severity", "Count"], sorted(d.severity_distribution.items(), key=lambda kv: -kv[1]))
    categories = table(["Category", "Count"], sorted(d.category_distribution.items(), key=lambda kv: -kv[1])[:12])
    return "\n\n".join(
        [
            "## Top Error Dashboard",
            "### Trend\n" + trend(d.error_trend, d.trend_bucket),
            "### Severity distribution\n" + severities,
            "### Category distribution\n" + categories,
            "### Top error patterns\n" + patterns_table(d.top_patterns),
            "### Recurring patterns\n" + patterns_table(d.recurring_patterns),
            "### Errors by process\n" + group_table(d.errors_by_process, "Process"),
            "### Errors by host\n" + group_table(d.errors_by_host, "Host"),
        ]
    )


def render_host_health_report(r: HostHealthReport) -> str:
    return "## Host Health Report\n\n" + table(
        ["Host", "Status", "Score", "Logs", "Errors", "Critical", "Error rate", "Anomalies", "Restarts", "Top issues"],
        [
            (
                h.hostname,
                h.status.value,
                f"{h.health_score:.1f}",
                h.total_logs,
                h.error_count,
                h.critical_count,
                f"{h.error_rate:.2%}",
                h.anomaly_count,
                h.restart_count,
                "; ".join(h.top_issues),
            )
            for h in r.hosts
        ],
    )


def render_recommended_actions(r: RecommendedActions) -> str:
    parts = ["## Recommended Actions"]
    if not r.actions:
        parts.append("_No actions recommended._")
    for n, a in enumerate(r.actions, 1):
        parts += [
            "",
            f"### {n}. [{a.priority.value}] {a.title}",
            f"*Owner: {a.owner or 'unassigned'} · Incidents: {', '.join(a.incident_ids) or '—'} · Source: {a.source}*",
            "",
            a.rationale,
            "",
            *(f"{i}. {step}" for i, step in enumerate(a.steps, 1)),
        ]
    return "\n".join(parts)


def render_report(report_type: ReportType, reports: AnalysisReports) -> str | None:
    report = reports.get(report_type)
    if report is None:
        return None
    titles = {i.id: i.title for i in reports.incident_report.incidents} if reports.incident_report else {}
    match report_type:
        case ReportType.EXECUTIVE_SUMMARY:
            return render_executive_summary(report)  # type: ignore[arg-type]
        case ReportType.INCIDENT_REPORT:
            return render_incident_report(report)  # type: ignore[arg-type]
        case ReportType.ROOT_CAUSE_REPORT:
            return render_root_cause_report(report, titles)  # type: ignore[arg-type]
        case ReportType.TOP_ERROR_DASHBOARD:
            return render_top_error_dashboard(report)  # type: ignore[arg-type]
        case ReportType.HOST_HEALTH_REPORT:
            return render_host_health_report(report)  # type: ignore[arg-type]
        case ReportType.RECOMMENDED_ACTIONS:
            return render_recommended_actions(report)  # type: ignore[arg-type]
    return None


def render_result(result: AnalysisResult) -> str:
    parts = ["# Log Analysis Report", ""]
    parts.append(
        f"*Run `{result.run_id}` · intent `{result.plan.intent.value}` · filter: {result.plan.filter.describe()}*"
    )
    if result.llm_model:
        parts.append(f"*Model: {result.llm_model}*")
    if result.question:
        parts += ["", f"> **Question:** {result.question}"]
    if result.answer:
        parts += ["", "## Answer", result.answer]
    for report_type in ReportType:
        rendered = render_report(report_type, result.reports)
        if rendered:
            parts += ["", rendered]
    if result.anomalies and result.anomalies.anomalies:
        parts += ["", "## Anomalies", anomalies_table(result.anomalies.anomalies)]
    if result.search_hits:
        parts += ["", "## Semantic Search Results", hits_table(result.search_hits)]
    notes = [*result.plan.notes, *result.warnings]
    if notes:
        parts += ["", "## Notes", *(f"- {n}" for n in notes)]
    return "\n".join(parts) + "\n"
