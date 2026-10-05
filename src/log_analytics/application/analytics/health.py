"""Host health scoring (0-100) from error rate, criticals, anomalies, failures and restarts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from log_analytics.domain.models import AnomalyFindings, ErrorFindings, FailureKind, Incident, Severity
from log_analytics.domain.reports import HealthStatus, HostHealth


@dataclass(frozen=True, slots=True)
class HealthWeights:
    error_rate_factor: float = 400.0  # 10% error rate costs 40 points
    error_rate_cap: float = 40.0
    critical_cost: float = 5.0
    critical_cap: float = 25.0
    anomaly_cost: float = 4.0
    anomaly_cap: float = 20.0
    restart_cost: float = 3.0
    restart_cap: float = 15.0
    failure_cost: float = 5.0
    failure_cap: float = 15.0
    healthy_floor: float = 85.0
    degraded_floor: float = 60.0


DEFAULT_WEIGHTS = HealthWeights()


def status_for(score: float, weights: HealthWeights = DEFAULT_WEIGHTS) -> HealthStatus:
    if score >= weights.healthy_floor:
        return HealthStatus.HEALTHY
    if score >= weights.degraded_floor:
        return HealthStatus.DEGRADED
    return HealthStatus.CRITICAL


class HostHealthScorer:
    def __init__(self, weights: HealthWeights | None = None) -> None:
        self._w = weights or HealthWeights()

    def score(
        self,
        errors: ErrorFindings,
        anomalies: AnomalyFindings | None,
        incidents: Sequence[Incident] | None = None,
    ) -> list[HostHealth]:
        w = self._w
        # Only significant (ERROR+) anomalies cost health; WARNING-level ones (busy hours, scheduled jobs) are context.
        anomaly_count = Counter(
            a.hostname for a in (anomalies.anomalies if anomalies else []) if a.severity.rank >= Severity.ERROR.rank
        )
        restarts: Counter[str] = Counter()
        failures: Counter[str] = Counter()
        for failure in errors.process_failures:
            target = restarts if failure.kind is FailureKind.RESTART else failures
            target[failure.hostname] += failure.count
        issues: dict[str, list[str]] = defaultdict(list)
        for incident in incidents or []:
            for host in incident.hostnames:
                if len(issues[host]) < 3:
                    issues[host].append(incident.title)

        report = []
        for stat in errors.by_host:
            host = stat.key
            score = 100.0
            score -= min(w.error_rate_cap, stat.error_rate * w.error_rate_factor)
            score -= min(w.critical_cap, stat.criticals * w.critical_cost)
            score -= min(w.anomaly_cap, anomaly_count[host] * w.anomaly_cost)
            score -= min(w.restart_cap, restarts[host] * w.restart_cost)
            score -= min(w.failure_cap, failures[host] * w.failure_cost)
            score = round(max(score, 0.0), 1)
            top_issues = issues.get(host) or ([f"Top error: {stat.top_pattern}"] if stat.top_pattern else [])
            report.append(
                HostHealth(
                    hostname=host,
                    status=status_for(score, w),
                    health_score=score,
                    total_logs=stat.total,
                    error_count=stat.errors,
                    critical_count=stat.criticals,
                    warning_count=stat.warnings,
                    error_rate=stat.error_rate,
                    anomaly_count=anomaly_count[host],
                    restart_count=restarts[host],
                    top_issues=top_issues,
                )
            )
        return sorted(report, key=lambda h: (h.health_score, h.hostname))
