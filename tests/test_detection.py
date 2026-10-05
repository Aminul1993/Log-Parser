"""Error, anomaly, incident and health detection on the planted synthetic scenarios."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from log_analytics.application.analytics.anomalies import AnomalyConfig, AnomalyDetector
from log_analytics.application.analytics.errors import ErrorDetector
from log_analytics.application.analytics.health import HostHealthScorer
from log_analytics.application.analytics.incidents import IncidentCorrelator
from log_analytics.application.analytics.knowledge import KnowledgeBase
from log_analytics.domain.models import AnomalyKind, FailureKind, IncidentKind, Severity
from log_analytics.domain.reports import HealthStatus
from tests.conftest import BASE_UTC


@pytest.fixture(scope="module")
def errors(frame: pl.DataFrame):  # type: ignore[no-untyped-def]
    return ErrorDetector().detect(frame)


@pytest.fixture(scope="module")
def anomalies(frame: pl.DataFrame):  # type: ignore[no-untyped-def]
    return AnomalyDetector().detect(frame)


# --------------------------------------------------------------------------- errors


def test_top_patterns_and_counts(errors) -> None:  # type: ignore[no-untyped-def]
    top = errors.top_patterns[0]
    assert top.process_name == "python" and "connection refused" in top.template and top.occurrences == 80
    assert "(attempt <N>)" in top.template  # 80 distinct lines collapsed into one template
    assert errors.total_criticals == 1


def test_recurring_and_critical_patterns(errors) -> None:  # type: ignore[no-untyped-def]
    recurring = {p.template for p in errors.recurring_patterns}
    assert "ERROR byte indices must be integers or slices, not str" in recurring
    assert all("connection refused" not in t for t in recurring)  # a single 5-minute burst is not "recurring"
    assert [p.process_name for p in errors.critical_errors] == ["kernel"]


def test_group_stats(errors) -> None:  # type: ignore[no-untyped-def]
    hosts = {s.key: s for s in errors.by_host}
    assert set(hosts) == {"web-01", "db-01"}
    assert hosts["web-01"].errors == 92 and hosts["web-01"].top_pattern.startswith("ERROR could not connect")
    python = next(s for s in errors.by_process if s.key == "python")
    assert python.distinct_pids == 2


def test_process_failures(errors) -> None:  # type: ignore[no-untyped-def]
    kinds = {(f.kind, f.process_name) for f in errors.process_failures}
    assert (FailureKind.OOM_KILL, "kernel") in kinds
    assert (FailureKind.UNIT_FAILED, "postgresql.service") in kinds
    restart = next(f for f in errors.process_failures if f.kind is FailureKind.RESTART)
    assert restart.process_name == "python" and restart.count == 1
    assert "pid 1000" in restart.evidence[0] and "no errors in its last minute" in restart.evidence[0]
    assert restart.event_times[0] == BASE_UTC + timedelta(hours=3)


# --------------------------------------------------------------------------- anomalies


def _kinds(findings, kind: AnomalyKind):  # type: ignore[no-untyped-def]
    return [a for a in findings.anomalies if a.kind is kind]


def test_error_spike_detected_at_planted_window(anomalies) -> None:  # type: ignore[no-untyped-def]
    spikes = _kinds(anomalies, AnomalyKind.ERROR_SPIKE)
    assert any(a.hostname == "web-01" and a.start == BASE_UTC + timedelta(hours=2) for a in spikes)


def test_heartbeat_silence_gap(anomalies) -> None:  # type: ignore[no-untyped-def]
    gaps = [a for a in _kinds(anomalies, AnomalyKind.SILENCE_GAP) if a.process_name == "CRON"]
    assert len(gaps) == 1
    assert gaps[0].start == BASE_UTC + timedelta(hours=3, minutes=59, seconds=1)
    assert gaps[0].observed == pytest.approx(46.0)
    assert not [a for a in _kinds(anomalies, AnomalyKind.SILENCE_GAP) if a.process_name == "python"]


def test_service_outage_burst(anomalies) -> None:  # type: ignore[no-untyped-def]
    outages = _kinds(anomalies, AnomalyKind.SERVICE_OUTAGE)
    assert [(a.hostname, a.process_name) for a in outages] == [("web-01", "python")]
    assert outages[0].observed == 80


def test_no_spikes_on_flat_series() -> None:
    start = datetime(2026, 9, 30, tzinfo=UTC)
    flat = pl.DataFrame(
        {
            "timestamp": [start + timedelta(seconds=10 * i) for i in range(4000)],
            "hostname": "h",
            "process_name": "app",
            "message": "ok",
            "severity": "INFO",
        }
    )
    found = AnomalyDetector().detect(flat)
    assert found.anomalies == []
    assert sum(b.total for b in found.volume_series) == 4000


def test_daily_recurring_spike_is_labelled_and_downgraded() -> None:
    rows = []
    day0 = datetime(2026, 9, 26, tzinfo=UTC)
    for day in range(4):
        base = day0 + timedelta(days=day)
        rows += [base + timedelta(minutes=m) for m in range(0, 24 * 60, 1)]  # steady 1/min
        rows += [base + timedelta(hours=3, seconds=s) for s in range(0, 240, 1) for _ in range(5)]  # 03:00 batch
    frame = pl.DataFrame(
        {"timestamp": sorted(rows), "hostname": "h", "process_name": "app", "message": "x", "severity": "INFO"}
    )
    spikes = _kinds(AnomalyDetector(AnomalyConfig(volume_extreme_min=100)).detect(frame), AnomalyKind.VOLUME_SPIKE)
    at_three = [a for a in spikes if a.start.hour == 3]
    assert len(at_three) >= 3
    assert all("recurs daily at 03:00" in a.description and a.severity is Severity.WARNING for a in at_three)


# --------------------------------------------------------------------------- incidents & health


@pytest.fixture(scope="module")
def incidents(frame, errors, anomalies):  # type: ignore[no-untyped-def]
    return IncidentCorrelator(KnowledgeBase()).correlate(frame, errors, anomalies)


def test_incidents_cover_planted_scenarios(incidents) -> None:  # type: ignore[no-untyped-def]
    kinds = {i.kind for i in incidents}
    assert {IncidentKind.CRITICAL_ERROR, IncidentKind.PROCESS_FAILURE, IncidentKind.SILENCE_GAP} <= kinds
    assert IncidentKind.ERROR_SPIKE in kinds or IncidentKind.SERVICE_OUTAGE in kinds
    rules = {i.knowledge_rule for i in incidents}
    assert {"oom-kill", "upstream-unavailable", "systemd-unit-failure", "ssh-bruteforce"} <= rules
    assert len({i.id for i in incidents}) == len(incidents)
    assert [i.score for i in incidents] == sorted((i.score for i in incidents), reverse=True)


def test_spike_and_its_dominant_pattern_are_not_duplicated(incidents) -> None:  # type: ignore[no-untyped-def]
    refused = [i for i in incidents if any("connection refused" in e for e in i.evidence)]
    assert len(refused) <= 2  # spike/outage, never an extra stand-alone pattern incident as well
    assert not any(i.kind is IncidentKind.ERROR_PATTERN and "connection refused" in i.title for i in incidents)


def test_acute_incidents_get_correlated_signals(incidents) -> None:  # type: ignore[no-untyped-def]
    oom = next(i for i in incidents if i.knowledge_rule == "oom-kill")
    assert any("unit_failed" in s for s in oom.correlated_signals)


def test_quota_limits_incidents_per_kind(frame, errors, anomalies) -> None:  # type: ignore[no-untyped-def]
    limited = IncidentCorrelator(KnowledgeBase(), max_per_kind=1, quotas={}).correlate(frame, errors, anomalies)
    kinds = [i.kind for i in limited]
    assert len(kinds) == len(set(kinds))


def test_host_health(errors, anomalies, incidents) -> None:  # type: ignore[no-untyped-def]
    report = {h.hostname: h for h in HostHealthScorer().score(errors, anomalies, incidents)}
    assert report["db-01"].critical_count == 1 and report["web-01"].restart_count == 1
    assert report["web-01"].health_score < 100 and report["db-01"].status in set(HealthStatus)
