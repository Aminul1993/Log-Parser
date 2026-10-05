"""Incident correlation: turn raw findings (patterns, anomalies, failures) into ranked incidents.

* Anomaly windows driven by the same failure signature (or the same dominant pattern) merge into one
  incident: one cause, many windows. A bursty bug therefore cannot crowd every other problem out.
* Error patterns of the same process that match the same known signature merge too (e.g. four
  variants of "invalid token" become one authentication incident).
* A per-kind quota keeps the list diverse; candidates sharing a key with a higher-ranked incident are dropped.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

import polars as pl

from log_analytics.application.analytics.knowledge import LOG_FLOOD_RULE, KnowledgeBase, KnowledgeRule
from log_analytics.application.analytics.patterns import ERROR_VALUES, fingerprint
from log_analytics.domain.models import (
    Anomaly,
    AnomalyFindings,
    AnomalyKind,
    Category,
    ErrorFindings,
    ErrorPattern,
    FailureKind,
    Incident,
    IncidentKind,
    ProcessFailure,
    Severity,
)

_ANOMALY_TO_INCIDENT = {
    AnomalyKind.ERROR_SPIKE: IncidentKind.ERROR_SPIKE,
    AnomalyKind.SERVICE_OUTAGE: IncidentKind.SERVICE_OUTAGE,
    AnomalyKind.SILENCE_GAP: IncidentKind.SILENCE_GAP,
    AnomalyKind.VOLUME_SPIKE: IncidentKind.VOLUME_SPIKE,
}
_FAILURE_SEVERITY = {
    FailureKind.OOM_KILL: Severity.CRITICAL,
    FailureKind.CRASH: Severity.CRITICAL,
    FailureKind.UNIT_FAILED: Severity.ERROR,
    FailureKind.EXIT_FAILURE: Severity.ERROR,
    FailureKind.RESTART: Severity.WARNING,
}
_FAILURE_TITLE = {
    FailureKind.OOM_KILL: "OOM kill",
    FailureKind.CRASH: "crash",
    FailureKind.UNIT_FAILED: "unit failure",
    FailureKind.EXIT_FAILURE: "abnormal exit",
    FailureKind.RESTART: "restarts",
}
DEFAULT_QUOTAS = {IncidentKind.ERROR_PATTERN: 5}
ACUTE_MAX_DURATION = timedelta(hours=2)


def _score(severity: Severity, events: int, bonus: float = 0.0) -> float:
    return round(severity.rank * 10 + math.log1p(events) * 2 + bonus, 2)


@dataclass(slots=True)
class _Candidate:
    incident: Incident
    keys: set[tuple] = field(default_factory=set)
    group: tuple | None = None  # candidates with the same non-None group are merged
    anomaly: Anomaly | None = None
    rule: KnowledgeRule | None = None


class IncidentCorrelator:
    def __init__(
        self,
        knowledge: KnowledgeBase,
        *,
        max_incidents: int = 10,
        max_per_kind: int = 3,
        quotas: dict[IncidentKind, int] | None = None,
        evidence_per_incident: int = 3,
        correlation_window: timedelta = timedelta(minutes=30),
    ) -> None:
        self._kb = knowledge
        self._max = max_incidents
        self._per_kind = max_per_kind
        self._quotas = DEFAULT_QUOTAS if quotas is None else quotas
        self._evidence = evidence_per_incident
        self._window = correlation_window

    def correlate(self, frame: pl.DataFrame, errors: ErrorFindings, anomalies: AnomalyFindings) -> list[Incident]:
        rows = frame.select("timestamp", "hostname", "process_name", "template", "message", "category", "severity")
        error_rows = rows.filter(pl.col("severity").is_in(ERROR_VALUES))
        patterns = errors.all_patterns()
        windows = [
            self._from_anomaly(a, rows if a.kind is AnomalyKind.VOLUME_SPIKE else error_rows)
            for a in anomalies.anomalies
            if not (a.kind is AnomalyKind.VOLUME_SPIKE and a.severity.rank < Severity.ERROR.rank)
        ]
        pattern_candidates = [
            self._from_pattern(p, IncidentKind.ERROR_PATTERN)
            for p in (*errors.recurring_patterns, *errors.top_patterns)
        ]
        candidates = [
            *self._merge(windows, patterns, self._merged_window),
            *(self._from_pattern(p, IncidentKind.CRITICAL_ERROR) for p in errors.critical_errors),
            *(self._from_failure(f) for f in errors.process_failures),
            *self._merge(self._unique(pattern_candidates), patterns, self._merged_patterns),
        ]

        seen: set[tuple] = set()
        per_kind: Counter[IncidentKind] = Counter()
        selected: list[Incident] = []
        for candidate in sorted(candidates, key=lambda c: c.incident.score, reverse=True):
            incident = candidate.incident
            quota = self._quotas.get(incident.kind, self._per_kind)
            if candidate.keys & seen or per_kind[incident.kind] >= quota:
                continue
            seen |= candidate.keys
            per_kind[incident.kind] += 1
            selected.append(incident)
            if len(selected) >= self._max:
                break
        return [self._annotate(i, errors, anomalies, patterns) for i in selected]

    # ------------------------------------------------------------------ builders

    def _from_anomaly(self, anomaly: Anomaly, source: pl.DataFrame) -> _Candidate:
        window = source.filter(
            pl.col("timestamp").is_between(anomaly.start, anomaly.end, closed="left")
            & (pl.col("hostname") == anomaly.hostname)
        )
        if anomaly.process_name and anomaly.kind is AnomalyKind.SERVICE_OUTAGE:
            window = window.filter(pl.col("process_name") == anomaly.process_name)
        top = (
            window.group_by("process_name", "template")
            .agg(
                pl.len().alias("n"),
                pl.col("message").first().alias("sample"),
                pl.col("category").mode().sort().first().alias("category"),
            )
            .sort(["n", "template"], descending=[True, False])
            .head(self._evidence)
            .to_dicts()
        )
        kind = _ANOMALY_TO_INCIDENT[anomaly.kind]
        if top:
            category = Category(top[0]["category"])
        elif anomaly.kind is AnomalyKind.SILENCE_GAP:
            category = Category.SERVICE
        else:
            category = Category.OTHER
        subject = anomaly.process_name or "all logging"
        titles = {
            IncidentKind.ERROR_SPIKE: f"Error spike on {anomaly.hostname}: {anomaly.observed:.0f} errors in one "
            f"interval (~{anomaly.expected:.0f} expected)",
            IncidentKind.SERVICE_OUTAGE: f"Possible service outage: {subject} on {anomaly.hostname}",
            IncidentKind.SILENCE_GAP: f"{subject} silent on {anomaly.hostname} "
            f"({(anomaly.end - anomaly.start).total_seconds() / 60:.0f} min)",
            IncidentKind.VOLUME_SPIKE: f"Log volume spike on {anomaly.hostname}: {anomaly.observed:.0f} lines in one "
            f"interval (~{anomaly.expected:.0f} expected)",
        }
        is_volume = kind is IncidentKind.VOLUME_SPIKE
        events = int(window.height) or int(anomaly.observed)
        fingerprints = [] if is_volume else [fingerprint(r["process_name"], r["template"]) for r in top]
        evidence = [
            anomaly.description,
            *(
                (f"{r['n']:,}x {r['process_name']}: {r['sample'][:240]}" if is_volume else r["sample"][:300])
                for r in top
            ),
        ]
        rule = None if is_volume or not top else self._kb.match(f"{top[0]['template']} {top[0]['sample']}")
        primary = fingerprints[0] if fingerprints and kind is IncidentKind.ERROR_SPIKE else None
        incident = Incident(
            id=self._id(kind, anomaly.hostname, anomaly.process_name or "", anomaly.start.isoformat()),
            kind=kind,
            title=titles[kind],
            severity=anomaly.severity,
            category=category,
            start=anomaly.start,
            end=anomaly.end,
            hostnames=[anomaly.hostname],
            processes=sorted({r["process_name"] for r in top} | ({anomaly.process_name} - {None})),
            event_count=events,
            # Volume spikes rank on intensity only: their raw line counts would dwarf real error incidents.
            score=_score(anomaly.severity, 0 if is_volume else events, min(anomaly.score, 20.0) / 2),
            pattern_fingerprints=fingerprints,
            evidence=evidence,
        )
        signature = rule.name if rule else primary
        return _Candidate(
            incident=incident,
            keys={("pattern", primary)} if primary else set(),
            group=(kind, anomaly.hostname, anomaly.process_name, signature),
            anomaly=anomaly,
            rule=rule,
        )

    def _from_pattern(self, pattern: ErrorPattern, kind: IncidentKind) -> _Candidate:
        name = pattern.label or pattern.template[:90]
        if kind is IncidentKind.CRITICAL_ERROR:
            title = f"Critical error in {pattern.process_name}: {name}"
        else:
            adjective = "Recurring" if pattern.recurring else "Frequent"
            title = f"{adjective} error in {pattern.process_name}: {name}"
        incident = Incident(
            id=self._id("pattern", pattern.fingerprint),
            kind=kind,
            title=title,
            severity=pattern.severity,
            category=pattern.category,
            start=pattern.first_seen,
            end=pattern.last_seen,
            hostnames=pattern.hostnames,
            processes=[pattern.process_name],
            event_count=pattern.occurrences,
            score=_score(pattern.severity, pattern.occurrences, min(pattern.active_hours, 24) / 4),
            pattern_fingerprints=[pattern.fingerprint],
            evidence=[pattern.sample[:300]],
        )
        rule = self._kb.match(f"{pattern.template} {pattern.sample}")
        return _Candidate(
            incident=incident,
            keys={("pattern", pattern.fingerprint)},
            group=(kind, pattern.process_name, rule.name) if rule and kind is IncidentKind.ERROR_PATTERN else None,
            rule=rule,
        )

    def _from_failure(self, failure: ProcessFailure) -> _Candidate:
        severity = _FAILURE_SEVERITY[failure.kind]
        if failure.kind is FailureKind.RESTART and failure.count >= 3:
            severity = Severity.ERROR
        category = {
            FailureKind.OOM_KILL: Category.MEMORY,
            FailureKind.UNIT_FAILED: Category.SERVICE,
        }.get(failure.kind, Category.APPLICATION)
        incident = Incident(
            id=self._id("failure", failure.hostname, failure.process_name, failure.kind.value),
            kind=IncidentKind.PROCESS_FAILURE,
            title=f"Process {_FAILURE_TITLE[failure.kind]}: {failure.process_name} on {failure.hostname} "
            f"({failure.count}x)",
            severity=severity,
            category=category,
            start=failure.first_seen,
            end=failure.last_seen,
            hostnames=[failure.hostname],
            processes=[failure.process_name],
            event_count=failure.count,
            score=_score(severity, failure.count, 2.0),
            evidence=failure.evidence[: self._evidence + 2],
        )
        return _Candidate(incident=incident, keys={("failure", failure.hostname, failure.process_name, failure.kind)})

    # ------------------------------------------------------------------ merging

    @staticmethod
    def _unique(candidates: list[_Candidate]) -> list[_Candidate]:
        seen: set[str] = set()
        unique = []
        for candidate in candidates:
            if candidate.incident.id not in seen:
                seen.add(candidate.incident.id)
                unique.append(candidate)
        return unique

    def _merge(
        self,
        candidates: list[_Candidate],
        patterns: dict[str, ErrorPattern],
        build: Callable[[tuple, list[_Candidate], dict[str, ErrorPattern]], Incident],
    ) -> list[_Candidate]:
        groups: dict[tuple, list[_Candidate]] = {}
        result: list[_Candidate] = []
        for candidate in candidates:
            if candidate.group is None:
                result.append(candidate)
            else:
                groups.setdefault(candidate.group, []).append(candidate)
        for group_key, members in groups.items():
            members.sort(key=lambda c: c.incident.score, reverse=True)
            if len(members) == 1:
                members[0].keys.add(("group", group_key))
                result.append(members[0])
                continue
            incident = build(group_key, members, patterns)
            keys = set().union(*(c.keys for c in members)) | {("group", group_key)}
            result.append(_Candidate(incident=incident, keys=keys, group=group_key, rule=members[0].rule))
        return result

    def _merged_window(
        self, group_key: tuple, members: list[_Candidate], patterns: dict[str, ErrorPattern]
    ) -> Incident:
        lead = members[0].incident
        kind, host, process, signature = group_key
        primary = lead.pattern_fingerprints[0] if lead.pattern_fingerprints else None
        name = members[0].rule.title if members[0].rule else ""
        if not name and primary in patterns:
            name = patterns[primary].label or patterns[primary].template[:80]
        n = len(members)
        titles = {
            IncidentKind.ERROR_SPIKE: f"Repeated error spikes on {host} ({n} windows)" + (f": {name}" if name else ""),
            IncidentKind.SERVICE_OUTAGE: f"Repeated outage bursts: {process} on {host} ({n} windows)",
            IncidentKind.SILENCE_GAP: f"{process or 'All logging'} repeatedly silent on {host} ({n} gaps)",
            IncidentKind.VOLUME_SPIKE: f"Repeated log volume spikes on {host} ({n} windows)",
        }
        windows = [
            f"window {c.anomaly.start:%Y-%m-%d %H:%M}–{c.anomaly.end:%H:%M}: {c.anomaly.description}"
            for c in members[:5]
            if c.anomaly
        ]
        return lead.model_copy(
            update={
                "id": self._id(kind, host, process or "", signature or ""),
                "title": titles[kind],
                "severity": max((c.incident.severity for c in members), key=lambda s: s.rank),
                "start": min(c.incident.start for c in members),
                "end": max(c.incident.end for c in members),
                "processes": sorted({p for c in members for p in c.incident.processes}),
                "event_count": sum(c.incident.event_count for c in members),
                "score": round(lead.score + math.log1p(n) * 2, 2),
                "pattern_fingerprints": list(
                    dict.fromkeys(fp for c in members for fp in c.incident.pattern_fingerprints)
                )[:5],
                "correlated_signals": windows,
            }
        )

    def _merged_patterns(
        self, group_key: tuple, members: list[_Candidate], patterns: dict[str, ErrorPattern]
    ) -> Incident:
        lead = members[0].incident
        kind, process, rule_name = group_key
        events = sum(c.incident.event_count for c in members)
        severity = max((c.incident.severity for c in members), key=lambda s: s.rank)
        bonus = max(min(patterns[fp].active_hours, 24) / 4 for c in members for fp in c.incident.pattern_fingerprints)
        title = f"{lead.title} (+{len(members) - 1} related pattern{'s' if len(members) > 2 else ''})"
        return lead.model_copy(
            update={
                "id": self._id(kind, process, rule_name),
                "title": title,
                "severity": severity,
                "start": min(c.incident.start for c in members),
                "end": max(c.incident.end for c in members),
                "hostnames": sorted({h for c in members for h in c.incident.hostnames}),
                "event_count": events,
                "score": _score(severity, events, bonus),
                "pattern_fingerprints": [fp for c in members for fp in c.incident.pattern_fingerprints][:6],
                "evidence": [c.incident.evidence[0] for c in members[:4]],
            }
        )

    # ------------------------------------------------------------------ enrichment

    def _annotate(
        self,
        incident: Incident,
        errors: ErrorFindings,
        anomalies: AnomalyFindings,
        patterns: dict[str, ErrorPattern],
    ) -> Incident:
        templates = [patterns[fp].template for fp in incident.pattern_fingerprints if fp in patterns]
        rule: KnowledgeRule | None = LOG_FLOOD_RULE
        if incident.kind is not IncidentKind.VOLUME_SPIKE:
            # Match the dominant evidence first so a secondary line cannot hijack the explanation.
            primary_text = " ".join([incident.title, *incident.evidence[:2], *templates[:1]])
            rule = self._kb.match(primary_text) or self._kb.match(" ".join([*incident.evidence, *templates]))
        signals = list(incident.correlated_signals)
        if incident.end - incident.start <= ACUTE_MAX_DURATION:
            lo, hi = incident.start - self._window, incident.end + self._window
            hosts = set(incident.hostnames)
            for anomaly in anomalies.anomalies:
                overlaps = anomaly.hostname in hosts and anomaly.start <= hi and anomaly.end >= lo
                is_self = anomaly.start == incident.start and anomaly.hostname == incident.hostnames[0]
                if overlaps and not is_self:
                    signals.append(f"{anomaly.kind.value} at {anomaly.start:%Y-%m-%d %H:%M}: {anomaly.description}")
            for failure in errors.process_failures:
                hits = [t for t in failure.event_times if lo <= t <= hi] if failure.hostname in hosts else []
                if hits:
                    signals.append(
                        f"{failure.kind.value} of {failure.process_name} at "
                        + ", ".join(f"{t:%Y-%m-%d %H:%M}" for t in hits[:3])
                    )
        else:
            for fp in incident.pattern_fingerprints[:2]:
                if fp in patterns:
                    p = patterns[fp]
                    signals.append(
                        f"chronic: '{p.template[:60]}' active in {p.active_hours} distinct hours between "
                        f"{p.first_seen:%Y-%m-%d %H:%M} and {p.last_seen:%Y-%m-%d %H:%M}"
                    )
        return incident.model_copy(
            update={
                "correlated_signals": list(dict.fromkeys(signals))[:8],
                "knowledge_rule": rule.name if rule else None,
                "heuristic_cause": rule.cause if rule else None,
                "heuristic_remediation": list(rule.remediation) if rule else [],
            }
        )

    @staticmethod
    def _id(*parts: str | object) -> str:
        return "INC-" + fingerprint(*(str(p) for p in parts))[:8].upper()
