"""Domain entities and value objects.

Everything here is immutable and framework-free (pydantic is used only for validation and
serialisation). Timestamps are always timezone-aware UTC.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def to_utc(value: datetime) -> datetime:
    """Interpret naive datetimes as UTC; convert aware ones to UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


# --------------------------------------------------------------------------- classification vocabulary


class Severity(StrEnum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"

    @property
    def rank(self) -> int:
        return SEVERITY_RANK[self]

    @classmethod
    def at_least(cls, floor: Severity) -> list[Severity]:
        return [s for s in cls if s.rank >= floor.rank]

    @classmethod
    def from_rank(cls, rank: int) -> Severity:
        return next(s for s, r in SEVERITY_RANK.items() if r == rank)


SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 4,
    Severity.ERROR: 3,
    Severity.WARNING: 2,
    Severity.INFO: 1,
    Severity.DEBUG: 0,
}
ERROR_SEVERITIES: tuple[Severity, ...] = (Severity.CRITICAL, Severity.ERROR)


class Category(StrEnum):
    CLOUD_AGENT = "cloud_agent"
    MEMORY = "memory"
    STORAGE = "storage"
    MAIL = "mail"
    SCHEDULER = "scheduler"
    DATABASE = "database"
    AUTH = "auth"
    NETWORK = "network"
    KERNEL = "kernel"
    SERVICE = "service"
    HTTP = "http"
    APPLICATION = "application"
    OTHER = "other"


# --------------------------------------------------------------------------- querying


class KeywordMode(StrEnum):
    ANY = "any"
    ALL = "all"


class LogFilter(DomainModel):
    """Selection criteria shared by the API, the CLI and the NL query planner."""

    start: datetime | None = None
    end: datetime | None = None
    hostnames: tuple[str, ...] = Field(default=(), max_length=100)
    processes: tuple[str, ...] = Field(default=(), max_length=100)
    keywords: tuple[str, ...] = Field(default=(), max_length=50)
    keyword_mode: KeywordMode = KeywordMode.ANY
    min_severity: Severity | None = None

    @field_validator("start", "end")
    @classmethod
    def _as_utc(cls, value: datetime | None) -> datetime | None:
        return to_utc(value) if value is not None else None

    @field_validator("hostnames", "processes", "keywords", mode="before")
    @classmethod
    def _clean_terms(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = [value]
        cleaned = (str(v).strip() for v in value)
        return tuple(dict.fromkeys(v[:200] for v in cleaned if v))

    @model_validator(mode="after")
    def _check_range(self) -> LogFilter:
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must not be after end")
        return self

    def merged_with(self, override: LogFilter) -> LogFilter:
        """Return a copy where every field explicitly set on ``override`` wins."""
        data = self.model_dump()
        data.update(override.model_dump(exclude_defaults=True))
        return LogFilter.model_validate(data)

    def describe(self) -> str:
        parts = []
        if self.start or self.end:
            parts.append(f"time {_fmt(self.start) or '…'} → {_fmt(self.end) or '…'}")
        if self.hostnames:
            parts.append(f"hosts={','.join(self.hostnames)}")
        if self.processes:
            parts.append(f"processes={','.join(self.processes)}")
        if self.keywords:
            parts.append(f"keywords({self.keyword_mode})={','.join(self.keywords)}")
        if self.min_severity:
            parts.append(f"severity>={self.min_severity}")
        return "; ".join(parts) or "all logs"


def _fmt(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC") if value else ""


class Intent(StrEnum):
    FULL_ANALYSIS = "full_analysis"
    CRITICAL_ISSUES = "critical_issues"
    ERRORS_BY_HOST = "errors_by_host"
    FAILING_PROCESSES = "failing_processes"
    RECURRING_PATTERNS = "recurring_patterns"
    HOST_ANALYSIS = "host_analysis"
    ANOMALIES = "anomalies"
    ROOT_CAUSE = "root_cause"
    SEMANTIC_SEARCH = "semantic_search"

    @property
    def needs_deep_analysis(self) -> bool:
        """Deep intents run root-cause, incident-summary and recommendation agents (LLM-heavy)."""
        return self in {Intent.FULL_ANALYSIS, Intent.CRITICAL_ISSUES, Intent.HOST_ANALYSIS, Intent.ROOT_CAUSE}


class QueryPlan(DomainModel):
    intent: Intent
    filter: LogFilter
    question: str | None = None
    search_text: str | None = None
    source: Literal["request", "rules", "llm"] = "request"
    notes: list[str] = Field(default_factory=list)


class AnalysisRequest(DomainModel):
    question: str | None = Field(default=None, max_length=2000)
    filter: LogFilter = Field(default_factory=LogFilter)
    intent: Intent | None = None

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None


# --------------------------------------------------------------------------- raw data views


class LogRecord(DomainModel):
    timestamp: datetime
    hostname: str
    process: str
    process_name: str
    pid: int | None
    message: str
    severity: Severity
    category: Category


class LogPage(DomainModel):
    total: int
    limit: int
    offset: int
    items: list[LogRecord]


class CountItem(DomainModel):
    key: str
    count: int


class DatasetCatalog(DomainModel):
    """Cheap metadata used for entity resolution in NL queries and for UI pickers."""

    total_rows: int
    start: datetime | None
    end: datetime | None
    hostnames: list[str]
    process_names: list[str]


class DatasetProfile(DomainModel):
    total_rows: int
    start: datetime | None
    end: datetime | None
    host_count: int
    process_count: int
    severity_counts: dict[str, int] = Field(default_factory=dict)
    category_counts: dict[str, int] = Field(default_factory=dict)
    top_hosts: list[CountItem] = Field(default_factory=list)
    top_processes: list[CountItem] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(self.severity_counts.get(s.value, 0) for s in ERROR_SEVERITIES)


# --------------------------------------------------------------------------- findings


class ErrorPattern(DomainModel):
    """A family of log lines that share a template (variable parts masked)."""

    fingerprint: str
    template: str
    sample: str
    process_name: str
    hostnames: list[str]
    severity: Severity
    category: Category
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    active_hours: int
    recurring: bool
    label: str | None = None
    explanation: str | None = None
    likely_impact: str | None = None


class GroupStat(DomainModel):
    key: str
    total: int
    errors: int
    criticals: int
    warnings: int
    error_rate: float
    distinct_pids: int | None = None
    top_pattern: str | None = None


class FailureKind(StrEnum):
    CRASH = "crash"
    OOM_KILL = "oom_kill"
    UNIT_FAILED = "unit_failed"
    EXIT_FAILURE = "exit_failure"
    RESTART = "restart"


class ProcessFailure(DomainModel):
    hostname: str
    process_name: str
    kind: FailureKind
    count: int
    first_seen: datetime
    last_seen: datetime
    evidence: list[str]
    event_times: list[datetime] = Field(default_factory=list, description="Up to 50 individual occurrence times")


class ErrorFindings(DomainModel):
    total_errors: int
    total_criticals: int
    total_warnings: int
    critical_errors: list[ErrorPattern]
    top_patterns: list[ErrorPattern]
    recurring_patterns: list[ErrorPattern]
    by_host: list[GroupStat]
    by_process: list[GroupStat]
    process_failures: list[ProcessFailure]

    def all_patterns(self) -> dict[str, ErrorPattern]:
        merged: dict[str, ErrorPattern] = {}
        for pattern in (*self.critical_errors, *self.top_patterns, *self.recurring_patterns):
            merged.setdefault(pattern.fingerprint, pattern)
        return merged


class AnomalyKind(StrEnum):
    VOLUME_SPIKE = "volume_spike"
    ERROR_SPIKE = "error_spike"
    SILENCE_GAP = "silence_gap"
    SERVICE_OUTAGE = "service_outage"


class Anomaly(DomainModel):
    kind: AnomalyKind
    hostname: str
    process_name: str | None = None
    start: datetime
    end: datetime
    observed: float
    expected: float
    score: float
    severity: Severity
    description: str


class TimeBucket(DomainModel):
    start: datetime
    total: int
    errors: int


class AnomalyFindings(DomainModel):
    bucket: str
    anomalies: list[Anomaly]
    volume_series: list[TimeBucket]
    series_bucket: str


class IncidentKind(StrEnum):
    CRITICAL_ERROR = "critical_error"
    ERROR_SPIKE = "error_spike"
    ERROR_PATTERN = "error_pattern"
    PROCESS_FAILURE = "process_failure"
    SERVICE_OUTAGE = "service_outage"
    SILENCE_GAP = "silence_gap"
    VOLUME_SPIKE = "volume_spike"


class Incident(DomainModel):
    id: str
    kind: IncidentKind
    title: str
    severity: Severity
    category: Category
    start: datetime
    end: datetime
    hostnames: list[str]
    processes: list[str]
    event_count: int
    score: float
    pattern_fingerprints: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    correlated_signals: list[str] = Field(default_factory=list)
    knowledge_rule: str | None = None
    heuristic_cause: str | None = None
    heuristic_remediation: list[str] = Field(default_factory=list)
    classification: str | None = None
    impact: str | None = None
    summary: str | None = None


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RootCause(DomainModel):
    incident_id: str
    probable_cause: str
    confidence: Confidence
    reasoning: str
    contributing_factors: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    similar_history: list[str] = Field(default_factory=list)
    source: Literal["llm", "heuristic"]


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Recommendation(DomainModel):
    priority: Priority
    title: str
    steps: list[str]
    rationale: str
    incident_ids: list[str] = Field(default_factory=list)
    owner: str | None = None
    source: Literal["llm", "heuristic"]


class ExecutiveNarrative(DomainModel):
    headline: str
    narrative: str


# --------------------------------------------------------------------------- RAG


class PatternDocument(DomainModel):
    """Unit of semantic indexing: one log template with its statistics (not one raw line)."""

    fingerprint: str
    process_name: str
    template: str
    sample: str
    hostnames: list[str]
    severity: Severity
    category: Category
    occurrences: int
    first_seen: datetime
    last_seen: datetime

    def content(self) -> str:
        hosts = ", ".join(self.hostnames[:5]) + (" …" if len(self.hostnames) > 5 else "")
        return (
            f"[{self.severity}] {self.category} | process={self.process_name} | hosts={hosts} | "
            f"seen {self.occurrences}x from {_fmt(self.first_seen)} to {_fmt(self.last_seen)}\n"
            f"pattern: {self.template}\nexample: {self.sample[:600]}"
        )


class SearchFilters(DomainModel):
    hostname: str | None = None
    process_name: str | None = None
    min_severity: Severity | None = None
    categories: tuple[Category, ...] = ()


class SearchHit(DomainModel):
    fingerprint: str
    process_name: str
    template: str
    sample: str
    hostnames: list[str]
    severity: Severity
    category: Category
    occurrences: int
    first_seen: datetime | None
    last_seen: datetime | None
    score: float
    similarity: float


class IndexStats(DomainModel):
    documents_indexed: int
    patterns_found: int
    embedding_model: str
    duration_ms: float


# --------------------------------------------------------------------------- background runs


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RunKind(StrEnum):
    ANALYSIS = "analysis"
    INDEX = "index"


class AnalysisRun(DomainModel):
    id: str
    kind: RunKind
    status: RunStatus
    request: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def new(cls, kind: RunKind, request: dict[str, Any]) -> AnalysisRun:
        now = datetime.now(UTC)
        return cls(
            id=uuid.uuid4().hex, kind=kind, status=RunStatus.PENDING, request=request, created_at=now, updated_at=now
        )

    def transition(
        self, status: RunStatus, *, result: dict[str, Any] | None = None, error: str | None = None
    ) -> AnalysisRun:
        return self.model_copy(
            update={"status": status, "result": result, "error": error, "updated_at": datetime.now(UTC)}
        )
