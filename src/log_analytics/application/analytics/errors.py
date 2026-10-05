"""Error detection: critical errors, frequent/recurring patterns, per-host/process stats, process failures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import polars as pl

from log_analytics.application.analytics.patterns import ERROR_VALUES, SEVERITY_RANK_MAP, fingerprint
from log_analytics.domain.models import (
    Category,
    ErrorFindings,
    ErrorPattern,
    FailureKind,
    GroupStat,
    ProcessFailure,
    Severity,
)

FAILURE_RULES: tuple[tuple[FailureKind, str], ...] = (
    (
        FailureKind.OOM_KILL,
        r"(?i)(out of memory: kill|oom-kill|invoked oom-killer|oom_reaper|memory cgroup out of memory)",
    ),
    (
        FailureKind.CRASH,
        r"(?i)(segfault|segmentation fault|core dumped|general protection fault|fatal python error|\bpanic:)",
    ),
    (
        FailureKind.UNIT_FAILED,
        r"(?i)(failed with result|entered failed state|start request repeated too quickly|failed to start"
        r"|main process exited, code=(?:killed|dumped)|main process exited, code=exited, status=[1-9])",
    ),
    (
        FailureKind.EXIT_FAILURE,
        r"(?i)(exited with (?:code|status) [1-9]|non-zero exit|terminated by signal|killed by signal"
        r"|exit status [1-9])",
    ),
)
UNIT_RE = r"([\w@.:-]+\.(?:service|socket|timer|mount|scope))\b"
_RANK_TO_SEVERITY = {rank: name for name, rank in SEVERITY_RANK_MAP.items()}


def aggregate_patterns(
    frame: pl.DataFrame, *, severities: Sequence[str] | None = None, sample_len: int = 400
) -> pl.DataFrame:
    """Collapse rows into one row per ``(process_name, template)`` with statistics and a fingerprint."""
    if severities is not None:
        frame = frame.filter(pl.col("severity").is_in(list(severities)))
    grouped = (
        frame.lazy()
        .with_columns(pl.col("severity").replace_strict(SEVERITY_RANK_MAP, return_dtype=pl.Int8).alias("_rank"))
        .group_by("process_name", "template")
        .agg(
            pl.len().alias("occurrences"),
            pl.col("timestamp").min().alias("first_seen"),
            pl.col("timestamp").max().alias("last_seen"),
            pl.col("hostname").unique().sort().head(20).alias("hostnames"),
            pl.col("message").first().str.slice(0, sample_len).alias("sample"),
            pl.col("_rank").max().alias("severity_rank"),
            pl.col("category").mode().sort().first().alias("category"),
            pl.col("timestamp").dt.truncate("1h").n_unique().alias("active_hours"),
        )
        .sort(["occurrences", "process_name", "template"], descending=[True, False, False])
        .collect()
    )
    fingerprints = [fingerprint(p, t) for p, t in zip(grouped["process_name"], grouped["template"], strict=True)]
    return grouped.with_columns(
        pl.Series("fingerprint", fingerprints, dtype=pl.String),
        pl.col("severity_rank").replace_strict(_RANK_TO_SEVERITY, return_dtype=pl.String).alias("severity"),
    )


@dataclass(frozen=True, slots=True)
class ErrorDetectionConfig:
    top_n: int = 20
    recurring_min_occurrences: int = 5
    recurring_min_hours: int = 2
    min_pid_lines: int = 50
    restart_tolerance: timedelta = timedelta(minutes=10)
    evidence_samples: int = 3


class ErrorDetector:
    def __init__(self, config: ErrorDetectionConfig | None = None) -> None:
        self._cfg = config or ErrorDetectionConfig()

    def detect(self, frame: pl.DataFrame) -> ErrorFindings:
        severity = pl.col("severity")
        totals = frame.select(
            severity.is_in(ERROR_VALUES).sum().alias("errors"),
            (severity == Severity.CRITICAL.value).sum().alias("criticals"),
            (severity == Severity.WARNING.value).sum().alias("warnings"),
        ).row(0, named=True)

        patterns = self._to_models(aggregate_patterns(frame, severities=ERROR_VALUES))
        top_n = self._cfg.top_n
        recurring = sorted(
            (p for p in patterns if p.recurring), key=lambda p: (p.active_hours, p.occurrences), reverse=True
        )
        return ErrorFindings(
            total_errors=int(totals["errors"] or 0),
            total_criticals=int(totals["criticals"] or 0),
            total_warnings=int(totals["warnings"] or 0),
            critical_errors=[p for p in patterns if p.severity is Severity.CRITICAL][:top_n],
            top_patterns=patterns[:top_n],
            recurring_patterns=recurring[:top_n],
            by_host=self._group_stats(frame, "hostname"),
            by_process=self._group_stats(frame, "process_name", with_pids=True),
            process_failures=[*self._explicit_failures(frame), *self._restarts(frame)],
        )

    # ------------------------------------------------------------------ helpers

    def _to_models(self, patterns: pl.DataFrame) -> list[ErrorPattern]:
        cfg = self._cfg
        return [
            ErrorPattern(
                fingerprint=row["fingerprint"],
                template=row["template"],
                sample=row["sample"],
                process_name=row["process_name"],
                hostnames=row["hostnames"],
                severity=Severity(row["severity"]),
                category=Category(row["category"]),
                occurrences=row["occurrences"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                active_hours=row["active_hours"],
                recurring=row["occurrences"] >= cfg.recurring_min_occurrences
                and row["active_hours"] >= cfg.recurring_min_hours,
            )
            for row in patterns.iter_rows(named=True)
        ]

    def _group_stats(self, frame: pl.DataFrame, key: str, *, with_pids: bool = False) -> list[GroupStat]:
        if frame.is_empty():
            return []
        severity = pl.col("severity")
        aggregations = [
            pl.len().alias("total"),
            severity.is_in(ERROR_VALUES).sum().alias("errors"),
            (severity == Severity.CRITICAL.value).sum().alias("criticals"),
            (severity == Severity.WARNING.value).sum().alias("warnings"),
        ]
        if with_pids:
            aggregations.append(pl.col("pid").drop_nulls().n_unique().alias("distinct_pids"))
        stats = frame.group_by(key).agg(aggregations)
        top = (
            frame.filter(severity.is_in(ERROR_VALUES))
            .group_by(key, "template")
            .agg(pl.len().alias("n"))
            .sort(["n", "template"], descending=[True, False])
            .group_by(key, maintain_order=True)
            .first()
            .select(key, pl.col("template").alias("top_pattern"))
        )
        stats = (
            stats.join(top, on=key, how="left")
            .with_columns((pl.col("errors") / pl.col("total")).alias("error_rate"))
            .sort(["errors", "total", key], descending=[True, True, False])
        )
        return [
            GroupStat(
                key=str(row[key]),
                total=row["total"],
                errors=row["errors"],
                criticals=row["criticals"],
                warnings=row["warnings"],
                error_rate=round(row["error_rate"], 6),
                distinct_pids=row.get("distinct_pids"),
                top_pattern=row["top_pattern"],
            )
            for row in stats.iter_rows(named=True)
        ]

    def _explicit_failures(self, frame: pl.DataFrame) -> list[ProcessFailure]:
        message = pl.col("message")
        kind = None
        for failure_kind, pattern in FAILURE_RULES:
            condition = message.str.contains(pattern)
            literal = pl.lit(failure_kind.value)
            kind = pl.when(condition).then(literal) if kind is None else kind.when(condition).then(literal)
        assert kind is not None
        hits = (
            frame.lazy()
            .with_columns(kind.otherwise(None).alias("failure_kind"))
            .filter(pl.col("failure_kind").is_not_null())
            .with_columns(pl.coalesce(message.str.extract(UNIT_RE, 1), pl.col("process_name")).alias("subject"))
            .group_by("hostname", "subject", "failure_kind")
            .agg(
                pl.len().alias("count"),
                pl.col("timestamp").min().alias("first_seen"),
                pl.col("timestamp").max().alias("last_seen"),
                message.unique(maintain_order=True).head(self._cfg.evidence_samples).alias("evidence"),
                pl.col("timestamp").unique().sort().head(50).alias("event_times"),
            )
            .sort(["count", "subject"], descending=[True, False])
            .collect()
        )
        return [
            ProcessFailure(
                hostname=row["hostname"],
                process_name=row["subject"],
                kind=FailureKind(row["failure_kind"]),
                count=row["count"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                evidence=[e[:400] for e in row["evidence"]],
                event_times=row["event_times"],
            )
            for row in hits.iter_rows(named=True)
        ]

    def _restarts(self, frame: pl.DataFrame) -> list[ProcessFailure]:
        """A long-running PID replaced by a new PID of the same program on the same host ⇒ restart."""
        pids = (
            frame.lazy()
            .filter(pl.col("pid").is_not_null())
            .group_by("hostname", "process_name", "pid")
            .agg(
                pl.len().alias("lines"),
                pl.col("timestamp").min().alias("first"),
                pl.col("timestamp").max().alias("last"),
            )
            .filter(pl.col("lines") >= self._cfg.min_pid_lines)
            .sort("hostname", "process_name", "first")
            .collect()
        )
        failures: list[ProcessFailure] = []
        for (hostname, process_name), group in pids.group_by(["hostname", "process_name"], maintain_order=True):
            rows = group.to_dicts()
            if len(rows) < 2:
                continue
            latest = rows[0]
            events, restart_times = [], []
            for row in rows[1:]:
                if row["first"] >= latest["last"] - self._cfg.restart_tolerance:
                    restart_times.append(row["first"])
                    errors_before = self._errors_before_exit(frame, hostname, process_name, latest)
                    context = (
                        f"preceded by {errors_before} error line(s) in its last minute (crash?)"
                        if errors_before
                        else "no errors in its last minute (planned restart/deploy?)"
                    )
                    events.append(
                        f"pid {latest['pid']} last seen {latest['last']:%Y-%m-%d %H:%M:%S} → "
                        f"pid {row['pid']} first seen {row['first']:%Y-%m-%d %H:%M:%S}; {context}"
                    )
                if row["last"] >= latest["last"]:
                    latest = row
            if events:
                failures.append(
                    ProcessFailure(
                        hostname=str(hostname),
                        process_name=str(process_name),
                        kind=FailureKind.RESTART,
                        count=len(events),
                        first_seen=min(restart_times),
                        last_seen=max(restart_times),
                        evidence=events[:10],
                        event_times=restart_times[:50],
                    )
                )
        return failures

    @staticmethod
    def _errors_before_exit(frame: pl.DataFrame, hostname: object, process_name: object, pid_row: dict) -> int:
        last = pid_row["last"]
        return frame.filter(
            (pl.col("hostname") == hostname)
            & (pl.col("process_name") == process_name)
            & (pl.col("pid") == pid_row["pid"])
            & pl.col("timestamp").is_between(last - timedelta(minutes=1), last)
            & pl.col("severity").is_in(ERROR_VALUES)
        ).height
