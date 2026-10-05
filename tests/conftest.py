"""Shared fixtures: a synthetic syslog dataset with planted, known scenarios.

Timeline (2026-09-30, UTC, 00:00-06:00):
* web-01 python app: an access-log line every 20s; PID 1000 restarts as PID 2000 at 03:00 (no errors first)
* web-01 02:00-02:05: 80 x "connection refused to db-01:5432"  → error spike + service outage
* web-01 every 30 min: "ERROR byte indices must be integers…"  → recurring code bug
* web-01 CRON every minute, silent 04:00-04:45                → heartbeat silence gap
* web-01 one line carrying a JWT, a password and an e-mail      → redaction
* db-01 postgres heartbeat every 30s; 01:00 20 x sshd failed passwords; 05:00 OOM kill + systemd unit failure
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from log_analytics.application.analytics.classification import LogEnricher
from log_analytics.config import Settings
from log_analytics.infrastructure.parquet_repository import ParquetLogRepository

BASE = datetime(2026, 9, 30, 0, 0, 0)
BASE_UTC = BASE.replace(tzinfo=UTC)
JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxMjM0NTZ9.c2lnbmF0dXJlLXZhbHVlLXg"


def syslog_ts(dt: datetime) -> str:
    return f"{dt:%b} {dt.day} {dt:%H:%M:%S}"


def _every(start: timedelta, end: timedelta, step: timedelta) -> Iterator[datetime]:
    t = BASE + start
    while t < BASE + end:
        yield t
        t += step


def synthetic_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []

    def add(dt: datetime, host: str, process: str, message: str) -> None:
        rows.append((syslog_ts(dt), host, process, message))

    h, m, s = timedelta(hours=1), timedelta(minutes=1), timedelta(seconds=1)
    for t in _every(0 * h, 6 * h, 20 * s):
        pid = 1000 if t < BASE + 3 * h else 2000
        add(
            t,
            "web-01",
            f"python[{pid}]",
            f'[30/Sep/2026 {t:%H:%M:%S}] "GET /api/items/?page={t.minute} HTTP/1.1" 200 5{t.second}',
        )
    for i in range(80):
        add(
            BASE + 2 * h + i * 3.5 * s,
            "web-01",
            "python[1000]",
            f"ERROR could not connect: connection refused to db-01:5432 (attempt {i})",
        )
    for k in range(12):
        add(
            BASE + k * 30 * m + 7 * s,
            "web-01",
            f"python[{1000 if k < 6 else 2000}]",
            "ERROR byte indices must be integers or slices, not str",
        )
    for t in _every(1 * s, 6 * h, 1 * m):
        if not (BASE + 4 * h <= t < BASE + 4 * h + 45 * m):
            add(t, "web-01", f"CRON[{5000 + int((t - BASE).total_seconds() // 60)}]", "(root) CMD (/usr/bin/backup.sh)")
    add(BASE + 10 * m, "web-01", "python[1000]:", "INFO cache warmed")
    add(
        BASE + 11 * m,
        "web-01",
        "python[1000]",
        f"headers --> {{'Authorization': 'Bearer {JWT}', 'password': 'hunter2'}} user=alice@example.com",
    )
    for t in _every(0 * h, 6 * h, 30 * s):
        add(t, "db-01", "postgres[700]", "LOG:  checkpoint complete: wrote 12 buffers (0.1%)")
    for i in range(20):
        add(
            BASE + h + i * 2 * s,
            "db-01",
            f"sshd[{9000 + i}]",
            f"Failed password for invalid user admin from 203.0.113.{i} port {40000 + i} ssh2",
        )
    add(
        BASE + 5 * h,
        "db-01",
        "kernel",
        "Out of memory: Killed process 700 (postgres) total-vm:8123456kB, anon-rss:7000000kB",
    )
    add(BASE + 5 * h + s, "db-01", "systemd[1]", "postgresql.service: Main process exited, code=killed, status=9/KILL")
    add(BASE + 5 * h + s, "db-01", "systemd[1]", "postgresql.service: Failed with result 'signal'.")
    return rows


def write_parquet(path: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    pl.DataFrame(rows, schema=["Timestamp", "Hostname", "Process", "Message"], orient="row").write_parquet(path)
    return path


@pytest.fixture(scope="session")
def parquet_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_parquet(tmp_path_factory.mktemp("data") / "syslog.parquet", synthetic_rows())


@pytest.fixture(scope="session")
def repository(parquet_path: Path) -> ParquetLogRepository:
    return ParquetLogRepository(str(parquet_path), log_year=2026)


@pytest.fixture(scope="session")
def frame(repository: ParquetLogRepository) -> pl.DataFrame:
    """Normalised + classified synthetic dataset, sorted by time."""
    return LogEnricher().enrich(repository.scan()).sort("timestamp").collect()


@pytest.fixture
def settings(parquet_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        parquet_path=str(parquet_path),
        log_year=2026,
        llm_enabled=False,
        database_url=None,
        log_json=False,
        log_level="WARNING",
        environment="test",
    )
