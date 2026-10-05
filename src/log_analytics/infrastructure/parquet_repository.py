"""Parquet log repository: lazy Polars scans (predicate pushdown, streaming) with an optional in-memory cache.

Normalises the raw ``Timestamp, Hostname, Process, Message`` schema into
``timestamp (UTC), hostname, process, process_name, pid, message``.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import structlog

from log_analytics.domain.errors import DataSourceError

log = structlog.get_logger(__name__)

REQUIRED_COLUMNS = ("Timestamp", "Hostname", "Process", "Message")
_SYSLOG_TS = re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}$")
_ISO_AWARE = "%Y-%m-%dT%H:%M:%S%.f%#z"  # %.f: optional fraction; %#z: "Z", "+05:30" or "+0530"
_ISO_NAIVE = ("%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f")
_ROLLOVER_SLACK = timedelta(days=2)


class ParquetLogRepository:
    def __init__(
        self,
        source: str,
        *,
        log_year: int | None = None,
        log_timezone: str = "UTC",
        cache: bool = True,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._source = source
        self._year = log_year
        self._tz = log_timezone
        self._cache_enabled = cache
        self._clock = clock
        self._lock = threading.Lock()
        self._cached: tuple[str, pl.DataFrame] | None = None

    def describe_source(self) -> str:
        return self._source

    # ------------------------------------------------------------------ port

    def version(self) -> str:
        digest = hashlib.sha1(usedforsecurity=False)
        for path in self._files():
            stat = os.stat(path)
            digest.update(f"{path}:{stat.st_mtime_ns}:{stat.st_size};".encode())
        return digest.hexdigest()

    def scan(self) -> pl.LazyFrame:
        if not self._cache_enabled:
            return self._normalized(self._files())
        version = self.version()
        with self._lock:
            if self._cached is None or self._cached[0] != version:
                started = datetime.now(UTC)
                frame = self._normalized(self._files()).collect(engine="streaming").rechunk()
                self._cached = (version, frame)
                log.info(
                    "dataset_cached",
                    source=self._source,
                    rows=frame.height,
                    size_mb=round(frame.estimated_size("mb"), 1),
                    duration_ms=round((datetime.now(UTC) - started).total_seconds() * 1000),
                )
            return self._cached[1].lazy()

    # ------------------------------------------------------------------ internals

    def _files(self) -> list[str]:
        if glob.has_magic(self._source):
            files = sorted(glob.glob(self._source, recursive=True))
        else:
            files = [self._source] if Path(self._source).is_file() else []
        if not files:
            raise DataSourceError(f"No Parquet files found at '{self._source}' (set LA_PARQUET_PATH)")
        return files

    def _normalized(self, files: list[str]) -> pl.LazyFrame:
        try:
            frame = pl.scan_parquet(files)
            schema = frame.collect_schema()
        except (OSError, pl.exceptions.PolarsError) as exc:
            raise DataSourceError(f"Cannot read Parquet source '{self._source}': {exc}") from exc
        columns = self._resolve_columns(schema.names())
        frame = frame.select([pl.col(actual).alias(canonical) for canonical, actual in columns.items()])
        timestamp = self._timestamp_expr(frame, schema[columns["Timestamp"]])
        process = pl.col("Process").cast(pl.String).str.strip_chars().str.strip_chars_end(":")
        return frame.select(
            timestamp.alias("timestamp"),
            pl.col("Hostname").cast(pl.String).str.strip_chars().fill_null("unknown").alias("hostname"),
            process.fill_null("unknown").alias("process"),
            process.str.replace(r"\[\d+\]$", "").fill_null("unknown").alias("process_name"),
            process.str.extract(r"\[(\d+)\]$", 1).cast(pl.Int64, strict=False).alias("pid"),
            pl.col("Message").cast(pl.String).fill_null("").alias("message"),
        ).filter(pl.col("timestamp").is_not_null())

    @staticmethod
    def _resolve_columns(names: list[str]) -> dict[str, str]:
        by_lower = {n.lower(): n for n in names}
        missing = [c for c in REQUIRED_COLUMNS if c.lower() not in by_lower]
        if missing:
            raise DataSourceError(f"Parquet schema is missing column(s) {missing}; found {names}")
        return {c: by_lower[c.lower()] for c in REQUIRED_COLUMNS}

    def _timestamp_expr(self, frame: pl.LazyFrame, dtype: pl.DataType) -> pl.Expr:
        col = pl.col("Timestamp")
        if isinstance(dtype, pl.Datetime):
            return self._to_utc(col, aware=dtype.time_zone is not None).dt.cast_time_unit("us")
        if dtype == pl.Date:
            return self._to_utc(col.cast(pl.Datetime("us")), aware=False)
        if dtype != pl.String:
            raise DataSourceError(f"Unsupported Timestamp column type {dtype}")
        # Pick parsers from a sample so the common single-format case pays for one parse only.
        sample = frame.select(col.drop_nulls().head(200)).collect()["Timestamp"].to_list()
        syslog_like = [bool(_SYSLOG_TS.match(s.strip())) for s in sample]
        if sample and all(syslog_like):
            return self._syslog_expr(col)
        if not any(syslog_like):
            return self._iso_expr(col)
        return pl.coalesce(self._syslog_expr(col), self._iso_expr(col))  # mixed RFC 3164 / RFC 3339 sources

    def _iso_expr(self, col: pl.Expr) -> pl.Expr:
        """RFC 3339 / ISO 8601 with explicit formats (no inference): offset-aware first, then naive (local tz)."""
        text = col.str.strip_chars()
        aware = text.str.to_datetime(_ISO_AWARE, strict=False, time_unit="us", time_zone="UTC")
        naive = [self._to_utc(text.str.to_datetime(f, strict=False, time_unit="us"), aware=False) for f in _ISO_NAIVE]
        return pl.coalesce(aware, *naive)

    def _to_utc(self, expr: pl.Expr, *, aware: bool) -> pl.Expr:
        if not aware:
            expr = expr.dt.replace_time_zone(self._tz, ambiguous="earliest", non_existent="null")
        return expr.dt.convert_time_zone("UTC")

    def _syslog_expr(self, col: pl.Expr) -> pl.Expr:
        """RFC 3164 timestamps ("Sep 27 00:00:06") have no year: supply it and handle Dec→Jan rollover."""
        year = self._year or self._clock().year
        text = pl.lit(f"{year} ") + col.str.strip_chars().str.replace_all(r"\s+", " ")
        ts = self._to_utc(text.str.strptime(pl.Datetime("us"), "%Y %b %d %H:%M:%S", strict=False), aware=False)
        if self._year is None:
            horizon = self._clock() + _ROLLOVER_SLACK
            ts = pl.when(ts > horizon).then(ts.dt.offset_by("-1y")).otherwise(ts)
        return ts
