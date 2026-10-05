"""Streaming syslog → Parquet conversion (the scalable successor of the original ``parser.py``).

Lines are scanned lazily and parsed with one vectorised regex, then streamed to Parquet, so multi-GB
inputs never need to fit in memory. Supports classic RFC 3164 ("Sep 27 00:00:06 host proc[pid]: msg")
and RFC 3339 high-precision timestamps.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from log_analytics.domain.errors import DataSourceError

SYSLOG_LINE_RE = (
    r"^(?P<Timestamp>[A-Z][a-z]{2}\s+\d{1,2}\s\d{2}:\d{2}:\d{2}"
    r"|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"\s+(?P<Hostname>\S+)\s+(?P<Process>[^\s:\[]+(?:\[\d+\])?):?\s?(?P<Message>.*)$"
)


def parse_lines(lines: pl.LazyFrame) -> pl.LazyFrame:
    """``raw`` text column → ``Timestamp, Hostname, Process, Message`` (unparseable lines dropped)."""
    return (
        lines.select(pl.col("raw").str.strip_chars_end("\r\n").str.extract_groups(SYSLOG_LINE_RE).alias("p"))
        .unnest("p")
        .filter(pl.col("Timestamp").is_not_null())
        .with_columns(pl.col("Timestamp").str.replace_all(r"\s+", " "))
    )


def ingest_syslog(inputs: Sequence[str | Path], output: str | Path, *, compression: str = "zstd") -> int:
    """Convert one or more syslog files to a single Parquet file. Returns the number of rows written."""
    paths = [Path(p) for p in inputs]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise DataSourceError(f"Input file(s) not found: {missing}")
    scans = [
        pl.scan_csv(
            path,
            has_header=False,
            new_columns=["raw"],
            separator="\x1f",  # unit separator: never present in syslog, so each line is one field
            quote_char=None,
            infer_schema=False,
            encoding="utf8-lossy",
            truncate_ragged_lines=True,
        )
        for path in paths
    ]
    parse_lines(pl.concat(scans)).sink_parquet(Path(output), compression=compression)  # type: ignore[arg-type]
    return int(pl.scan_parquet(output).select(pl.len()).collect().item())
