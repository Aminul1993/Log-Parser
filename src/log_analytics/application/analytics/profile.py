"""Dataset profiling: volumes, time span, severity/category distributions."""

from __future__ import annotations

import polars as pl

from log_analytics.domain.models import CountItem, DatasetProfile


def _counts(frame: pl.DataFrame, column: str, limit: int | None = None) -> list[CountItem]:
    counted = frame.group_by(column).agg(pl.len().alias("count")).sort(["count", column], descending=[True, False])
    if limit:
        counted = counted.head(limit)
    return [CountItem(key=str(row[column]), count=int(row["count"])) for row in counted.iter_rows(named=True)]


def profile_frame(frame: pl.DataFrame, *, top: int = 10) -> DatasetProfile:
    if frame.is_empty():
        return DatasetProfile(total_rows=0, start=None, end=None, host_count=0, process_count=0)
    classified = "severity" in frame.columns
    return DatasetProfile(
        total_rows=frame.height,
        start=frame["timestamp"].min(),
        end=frame["timestamp"].max(),
        host_count=frame["hostname"].n_unique(),
        process_count=frame["process_name"].n_unique(),
        severity_counts={i.key: i.count for i in _counts(frame, "severity")} if classified else {},
        category_counts={i.key: i.count for i in _counts(frame, "category")} if classified else {},
        top_hosts=_counts(frame, "hostname", top),
        top_processes=_counts(frame, "process_name", top),
    )
