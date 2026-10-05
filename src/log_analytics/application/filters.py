"""Translate a domain ``LogFilter`` into Polars predicates.

Row filters (time, host, process) run on normalised columns and are pushed down into the Parquet scan.
Keyword filters run on the *redacted* message, after classification, so results never depend on
whether secrets were matched and behave the same with or without the classification cache.
"""

from __future__ import annotations

import polars as pl

from log_analytics.application.analytics.patterns import SEVERITY_RANK_MAP, rust_regex_escape
from log_analytics.domain.models import KeywordMode, LogFilter, Severity


def apply_row_filter(frame: pl.LazyFrame, flt: LogFilter) -> pl.LazyFrame:
    predicates: list[pl.Expr] = []
    if flt.start is not None:
        predicates.append(pl.col("timestamp") >= flt.start)
    if flt.end is not None:
        predicates.append(pl.col("timestamp") <= flt.end)
    if flt.hostnames:
        predicates.append(pl.col("hostname").str.to_lowercase().is_in([h.lower() for h in flt.hostnames]))
    if flt.processes:
        names = [p.lower() for p in flt.processes]
        predicates.append(
            pl.col("process_name").str.to_lowercase().is_in(names) | pl.col("process").str.to_lowercase().is_in(names)
        )
    return frame.filter(pl.all_horizontal(predicates)) if predicates else frame


def apply_keyword_filter(frame: pl.LazyFrame, flt: LogFilter) -> pl.LazyFrame:
    if not flt.keywords:
        return frame
    matches = [pl.col("message").str.contains(f"(?i){rust_regex_escape(k)}") for k in flt.keywords]
    combined = pl.any_horizontal(matches) if flt.keyword_mode is KeywordMode.ANY else pl.all_horizontal(matches)
    return frame.filter(combined)


def apply_filter(frame: pl.LazyFrame, flt: LogFilter) -> pl.LazyFrame:
    """Row and keyword filters together (severity is applied separately, after classification)."""
    return apply_keyword_filter(apply_row_filter(frame, flt), flt)


def apply_severity_floor(frame: pl.LazyFrame, floor: Severity | None) -> pl.LazyFrame:
    if floor is None or floor is Severity.DEBUG:
        return frame
    allowed = [name for name, rank in SEVERITY_RANK_MAP.items() if rank >= floor.rank]
    return frame.filter(pl.col("severity").is_in(allowed))
