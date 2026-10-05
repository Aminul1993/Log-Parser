from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from pydantic import ValidationError

from log_analytics.application.filters import apply_filter, apply_severity_floor
from log_analytics.domain.models import KeywordMode, LogFilter, Severity
from tests.conftest import BASE_UTC


def _count(frame: pl.DataFrame, flt: LogFilter) -> int:
    return apply_filter(frame.lazy(), flt).collect().height


def test_time_range_inclusive(frame: pl.DataFrame) -> None:
    flt = LogFilter(start=BASE_UTC + timedelta(hours=2), end=BASE_UTC + timedelta(hours=2, minutes=5))
    out = apply_filter(frame.lazy(), flt).collect()
    assert out["timestamp"].min() >= flt.start and out["timestamp"].max() <= flt.end
    assert out.filter(pl.col("message").str.contains("connection refused")).height == 80


def test_naive_datetimes_are_utc() -> None:
    assert LogFilter(start=datetime(2026, 9, 30)).start == BASE_UTC


def test_hostname_case_insensitive(frame: pl.DataFrame) -> None:
    assert _count(frame, LogFilter(hostnames=("WEB-01",))) == frame.filter(pl.col("hostname") == "web-01").height


def test_process_name_or_full_tag(frame: pl.DataFrame) -> None:
    by_name = _count(frame, LogFilter(processes=("python",)))
    by_tag = _count(frame, LogFilter(processes=("python[2000]",)))
    assert by_name > by_tag > 0


def test_keywords_any_and_all(frame: pl.DataFrame) -> None:
    any_count = _count(frame, LogFilter(keywords=("connection refused", "byte indices")))
    all_count = _count(frame, LogFilter(keywords=("connection", "refused"), keyword_mode=KeywordMode.ALL))
    assert any_count == 92 and all_count == 80


def test_keywords_are_literals_not_regex(frame: pl.DataFrame) -> None:
    assert _count(frame, LogFilter(keywords=("(0.1%)",))) > 0
    assert _count(frame, LogFilter(keywords=("a+b(",))) == 0


def test_severity_floor(frame: pl.DataFrame) -> None:
    out = apply_severity_floor(frame.lazy(), Severity.ERROR).collect()
    assert set(out["severity"].unique()) <= {"ERROR", "CRITICAL"} and out.height > 0


def test_filter_validation() -> None:
    with pytest.raises(ValidationError):
        LogFilter(start=datetime(2026, 10, 2), end=datetime(2026, 10, 1))
    assert LogFilter(hostnames=["  a ", "a", ""]).hostnames == ("a",)


def test_merged_with_prefers_explicit_override() -> None:
    parsed = LogFilter(hostnames=("a",), keywords=("x",))
    merged = parsed.merged_with(LogFilter(hostnames=("b",)))
    assert merged.hostnames == ("b",) and merged.keywords == ("x",)
