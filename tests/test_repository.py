from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from log_analytics.domain.errors import DataSourceError
from log_analytics.infrastructure.parquet_repository import ParquetLogRepository
from tests.conftest import write_parquet


def _repo(path: Path, **kwargs) -> ParquetLogRepository:  # type: ignore[no-untyped-def]
    return ParquetLogRepository(str(path), **kwargs)


def test_normalises_schema(repository: ParquetLogRepository) -> None:
    frame = repository.scan().collect()
    assert frame.columns == ["timestamp", "hostname", "process", "process_name", "pid", "message"]
    assert frame.schema["timestamp"] == pl.Datetime("us", "UTC")
    assert frame["timestamp"].null_count() == 0
    row = frame.filter(pl.col("process") == "python[1000]").row(0, named=True)
    assert row["process_name"] == "python" and row["pid"] == 1000


def test_strips_trailing_colon_from_process(repository: ParquetLogRepository) -> None:
    frame = repository.scan().filter(pl.col("message") == "INFO cache warmed").collect()
    assert frame["process"].to_list() == ["python[1000]"]


def test_single_digit_day_and_extra_spaces(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path / "a.parquet",
        [("Oct 1 03:00:51", "h", "kernel", "a"), ("Oct  2 04:05:06", "h", "cron[1]", "b")],
    )
    stamps = _repo(path, log_year=2026).scan().collect()["timestamp"].to_list()
    assert stamps == [datetime(2026, 10, 1, 3, 0, 51, tzinfo=UTC), datetime(2026, 10, 2, 4, 5, 6, tzinfo=UTC)]


def test_year_rollover_when_year_inferred(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path / "a.parquet", [("Dec 31 23:59:59", "h", "p", "late"), ("Jan 1 00:00:01", "h", "p", "new")]
    )
    repo = _repo(path, clock=lambda: datetime(2027, 1, 1, 0, 5, tzinfo=UTC))
    stamps = repo.scan().collect()["timestamp"].to_list()
    assert stamps[0].year == 2026 and stamps[1].year == 2027


def test_local_timezone_is_converted_to_utc(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "a.parquet", [("Sep 30 10:00:00", "h", "p", "m")])
    [stamp] = _repo(path, log_year=2026, log_timezone="Asia/Kolkata").scan().collect()["timestamp"].to_list()
    assert stamp == datetime(2026, 9, 30, 4, 30, tzinfo=UTC)


def test_iso_timestamps(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "a.parquet", [("2026-09-30T10:00:00.250+02:00", "h", "p", "m")])
    [stamp] = _repo(path).scan().collect()["timestamp"].to_list()
    assert stamp == datetime(2026, 9, 30, 8, 0, 0, 250000, tzinfo=UTC)


def test_native_datetime_column(tmp_path: Path) -> None:
    path = tmp_path / "a.parquet"
    pl.DataFrame(
        {"timestamp": [datetime(2026, 9, 30, 1)], "hostname": ["h"], "process": ["p[1]"], "message": ["m"]}
    ).write_parquet(path)
    frame = _repo(path).scan().collect()  # column names are matched case-insensitively
    assert frame["timestamp"].to_list() == [datetime(2026, 9, 30, 1, tzinfo=UTC)]


def test_missing_column_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pl.DataFrame({"Timestamp": ["Sep 30 00:00:00"], "Hostname": ["h"]}).write_parquet(path)
    with pytest.raises(DataSourceError, match="missing column"):
        _repo(path).scan()


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DataSourceError, match="No Parquet files"):
        _repo(tmp_path / "nope.parquet").scan()


def test_glob_source_reads_all_files(tmp_path: Path) -> None:
    write_parquet(tmp_path / "a.parquet", [("Sep 30 00:00:00", "h1", "p", "m")])
    write_parquet(tmp_path / "b.parquet", [("Sep 30 00:00:01", "h2", "p", "m")])
    assert _repo(tmp_path / "*.parquet", log_year=2026).scan().collect().height == 2


def test_cache_invalidated_when_file_changes(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "a.parquet", [("Sep 30 00:00:00", "h", "p", "one")])
    repo = _repo(path, log_year=2026, cache=True)
    assert repo.scan().collect().height == 1
    write_parquet(path, [("Sep 30 00:00:00", "h", "p", "one"), ("Sep 30 00:00:01", "h", "p", "two")])
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert repo.scan().collect().height == 2
