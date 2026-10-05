from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from log_analytics.application.reporting.markdown import render_result, sparkline, table
from log_analytics.bootstrap import build_container
from log_analytics.cli import main
from log_analytics.config import Settings
from log_analytics.domain.models import AnalysisRequest
from log_analytics.infrastructure.ingest import ingest_syslog
from log_analytics.infrastructure.parquet_repository import ParquetLogRepository

RAW = """Sep 27 00:00:06 ip-127-0-0-1 systemd[1]: logrotate.service: Deactivated successfully.
Oct  1 03:00:51 ip-127-0-0-1 python[2990436]: INSTANCE ########### GET : /settings/
Oct 1 03:00:52 ip-127-0-0-1 kernel: [12345.6789] Out of memory: Killed process 42 (java)
2026-10-01T03:00:53.123456+00:00 web-02 nginx[77]: upstream timed out (110: Connection timed out)
this line is not syslog
Oct 1 03:00:54 ip-127-0-0-1 postfix/smtp[11]: connect to mx[1.2.3.4]:25: Connection timed out
"""


def test_ingest_parses_classic_and_rfc3339(tmp_path: Path) -> None:
    source = tmp_path / "syslog"
    source.write_text(RAW, encoding="utf-8")
    output = tmp_path / "out.parquet"
    assert ingest_syslog([source], output) == 5
    frame = pl.read_parquet(output)
    assert frame.columns == ["Timestamp", "Hostname", "Process", "Message"]
    assert frame["Process"].to_list() == ["systemd[1]", "python[2990436]", "kernel", "nginx[77]", "postfix/smtp[11]"]
    assert frame["Timestamp"][1] == "Oct 1 03:00:51"
    assert frame["Message"][2].startswith("[12345.6789] Out of memory")
    repo = ParquetLogRepository(str(output), log_year=2026).scan().collect()
    assert repo.height == 5  # mixed RFC 3164 and RFC 3339 timestamps are both parsed
    nginx = repo.filter(pl.col("process_name") == "nginx").row(0, named=True)
    assert nginx["hostname"] == "web-02" and nginx["timestamp"].isoformat() == "2026-10-01T03:00:53.123456+00:00"


def test_markdown_helpers() -> None:
    assert table(["a"], []) == "_No data._"
    assert "a \\| b" in table(["x"], [("a | b",)])
    assert sparkline([0, 5, 10]) == "▁▄█"


async def test_render_full_report(settings: Settings) -> None:
    container = await build_container(settings)
    result = await container.workflow.run(AnalysisRequest(question="What are today's critical issues?"))
    text = render_result(result)
    for heading in (
        "## Answer",
        "## Executive Summary",
        "## Incident Report",
        "## Root Cause Analysis",
        "## Top Error Dashboard",
        "## Host Health Report",
        "## Recommended Actions",
    ):
        assert heading in text
    await container.aclose()


def test_cli_analyze_writes_markdown(tmp_path: Path, parquet_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LA_LOG_YEAR", "2026")
    monkeypatch.setenv("LA_LOG_LEVEL", "WARNING")
    output = tmp_path / "report.md"
    code = main(
        [
            "--parquet",
            str(parquet_path),
            "--no-llm",
            "analyze",
            "-q",
            "Show recurring error patterns.",
            "-o",
            str(output),
        ]
    )
    assert code == 0 and "byte indices" in output.read_text(encoding="utf-8")


REAL = Path(__file__).resolve().parents[1] / "syslog.parquet"


@pytest.mark.realdata
@pytest.mark.skipif(not REAL.exists() or os.getenv("LA_SKIP_REALDATA") == "1", reason="real dataset not present")
async def test_real_dataset_end_to_end() -> None:
    settings = Settings(_env_file=None, parquet_path=str(REAL), llm_enabled=False, log_level="WARNING")  # type: ignore[call-arg]
    container = await build_container(settings)
    result = await container.workflow.run(AnalysisRequest())
    assert result.reports.executive_summary.total_logs > 1_000_000
    assert result.reports.incident_report.incidents
    assert result.timings_ms["total"] < 120_000
    await container.aclose()
