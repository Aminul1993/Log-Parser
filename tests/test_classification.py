from __future__ import annotations

import polars as pl
import pytest

from log_analytics.application.analytics.classification import LogEnricher
from log_analytics.application.analytics.patterns import redact_text, template_expr
from tests.conftest import JWT


def _classify(process_name: str, message: str) -> dict:
    frame = pl.LazyFrame({"process_name": [process_name], "message": [message]})
    return LogEnricher().enrich(frame).collect().row(0, named=True)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Out of memory: Killed process 700 (postgres)", "CRITICAL"),
        ("ERROR byte indices must be integers or slices, not str", "ERROR"),
        ("2026-09-30 00:59:44.3594 WARN EC2RoleProvider Failed to connect", "WARNING"),  # explicit level beats keyword
        ("2026-09-30 01:42:25.9873 ERROR EC2RoleProvider Failed to connect", "ERROR"),
        ('[30/Sep/2026 10:00:00] "GET /x HTTP/1.1" 500 12', "ERROR"),
        ('[30/Sep/2026 10:00:00] "GET /x HTTP/1.1" 403 12', "WARNING"),
        ('[30/Sep/2026 10:00:00] "GET /x HTTP/1.1" 404 12', "INFO"),
        ("TypeError: unsupported operand", "ERROR"),
        ("{'Description': 'EMERGENCY SWITCH WITH NC'}", "INFO"),  # business data, not a syslog EMERG
        ("DEBUG cache key computed", "DEBUG"),
        ("logrotate.service: Deactivated successfully.", "INFO"),
    ],
)
def test_severity(message: str, expected: str) -> None:
    assert _classify("app", message)["severity"] == expected


@pytest.mark.parametrize(
    ("process", "message", "expected"),
    [
        (
            "amazon-ssm-agent.amazon-ssm-agent",
            "Failed to connect to Systems Manager with SSM role credentials",
            "cloud_agent",
        ),
        ("postfix/smtp", "connect to mx[1.2.3.4]:25: Connection timed out", "mail"),
        ("CRON", "(root) CMD (/usr/bin/backup.sh)", "scheduler"),
        ("sshd", "Failed password for invalid user admin", "auth"),
        ("kernel", "Out of memory: Killed process 700", "memory"),
        ("systemd", "nginx.service: Failed with result 'exit-code'.", "service"),
        ("systemd", "postgresql.service: Failed with result 'signal'.", "database"),
        ("python", "Cannot resolve keyword 'date' into field", "application"),
        ("python", '[30/Sep/2026 10:00:00] "GET /api/items/ HTTP/1.1" 200 5', "http"),
        ("postgres", "LOG:  checkpoint complete", "database"),
    ],
)
def test_category(process: str, message: str, expected: str) -> None:
    assert _classify(process, message)["category"] == expected


def test_redaction_in_frames_and_text() -> None:
    message = f"Authorization: Bearer {JWT} password=hunter2 api_key: 'abc123' mail alice@example.com"
    redacted = _classify("app", message)["message"]
    assert JWT not in redacted and "hunter2" not in redacted and "abc123" not in redacted
    assert "alice@example.com" not in redacted
    assert redacted == redact_text(message)
    assert "<JWT>" in redacted and "password=<REDACTED>" in redacted and "<EMAIL>" in redacted


def test_redaction_keeps_non_secret_fields() -> None:
    assert redact_text("'password_expiry_date': datetime.date(2026, 8, 27)").startswith("'password_expiry_date'")


def _template(message: str) -> str:
    return pl.select(template_expr(pl.lit(message))).item()


def test_templates_collapse_variable_tokens() -> None:
    a = _template("connect to mx[52.101.145.2]:25: Connection timed out (attempt 3)")
    b = _template("connect to mx[52.101.144.0]:25: Connection timed out (attempt 17)")
    assert a == b == "connect to mx[<IP>]:<N>: Connection timed out (attempt <N>)"


def test_templates_mask_ipv6_uuid_hex_but_not_words_with_digits() -> None:
    assert _template("to [2a01:111:f403:cc2c::1]:25 down") == _template("to [2a01:111:f403:cc2d::]:25 down")
    assert "<UUID>" in _template("job 123e4567-e89b-12d3-a456-426614174000 failed")
    assert "<HEX>" in _template("object at 0x7e305f754320")
    assert "EC2RoleProvider" in _template("EC2RoleProvider failed")
    assert _template("at 10:46:27 ok") == "at <N> ok"  # a clock time is not an IPv6 address
