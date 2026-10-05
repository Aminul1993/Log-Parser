from __future__ import annotations

from datetime import UTC, datetime

import pytest

from log_analytics.application.query_parser import RuleBasedQueryParser
from log_analytics.domain.models import Intent

ANCHOR = datetime(2026, 10, 1, 15, 28, tzinfo=UTC)


@pytest.fixture
def parser() -> RuleBasedQueryParser:
    return RuleBasedQueryParser(
        ["web-01", "db-01", "ip-127-0-0-1"], ["python", "CRON", "postfix/smtp", "sshd"], ANCHOR
    )


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("What are today's critical issues?", Intent.CRITICAL_ISSUES),
        ("Summarize errors by host.", Intent.ERRORS_BY_HOST),
        ("Which processes are failing most frequently?", Intent.FAILING_PROCESSES),
        ("Show recurring error patterns.", Intent.RECURRING_PATTERNS),
        ("Analyze issues for hostname XYZ.", Intent.HOST_ANALYSIS),
        ("Why did errors spike on web-01?", Intent.ROOT_CAUSE),
        ("Any unusual spikes in log volume?", Intent.ANOMALIES),
        ("Find logs similar to ssm credential failures", Intent.SEMANTIC_SEARCH),
        ("Give me an overview", Intent.FULL_ANALYSIS),
    ],
)
def test_intents(parser: RuleBasedQueryParser, question: str, intent: Intent) -> None:
    assert parser.parse(question).intent is intent


def test_unmatched_question_has_no_intent(parser: RuleBasedQueryParser) -> None:
    assert parser.parse("hello there").intent is None


def test_today_resolves_against_anchor(parser: RuleBasedQueryParser) -> None:
    parsed = parser.parse("What are today's critical issues?")
    assert parsed.filter.start == datetime(2026, 10, 1, tzinfo=UTC) and parsed.filter.end == ANCHOR
    assert any("Relative time resolved" in n for n in parsed.notes)


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("errors in the last 6 hours", datetime(2026, 10, 1, 9, 28, tzinfo=UTC), ANCHOR),
        ("past hour", datetime(2026, 10, 1, 14, 28, tzinfo=UTC), ANCHOR),
        ("yesterday", datetime(2026, 9, 30, tzinfo=UTC), datetime(2026, 9, 30, 23, 59, 59, 999999, tzinfo=UTC)),
        (
            "what happened on 2026-09-28",
            datetime(2026, 9, 28, tzinfo=UTC),
            datetime(2026, 9, 28, 23, 59, 59, 999999, tzinfo=UTC),
        ),
    ],
)
def test_time_expressions(parser: RuleBasedQueryParser, text: str, start: datetime, end: datetime) -> None:
    assert parser.resolve_time(text) == (start, end)


def test_unknown_explicit_hostname_is_kept_and_flagged(parser: RuleBasedQueryParser) -> None:
    parsed = parser.parse("Analyze issues for hostname XYZ.")
    assert parsed.filter.hostnames == ("XYZ",)
    assert any("does not appear" in n for n in parsed.notes)


def test_hosts_resolved_and_mentioned(parser: RuleBasedQueryParser) -> None:
    assert parser.parse("health of host WEB-01 and db-01").filter.hostnames == ("web-01", "db-01")
    assert parser.parse("issues on server web01").filter.hostnames == ("web-01",)  # fuzzy match


def test_verbs_after_host_words_are_not_hostnames(parser: RuleBasedQueryParser) -> None:
    assert parser.parse("Which server has the most errors?").filter.hostnames == ()
    assert parser.parse("Summarize errors by host and process").filter.hostnames == ()


def test_processes_and_keywords(parser: RuleBasedQueryParser) -> None:
    parsed = parser.parse('Show python errors containing timeout and "deadlock detected"')
    assert parsed.filter.processes == ("python",)
    assert set(parsed.filter.keywords) == {"timeout", "deadlock detected"}
    assert parser.parse("Which processes are failing?").filter.processes == ()


def test_search_text_strips_command_words(parser: RuleBasedQueryParser) -> None:
    assert parser.parse("Find logs about ssm agent credentials").search_text == "ssm agent credentials"
