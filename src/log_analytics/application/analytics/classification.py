"""Rule-based, vectorised log classification: severity, category, template (+ secret redaction).

Rules run over every row in Rust via Polars, so classifying millions of lines takes about a second.
The LLM classification agent then refines categories per *template* (not per row), which scales.
"""

from __future__ import annotations

import polars as pl

from log_analytics.application.analytics.patterns import redact_expr, template_expr
from log_analytics.domain.models import Category, Severity

# Explicit level tokens emitted by logging frameworks ("2026-10-01 ERROR …", "[WARN]", "level=error").
EXPLICIT_LEVEL_RE = (
    r"(?:^|[\s\[(|<])(EMERG|ALERT|CRIT|CRITICAL|FATAL|ERROR|ERR|WARN|WARNING|NOTICE|INFO|DEBUG|TRACE)"
    r"(?:[\s\])|>:,]|$)"
)
LEVEL_MAP: dict[str, str] = {
    "EMERG": "CRITICAL",
    "ALERT": "CRITICAL",
    "CRIT": "CRITICAL",
    "CRITICAL": "CRITICAL",
    "FATAL": "CRITICAL",
    "ERROR": "ERROR",
    "ERR": "ERROR",
    "WARN": "WARNING",
    "WARNING": "WARNING",
    "NOTICE": "INFO",
    "INFO": "INFO",
    "DEBUG": "DEBUG",
    "TRACE": "DEBUG",
}

# Keyword heuristics, evaluated in order (first match wins).
SEVERITY_RULES: tuple[tuple[Severity, str], ...] = (
    (
        Severity.CRITICAL,
        # Syslog "EMERG"/"ALERT" levels come through EXPLICIT_LEVEL_RE; the bare word "emergency" is not used here
        # because it is common in business data ("EMERGENCY SWITCH" product descriptions).
        r"(?i)\b(kernel panic|panic|fatal|segfault|segmentation fault|kernel bug|out of memory"
        r"|oom-kill(?:er)?|killed process|core dumped|data corruption|corrupted|read-only file system"
        r"|no space left on device|disk full)\b",
    ),
    (
        Severity.ERROR,
        r"(?i)(\b(error|errors|err|exception|traceback|failed|failure|fails|refused|unreachable|timed out"
        r"|timeout|denied|forbidden|unauthori[sz]ed|unable to|cannot|could not|couldn't|invalid|aborted|abort"
        r"|crash|crashed|broken pipe|reset by peer|non-zero)\b|\b\w+(?:Error|Exception)\b"
        r"|HTTP/\d(?:\.\d)?\" 5\d\d\b)",
    ),
    (
        Severity.WARNING,
        r"(?i)(\b(warn|warning|warnings|deprecated|deprecation|retry|retrying|retries|slow|degraded"
        r"|throttl\w*|backoff|deferred|not found|missing)\b|HTTP/\d(?:\.\d)?\" (?:401|403|429)\b)",
    ),
    (Severity.DEBUG, r"(?i)^\s*(debug|trace)\b|\blevel=debug\b"),
)

# Matched against "<process_name> <message>", evaluated in order (first match wins).
CATEGORY_RULES: tuple[tuple[Category, str], ...] = (
    (
        Category.CLOUD_AGENT,
        r"(?i)(amazon-ssm-agent|ssm-agent|cloud-init|waagent|google_guest_agent|ec2metadata|ec2roleprovider|\bimds\b)",
    ),
    (Category.MEMORY, r"(?i)(out of memory|\boom\b|oom-kill|cannot allocate memory|memory cgroup|\bmalloc\b)"),
    (
        Category.STORAGE,
        r"(?i)(no space left|\bdisk\b|filesystem|i/o error|\bext4\b|\bxfs\b|\binode|read-only file system|\bmount)",
    ),
    (Category.MAIL, r"(?i)^(postfix|sendmail|exim|dovecot)\S*|\bsmtp\b"),
    (Category.SCHEDULER, r"(?i)^(cron|crond|anacron|atd)\b|\.timer\b"),
    (
        Category.DATABASE,
        r"(?i)(postgres|psycopg|mysql|mariadb|sqlite|database|deadlock|\bora-\d+|mongo|redis|sqlalchemy"
        r"|integrityerror|operationalerror)",
    ),
    (
        Category.AUTH,
        r"(?i)(\bauth|credential|password|permission|unauthori[sz]ed|forbidden|\blogin|\blogout|\btoken"
        r"|^sshd|\bsudo\b|pam_|invalid user|access denied|\bjwt\b|HTTP/\d(?:\.\d)?\" 40[13]\b)",
    ),
    (
        Category.NETWORK,
        r"(?i)(connection (?:refused|reset|timed out|closed)|timed out|timeout|unreachable|\bdns\b|name resolution"
        r"|resolve host|\bsocket|econn|no route|network|\btls\b|\bssl\b|handshake)",
    ),
    (Category.KERNEL, r"(?i)^kernel\b"),
    (Category.SERVICE, r"(?i)^systemd\b|\.service\b"),
    (Category.HTTP, r"\"(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) \S+ HTTP/"),
    (
        Category.APPLICATION,
        r"(?i)^(python\d?|java|node|gunicorn|uwsgi|celery|php\S*|ruby|dotnet|nginx|apache2|httpd)\b",
    ),
)


def _first_match(rules: tuple[tuple[str, str], ...], subject: pl.Expr, default: str) -> pl.Expr:
    chain = None
    for value, pattern in rules:
        condition = subject.str.contains(pattern)
        chain = pl.when(condition).then(pl.lit(value)) if chain is None else chain.when(condition).then(pl.lit(value))
    return chain.otherwise(pl.lit(default)) if chain is not None else pl.lit(default)


def severity_expr(message: pl.Expr) -> pl.Expr:
    heuristic = _first_match(tuple((s.value, p) for s, p in SEVERITY_RULES), message, Severity.INFO.value)
    explicit = message.str.extract(EXPLICIT_LEVEL_RE, 1).replace_strict(LEVEL_MAP, default=None)
    # Critical keywords always win; otherwise an explicit level token beats keyword guessing.
    return pl.when(heuristic == Severity.CRITICAL.value).then(heuristic).otherwise(pl.coalesce(explicit, heuristic))


def category_expr(process_name: pl.Expr, message: pl.Expr) -> pl.Expr:
    subject = pl.concat_str([process_name, pl.lit(" "), message])
    return _first_match(tuple((c.value, p) for c, p in CATEGORY_RULES), subject, Category.OTHER.value)


class LogEnricher:
    """Adds ``severity``, ``category`` and ``template`` columns (and redacts secrets in ``message``)."""

    def __init__(self, *, redact_secrets: bool = True, redact_emails: bool = True) -> None:
        self._redact_secrets = redact_secrets
        self._redact_emails = redact_emails

    def enrich(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        if self._redact_secrets:
            frame = frame.with_columns(redact_expr(pl.col("message"), emails=self._redact_emails))
        message = pl.col("message")
        return frame.with_columns(
            severity_expr(message).alias("severity"),
            category_expr(pl.col("process_name"), message).alias("category"),
            template_expr(message).alias("template"),
        )
