"""Regex vocabularies shared by the vectorised (Polars / Rust regex) and scalar (Python ``re``) paths.

Patterns avoid look-around and back-references so they compile under both engines. Replacement
strings use Rust's ``${n}`` group syntax and are converted for Python automatically.
"""

from __future__ import annotations

import hashlib
import re
from datetime import timedelta

import polars as pl

from log_analytics.domain.models import ERROR_SEVERITIES, SEVERITY_RANK

ERROR_VALUES: list[str] = [s.value for s in ERROR_SEVERITIES]
SEVERITY_RANK_MAP: dict[str, int] = {s.value: r for s, r in SEVERITY_RANK.items()}

SECRET_RULES: tuple[tuple[str, str], ...] = (
    (r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}", "<JWT>"),
    (r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer <TOKEN>"),
    (r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", "<AWS_KEY>"),
    (
        r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret)"
        r"(['\"]?\s*[:=]\s*['\"]?)[^\s'\",}&]+",
        "${1}${2}<REDACTED>",
    ),
)
EMAIL_RULE: tuple[str, str] = (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", "<EMAIL>")

TEMPLATE_RULES: tuple[tuple[str, str], ...] = (
    (r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?", "<TS>"),
    (r"\b\d{1,2}/[A-Z][a-z]{2}/\d{4}[: ]\d{2}:\d{2}:\d{2}\b", "<TS>"),
    (r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<UUID>"),
    (r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?\b", "<IP>"),
    # IPv6: "::"-compressed form first (leftmost-first alternation), then full >=4-group form; clock times such
    # as 10:46:27 match neither.
    (
        r"(?i)\b(?:[0-9a-f]{1,4}:){1,6}:(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4})*)?|\b(?:[0-9a-f]{1,4}:){3,7}[0-9a-f]{1,4}\b",
        "<IP>",
    ),
    (r"(?i)\b0x[0-9a-f]+\b", "<HEX>"),
    (r"(?i)\b[0-9a-f]{16,}\b", "<HEX>"),
    (r"\?[^\s\"']+", "?<QUERY>"),
    (r"\b\d+(?:[.,:]\d+)*\b", "<N>"),
    (r"\s+", " "),
)
TEMPLATE_MAX_LEN = 240

# Signals of a service being unreachable / down (used by outage detection).
OUTAGE_RE = (
    r"(?i)(connection refused|service unavailable|no route to host|host unreachable|network is unreachable"
    r"|bad gateway|gateway time-?out|upstream (?:timed out|prematurely closed)|could not connect"
    r"|failed to connect|connection timed out|name or service not known|temporary failure in name resolution"
    r"|HTTP/\d(?:\.\d)?\" 50[234]\b)"
)


def _python_replacement(rust_repl: str) -> str:
    return re.sub(r"\$\{(\d+)\}", r"\\g<\1>", rust_repl)


def _secret_rules(emails: bool) -> tuple[tuple[str, str], ...]:
    return (*SECRET_RULES, EMAIL_RULE) if emails else SECRET_RULES


def redact_expr(expr: pl.Expr, *, emails: bool = True) -> pl.Expr:
    for pattern, replacement in _secret_rules(emails):
        expr = expr.str.replace_all(pattern, replacement)
    return expr


def template_expr(expr: pl.Expr) -> pl.Expr:
    """Mask variable tokens so lines from the same code path collapse to one template."""
    for pattern, replacement in TEMPLATE_RULES:
        expr = expr.str.replace_all(pattern, replacement)
    return expr.str.strip_chars().str.slice(0, TEMPLATE_MAX_LEN)


_PY_SECRET_RULES = {
    emails: [(re.compile(p), _python_replacement(r)) for p, r in _secret_rules(emails)] for emails in (True, False)
}


def redact_text(text: str, *, emails: bool = True) -> str:
    """Scalar counterpart of :func:`redact_expr` for strings bound for an LLM, a report or an index."""
    for pattern, replacement in _PY_SECRET_RULES[emails]:
        text = pattern.sub(replacement, text)
    return text


def fingerprint(*parts: str) -> str:
    """Stable identifier (unlike ``pl.Expr.hash``, which may change across Polars versions)."""
    return hashlib.sha1("\x1f".join(parts).encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def rust_regex_escape(text: str) -> str:
    """Escape a literal for the Rust regex engine (``re.escape`` emits escapes Rust rejects)."""
    return re.sub(r"([\\.+*?()|\[\]{}^$#&\-~])", r"\\\1", text)


_DURATION_RE = re.compile(r"^(\d+)(s|m|h|d)$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(spec: str) -> timedelta:
    """Parse the Polars-style interval strings used in settings (``30s``, ``5m``, ``1h``, ``1d``)."""
    match = _DURATION_RE.match(spec.strip())
    if not match:
        raise ValueError(f"Unsupported interval '{spec}'; use forms like 30s, 5m, 1h, 1d")
    return timedelta(seconds=int(match.group(1)) * _UNIT_SECONDS[match.group(2)])


def interval_spec(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    for unit in ("d", "h", "m"):
        if seconds % _UNIT_SECONDS[unit] == 0:
            return f"{seconds // _UNIT_SECONDS[unit]}{unit}"
    return f"{seconds}s"


NICE_INTERVALS = ("1m", "5m", "15m", "30m", "1h", "3h", "6h", "12h", "1d", "7d")


def coarsest_needed(span: timedelta, max_points: int, minimum: str = "1m") -> str:
    """Smallest 'nice' interval (>= minimum) that keeps ``span`` within ``max_points`` buckets."""
    floor = parse_duration(minimum)
    for spec in NICE_INTERVALS:
        delta = parse_duration(spec)
        if delta >= floor and span / delta <= max_points:
            return spec
    return NICE_INTERVALS[-1]
