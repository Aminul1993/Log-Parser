"""Deterministic natural-language query parsing: intent, time range and entity resolution.

Rules handle the common phrasings precisely and instantly; the LLM planner is consulted only when no
rule matches. Entities proposed by either path are resolved against the hosts/processes that
actually exist in the data, so a hallucinated hostname cannot silently widen or empty a query.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from log_analytics.domain.models import Intent, LogFilter

_I = re.IGNORECASE
INTENT_RULES: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (
        Intent.SEMANTIC_SEARCH,
        re.compile(
            r"^\s*(search|find|look ?up|grep)\b|\bsimilar to\b|\blogs? (?:about|mentioning|related to|like)\b", _I
        ),
    ),
    (Intent.ROOT_CAUSE, re.compile(r"\broot[- ]?causes?\b|\bwhy\b|\brca\b|\bwhat caused\b", _I)),
    (
        Intent.ERRORS_BY_HOST,
        re.compile(r"\b(by|per|each|across|for every) (host|hostname|server|node|machine)s?\b", _I),
    ),
    (
        Intent.FAILING_PROCESSES,
        re.compile(
            r"\bprocess(es)?\b.*\b(fail\w*|crash\w*|restart\w*|die|dying|down|unstable)"
            r"|\b(fail\w*|crash\w*|restart\w*)\b.*\bprocess(es)?\b",
            _I,
        ),
    ),
    (
        Intent.RECURRING_PATTERNS,
        re.compile(r"\b(recurr\w*|repeat\w*|patterns?|common errors?|top errors?|most frequent errors?)\b", _I),
    ),
    (Intent.ANOMALIES, re.compile(r"\b(anomal\w*|spikes?|unusual|bursts?|surges?|outages?|silen\w*|volume)\b", _I)),
    (Intent.CRITICAL_ISSUES, re.compile(r"\b(critical|severe|urgent|sev ?1|p1|major|biggest|most important)\b", _I)),
    (
        Intent.FULL_ANALYSIS,
        re.compile(
            r"\b(summar\w*|overview|report|health|status|what happened|analy[sz]\w*|issues?|problems?|errors?)\b", _I
        ),
    ),
)
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "get",
        "got",
        "had",
        "has",
        "have",
        "having",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "show",
        "shows",
        "showing",
        "the",
        "to",
        "was",
        "were",
        "with",
        "which",
        "that",
        "this",
        "those",
        "these",
        "what",
        "when",
        "where",
        "who",
        "errors",
        "error",
        "issues",
        "issue",
        "logs",
        "log",
        "health",
        "status",
        "name",
        "names",
        "process",
        "processes",
        "host",
        "hosts",
    ]
)
GENERIC_PROCESS_WORDS = frozenset(
    ["login", "init", "agent", "app", "api", "server", "service", "worker", "main", "error", "host", "log", "logs"]
)

_HOST_RE = re.compile(r"\b(host(?:name)?|server|node|machine|instance)\s*[:=]?\s*[\"'`]?([A-Za-z0-9][\w.-]*)", _I)
_HOSTLIKE_RE = re.compile(r"[\d._-]")
_PROCESS_RE = re.compile(r"\bprocess(?:es)?\s*[:=]?\s*[\"'`]?([A-Za-z][\w./@-]*)", _I)
_QUOTED_RE = re.compile(r"(?:^|\s)[\"“'`]([^\"”'`]{2,80})[\"”'`]")
_KEYWORD_RE = re.compile(r"\b(?:containing|mentioning|matching|with (?:the )?keyword)\s+([\w.:/@-]{2,80})", _I)
_RELATIVE_RE = re.compile(r"\b(?:last|past|previous)\s+(\d+)?\s*(minute|min|hour|hr|day|week)s?\b", _I)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_SEARCH_PREFIX_RE = re.compile(
    r"^\s*(?:search|find|look ?up|grep|show me|show)\s+(?:for\s+)?(?:logs?\s+)?"
    r"(?:about|mentioning|related to|similar to|like)?\s*",
    _I,
)
_UNITS = {"minute": "minutes", "min": "minutes", "hour": "hours", "hr": "hours", "day": "days", "week": "weeks"}


@dataclass(slots=True)
class ParsedQuery:
    intent: Intent | None
    filter: LogFilter
    search_text: str
    notes: list[str] = field(default_factory=list)


def _day_bounds(day: datetime) -> tuple[datetime, datetime]:
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1) - timedelta(microseconds=1)


class RuleBasedQueryParser:
    def __init__(self, known_hosts: Sequence[str], known_processes: Sequence[str], anchor: datetime) -> None:
        self._hosts = list(known_hosts)
        self._processes = list(known_processes)
        self._anchor = anchor

    def parse(self, question: str) -> ParsedQuery:
        notes: list[str] = []
        time_range = self.resolve_time(question)
        hostnames = self._hosts_in(question, notes)
        processes = self._processes_in(question)
        keywords = list(dict.fromkeys([*_QUOTED_RE.findall(question), *_KEYWORD_RE.findall(question)]))

        intent = next((intent for intent, pattern in INTENT_RULES if pattern.search(question)), None)
        if hostnames and intent in (None, Intent.FULL_ANALYSIS):
            intent = Intent.HOST_ANALYSIS
        if time_range:
            notes.append(
                f"Relative time resolved against {self._anchor:%Y-%m-%d %H:%M UTC}: "
                f"{time_range[0]:%Y-%m-%d %H:%M} → {time_range[1]:%Y-%m-%d %H:%M}"
            )
        flt = LogFilter(
            start=time_range[0] if time_range else None,
            end=time_range[1] if time_range else None,
            hostnames=hostnames,
            processes=processes,
            keywords=keywords,
        )
        search_text = _SEARCH_PREFIX_RE.sub("", question).strip() or question
        return ParsedQuery(intent=intent, filter=flt, search_text=search_text, notes=notes)

    # ------------------------------------------------------------------ pieces

    def resolve_time(self, text: str | None) -> tuple[datetime, datetime] | None:
        if not text:
            return None
        anchor = self._anchor
        lowered = text.lower()
        if match := _RELATIVE_RE.search(lowered):
            amount = int(match.group(1) or 1)
            delta = timedelta(**{_UNITS[match.group(2)]: amount})
            return anchor - delta, anchor
        if re.search(r"\byesterday\b", lowered):
            return _day_bounds(anchor - timedelta(days=1))
        if re.search(r"\btoday\b|\btonight\b|\bthis morning\b", lowered):
            return _day_bounds(anchor)[0], anchor
        if re.search(r"\bthis week\b", lowered):
            start = _day_bounds(anchor - timedelta(days=anchor.weekday()))[0]
            return start, anchor
        if match := _DATE_RE.search(lowered):
            try:
                day = datetime.fromisoformat(match.group(1)).replace(tzinfo=anchor.tzinfo)
            except ValueError:
                return None
            return _day_bounds(day)
        return None

    def resolve_hosts(self, candidates: Sequence[str], notes: list[str], *, strict: bool = False) -> list[str]:
        """Map names onto known hosts. Unknown names are kept (the user asked for them explicitly) unless
        ``strict`` and the token does not even look like a hostname (e.g. "which server *has* …")."""
        resolved: list[str] = []
        lowered = {h.lower(): h for h in self._hosts}
        for raw in candidates:
            name = raw.strip().rstrip(".,;:!?")
            if not name or name.lower() in STOPWORDS:
                continue
            if name.lower() in lowered:
                resolved.append(lowered[name.lower()])
                continue
            close = difflib.get_close_matches(name.lower(), list(lowered), n=1, cutoff=0.8)
            if close:
                resolved.append(lowered[close[0]])
                notes.append(f"Host '{name}' resolved to '{lowered[close[0]]}'.")
            elif not strict or _HOSTLIKE_RE.search(name) or name.isupper():
                resolved.append(name)
                sample = ", ".join(self._hosts[:5]) or "none"
                notes.append(f"Host '{name}' does not appear in the dataset (known hosts: {sample}).")
        return list(dict.fromkeys(resolved))

    def resolve_processes(self, candidates: Sequence[str]) -> list[str]:
        lowered = {p.lower(): p for p in self._processes}
        resolved = []
        for raw in candidates:
            name = raw.strip().rstrip(".,;:!?").lower()
            if not name or name in STOPWORDS:
                continue
            if name in lowered:
                resolved.append(lowered[name])
            elif close := difflib.get_close_matches(name, list(lowered), n=1, cutoff=0.85):
                resolved.append(lowered[close[0]])
        return list(dict.fromkeys(resolved))

    def _hosts_in(self, question: str, notes: list[str]) -> list[str]:
        explicit: list[str] = []
        for trigger, token in _HOST_RE.findall(question):
            explicit += self.resolve_hosts([token], notes, strict=trigger.lower() != "hostname")
        mentioned = [h for h in self._hosts if re.search(rf"(?<![\w.-]){re.escape(h)}(?![\w-])", question, _I)]
        return list(dict.fromkeys([*explicit, *mentioned]))

    def _processes_in(self, question: str) -> list[str]:
        explicit = self.resolve_processes(_PROCESS_RE.findall(question))
        mentioned = [
            p
            for p in self._processes
            if len(p) >= 3
            and p.lower() not in GENERIC_PROCESS_WORDS | STOPWORDS
            and re.search(rf"(?<![\w.-]){re.escape(p)}(?![\w-])", question, _I)
        ]
        return list(dict.fromkeys([*explicit, *mentioned]))
