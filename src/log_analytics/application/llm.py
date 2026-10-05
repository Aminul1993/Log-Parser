"""Fault-tolerant LLM façade used by every agent.

Agents receive ``(result | None, warnings)`` instead of exceptions, so a slow or absent model degrades
the narrative sections to heuristics instead of failing the run. A small circuit breaker stops a
dead Ollama from costing a full timeout per agent.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import structlog
from pydantic import BaseModel

from log_analytics.domain.errors import LLMError, LLMUnavailableError
from log_analytics.domain.ports import LLMClient

T = TypeVar("T", bound=BaseModel)
log = structlog.get_logger(__name__)


class LLMGateway:
    def __init__(
        self,
        client: LLMClient | None,
        *,
        enabled: bool = True,
        failure_threshold: int = 2,
        cooldown_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._enabled = enabled and client is not None
        self._threshold = failure_threshold
        self._cooldown = cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def available(self) -> bool:
        return self._enabled and not self._circuit_open()

    @property
    def model_name(self) -> str | None:
        return self._client.model_name if self._enabled and self._client else None

    async def json(self, schema: type[T], *, system: str, prompt: str, purpose: str) -> tuple[T | None, list[str]]:
        return await self._call(purpose, lambda c: c.complete_json(system=system, prompt=prompt, schema=schema))

    async def text(self, *, system: str, prompt: str, purpose: str) -> tuple[str | None, list[str]]:
        return await self._call(purpose, lambda c: c.complete(system=system, prompt=prompt))

    async def _call(self, purpose: str, invoke):  # type: ignore[no-untyped-def]
        if not self._enabled or self._client is None:
            return None, []
        if self._circuit_open():
            return None, [f"LLM skipped for {purpose}: model unavailable (circuit open); heuristic fallback used."]
        started = time.perf_counter()
        try:
            result = await invoke(self._client)
        except LLMError as exc:
            self._record_failure(exc)
            log.warning("llm_call_failed", purpose=purpose, error=str(exc), error_type=type(exc).__name__)
            return None, [f"LLM {purpose} failed ({type(exc).__name__}: {exc}); heuristic fallback used."]
        except Exception as exc:  # defensive: an adapter bug must not abort the whole analysis
            log.exception("llm_call_crashed", purpose=purpose)
            return None, [f"LLM {purpose} failed unexpectedly ({type(exc).__name__}); heuristic fallback used."]
        self._failures, self._opened_at = 0, None
        log.info("llm_call_succeeded", purpose=purpose, duration_ms=round((time.perf_counter() - started) * 1000))
        return result, []

    def _record_failure(self, exc: LLMError) -> None:
        if not isinstance(exc, LLMUnavailableError):
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = self._clock()

    def _circuit_open(self) -> bool:
        if self._opened_at is None:
            return False
        if self._clock() - self._opened_at >= self._cooldown:
            self._opened_at, self._failures = None, self._threshold - 1  # half-open: one trial call
            return False
        return True
