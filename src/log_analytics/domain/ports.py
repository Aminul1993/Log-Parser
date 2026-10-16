"""Ports (interfaces) the application layer depends on; infrastructure provides the adapters.

Polars frames are the deliberate interchange format between the data port and the analytics
services: keeping millions of rows columnar and vectorised is the whole point of the design.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Protocol, TypeVar, runtime_checkable

import polars as pl
from pydantic import BaseModel

from log_analytics.domain.models import AnalysisRun, PatternDocument, SearchFilters, SearchHit

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LogRepository(Protocol):
    """Source of normalised log rows: timestamp(UTC), hostname, process, process_name, pid, message.

    Methods are blocking; callers run them in worker threads.
    """

    def scan(self) -> pl.LazyFrame: ...

    def version(self) -> str:
        """Opaque token that changes whenever the underlying data changes (cache key)."""
        ...

    def describe_source(self) -> str: ...


@runtime_checkable
class LLMClient(Protocol):
    model_name: str

    async def complete(self, *, system: str, prompt: str, temperature: float | None = None) -> str: ...

    async def complete_json(self, *, system: str, prompt: str, schema: type[T]) -> T: ...

    async def healthy(self) -> bool: ...


@runtime_checkable
class EmbeddingClient(Protocol):
    embed_model_name: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(
        self, documents: Sequence[PatternDocument], embeddings: Sequence[Sequence[float]], *, model: str
    ) -> int: ...

    async def search(
        self, embedding: Sequence[float], *, query_text: str, k: int, filters: SearchFilters
    ) -> list[SearchHit]: ...

    async def count(self) -> int: ...

    async def healthy(self) -> bool: ...


@runtime_checkable
class RunRepository(Protocol):
    async def save(self, run: AnalysisRun) -> None: ...

    async def get(self, run_id: str) -> AnalysisRun | None: ...

    async def list(self, *, limit: int = 20) -> list[AnalysisRun]: ...

    async def fail_unfinished(self, *, reason: str) -> int:
        """Mark every pending/running run failed; returns how many. Called at startup, when no task of this
        process can own them yet (single replica: a previous process crashed or was killed)."""
        ...

    async def purge(self, *, older_than: timedelta) -> int:
        """Delete runs created before ``now - older_than``; returns how many."""
        ...
