"""RAG pipeline: index log *patterns* (templates + stats) into a vector store and retrieve them.

Embedding every raw line of a multi-million-line file is slow, expensive and mostly redundant; the
pattern is the semantic unit, and its occurrence counts and time span are what an analyst needs.
"""

from __future__ import annotations

import asyncio
import time

import polars as pl
import structlog

from log_analytics.application.analytics.errors import aggregate_patterns
from log_analytics.application.dataset import DatasetService
from log_analytics.domain.models import (
    Category,
    IndexStats,
    LogFilter,
    PatternDocument,
    SearchFilters,
    SearchHit,
    Severity,
)
from log_analytics.domain.ports import EmbeddingClient, VectorStore

log = structlog.get_logger(__name__)


class PatternIndexer:
    def __init__(
        self,
        dataset: DatasetService,
        embedder: EmbeddingClient,
        store: VectorStore,
        *,
        batch_size: int = 32,
        max_documents: int = 5000,
        document_prefix: str = "",
    ) -> None:
        self._dataset = dataset
        self._embedder = embedder
        self._store = store
        self._batch = batch_size
        self._max = max_documents
        self._prefix = document_prefix

    async def index(self, flt: LogFilter) -> IndexStats:
        started = time.perf_counter()
        frame = await self._dataset.load(flt)
        patterns = await asyncio.to_thread(aggregate_patterns, frame)
        documents = await asyncio.to_thread(self._documents, patterns)
        indexed = 0
        for offset in range(0, len(documents), self._batch):
            batch = documents[offset : offset + self._batch]
            vectors = await self._embedder.embed([self._prefix + d.content() for d in batch])
            indexed += await self._store.upsert(batch, vectors, model=self._embedder.embed_model_name)
            log.debug("rag_batch_indexed", indexed=indexed, total=len(documents))
        stats = IndexStats(
            documents_indexed=indexed,
            patterns_found=patterns.height,
            embedding_model=self._embedder.embed_model_name,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        log.info("rag_index_completed", **stats.model_dump())
        return stats

    def _documents(self, patterns: pl.DataFrame) -> list[PatternDocument]:
        # Most severe first, then most frequent: errors are always indexed even in noisy datasets.
        ranked = patterns.sort(["severity_rank", "occurrences"], descending=True).head(self._max)
        return [
            PatternDocument(
                fingerprint=row["fingerprint"],
                process_name=row["process_name"],
                template=row["template"],
                sample=row["sample"],
                hostnames=row["hostnames"],
                severity=Severity(row["severity"]),
                category=Category(row["category"]),
                occurrences=row["occurrences"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
            )
            for row in ranked.iter_rows(named=True)
        ]


class PatternRetriever:
    def __init__(self, embedder: EmbeddingClient, store: VectorStore, *, query_prefix: str = "") -> None:
        self._embedder = embedder
        self._store = store
        self._prefix = query_prefix

    async def search(self, query: str, *, k: int, filters: SearchFilters | None = None) -> list[SearchHit]:
        [vector] = await self._embedder.embed([self._prefix + query])
        return await self._store.search(vector, query_text=query, k=k, filters=filters or SearchFilters())
