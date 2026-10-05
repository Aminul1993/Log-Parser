"""Composition root: wires settings → adapters → services → agents → workflow. The only module that
knows every concrete class; everything else depends on ports and constructor-injected collaborators."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import structlog

from log_analytics.application.agents import (
    AnomalyDetectionAgent,
    DataLoadingAgent,
    ErrorDetectionAgent,
    IncidentCorrelationAgent,
    IncidentSummaryAgent,
    LogClassificationAgent,
    QueryPlannerAgent,
    RecommendationAgent,
    ReportComposerAgent,
    RootCauseAnalysisAgent,
    SemanticSearchAgent,
)
from log_analytics.application.analytics.anomalies import AnomalyConfig, AnomalyDetector
from log_analytics.application.analytics.classification import LogEnricher
from log_analytics.application.analytics.errors import ErrorDetectionConfig, ErrorDetector
from log_analytics.application.analytics.health import HostHealthScorer
from log_analytics.application.analytics.incidents import IncidentCorrelator
from log_analytics.application.analytics.knowledge import KnowledgeBase
from log_analytics.application.analytics.patterns import parse_duration
from log_analytics.application.dataset import DatasetService
from log_analytics.application.frame_store import FrameStore
from log_analytics.application.insights import InsightsService
from log_analytics.application.jobs import JobRunner
from log_analytics.application.llm import LLMGateway
from log_analytics.application.rag import PatternIndexer, PatternRetriever
from log_analytics.application.reporting.builder import ReportBuilder
from log_analytics.application.workflow import AnalysisWorkflow
from log_analytics.config import Settings
from log_analytics.domain.ports import EmbeddingClient, LLMClient, LogRepository, RunRepository, VectorStore
from log_analytics.infrastructure.ollama import OllamaClient
from log_analytics.infrastructure.parquet_repository import ParquetLogRepository
from log_analytics.infrastructure.pgvector_store import PgVectorStore
from log_analytics.infrastructure.postgres import create_pool
from log_analytics.infrastructure.run_repositories import InMemoryRunRepository, PostgresRunRepository

log = structlog.get_logger(__name__)
T = TypeVar("T")


@dataclass
class Container:
    settings: Settings
    dataset: DatasetService
    workflow: AnalysisWorkflow
    insights: InsightsService
    jobs: JobRunner
    runs: RunRepository
    llm: LLMGateway
    llm_client: LLMClient | None
    vector_store: VectorStore | None
    indexer: PatternIndexer | None
    retriever: PatternRetriever | None
    _closers: list[Any] = field(default_factory=list)

    async def readiness(self) -> dict[str, dict[str, Any]]:
        components: dict[str, dict[str, Any]] = {}
        try:
            catalog = await self.dataset.catalog()
            components["dataset"] = {"ok": True, "rows": catalog.total_rows, "source": self.dataset.describe_source()}
        except Exception as exc:
            components["dataset"] = {"ok": False, "error": str(exc)}
        if self.llm_client is not None and self.llm.enabled:
            ok = await self.llm_client.healthy()
            components["llm"] = {"ok": ok, "model": self.llm_client.model_name, "required": False}
        else:
            components["llm"] = {"ok": False, "enabled": False, "required": False}
        if self.vector_store is not None:
            components["vector_store"] = {"ok": await self.vector_store.healthy(), "required": False}
        else:
            components["vector_store"] = {"ok": False, "configured": False, "required": False}
        return components

    async def warmup(self) -> None:
        """Load, classify and cache the dataset in the background so the first request is fast."""
        try:
            await self.dataset.catalog()
            await asyncio.to_thread(self.dataset.warm)
        except Exception as exc:
            log.warning("dataset_warmup_failed", error=str(exc))

    async def aclose(self) -> None:
        await self.jobs.shutdown()
        for close in reversed(self._closers):
            try:
                await close()
            except Exception:
                log.exception("resource_close_failed")


def build_repository(settings: Settings) -> LogRepository:
    return ParquetLogRepository(
        settings.parquet_path,
        log_year=settings.log_year,
        log_timezone=settings.log_timezone,
        cache=settings.cache_dataset,
    )


async def build_container(
    settings: Settings,
    *,
    repository: LogRepository | None = None,
    llm_client: LLMClient | None = None,
    embedder: EmbeddingClient | None = None,
    vector_store: VectorStore | None = None,
    runs: RunRepository | None = None,
) -> Container:
    """Build the object graph. Keyword overrides exist for tests and alternative adapters."""
    closers: list[Any] = []
    repository = repository or build_repository(settings)
    dataset = DatasetService(
        repository,
        LogEnricher(redact_secrets=settings.redact_secrets, redact_emails=settings.redact_emails),
        cache_classified=settings.cache_dataset,
    )

    if llm_client is None and (settings.llm_enabled or settings.database_url):
        ollama = OllamaClient(
            settings.ollama_base_url,
            model=settings.ollama_model,
            embed_model=settings.ollama_embed_model,
            timeout_s=settings.ollama_timeout_s,
            max_concurrency=settings.ollama_max_concurrency,
            temperature=settings.llm_temperature,
            num_ctx=settings.ollama_num_ctx,
            keep_alive=settings.ollama_keep_alive,
            max_retries=settings.llm_max_retries,
        )
        closers.append(ollama.aclose)
        llm_client, embedder = ollama, embedder or ollama
    gateway = LLMGateway(llm_client, enabled=settings.llm_enabled)

    if vector_store is None and settings.database_url is not None:
        try:
            pool = await create_pool(settings.database_url.get_secret_value(), embedding_dim=settings.embedding_dim)
        except Exception as exc:
            if settings.require_database:
                raise
            log.warning("database_unavailable_running_degraded", error=str(exc))
        else:
            closers.append(pool.close)
            vector_store = PgVectorStore(pool)
            runs = runs or PostgresRunRepository(pool)
    runs = runs or InMemoryRunRepository(max_runs=settings.max_stored_runs)

    retriever = indexer = None
    if vector_store is not None and embedder is not None:
        retriever = PatternRetriever(embedder, vector_store, query_prefix=settings.embed_query_prefix)
        indexer = PatternIndexer(
            dataset,
            embedder,
            vector_store,
            batch_size=settings.rag_batch_size,
            max_documents=settings.rag_max_documents,
            document_prefix=settings.embed_document_prefix,
        )

    knowledge = KnowledgeBase()
    error_detector = ErrorDetector(ErrorDetectionConfig(top_n=settings.top_n_patterns))
    anomaly_detector = AnomalyDetector(
        AnomalyConfig(
            bucket=settings.anomaly_bucket,
            z_threshold=settings.anomaly_z_threshold,
            baseline_window=settings.anomaly_baseline_window,
            min_count=settings.anomaly_min_count,
            silence_gap=parse_duration(f"{settings.silence_gap_minutes}m"),
        )
    )
    correlator = IncidentCorrelator(knowledge, max_incidents=settings.max_incidents)
    builder = ReportBuilder(HostHealthScorer())
    frames = FrameStore()

    workflow = AnalysisWorkflow(
        planner=QueryPlannerAgent(dataset, gateway, anchor=settings.relative_time_anchor),
        loader=DataLoadingAgent(dataset, frames),
        classifier=LogClassificationAgent(dataset, frames, gateway, max_templates=settings.max_patterns_for_llm),
        error_detector=ErrorDetectionAgent(
            frames, error_detector, gateway, max_explanations=settings.max_patterns_for_llm
        ),
        anomaly_detector=AnomalyDetectionAgent(frames, anomaly_detector),
        correlator=IncidentCorrelationAgent(frames, correlator),
        root_cause=RootCauseAnalysisAgent(gateway, retriever, max_incidents=settings.max_incidents_for_llm),
        summarizer=IncidentSummaryAgent(gateway, max_incidents=settings.max_incidents_for_llm),
        recommender=RecommendationAgent(gateway, knowledge, max_incidents=settings.max_incidents_for_llm),
        searcher=SemanticSearchAgent(retriever, top_k=settings.rag_top_k),
        reporter=ReportComposerAgent(builder, gateway),
        frames=frames,
    )
    return Container(
        settings=settings,
        dataset=dataset,
        workflow=workflow,
        insights=InsightsService(dataset, error_detector, anomaly_detector, correlator, builder),
        jobs=JobRunner(runs, max_concurrency=settings.max_concurrent_jobs),
        runs=runs,
        llm=gateway,
        llm_client=llm_client,
        vector_store=vector_store,
        indexer=indexer,
        retriever=retriever,
        _closers=closers,
    )


async def run_with_container(settings: Settings, work: Callable[[Container], Awaitable[T]]) -> T:
    """Helper for CLI commands: build, run, always close."""
    container = await build_container(settings)
    try:
        return await work(container)
    finally:
        await container.aclose()
