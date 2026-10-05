from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status

from log_analytics.api.deps import ContainerDep
from log_analytics.api.schemas import JobAccepted, RagStats, SearchRequest, SearchResponse
from log_analytics.application.rag import PatternIndexer, PatternRetriever
from log_analytics.bootstrap import Container
from log_analytics.domain.errors import FeatureUnavailableError
from log_analytics.domain.models import LogFilter, RunKind, SearchFilters

router = APIRouter(prefix="/rag", tags=["rag"])
_UNAVAILABLE = "Vector store not configured or unreachable: set LA_DATABASE_URL (PostgreSQL with pgvector)"


def _require(container: Container) -> tuple[PatternIndexer, PatternRetriever]:
    if container.indexer is None or container.retriever is None or container.vector_store is None:
        raise FeatureUnavailableError(_UNAVAILABLE)
    return container.indexer, container.retriever


@router.post("/index", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED, summary="Index log patterns")
async def index(container: ContainerDep, request: Request, body: LogFilter | None = None) -> JobAccepted:
    indexer, _ = _require(container)
    flt = body or LogFilter()

    async def work(_: str) -> dict[str, Any]:
        return (await indexer.index(flt)).model_dump(mode="json")

    run = await container.jobs.submit(RunKind.INDEX, {"filter": flt.model_dump(mode="json")}, work)
    return JobAccepted(
        run_id=run.id, kind=run.kind, status=run.status, status_url=str(request.url_for("get_analysis", run_id=run.id))
    )


@router.post("/search", response_model=SearchResponse, summary="Hybrid semantic + keyword search over patterns")
async def search(body: SearchRequest, container: ContainerDep) -> SearchResponse:
    _, retriever = _require(container)
    filters = SearchFilters(
        hostname=body.hostname,
        process_name=body.process_name,
        min_severity=body.min_severity,
        categories=tuple(body.categories),
    )
    return SearchResponse(query=body.query, hits=await retriever.search(body.query, k=body.k, filters=filters))


@router.get("/stats", response_model=RagStats, summary="Indexed document count")
async def stats(container: ContainerDep) -> RagStats:
    _require(container)
    assert container.vector_store is not None
    return RagStats(documents=await container.vector_store.count())
