"""Semantic Search node (RAG retrieval) for search-style questions."""

from __future__ import annotations

from typing import Any

import structlog

from log_analytics.application.agents.base import Agent
from log_analytics.application.rag import PatternRetriever
from log_analytics.application.state import AnalysisState
from log_analytics.domain.models import SearchFilters

log = structlog.get_logger(__name__)


class SemanticSearchAgent(Agent):
    name = "semantic_search"

    def __init__(self, retriever: PatternRetriever | None, *, top_k: int) -> None:
        self._retriever = retriever
        self._k = top_k

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        plan = state["plan"]
        if self._retriever is None:
            return {
                "search_hits": [],
                "warnings": [
                    "Semantic search unavailable: set LA_DATABASE_URL and index patterns via POST /api/v1/rag/index."
                ],
            }
        flt = plan.filter
        filters = SearchFilters(
            hostname=flt.hostnames[0] if len(flt.hostnames) == 1 else None,
            process_name=flt.processes[0] if len(flt.processes) == 1 else None,
            min_severity=flt.min_severity,
        )
        try:
            hits = await self._retriever.search(plan.search_text or plan.question or "", k=self._k, filters=filters)
        except Exception as exc:
            log.warning("semantic_search_failed", error=str(exc))
            return {"search_hits": [], "warnings": [f"Semantic search failed ({type(exc).__name__}: {exc})."]}
        return {"search_hits": hits, "warnings": []}
