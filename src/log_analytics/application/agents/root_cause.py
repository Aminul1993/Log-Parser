"""Root Cause Analysis Agent: LLM RCA grounded in evidence, correlated signals, known signatures and
similar historical patterns retrieved from the vector store (RAG). Falls back to signature matching."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import structlog

from log_analytics.application import prompts
from log_analytics.application.agents.base import Agent
from log_analytics.application.llm import LLMGateway
from log_analytics.application.llm_schemas import RootCauseBatch, RootCauseItem
from log_analytics.application.rag import PatternRetriever
from log_analytics.application.state import AnalysisState
from log_analytics.domain.models import Confidence, Incident, RootCause, SearchHit

log = structlog.get_logger(__name__)


def _describe(hit: SearchHit) -> str:
    seen = f", last seen {hit.last_seen:%Y-%m-%d}" if hit.last_seen else ""
    return f"{hit.process_name}: {hit.template[:140]} ({hit.occurrences:,}x{seen}, similarity {hit.similarity:.2f})"


class RootCauseAnalysisAgent(Agent):
    name = "root_cause_analyst"

    def __init__(
        self, llm: LLMGateway, retriever: PatternRetriever | None, *, max_incidents: int, history_k: int = 3
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._max = max_incidents
        self._k = history_k

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        incidents = state.get("incidents") or []
        focus = incidents[: self._max]  # LLM + RAG budget; the rest get signature-based RCA
        history, warnings = await self._history(focus)
        by_id: dict[str, RootCauseItem] = {}
        if focus and self._llm.available:
            out, llm_warnings = await self._llm.json(
                RootCauseBatch,
                system=prompts.SYSTEM_SRE,
                prompt=prompts.root_cause(focus, history),
                purpose="root cause analysis",
            )
            warnings += llm_warnings
            ids = {i.id for i in focus}
            by_id = {item.incident_id: item for item in (out.items if out else []) if item.incident_id in ids}

        causes = []
        for incident in incidents:
            similar = [_describe(h) for h in history.get(incident.id, [])]
            item = by_id.get(incident.id)
            if item and item.probable_cause.strip():
                causes.append(
                    RootCause(
                        incident_id=incident.id,
                        probable_cause=item.probable_cause.strip(),
                        confidence=item.confidence,
                        reasoning=item.reasoning.strip(),
                        contributing_factors=[f.strip() for f in item.contributing_factors if f.strip()][:6],
                        evidence=incident.evidence[:3],
                        similar_history=similar,
                        source="llm",
                    )
                )
            else:
                causes.append(self._heuristic(incident, similar))
        return {"root_causes": causes, "warnings": warnings}

    async def _history(self, incidents: Sequence[Incident]) -> tuple[dict[str, list[SearchHit]], list[str]]:
        if self._retriever is None or not incidents:
            return {}, []
        retriever = self._retriever

        async def similar(incident: Incident) -> tuple[str, list[SearchHit]]:
            query = " ".join([incident.title, *incident.evidence[:1]])[:500]
            return incident.id, await retriever.search(query, k=self._k)

        try:
            return dict(await asyncio.gather(*(similar(i) for i in incidents))), []
        except Exception as exc:
            log.warning("rca_history_failed", error=str(exc))
            return {}, [f"Historical context retrieval failed ({type(exc).__name__}); RCA ran without RAG context."]

    @staticmethod
    def _heuristic(incident: Incident, similar: list[str]) -> RootCause:
        if incident.heuristic_cause:
            return RootCause(
                incident_id=incident.id,
                probable_cause=incident.heuristic_cause,
                confidence=Confidence.MEDIUM,
                reasoning=f"Matched the known failure signature '{incident.knowledge_rule}' in the incident evidence "
                f"({incident.event_count:,} events).",
                contributing_factors=incident.correlated_signals[:4],
                evidence=incident.evidence[:3],
                similar_history=similar,
                source="heuristic",
            )
        processes = ", ".join(incident.processes[:3]) or "unknown process"
        return RootCause(
            incident_id=incident.id,
            probable_cause=f"Undetermined: {incident.kind.value.replace('_', ' ')} in {processes}; no known signature "
            "matched.",
            confidence=Confidence.LOW,
            reasoning="No LLM analysis was available and no curated signature matched; triage the evidence manually.",
            contributing_factors=incident.correlated_signals[:4],
            evidence=incident.evidence[:3],
            similar_history=similar,
            source="heuristic",
        )
