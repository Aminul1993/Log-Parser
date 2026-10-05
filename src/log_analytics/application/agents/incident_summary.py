"""Incident Summary Agent: LLM incident classification/impact/summary and the executive narrative."""

from __future__ import annotations

from typing import Any

from log_analytics.application import prompts
from log_analytics.application.agents.base import Agent
from log_analytics.application.llm import LLMGateway
from log_analytics.application.llm_schemas import IncidentSummaryBatch
from log_analytics.application.state import AnalysisState
from log_analytics.domain.models import ExecutiveNarrative, Incident, RootCause, Severity

_IMPACT = {
    Severity.CRITICAL: "Likely service-impacting; immediate attention required.",
    Severity.ERROR: "Degraded functionality likely for affected requests or jobs.",
    Severity.WARNING: "Low immediate impact; monitor for escalation.",
}


class IncidentSummaryAgent(Agent):
    name = "incident_summarizer"

    def __init__(self, llm: LLMGateway, *, max_incidents: int) -> None:
        self._llm = llm
        self._max = max_incidents

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        incidents = state.get("incidents") or []
        causes = {c.incident_id: c for c in state.get("root_causes") or []}
        out, warnings = None, []
        if incidents and self._llm.available:
            out, warnings = await self._llm.json(
                IncidentSummaryBatch,
                system=prompts.SYSTEM_SRE,
                prompt=prompts.incident_summary(state["profile"], incidents[: self._max], list(causes.values())),
                purpose="incident summary",
            )
        items = {item.incident_id: item for item in out.items} if out else {}
        updated = []
        for incident in incidents:
            item = items.get(incident.id)
            if item and item.summary.strip():
                updated.append(
                    incident.model_copy(
                        update={
                            "classification": item.classification.strip()[:80],
                            "impact": item.impact.strip(),
                            "summary": item.summary.strip(),
                        }
                    )
                )
            else:
                updated.append(self._fallback(incident, causes.get(incident.id)))
        narrative = (
            ExecutiveNarrative(headline=out.headline.strip(), narrative=out.narrative.strip())
            if out and out.headline.strip() and out.narrative.strip()
            else None
        )
        return {"incidents": updated, "narrative": narrative, "warnings": warnings}

    @staticmethod
    def _fallback(incident: Incident, cause: RootCause | None) -> Incident:
        hosts = ", ".join(incident.hostnames[:3])
        summary = (
            f"{incident.event_count:,} events between {incident.start:%Y-%m-%d %H:%M} and "
            f"{incident.end:%Y-%m-%d %H:%M} UTC on {hosts}."
        )
        if cause:
            summary += f" Probable cause ({cause.confidence.value} confidence): {cause.probable_cause}"
        return incident.model_copy(
            update={
                "classification": incident.classification
                or f"{incident.category.value.replace('_', ' ')} / {incident.kind.value.replace('_', ' ')}",
                "impact": incident.impact or _IMPACT.get(incident.severity, "Informational."),
                "summary": incident.summary or summary,
            }
        )
