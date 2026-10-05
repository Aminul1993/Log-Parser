"""Recommendation Agent: prioritised remediation plan (LLM), backfilled from curated signatures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlencode

from log_analytics.application import prompts
from log_analytics.application.agents.base import Agent
from log_analytics.application.analytics.knowledge import KnowledgeBase
from log_analytics.application.llm import LLMGateway
from log_analytics.application.llm_schemas import ActionPlan
from log_analytics.application.state import AnalysisState
from log_analytics.domain.models import Incident, Priority, Recommendation, Severity

_PRIORITY = {Severity.CRITICAL: Priority.P1, Severity.ERROR: Priority.P2}


class RecommendationAgent(Agent):
    name = "recommender"

    def __init__(self, llm: LLMGateway, knowledge: KnowledgeBase, *, max_incidents: int) -> None:
        self._llm = llm
        self._kb = knowledge
        self._max = max_incidents

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        incidents = state.get("incidents") or []
        focus = incidents[: self._max]  # LLM budget; every incident still gets at least a curated action
        causes = state.get("root_causes") or []
        recommendations: list[Recommendation] = []
        warnings: list[str] = []
        if focus and self._llm.available:
            out, warnings = await self._llm.json(
                ActionPlan,
                system=prompts.SYSTEM_SRE,
                prompt=prompts.recommendations(focus, causes),
                purpose="remediation recommendations",
            )
            ids = {i.id for i in focus}
            for item in (out.items if out else [])[:10]:
                steps = [s.strip() for s in item.steps if s.strip()]
                if not item.title.strip() or not steps:
                    continue
                recommendations.append(
                    Recommendation(
                        priority=item.priority,
                        title=item.title.strip()[:160],
                        steps=steps[:8],
                        rationale=item.rationale.strip(),
                        incident_ids=[i for i in item.incident_ids if i in ids],
                        owner=item.owner.strip() or None,
                        source="llm",
                    )
                )
        covered = {iid for r in recommendations for iid in r.incident_ids}
        errors = state.get("errors")
        down = {s.key for s in errors.by_process if s.errors >= 100 and s.error_rate >= 0.9} if errors else set()
        recommendations += self._heuristic([i for i in incidents if i.id not in covered], down)
        recommendations.sort(key=lambda r: r.priority.value)
        return {"recommendations": recommendations, "warnings": warnings}

    def _heuristic(self, incidents: Sequence[Incident], down_processes: set[str]) -> list[Recommendation]:
        """Curated actions per signature. Priority follows severity, escalated to P1 when a component is
        effectively down (>=90% of its lines are errors), which severity alone does not capture."""
        groups: dict[str, list[Incident]] = {}
        for incident in incidents:
            groups.setdefault(incident.knowledge_rule or f"incident:{incident.id}", []).append(incident)
        recommendations = []
        for group in groups.values():
            worst = max(group, key=lambda i: i.severity.rank)
            priority = _PRIORITY.get(worst.severity, Priority.P3)
            if any(i.processes and all(p in down_processes for p in i.processes) for i in group):
                priority = Priority.P1
            rule = self._kb.get(worst.knowledge_rule) if worst.knowledge_rule else None
            if rule:
                recommendations.append(
                    Recommendation(
                        priority=priority,
                        title=rule.title,
                        steps=list(rule.remediation),
                        rationale=rule.cause,
                        incident_ids=[i.id for i in group],
                        owner=rule.owner,
                        source="heuristic",
                    )
                )
                continue
            query = urlencode(
                [("start", worst.start.isoformat()), ("end", worst.end.isoformat())]
                + [("hostname", h) for h in worst.hostnames[:3]]
                + [("process", p) for p in worst.processes[:3]]
            )
            recommendations.append(
                Recommendation(
                    priority=priority,
                    title=f"Triage: {worst.title[:120]}",
                    steps=[
                        f"Inspect the raw evidence: GET /api/v1/logs?{query}",
                        "Identify the first failing component in the window and check its upstream dependencies.",
                        "Once the cause is confirmed, add a signature to application/analytics/knowledge.py so future "
                        "runs explain it automatically.",
                    ],
                    rationale="No curated signature matched this incident; manual triage is required.",
                    incident_ids=[worst.id],
                    source="heuristic",
                )
            )
        return recommendations
