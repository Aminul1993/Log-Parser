"""Report Composer: builds the six reports and answers the user's question."""

from __future__ import annotations

from typing import Any

from log_analytics.application import prompts
from log_analytics.application.agents.base import Agent
from log_analytics.application.llm import LLMGateway
from log_analytics.application.reporting.answers import AnswerMaterial
from log_analytics.application.reporting.builder import ReportBuilder
from log_analytics.application.state import AnalysisState
from log_analytics.domain.reports import AnalysisResult


class ReportComposerAgent(Agent):
    name = "reporter"

    def __init__(self, builder: ReportBuilder, llm: LLMGateway) -> None:
        self._builder = builder
        self._llm = llm

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        plan = state["plan"]
        reports = self._builder.build(
            profile=state.get("profile"),
            errors=state.get("errors"),
            anomalies=state.get("anomalies"),
            incidents=state.get("incidents"),
            root_causes=state.get("root_causes"),
            recommendations=state.get("recommendations"),
            narrative=state.get("narrative"),
        )
        answer: str | None = None
        warnings: list[str] = []
        if plan.question:
            material = AnswerMaterial(
                profile=state.get("profile"),
                errors=state.get("errors"),
                anomalies=state.get("anomalies"),
                incidents=state.get("incidents"),
                hits=state.get("search_hits") or [],
                reports=reports,
            )
            if not material.empty and self._llm.available:
                answer, warnings = await self._llm.text(
                    system=prompts.SYSTEM_SRE,
                    prompt=prompts.answer(plan.question, plan.intent, material.context(plan.intent)),
                    purpose="answer generation",
                )
            answer = (answer or "").strip() or material.fallback(plan.intent)
        result = AnalysisResult(
            run_id=state["run_id"],
            question=plan.question,
            plan=plan,
            answer=answer,
            reports=reports,
            anomalies=state.get("anomalies"),
            search_hits=state.get("search_hits") or [],
            llm_model=self._llm.model_name,
        )
        return {"result": result, "warnings": warnings}
