"""Query Planner Agent: natural-language question → intent + validated filter."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from log_analytics.application import prompts
from log_analytics.application.agents.base import Agent
from log_analytics.application.dataset import DatasetService
from log_analytics.application.llm import LLMGateway
from log_analytics.application.llm_schemas import PlannedQuery
from log_analytics.application.query_parser import RuleBasedQueryParser
from log_analytics.application.state import AnalysisState
from log_analytics.domain.models import Intent, LogFilter, QueryPlan


class QueryPlannerAgent(Agent):
    name = "planner"

    def __init__(
        self,
        dataset: DatasetService,
        llm: LLMGateway,
        *,
        anchor: Literal["dataset_end", "now"] = "dataset_end",
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._dataset = dataset
        self._llm = llm
        self._anchor = anchor
        self._clock = clock

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        request = state["request"]
        base_notes = [] if self._llm.enabled else ["LLM disabled: narrative sections are generated heuristically."]
        if not request.question:
            plan = QueryPlan(
                intent=request.intent or Intent.FULL_ANALYSIS, filter=request.filter, source="request", notes=base_notes
            )
            return {"plan": plan, "warnings": []}

        catalog = await self._dataset.catalog()
        anchor = catalog.end if self._anchor == "dataset_end" and catalog.end else self._clock()
        parser = RuleBasedQueryParser(catalog.hostnames, catalog.process_names, anchor)
        parsed = parser.parse(request.question)
        intent, flt, search_text, source = parsed.intent, parsed.filter, parsed.search_text, "rules"
        notes, warnings = [*base_notes, *parsed.notes], []

        if intent is None and self._llm.available:
            planned, warnings = await self._llm.json(
                PlannedQuery,
                system=prompts.SYSTEM_SRE,
                prompt=prompts.query_plan(request.question, catalog, anchor),
                purpose="query planning",
            )
            if planned:
                intent, source = planned.intent, "llm"
                flt = self._from_llm(planned, parsed.filter, parser, set(catalog.hostnames), notes)
                search_text = (planned.search_text or "").strip() or search_text
        if intent is None:
            intent = Intent.FULL_ANALYSIS
            notes.append("The question did not map to a specific intent; ran a full analysis.")

        plan = QueryPlan(
            intent=request.intent or intent,
            filter=flt.merged_with(request.filter),
            question=request.question,
            search_text=search_text,
            source=source,  # type: ignore[arg-type]
            notes=notes,
        )
        return {"plan": plan, "warnings": warnings}

    @staticmethod
    def _from_llm(
        planned: PlannedQuery,
        rules: LogFilter,
        parser: RuleBasedQueryParser,
        known_hosts: set[str],
        notes: list[str],
    ) -> LogFilter:
        """Accept only LLM entities that exist in the data; times are re-resolved deterministically.

        LLM-proposed keywords are deliberately *not* applied as row filters (they would silently narrow
        detection); they only feed semantic search via ``search_text``.
        """
        time_range = parser.resolve_time(planned.time_range)
        hosts = [h for h in parser.resolve_hosts(planned.hostnames, notes, strict=True) if h in known_hosts]
        return LogFilter(
            start=time_range[0] if time_range else rules.start,
            end=time_range[1] if time_range else rules.end,
            hostnames=hosts or rules.hostnames,
            processes=parser.resolve_processes(planned.processes) or rules.processes,
            keywords=rules.keywords,
        )
