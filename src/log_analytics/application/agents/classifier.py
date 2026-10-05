"""Log Classification Agent: rule-based severity/category/template for every row, then LLM refinement
of categories (plus incident-type labels) for the most significant templates."""

from __future__ import annotations

import asyncio
from typing import Any

import polars as pl

from log_analytics.application import prompts
from log_analytics.application.agents.base import Agent
from log_analytics.application.analytics.errors import aggregate_patterns
from log_analytics.application.analytics.profile import profile_frame
from log_analytics.application.dataset import DatasetService
from log_analytics.application.frame_store import FrameStore
from log_analytics.application.llm import LLMGateway
from log_analytics.application.llm_schemas import TemplateLabels
from log_analytics.application.state import AnalysisState
from log_analytics.domain.models import Severity

_WARNING_AND_ABOVE = [s.value for s in Severity.at_least(Severity.WARNING)]


class LogClassificationAgent(Agent):
    name = "classifier"

    def __init__(self, dataset: DatasetService, frames: FrameStore, llm: LLMGateway, *, max_templates: int) -> None:
        self._dataset = dataset
        self._frames = frames
        self._llm = llm
        self._max = max_templates

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        run_id, plan = state["run_id"], state["plan"]
        raw = self._frames.get(run_id)
        frame = await asyncio.to_thread(self._dataset.classify, raw, plan.filter)
        labels: dict[str, str] = {}
        warnings: list[str] = []
        candidates = await asyncio.to_thread(self._candidates, frame)
        if candidates and self._llm.available:
            out, warnings = await self._llm.json(
                TemplateLabels,
                system=prompts.SYSTEM_SRE,
                prompt=prompts.classification(candidates),
                purpose="incident classification",
            )
            if out:
                by_fp = {c["fingerprint"]: c for c in candidates}
                refinements: dict[tuple[str, str], str] = {}
                for item in out.items:
                    candidate = by_fp.get(item.fingerprint)
                    if candidate is None:
                        continue  # hallucinated id
                    if item.label.strip():
                        labels[item.fingerprint] = item.label.strip()[:60]
                    if item.category.value != candidate["category"]:
                        refinements[(candidate["process_name"], candidate["template"])] = item.category.value
                if refinements:
                    frame = await asyncio.to_thread(self._refine, frame, refinements)
        self._frames.put(run_id, frame)
        profile = await asyncio.to_thread(profile_frame, frame)
        return {"profile": profile, "template_labels": labels, "warnings": warnings}

    def _candidates(self, frame: pl.DataFrame) -> list[dict[str, Any]]:
        patterns = aggregate_patterns(frame, severities=_WARNING_AND_ABOVE)
        return patterns.sort(["severity_rank", "occurrences"], descending=True).head(self._max).to_dicts()

    @staticmethod
    def _refine(frame: pl.DataFrame, refinements: dict[tuple[str, str], str]) -> pl.DataFrame:
        """Apply per-template LLM categories to every matching row (one join, not one call per row)."""
        keys = pl.DataFrame(
            {
                "process_name": [k[0] for k in refinements],
                "template": [k[1] for k in refinements],
                "_llm_category": list(refinements.values()),
            }
        )
        return (
            frame.join(keys, on=["process_name", "template"], how="left", maintain_order="left")
            .with_columns(pl.coalesce("_llm_category", "category").alias("category"))
            .drop("_llm_category")
        )
