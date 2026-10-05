"""Error Detection Agent: critical errors, frequent/recurring patterns, failures; LLM explains the top ones."""

from __future__ import annotations

import asyncio
from typing import Any

from log_analytics.application import prompts
from log_analytics.application.agents.base import Agent
from log_analytics.application.analytics.errors import ErrorDetector
from log_analytics.application.frame_store import FrameStore
from log_analytics.application.llm import LLMGateway
from log_analytics.application.llm_schemas import PatternExplanations
from log_analytics.application.state import AnalysisState
from log_analytics.domain.models import ErrorFindings, ErrorPattern


def _update_patterns(findings: ErrorFindings, updates: dict[str, dict[str, str]]) -> ErrorFindings:
    if not updates:
        return findings

    def apply(patterns: list[ErrorPattern]) -> list[ErrorPattern]:
        return [p.model_copy(update=updates[p.fingerprint]) if p.fingerprint in updates else p for p in patterns]

    return findings.model_copy(
        update={
            "critical_errors": apply(findings.critical_errors),
            "top_patterns": apply(findings.top_patterns),
            "recurring_patterns": apply(findings.recurring_patterns),
        }
    )


class ErrorDetectionAgent(Agent):
    name = "error_detector"

    def __init__(self, frames: FrameStore, detector: ErrorDetector, llm: LLMGateway, *, max_explanations: int) -> None:
        self._frames = frames
        self._detector = detector
        self._llm = llm
        self._max = max_explanations

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        frame = self._frames.get(state["run_id"])
        findings = await asyncio.to_thread(self._detector.detect, frame)
        updates: dict[str, dict[str, str]] = {
            fp: {"label": label} for fp, label in (state.get("template_labels") or {}).items()
        }
        warnings: list[str] = []
        targets = list(findings.all_patterns().values())[: self._max]
        if targets and self._llm.available:
            out, warnings = await self._llm.json(
                PatternExplanations,
                system=prompts.SYSTEM_SRE,
                prompt=prompts.error_explanation(targets),
                purpose="error explanation",
            )
            valid = {p.fingerprint for p in targets}
            for item in out.items if out else []:
                if item.fingerprint in valid and item.explanation.strip():
                    updates.setdefault(item.fingerprint, {}).update(
                        explanation=item.explanation.strip(), likely_impact=item.likely_impact.strip()
                    )
        return {"errors": _update_patterns(findings, updates), "warnings": warnings}
