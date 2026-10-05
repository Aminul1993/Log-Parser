"""Common agent behaviour: timing, structured logging and error reporting around ``run``."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import structlog

from log_analytics.application.state import AnalysisState

log = structlog.get_logger(__name__)


class Agent(ABC):
    name: ClassVar[str]

    async def __call__(self, state: AnalysisState) -> dict[str, Any]:
        bound = log.bind(agent=self.name, run_id=state.get("run_id"))
        started = time.perf_counter()
        try:
            update = await self.run(state)
        except Exception:
            bound.exception("agent_failed")
            raise
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        update["timings_ms"] = {self.name: elapsed}
        bound.info("agent_completed", duration_ms=elapsed, warnings=len(update.get("warnings", [])))
        return update

    @abstractmethod
    async def run(self, state: AnalysisState) -> dict[str, Any]:
        """Return a partial state update."""
