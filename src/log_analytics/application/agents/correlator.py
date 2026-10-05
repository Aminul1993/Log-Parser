"""Incident Correlation node (fan-in point): joins error and anomaly findings into ranked incidents."""

from __future__ import annotations

import asyncio
from typing import Any

from log_analytics.application.agents.base import Agent
from log_analytics.application.analytics.incidents import IncidentCorrelator
from log_analytics.application.frame_store import FrameStore
from log_analytics.application.state import AnalysisState


class IncidentCorrelationAgent(Agent):
    name = "correlator"

    def __init__(self, frames: FrameStore, correlator: IncidentCorrelator) -> None:
        self._frames = frames
        self._correlator = correlator

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        frame = self._frames.get(state["run_id"])
        incidents = await asyncio.to_thread(self._correlator.correlate, frame, state["errors"], state["anomalies"])
        return {"incidents": incidents, "warnings": []}
