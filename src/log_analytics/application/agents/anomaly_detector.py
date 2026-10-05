"""Anomaly Detection Agent: volume/error spikes, silence gaps and outage bursts (statistical, no LLM)."""

from __future__ import annotations

import asyncio
from typing import Any

from log_analytics.application.agents.base import Agent
from log_analytics.application.analytics.anomalies import AnomalyDetector
from log_analytics.application.frame_store import FrameStore
from log_analytics.application.state import AnalysisState


class AnomalyDetectionAgent(Agent):
    name = "anomaly_detector"

    def __init__(self, frames: FrameStore, detector: AnomalyDetector) -> None:
        self._frames = frames
        self._detector = detector

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        frame = self._frames.get(state["run_id"])
        return {"anomalies": await asyncio.to_thread(self._detector.detect, frame), "warnings": []}
