"""Data Loading Agent: pushes the plan's filter into the Parquet scan and stages the working set."""

from __future__ import annotations

import asyncio
from typing import Any

from log_analytics.application.agents.base import Agent
from log_analytics.application.analytics.profile import profile_frame
from log_analytics.application.dataset import DatasetService
from log_analytics.application.frame_store import FrameStore
from log_analytics.application.state import AnalysisState


class DataLoadingAgent(Agent):
    name = "data_loader"

    def __init__(self, dataset: DatasetService, frames: FrameStore) -> None:
        self._dataset = dataset
        self._frames = frames

    async def run(self, state: AnalysisState) -> dict[str, Any]:
        plan = state["plan"]
        frame = await self._dataset.load_raw(plan.filter)
        self._frames.put(state["run_id"], frame)
        profile = await asyncio.to_thread(profile_frame, frame)
        warnings = [] if frame.height else [f"No log records matched the filter ({plan.filter.describe()})."]
        return {"profile": profile, "warnings": warnings}
