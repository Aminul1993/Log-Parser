"""Data Loading Agent: pushes the plan's filter into the scan (or the classified cache) and stages the working set."""

from __future__ import annotations

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
        frame = await self._dataset.load_working_set(plan.filter)
        self._frames.put(state["run_id"], frame)
        if frame.height:
            return {"row_count": frame.height, "warnings": []}  # the classifier profiles the final working set
        # Nothing matched: the reporter runs next and reports on an empty profile.
        warning = f"No log records matched the filter ({plan.filter.describe()})."
        return {"row_count": 0, "profile": profile_frame(frame), "warnings": [warning]}
