"""Per-run DataFrame storage kept outside LangGraph state.

Graph state stays small and serialisable (so a checkpointer can be added later) while agents share
the working set by ``run_id``.
"""

from __future__ import annotations

import polars as pl

from log_analytics.domain.errors import LogAnalyticsError


class FrameStore:
    def __init__(self) -> None:
        self._frames: dict[str, pl.DataFrame] = {}

    def put(self, run_id: str, frame: pl.DataFrame) -> None:
        self._frames[run_id] = frame

    def get(self, run_id: str) -> pl.DataFrame:
        try:
            return self._frames[run_id]
        except KeyError:
            raise LogAnalyticsError(f"No working set loaded for run {run_id}") from None

    def drop(self, run_id: str) -> None:
        self._frames.pop(run_id, None)

    def __len__(self) -> int:
        return len(self._frames)
