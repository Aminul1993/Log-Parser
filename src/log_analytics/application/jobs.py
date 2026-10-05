"""In-process background job runner with bounded concurrency and persisted run state.

Suitable for a single API replica. For horizontal scale, swap for a queue (Arq/Celery/SQS) behind the
same ``submit`` contract; run state already lives in the ``RunRepository`` port.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from log_analytics.domain.models import AnalysisRun, RunKind, RunStatus
from log_analytics.domain.ports import RunRepository

log = structlog.get_logger(__name__)
Work = Callable[[str], Awaitable[dict[str, Any]]]


class JobRunner:
    def __init__(self, runs: RunRepository, *, max_concurrency: int = 2) -> None:
        self._runs = runs
        self._slots = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def active(self) -> int:
        return len(self._tasks)

    async def submit(self, kind: RunKind, request: dict[str, Any], work: Work) -> AnalysisRun:
        run = AnalysisRun.new(kind, request)
        await self._runs.save(run)
        task = asyncio.create_task(self._execute(run, work), name=f"{kind.value}:{run.id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        log.info("job_submitted", run_id=run.id, kind=kind.value)
        return run

    async def _execute(self, run: AnalysisRun, work: Work) -> None:
        async with self._slots:
            run = run.transition(RunStatus.RUNNING)
            await self._safe_save(run)
            try:
                result = await work(run.id)
            except asyncio.CancelledError:
                await self._safe_save(run.transition(RunStatus.FAILED, error="cancelled (service shutting down)"))
                raise
            except Exception as exc:
                log.exception("job_failed", run_id=run.id)
                await self._safe_save(run.transition(RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}"[:1000]))
            else:
                await self._safe_save(run.transition(RunStatus.SUCCEEDED, result=result))
                log.info("job_succeeded", run_id=run.id)

    async def _safe_save(self, run: AnalysisRun) -> None:
        try:
            await self._runs.save(run)
        except Exception:
            log.exception("job_state_persist_failed", run_id=run.id, status=run.status.value)

    async def shutdown(self, grace_s: float = 10.0) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=grace_s)
