"""Job runner capacity limits and run-history lifecycle (recovery after restarts, retention)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from log_analytics.application.jobs import JobRunner
from log_analytics.domain.errors import CapacityError
from log_analytics.domain.models import AnalysisRun, RunKind, RunStatus
from log_analytics.infrastructure.run_repositories import InMemoryRunRepository, _affected


def _gated() -> tuple[asyncio.Event, Any]:
    gate = asyncio.Event()

    async def work(_: str) -> dict[str, Any]:
        await gate.wait()
        return {"ok": True}

    return gate, work


async def _status(runs: InMemoryRunRepository, run_id: str, expected: RunStatus) -> AnalysisRun:
    for _ in range(200):
        run = await runs.get(run_id)
        if run is not None and run.status is expected:
            return run
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} never reached {expected}")


async def test_submit_rejects_when_queue_is_full() -> None:
    runs = InMemoryRunRepository()
    runner = JobRunner(runs, max_concurrency=1, max_pending=2)
    gate, work = _gated()
    first = await runner.submit(RunKind.ANALYSIS, {}, work)
    await runner.submit(RunKind.ANALYSIS, {}, work)  # queued behind the first
    with pytest.raises(CapacityError, match="2 jobs"):
        await runner.submit(RunKind.ANALYSIS, {}, work)
    assert len(await runs.list()) == 2  # the rejected request left no run behind
    gate.set()
    await _status(runs, first.id, RunStatus.SUCCEEDED)
    await runner.shutdown()


async def test_run_now_shares_slots_with_background_jobs() -> None:
    runs = InMemoryRunRepository()
    runner = JobRunner(runs, max_concurrency=1, slot_wait_s=0.05)
    gate, work = _gated()
    job = await runner.submit(RunKind.ANALYSIS, {}, work)
    await _status(runs, job.id, RunStatus.RUNNING)  # the job holds the only slot
    with pytest.raises(CapacityError, match="busy"):
        await runner.run_now(lambda: asyncio.sleep(0, result="never"))
    gate.set()
    await _status(runs, job.id, RunStatus.SUCCEEDED)
    assert await runner.run_now(lambda: asyncio.sleep(0, result="free")) == "free"


async def test_run_now_releases_its_slot_after_errors_and_timeouts() -> None:
    runner = JobRunner(InMemoryRunRepository(), max_concurrency=1, slot_wait_s=0.05)

    async def boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await runner.run_now(boom)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await runner.run_now(lambda: asyncio.sleep(5))
    assert await runner.run_now(lambda: asyncio.sleep(0, result="still usable")) == "still usable"


async def test_in_memory_runs_fail_unfinished_and_purge() -> None:
    runs = InMemoryRunRepository()
    pending = AnalysisRun.new(RunKind.ANALYSIS, {})
    running = AnalysisRun.new(RunKind.INDEX, {}).transition(RunStatus.RUNNING)
    done = AnalysisRun.new(RunKind.ANALYSIS, {}).transition(RunStatus.SUCCEEDED, result={"x": 1})
    old = done.model_copy(update={"id": "old", "created_at": datetime.now(UTC) - timedelta(days=40)})
    for run in (pending, running, done, old):
        await runs.save(run)

    assert await runs.fail_unfinished(reason="restarted") == 2
    for run_id in (pending.id, running.id):
        failed = await runs.get(run_id)
        assert failed is not None and failed.status is RunStatus.FAILED and failed.error == "restarted"
    assert (await runs.get(done.id)) == done

    assert await runs.purge(older_than=timedelta(days=30)) == 1
    assert await runs.get("old") is None and await runs.get(done.id) is not None


def test_command_tag_row_counts() -> None:
    assert _affected("UPDATE 3") == 3
    assert _affected("DELETE 0") == 0
