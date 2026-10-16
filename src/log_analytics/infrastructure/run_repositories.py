"""Run-history repositories: PostgreSQL (durable) and in-memory (bounded, single replica)."""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime, timedelta

import asyncpg

from log_analytics.domain.models import AnalysisRun, RunStatus

_UNFINISHED = (RunStatus.PENDING, RunStatus.RUNNING)


def _affected(status: str) -> int:
    """Row count from an asyncpg command tag such as ``UPDATE 3`` or ``DELETE 0``."""
    return int(status.rsplit(" ", 1)[-1])


class InMemoryRunRepository:
    def __init__(self, *, max_runs: int = 200) -> None:
        self._runs: OrderedDict[str, AnalysisRun] = OrderedDict()
        self._max = max_runs

    async def save(self, run: AnalysisRun) -> None:
        self._runs[run.id] = run
        self._runs.move_to_end(run.id)
        while len(self._runs) > self._max:
            self._runs.popitem(last=False)

    async def get(self, run_id: str) -> AnalysisRun | None:
        return self._runs.get(run_id)

    async def list(self, *, limit: int = 20) -> list[AnalysisRun]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)[:limit]

    async def fail_unfinished(self, *, reason: str) -> int:
        stale = [run for run in self._runs.values() if run.status in _UNFINISHED]
        for run in stale:
            self._runs[run.id] = run.transition(RunStatus.FAILED, error=reason)
        return len(stale)

    async def purge(self, *, older_than: timedelta) -> int:
        cutoff = datetime.now(UTC) - older_than
        expired = [run_id for run_id, run in self._runs.items() if run.created_at < cutoff]
        for run_id in expired:
            del self._runs[run_id]
        return len(expired)


class PostgresRunRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, run: AnalysisRun) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO analysis_runs (id, kind, status, request, result, error, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status, result = EXCLUDED.result,
                    error = EXCLUDED.error, updated_at = EXCLUDED.updated_at
                """,
                run.id,
                run.kind.value,
                run.status.value,
                run.request,
                run.result,
                run.error,
                run.created_at,
                run.updated_at,
            )

    async def get(self, run_id: str) -> AnalysisRun | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM analysis_runs WHERE id = $1", run_id)
        return AnalysisRun.model_validate(dict(row)) if row else None

    async def list(self, *, limit: int = 20) -> list[AnalysisRun]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, kind, status, request, NULL::jsonb AS result, error, created_at, updated_at "
                "FROM analysis_runs ORDER BY created_at DESC LIMIT $1",
                limit,
            )
        return [AnalysisRun.model_validate(dict(r)) for r in rows]

    async def fail_unfinished(self, *, reason: str) -> int:
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                "UPDATE analysis_runs SET status = $1, error = $2, updated_at = now() WHERE status = ANY($3::text[])",
                RunStatus.FAILED.value,
                reason,
                [s.value for s in _UNFINISHED],
            )
        return _affected(status)

    async def purge(self, *, older_than: timedelta) -> int:
        async with self._pool.acquire() as conn:
            status = await conn.execute("DELETE FROM analysis_runs WHERE created_at < now() - $1::interval", older_than)
        return _affected(status)
