"""FastAPI dependencies: container access, API-key auth and the shared log-filter query parameters."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, TypeVar

from fastapi import Depends, Header, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from log_analytics.bootstrap import Container
from log_analytics.domain.models import KeywordMode, LogFilter, Severity

T = TypeVar("T")


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


async def run_bounded(container: Container, work: Callable[[], Awaitable[T]]) -> T:
    """Run a heavy, request-scoped computation under the shared analysis concurrency limit."""
    try:
        async with asyncio.timeout(container.settings.query_timeout_s):
            return await container.jobs.run_now(work)
    except TimeoutError:
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT, "Computation exceeded the synchronous timeout (LA_QUERY_TIMEOUT_S)."
        ) from None


async def require_api_key(
    container: ContainerDep, x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None
) -> None:
    expected = container.settings.api_key
    if expected is None:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), expected.get_secret_value().encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key", {"WWW-Authenticate": "ApiKey"}
        )


def log_filter(
    start: Annotated[datetime | None, Query(description="Inclusive lower bound (ISO 8601; naive = UTC)")] = None,
    end: Annotated[datetime | None, Query(description="Inclusive upper bound (ISO 8601; naive = UTC)")] = None,
    hostname: Annotated[list[str] | None, Query(description="Repeatable")] = None,
    process: Annotated[list[str] | None, Query(description="Process name (python) or full tag (python[123])")] = None,
    keyword: Annotated[list[str] | None, Query(description="Case-insensitive literal match on message")] = None,
    keyword_mode: KeywordMode = KeywordMode.ANY,
    min_severity: Severity | None = None,
) -> LogFilter:
    try:
        return LogFilter(
            start=start,
            end=end,
            hostnames=hostname or (),
            processes=process or (),
            keywords=keyword or (),
            keyword_mode=keyword_mode,
            min_severity=min_severity,
        )
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


FilterDep = Annotated[LogFilter, Depends(log_filter)]
