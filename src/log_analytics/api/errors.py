"""Map domain exceptions to HTTP responses."""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from log_analytics.domain.errors import (
    CapacityError,
    ConflictError,
    DataSourceError,
    FeatureUnavailableError,
    LLMError,
    LogAnalyticsError,
    NotFoundError,
)

log = structlog.get_logger(__name__)

_STATUS: list[tuple[type[Exception], int]] = [
    (NotFoundError, 404),
    (ConflictError, 409),
    (CapacityError, 429),
    (FeatureUnavailableError, 503),
    (DataSourceError, 503),
    (LLMError, 503),
    (LogAnalyticsError, 400),
]
RETRY_AFTER_S = 30


def _request_id() -> str | None:
    return structlog.contextvars.get_contextvars().get("request_id")


def register_exception_handlers(app: FastAPI) -> None:
    async def domain_error(_: Request, exc: Exception) -> JSONResponse:
        status = next(code for kind, code in _STATUS if isinstance(exc, kind))
        log.warning("request_failed", error=str(exc), error_type=type(exc).__name__, status=status)
        headers = {"Retry-After": str(RETRY_AFTER_S)} if isinstance(exc, CapacityError) else None
        return JSONResponse(
            status_code=status, content={"detail": str(exc), "request_id": _request_id()}, headers=headers
        )

    async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error", error_type=type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": _request_id()})

    app.add_exception_handler(LogAnalyticsError, domain_error)
    app.add_exception_handler(Exception, unexpected_error)
