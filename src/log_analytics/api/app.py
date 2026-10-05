"""FastAPI application factory. Run with:  uvicorn log_analytics.api.app:create_app --factory"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from log_analytics import __version__
from log_analytics.api.deps import require_api_key
from log_analytics.api.errors import register_exception_handlers
from log_analytics.api.middleware import RequestContextMiddleware
from log_analytics.api.routers import analyses, health, logs, rag, workflow
from log_analytics.bootstrap import Container, build_container
from log_analytics.config import Settings, get_settings
from log_analytics.infrastructure.logging import configure_logging


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else get_settings())
    configure_logging(settings.log_level, json=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        app.state.container = container or await build_container(settings)
        warmup = asyncio.create_task(app.state.container.warmup())  # load + cache the dataset in background
        try:
            yield
        finally:
            warmup.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warmup
            if owned:
                await app.state.container.aclose()

    app = FastAPI(
        title="Log Analytics API",
        version=__version__,
        description="LangGraph + Ollama log analytics over Parquet syslog data, with RAG on PostgreSQL/pgvector.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
        )
    register_exception_handlers(app)
    app.include_router(health.router)
    protected = [Depends(require_api_key)]
    for module in (logs, analyses, rag, workflow):
        app.include_router(module.router, prefix="/api/v1", dependencies=protected)
    return app
