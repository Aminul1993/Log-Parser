from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from log_analytics import __version__
from log_analytics.api.deps import ContainerDep

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/ready", summary="Readiness probe (dataset required and warmed up; LLM and vector store optional)")
async def ready(container: ContainerDep) -> JSONResponse:
    components: dict[str, Any] = await container.readiness()
    dataset_ok = components["dataset"]["ok"]
    optional_ok = all(c["ok"] for name, c in components.items() if name != "dataset")
    if dataset_ok:
        status = "ready" if optional_ok else "degraded"
    else:
        status = "warming" if components["dataset"].get("warming") else "unavailable"
    return JSONResponse(status_code=200 if dataset_ok else 503, content={"status": status, "components": components})
