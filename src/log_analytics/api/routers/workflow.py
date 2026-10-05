from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from log_analytics.api.deps import ContainerDep

router = APIRouter(tags=["workflow"])


@router.get(
    "/workflow/diagram",
    response_class=PlainTextResponse,
    summary="Live Mermaid diagram generated from the compiled LangGraph",
)
async def diagram(container: ContainerDep) -> PlainTextResponse:
    return PlainTextResponse(container.workflow.mermaid(), media_type="text/plain")
