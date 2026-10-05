"""LLM gateway (degradation + circuit breaker) and the Ollama HTTP adapter (against a mock transport)."""

from __future__ import annotations

import json

import httpx
import pytest

from log_analytics.application.llm import LLMGateway
from log_analytics.application.llm_schemas import TemplateLabels
from log_analytics.domain.errors import LLMResponseError, LLMUnavailableError
from log_analytics.infrastructure.ollama import OllamaClient
from tests.fakes import FakeLLM

# --------------------------------------------------------------------------- gateway


async def test_gateway_disabled_returns_nothing() -> None:
    gateway = LLMGateway(FakeLLM(), enabled=False)
    assert not gateway.available and gateway.model_name is None
    assert await gateway.text(system="s", prompt="p", purpose="x") == (None, [])


async def test_gateway_converts_failures_to_warnings_and_opens_circuit() -> None:
    now = [0.0]
    client = FakeLLM(fail=True)
    gateway = LLMGateway(client, failure_threshold=2, cooldown_s=60, clock=lambda: now[0])
    for _ in range(2):
        result, warnings = await gateway.json(TemplateLabels, system="s", prompt="p", purpose="classification")
        assert result is None and "heuristic fallback" in warnings[0]
    assert not gateway.available
    _, warnings = await gateway.json(TemplateLabels, system="s", prompt="p", purpose="classification")
    assert "circuit open" in warnings[0] and len(client.calls) == 2  # third call never reached the model
    now[0] = 61.0
    client.fail = False
    result, warnings = await gateway.text(system="s", prompt="p", purpose="answer")  # half-open trial succeeds
    assert result and not warnings and gateway.available


# --------------------------------------------------------------------------- Ollama adapter


def _client(handler, **kwargs) -> OllamaClient:  # type: ignore[no-untyped-def]
    return OllamaClient(
        "http://ollama:11434",
        model="llama3.1:8b",
        embed_model="nomic-embed-text",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _chat(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}, "eval_count": 5})


async def test_structured_output_sends_schema_and_parses() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _chat(json.dumps({"items": [{"fingerprint": "f1", "category": "mail", "label": "SMTP blocked"}]}))

    client = _client(handler)
    out = await client.complete_json(system="s", prompt="p", schema=TemplateLabels)
    assert out.items[0].label == "SMTP blocked"
    body = seen[0]
    assert body["model"] == "llama3.1:8b" and body["stream"] is False
    assert body["format"]["title"] == "TemplateLabels" and body["options"]["temperature"] == 0.0
    await client.aclose()


async def test_invalid_json_gets_one_corrective_retry() -> None:
    replies = iter(['{"items": "nope"}', json.dumps({"items": []})])
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _chat(next(replies))

    out = await _client(handler).complete_json(system="s", prompt="p", schema=TemplateLabels)
    assert out.items == [] and len(requests) == 2
    assert "did not match the schema" in requests[1]["messages"][-1]["content"]


async def test_persistently_invalid_json_raises_response_error() -> None:
    client = _client(lambda request: _chat("not json"))
    with pytest.raises(LLMResponseError):
        await client.complete_json(system="s", prompt="p", schema=TemplateLabels)


async def test_retries_transient_http_errors() -> None:
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return _chat("hello") if status == 200 else httpx.Response(status)

    assert await _client(handler).complete(system="s", prompt="p") == "hello"


async def test_missing_model_is_unavailable_with_hint() -> None:
    client = _client(lambda request: httpx.Response(404, json={"error": "model 'llama3.1:8b' not found"}))
    with pytest.raises(LLMUnavailableError, match=r"ollama pull llama3.1:8b"):
        await client.complete(system="s", prompt="p")


async def test_connection_errors_exhaust_retries() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailableError, match="Cannot reach Ollama"):
        await _client(handler, max_retries=1).complete(system="s", prompt="p")
    assert len(calls) == 2


async def test_embeddings_and_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.1:8b", "model": "llama3.1:8b"}]})
        body = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})

    client = _client(handler)
    assert await client.embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert await client.healthy() is True
    assert await _client(lambda r: httpx.Response(200, json={"models": []})).healthy() is False
