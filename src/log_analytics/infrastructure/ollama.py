"""Async Ollama adapter for chat (structured JSON outputs) and embeddings.

Uses Ollama's native HTTP API: ``/api/chat`` with ``format=<JSON Schema>`` for constrained decoding and
``/api/embed`` for batched embeddings. Works with any pulled model (llama3.x, qwen2.5, mistral, …).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from typing import Any, TypeVar

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from log_analytics.domain.errors import LLMResponseError, LLMUnavailableError

T = TypeVar("T", bound=BaseModel)
log = structlog.get_logger(__name__)
_RETRYABLE = {429, 500, 502, 503, 504}


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        embed_model: str,
        timeout_s: float = 300.0,
        max_concurrency: int = 2,
        temperature: float = 0.1,
        num_ctx: int = 8192,
        keep_alive: str = "10m",
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model_name = model
        self.embed_model_name = embed_model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._retries = max_retries
        self._slots = asyncio.Semaphore(max_concurrency)
        self._http = httpx.AsyncClient(
            base_url=self._base_url, timeout=httpx.Timeout(timeout_s, connect=5.0), transport=transport
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ LLMClient

    async def complete(self, *, system: str, prompt: str, temperature: float | None = None) -> str:
        data = await self._chat(_messages(system, prompt), temperature=temperature)
        return str(data.get("message", {}).get("content", ""))

    async def complete_json(self, *, system: str, prompt: str, schema: type[T]) -> T:
        messages = _messages(system, prompt)
        json_schema = schema.model_json_schema()
        error: ValidationError | None = None
        for _ in range(2):  # one corrective retry with the validation error fed back
            data = await self._chat(messages, fmt=json_schema, temperature=0.0)
            content = str(data.get("message", {}).get("content", ""))
            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                error = exc
                messages = [
                    *messages,
                    {"role": "assistant", "content": content[:4000]},
                    {
                        "role": "user",
                        "content": f"That JSON did not match the schema ({exc.error_count()} error(s): "
                        f"{str(exc)[:600]}). Return corrected JSON only.",
                    },
                ]
        count = error.error_count() if error else "?"
        raise LLMResponseError(f"{self.model_name} output failed {schema.__name__} validation ({count} errors)")

    async def healthy(self) -> bool:
        """True when the server answers and the chat model is installed."""
        try:
            response = await self._http.get("/api/tags", timeout=3.0)
            response.raise_for_status()
            models = response.json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return False
        names = {m.get("name") for m in models} | {m.get("model") for m in models}
        return self.model_name in names or f"{self.model_name}:latest" in names

    # ------------------------------------------------------------------ EmbeddingClient

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        data = await self._post(
            "/api/embed",
            {"model": self.embed_model_name, "input": list(texts), "truncate": True, "keep_alive": self._keep_alive},
            model=self.embed_model_name,
        )
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise LLMResponseError("Unexpected /api/embed response shape")
        return vectors

    # ------------------------------------------------------------------ transport

    async def _chat(
        self, messages: list[dict[str, str]], *, fmt: dict[str, Any] | None = None, temperature: float | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_ctx": self._num_ctx,
            },
        }
        if fmt is not None:
            payload["format"] = fmt
        return await self._post("/api/chat", payload, model=self.model_name)

    async def _post(self, path: str, payload: dict[str, Any], *, model: str) -> dict[str, Any]:
        async with self._slots:
            error: Exception = LLMUnavailableError("Ollama request failed")
            for attempt in range(self._retries + 1):
                started = time.perf_counter()
                try:
                    response = await self._http.post(path, json=payload)
                except httpx.TimeoutException as exc:
                    # A slow model will not get faster on retry; fail fast and let the gateway degrade.
                    raise LLMUnavailableError(f"Ollama timed out after {self._timeout:.0f}s ({path})") from exc
                except httpx.TransportError as exc:
                    error = LLMUnavailableError(f"Cannot reach Ollama at {self._base_url}: {exc}")
                else:
                    if response.status_code == 404:
                        raise LLMUnavailableError(
                            f"Model '{model}' is not available on Ollama (run `ollama pull {model}`): "
                            f"{response.text[:200]}"
                        )
                    if response.status_code in _RETRYABLE:
                        error = LLMUnavailableError(f"Ollama returned HTTP {response.status_code}")
                    elif response.is_error:
                        raise LLMResponseError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")
                    else:
                        data = response.json()
                        log.debug(
                            "ollama_call",
                            path=path,
                            model=model,
                            duration_ms=round((time.perf_counter() - started) * 1000),
                            prompt_tokens=data.get("prompt_eval_count"),
                            output_tokens=data.get("eval_count"),
                        )
                        return data
                if attempt < self._retries:
                    await asyncio.sleep(min(8.0, 0.5 * 2**attempt) + random.uniform(0, 0.25))
            raise error


def _messages(system: str, prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
