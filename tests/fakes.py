"""Test doubles for the LLM, embedding and vector-store ports."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from log_analytics.domain.errors import LLMUnavailableError
from log_analytics.domain.models import PatternDocument, SearchFilters, SearchHit

T = TypeVar("T", bound=BaseModel)
_ID_RE = re.compile(r'"(?:fingerprint|incident_id)":"([^"]+)"')


class FakeLLM:
    """Echoes the ids found in the prompt's evidence back in schema-shaped answers."""

    model_name = "fake-llm"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def complete(self, *, system: str, prompt: str, temperature: float | None = None) -> str:
        self.calls.append("answer")
        if self.fail:
            raise LLMUnavailableError("fake outage")
        return "LLM answer: errors are concentrated on web-01."

    async def complete_json(self, *, system: str, prompt: str, schema: type[T]) -> T:
        self.calls.append(schema.__name__)
        if self.fail:
            raise LLMUnavailableError("fake outage")
        ids = list(dict.fromkeys(_ID_RE.findall(prompt)))
        data: dict[str, Any]
        match schema.__name__:
            case "TemplateLabels":
                data = {
                    "items": [{"fingerprint": i, "category": "application", "label": f"label-{i[:4]}"} for i in ids]
                }
            case "PatternExplanations":
                data = {
                    "items": [{"fingerprint": i, "explanation": f"explained {i}", "likely_impact": "some"} for i in ids]
                }
            case "RootCauseBatch":
                items = [
                    {
                        "incident_id": i,
                        "probable_cause": f"LLM cause {i}",
                        "confidence": "high",
                        "reasoning": "because",
                        "contributing_factors": ["factor"],
                    }
                    for i in ids
                ]
                items.append({**items[0], "incident_id": "INC-HALLUCINATED"} if items else {})
                data = {"items": [i for i in items if i]}
            case "IncidentSummaryBatch":
                data = {
                    "headline": "LLM headline",
                    "narrative": "LLM narrative",
                    "items": [
                        {"incident_id": i, "classification": "Outage", "impact": "High", "summary": "S"} for i in ids
                    ],
                }
            case "ActionPlan":
                data = {
                    "items": [
                        {
                            "priority": "P1",
                            "title": "LLM action",
                            "steps": ["do it"],
                            "rationale": "r",
                            "incident_ids": [*ids, "INC-HALLUCINATED"],
                            "owner": "platform",
                        }
                    ]
                }
            case "PlannedQuery":
                data = {"intent": "full_analysis", "hostnames": ["no-such-host"], "processes": [], "keywords": ["x"]}
            case _:
                raise AssertionError(f"unexpected schema {schema.__name__}")
        return schema.model_validate(data)

    async def healthy(self) -> bool:
        return not self.fail


class FakeEmbedder:
    """Deterministic hashed bag-of-words embeddings (cosine-comparable)."""

    embed_model_name = "fake-embed"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in re.findall(r"[a-z]{3,}", text.lower()):
                vector[int(hashlib.md5(token.encode(), usedforsecurity=False).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.docs: dict[str, tuple[PatternDocument, list[float]]] = {}

    async def upsert(
        self, documents: Sequence[PatternDocument], embeddings: Sequence[Sequence[float]], *, model: str
    ) -> int:
        for doc, vector in zip(documents, embeddings, strict=True):
            self.docs[doc.fingerprint] = (doc, list(vector))
        return len(documents)

    async def search(
        self, embedding: Sequence[float], *, query_text: str, k: int, filters: SearchFilters
    ) -> list[SearchHit]:
        scored = []
        for doc, vector in self.docs.values():
            if filters.process_name and doc.process_name != filters.process_name:
                continue
            if filters.hostname and filters.hostname not in doc.hostnames:
                continue
            similarity = sum(a * b for a, b in zip(embedding, vector, strict=True))
            scored.append((similarity, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                **doc.model_dump(exclude={"first_seen", "last_seen"}),
                first_seen=doc.first_seen,
                last_seen=doc.last_seen,
                score=round(sim, 4),
                similarity=round(sim, 4),
            )
            for sim, doc in scored[:k]
        ]

    async def count(self) -> int:
        return len(self.docs)

    async def healthy(self) -> bool:
        return True
