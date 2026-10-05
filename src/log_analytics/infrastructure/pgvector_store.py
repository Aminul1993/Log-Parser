"""pgvector-backed store for log patterns with hybrid retrieval.

Search fuses two rankings with Reciprocal Rank Fusion: cosine similarity over embeddings (semantics)
and PostgreSQL full-text rank (exact tokens such as error codes, unit names, hostnames), which pure
vector search tends to miss in log data.
"""

from __future__ import annotations

from collections.abc import Sequence

import asyncpg
import numpy as np

from log_analytics.domain.models import Category, PatternDocument, SearchFilters, SearchHit, Severity

RRF_K = 60

UPSERT_SQL = """
INSERT INTO log_patterns (fingerprint, process_name, template, sample, hostnames, severity, severity_rank,
                          category, occurrences, first_seen, last_seen, content, embedding, embedding_model)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (fingerprint) DO UPDATE SET
    sample = EXCLUDED.sample,
    hostnames = EXCLUDED.hostnames,
    severity = EXCLUDED.severity,
    severity_rank = EXCLUDED.severity_rank,
    category = EXCLUDED.category,
    occurrences = EXCLUDED.occurrences,
    first_seen = LEAST(log_patterns.first_seen, EXCLUDED.first_seen),
    last_seen = GREATEST(log_patterns.last_seen, EXCLUDED.last_seen),
    content = EXCLUDED.content,
    embedding = EXCLUDED.embedding,
    embedding_model = EXCLUDED.embedding_model,
    indexed_at = now()
"""

_FILTERS = """
    ($3::text IS NULL OR process_name = $3)
    AND ($4::text IS NULL OR $4 = ANY(hostnames))
    AND severity_rank >= $5
    AND ($6::text[] IS NULL OR category = ANY($6))
"""

SEARCH_SQL = f"""
WITH semantic AS (
    SELECT fingerprint, row_number() OVER (ORDER BY distance) AS rnk
    FROM (
        SELECT fingerprint, embedding <=> $1 AS distance
        FROM log_patterns
        WHERE {_FILTERS}
        ORDER BY embedding <=> $1
        LIMIT $7
    ) s
),
lexical AS (
    SELECT fingerprint, row_number() OVER (ORDER BY rank DESC) AS rnk
    FROM (
        SELECT fingerprint, ts_rank_cd(content_tsv, query) AS rank
        FROM log_patterns, websearch_to_tsquery('simple', $2) AS query
        WHERE content_tsv @@ query AND {_FILTERS}
        ORDER BY rank DESC
        LIMIT $7
    ) l
),
fused AS (
    SELECT COALESCE(s.fingerprint, l.fingerprint) AS fingerprint,
           (COALESCE(1.0 / ($8 + s.rnk), 0) + COALESCE(1.0 / ($8 + l.rnk), 0))::float8 AS score
    FROM semantic s FULL OUTER JOIN lexical l ON s.fingerprint = l.fingerprint
)
SELECT p.fingerprint, p.process_name, p.template, p.sample, p.hostnames, p.severity, p.category,
       p.occurrences, p.first_seen, p.last_seen, f.score, (1 - (p.embedding <=> $1))::float8 AS similarity
FROM fused f JOIN log_patterns p ON p.fingerprint = f.fingerprint
ORDER BY f.score DESC, similarity DESC
LIMIT $9
"""


class PgVectorStore:
    def __init__(self, pool: asyncpg.Pool, *, candidate_pool: int = 50) -> None:
        self._pool = pool
        self._candidates = candidate_pool

    async def upsert(
        self, documents: Sequence[PatternDocument], embeddings: Sequence[Sequence[float]], *, model: str
    ) -> int:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        rows = [
            (
                d.fingerprint,
                d.process_name,
                d.template,
                d.sample,
                d.hostnames,
                d.severity.value,
                d.severity.rank,
                d.category.value,
                d.occurrences,
                d.first_seen,
                d.last_seen,
                d.content(),
                np.asarray(e, dtype=np.float32),
                model,
            )
            for d, e in zip(documents, embeddings, strict=True)
        ]
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.executemany(UPSERT_SQL, rows)
        return len(rows)

    async def search(
        self, embedding: Sequence[float], *, query_text: str, k: int, filters: SearchFilters
    ) -> list[SearchHit]:
        categories = [c.value for c in filters.categories] or None
        floor = filters.min_severity.rank if filters.min_severity else 0
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute("SET LOCAL hnsw.ef_search = 100")
            rows = await conn.fetch(
                SEARCH_SQL,
                np.asarray(embedding, dtype=np.float32),
                query_text,
                filters.process_name,
                filters.hostname,
                floor,
                categories,
                max(self._candidates, k),
                RRF_K,
                k,
            )
        return [
            SearchHit(
                fingerprint=r["fingerprint"],
                process_name=r["process_name"],
                template=r["template"],
                sample=r["sample"],
                hostnames=list(r["hostnames"]),
                severity=Severity(r["severity"]),
                category=Category(r["category"]),
                occurrences=r["occurrences"],
                first_seen=r["first_seen"],
                last_seen=r["last_seen"],
                score=round(r["score"], 6),
                similarity=round(r["similarity"], 4),
            )
            for r in rows
        ]

    async def count(self) -> int:
        async with self._pool.acquire() as conn:
            return int(await conn.fetchval("SELECT count(*) FROM log_patterns"))

    async def healthy(self) -> bool:
        try:
            async with self._pool.acquire(timeout=3) as conn:
                return await conn.fetchval("SELECT 1") == 1
        except Exception:
            return False
