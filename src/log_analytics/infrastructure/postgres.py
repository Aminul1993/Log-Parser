"""PostgreSQL connection pool and idempotent schema bootstrap (pgvector + run history)."""

from __future__ import annotations

import json

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

from log_analytics.domain.errors import DataSourceError

log = structlog.get_logger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS log_patterns (
    fingerprint      TEXT PRIMARY KEY,
    process_name     TEXT NOT NULL,
    template         TEXT NOT NULL,
    sample           TEXT NOT NULL,
    hostnames        TEXT[] NOT NULL DEFAULT '{{}}',
    severity         TEXT NOT NULL,
    severity_rank    SMALLINT NOT NULL,
    category         TEXT NOT NULL,
    occurrences      BIGINT NOT NULL,
    first_seen       TIMESTAMPTZ,
    last_seen        TIMESTAMPTZ,
    content          TEXT NOT NULL,
    content_tsv      TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    embedding        VECTOR({dim}) NOT NULL,
    embedding_model  TEXT NOT NULL,
    indexed_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS log_patterns_embedding_hnsw ON log_patterns USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS log_patterns_content_tsv ON log_patterns USING gin (content_tsv);
CREATE INDEX IF NOT EXISTS log_patterns_process ON log_patterns (process_name);
CREATE INDEX IF NOT EXISTS log_patterns_severity ON log_patterns (severity_rank);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    request     JSONB NOT NULL,
    result      JSONB,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS analysis_runs_created ON analysis_runs (created_at DESC);
"""


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def create_pool(dsn: str, *, embedding_dim: int, min_size: int = 1, max_size: int = 10) -> asyncpg.Pool:
    """Create the extension and schema first (the vector codec needs the type to exist), then the pool."""
    conn = await asyncpg.connect(dsn, timeout=10)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(SCHEMA_SQL.format(dim=int(embedding_dim)))
        existing = await conn.fetchval(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'log_patterns'::regclass AND attname = 'embedding'"
        )
    finally:
        await conn.close()
    if existing not in (None, -1, embedding_dim):
        raise DataSourceError(
            f"log_patterns.embedding is vector({existing}) but LA_EMBEDDING_DIM={embedding_dim}; "
            "drop the table or use a matching embedding model"
        )
    pool = await asyncpg.create_pool(
        dsn, min_size=min_size, max_size=max_size, init=_init_connection, command_timeout=60
    )
    log.info("postgres_pool_ready", embedding_dim=embedding_dim)
    return pool
