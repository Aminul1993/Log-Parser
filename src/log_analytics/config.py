"""Application settings. Twelve-factor: every value is overridable through an ``LA_*`` environment variable."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: docker-compose passes unset optional values as "" (e.g. LA_API_KEY=), which must mean "unset".
    model_config = SettingsConfigDict(
        env_prefix="LA_", env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    # --- service
    app_name: str = "log-analytics"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_json: bool = True
    api_key: SecretStr | None = None
    # Production refuses to serve without an API key unless this is set (trusted networks only).
    allow_anonymous: bool = False
    cors_origins: list[str] = Field(default_factory=list)
    query_timeout_s: float = Field(default=600.0, gt=0)
    # Heavy computations (background jobs, /query, dashboards) share these slots.
    max_concurrent_jobs: int = Field(default=2, ge=1)
    max_pending_jobs: int = Field(default=20, ge=1)
    capacity_wait_s: float = Field(default=30.0, gt=0)
    max_stored_runs: int = Field(default=200, ge=1)
    run_retention_days: int = Field(default=30, ge=1)

    # --- data source
    parquet_path: str = "syslog.parquet"
    log_year: int | None = Field(default=None, ge=1970, le=2200)
    log_timezone: str = "UTC"
    cache_dataset: bool = True
    redact_secrets: bool = True
    redact_emails: bool = True
    relative_time_anchor: Literal["dataset_end", "now"] = "dataset_end"

    # --- LLM (Ollama)
    llm_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_s: float = Field(default=300.0, gt=0)
    ollama_max_concurrency: int = Field(default=2, ge=1)
    ollama_num_ctx: int = Field(default=8192, ge=2048)
    ollama_keep_alive: str = "10m"
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    max_patterns_for_llm: int = Field(default=12, ge=1, le=50)
    max_incidents_for_llm: int = Field(default=5, ge=1, le=20)

    # --- RAG / PostgreSQL + pgvector
    database_url: SecretStr | None = None
    require_database: bool = False
    embedding_dim: int = Field(default=768, ge=8, le=16000)
    embed_document_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    rag_batch_size: int = Field(default=32, ge=1, le=512)
    rag_max_documents: int = Field(default=5000, ge=1)
    rag_top_k: int = Field(default=8, ge=1, le=100)

    # --- detection tuning
    anomaly_bucket: str = "5m"
    anomaly_z_threshold: float = Field(default=3.5, gt=0)
    anomaly_baseline_window: int = Field(default=12, ge=3)
    anomaly_min_count: int = Field(default=20, ge=1)
    silence_gap_minutes: int = Field(default=15, ge=1)
    top_n_patterns: int = Field(default=20, ge=1, le=200)
    max_incidents: int = Field(default=10, ge=1, le=100)


@lru_cache
def get_settings() -> Settings:
    return Settings()
