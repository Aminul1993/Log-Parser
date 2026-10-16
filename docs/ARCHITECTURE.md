# Architecture

## Design principles

1. **Deterministic facts first, LLM interpretation second.** Counting, grouping, spike detection and
   correlation over millions of rows run in vectorised Polars. The LLM never sees raw volumes. It gets
   compact, redacted evidence (top templates, windows, counts) and explains, classifies and recommends.
   Every LLM-produced section has a deterministic fallback, so the system still works with no model.
2. **Clean Architecture.** `domain` (entities + ports) ← `application` (use cases, analytics, agents)
   ← `infrastructure` (Parquet, Ollama, pgvector) / `api` (FastAPI). Only `bootstrap.py` (the
   composition root) knows concrete classes; everything else gets its collaborators through constructor
   injection.
3. **Patterns, not lines.** Log lines collapse into *templates* (variable tokens masked). Templates are the
   unit of classification by the LLM, explanation, incident building and semantic indexing. 1.46M lines
   become a few thousand templates.
4. **Safe by default.** Secrets (JWTs, bearer tokens, passwords, API keys, AWS keys) and e-mail addresses
   are redacted before anything is classified, returned, embedded or sent to a model. Log text is wrapped
   as untrusted `<evidence>` in prompts, and every id an LLM returns is checked against known ids.

## System context and deployment

```mermaid
flowchart LR
    user([Engineer / SRE / Dashboard]) -->|HTTPS + X-API-Key| api
    cli([CLI: log-analytics]) --> core

    subgraph api_container["api container (FastAPI, 1 worker)"]
        api[REST API /api/v1] --> core[Application core<br/>LangGraph workflow + analytics]
        core --> jobs[Background job runner]
    end

    core -->|lazy scan, predicate pushdown| parquet[(Parquet files<br/>/data/*.parquet, read-only)]
    core -->|/api/chat JSON-schema outputs<br/>/api/embed| ollama[[Ollama<br/>llama3.1 / qwen2.5 / mistral<br/>nomic-embed-text]]
    core -->|asyncpg| pg[(PostgreSQL 17 + pgvector<br/>log_patterns · analysis_runs)]
    raw[(raw syslog files)] -->|log-analytics ingest<br/>streaming| parquet
```

## Layers and dependencies

```mermaid
flowchart TB
    subgraph delivery["Delivery"]
        API["api/<br/>FastAPI routers · auth · middleware · error mapping"]
        CLI["cli.py"]
    end
    subgraph app["Application"]
        WF["workflow.py<br/>LangGraph StateGraph"]
        AG["agents/<br/>planner · loader · classifier · error · anomaly ·<br/>correlator · RCA · summary · recommender · search · reporter"]
        AN["analytics/<br/>classification · patterns · errors · anomalies ·<br/>incidents · knowledge · health · profile"]
        SV["dataset · query_parser · llm gateway · rag · jobs · insights · reporting"]
    end
    subgraph domain["Domain"]
        M["models · reports · errors"]
        P["ports (Protocols)<br/>LogRepository · LLMClient · EmbeddingClient · VectorStore · RunRepository"]
    end
    subgraph infra["Infrastructure"]
        PQ["ParquetLogRepository"]
        OL["OllamaClient"]
        PG["PgVectorStore · PostgresRunRepository"]
        IN["ingest (syslog → Parquet)"]
    end
    BOOT["bootstrap.py<br/>composition root"]

    API --> WF & SV
    CLI --> BOOT
    API --> BOOT
    WF --> AG --> AN & SV
    AG & AN & SV --> M & P
    PQ & OL & PG -.implements.-> P
    BOOT --> PQ & OL & PG & WF
```

## LangGraph workflow

`GET /api/v1/workflow/diagram` returns this graph as generated live from the compiled `StateGraph`.
The annotated version:

```mermaid
flowchart TD
    START([START]) --> planner
    planner["🧭 Query Planner<br/><i>rules → LLM fallback; entity resolution</i>"]
    planner -->|intent = semantic_search| search["🔎 Semantic Search (RAG)"]
    planner -->|otherwise| loader["📥 Data Loading Agent<br/><i>filter pushdown into Parquet scan</i>"]
    loader -->|0 rows| reporter
    loader -->|rows| classifier["🏷️ Log Classification Agent<br/><i>rules per row · LLM per template</i>"]
    classifier --> errors["🚨 Error Detection Agent<br/><i>critical · recurring · failures · LLM explanations</i>"]
    classifier --> anomalies["📈 Anomaly Detection Agent<br/><i>spikes · silence gaps · outages · seasonality</i>"]
    errors --> correlator["🔗 Incident Correlator<br/><i>fan-in · merge by signature · quotas</i>"]
    anomalies --> correlator
    correlator -->|deep intent and incidents| rca["🧠 Root Cause Analysis Agent<br/><i>LLM + known signatures + RAG history</i>"]
    correlator -->|statistical intent| reporter
    rca --> summary["📝 Incident Summary Agent<br/><i>classification · impact · exec narrative</i>"]
    summary --> recommend["🛠️ Recommendation Agent<br/><i>prioritised P1–P3 actions</i>"]
    recommend --> reporter["📊 Report Composer<br/><i>6 reports + answer</i>"]
    search --> reporter
    reporter --> END([END])
```

| Node | Responsibility | LLM use (all optional) |
|---|---|---|
| `planner` | NL question → intent + validated `LogFilter`. Rules handle common phrasings. Relative time resolves against the dataset end (or now). Hosts and processes are matched against the data. | Only when no rule matches (`PlannedQuery`). The LLM's entities are re-validated and times re-resolved deterministically. |
| `data_loader` | Loads the working set once and stages it in the `FrameStore` (DataFrames stay outside graph state). With the cache on, that is one filter over the classified frame; otherwise time/host/process filters are pushed into the Parquet scan and the classifier enriches the rows. | – |
| `classifier` | Severity, category and template for every row; secret redaction; keyword and severity filters. | Classifies the top templates (`TemplateLabels`). Categories are applied to all matching rows with one join. |
| `error_detector` | Critical errors, frequent/recurring patterns, per-host/process stats, explicit failures (OOM, crash, unit failures, abnormal exits), restarts (PID replacement + whether errors preceded it). | Explains the top patterns (`PatternExplanations`). |
| `anomaly_detector` | Robust z-score against a trailing rolling median (MAD scale, Poisson floor) for volume and error spikes; daily-seasonality labelling; host silence; heartbeat gaps for regular processes; outage bursts. | – |
| `correlator` | Turns findings into ranked incidents: merges windows/patterns that share a known signature, enforces per-kind quotas, attaches time-correlated signals and knowledge-base hints. | – |
| `root_cause_analyst` | RCA for every incident. RAG retrieval of similar historical patterns. Falls back to curated signatures. | `RootCauseBatch` for the top N incidents. |
| `incident_summarizer` | Incident classification, impact and summary; executive narrative. | `IncidentSummaryBatch`. |
| `recommender` | Prioritised remediation. Curated actions fill in for incidents the LLM did not cover. | `ActionPlan`. |
| `semantic_search` | Hybrid retrieval over indexed patterns. | – |
| `reporter` | Builds the six reports; answers the question with intent-specific context. | Free-text answer, with a deterministic Markdown fallback. |

Routing keeps cost proportional to the question. *"Summarize errors by host"* uses at most 3 LLM calls
(classification, explanations, answer). A full analysis uses 6. When the model is down, a circuit
breaker in `LLMGateway` stops paying a timeout per agent.

State (`application/state.py`) is a small typed dict of domain objects. Parallel branches merge through
reducers (`warnings` concatenates, `timings_ms` merges), which is also what lets error and anomaly
detection run concurrently.

## RAG pipeline

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI
    participant J as JobRunner
    participant D as DatasetService
    participant O as Ollama /api/embed
    participant V as pgvector (log_patterns)

    C->>API: POST /api/v1/rag/index {filter}
    API->>J: submit(index job) → 202 {run_id}
    J->>D: load + classify (cached per data version)
    D-->>J: rows → aggregate_patterns() → PatternDocument[] (severity-first, capped)
    loop batches of 32
        J->>O: embed("search_document: " + content)
        J->>V: UPSERT … ON CONFLICT (fingerprint)
    end
    C->>API: POST /api/v1/rag/search {query, filters}
    API->>O: embed("search_query: " + query)
    API->>V: HNSW cosine top-50 ∪ full-text top-50 → Reciprocal Rank Fusion
    V-->>API: SearchHit[] (template, counts, first/last seen, score, similarity)
```

* **Indexing unit:** a pattern document (template + example + process, hosts, severity, category,
  occurrences, time span), keyed by a stable SHA-1 fingerprint, so re-indexing is idempotent.
* **Hybrid retrieval:** `embedding <=> query` (HNSW, cosine) fused with `ts_rank_cd` over a generated
  `tsvector` column. Exact tokens such as error codes and unit names match even when embeddings miss them.
* **In the workflow:** the RCA agent retrieves similar historical patterns per incident and passes
  them to the model as `similar_historical_patterns`.

## Analysis job lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant R as RunRepository (Postgres | memory)
    participant J as JobRunner (bounded semaphore)
    participant G as LangGraph workflow

    C->>API: POST /api/v1/analyses {question?, filter, intent?}
    alt queue full (LA_MAX_PENDING_JOBS)
        API-->>C: 429 + Retry-After
    else accepted
        API->>R: save(run: pending)
        API->>J: create_task(work)
        API-->>C: 202 {run_id, status_url}
    end
    J->>R: save(running)
    J->>G: ainvoke(state)
    G-->>J: AnalysisResult
    J->>R: save(succeeded, result JSON)
    C->>API: GET /api/v1/analyses/{id}/reports/{type}?format=markdown
    API->>R: get(id) → render
```

* **Capacity:** background jobs and request-scoped heavy work (`/query`, `/insights/*`, `/logs/stats`) share the
  same `LA_MAX_CONCURRENT_JOBS` slots. A request that cannot get a slot within `LA_CAPACITY_WAIT_S` gets `429`,
  so memory stays bounded by the number of slots, not by the number of clients.
* **Restarts:** at startup, runs still `pending`/`running` belong to a process that no longer exists (crash, OOM
  kill) and are marked failed. That is only safe with one replica per database; several replicas need run
  ownership (heartbeats) first.
* **Retention:** runs older than `LA_RUN_RETENTION_DAYS` are deleted at startup and every 6 hours.

## Data model (PostgreSQL)

```mermaid
erDiagram
    log_patterns {
        text fingerprint PK
        text process_name
        text template
        text sample
        text_array hostnames
        text severity
        smallint severity_rank
        text category
        bigint occurrences
        timestamptz first_seen
        timestamptz last_seen
        text content
        tsvector content_tsv "generated"
        vector embedding "HNSW cosine"
        text embedding_model
    }
    analysis_runs {
        text id PK
        text kind "analysis | index"
        text status
        jsonb request
        jsonb result
        text error
        timestamptz created_at
    }
```

The schema is created idempotently at startup (`infrastructure/postgres.py`). If an existing table's
vector dimension doesn't match `LA_EMBEDDING_DIM`, startup fails with a clear error.

## Performance (measured on the bundled 1.46M-row file, 12-core laptop)

| Step | Time |
|---|---|
| `ingest` 269 MB raw syslog → Parquet (streaming) | 2.2 s |
| Load + normalise Parquet into cache | 0.2–0.3 s |
| Rule classification of all rows (once per data version, at warm-up) | ~4.4 s |
| Deterministic analysis of all rows (load → reports), warm | ~0.7 s |
| LLM prompts per agent | ≤ ~2.5k tokens (fits `num_ctx` 8192) |

With `LA_CACHE_DATASET=false`, every query streams from Parquet and classifies only the rows it selects.
Use that for datasets larger than memory.

## Extension points

* **New failure signature:** add a `KnowledgeRule` in `application/analytics/knowledge.py`. It improves
  RCA, recommendations and incident merging, with or without an LLM.
* **New log source:** implement `LogRepository` (e.g. S3 Parquet, ClickHouse) and wire it in `bootstrap.py`.
* **Different LLM provider:** implement `LLMClient` / `EmbeddingClient`. Agents depend only on the ports.
* **Horizontal scale:** replace `JobRunner` with a queue (Arq/Celery/SQS) behind the same `submit`
  contract; run state already lives in PostgreSQL.
