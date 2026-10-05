# Log-Parser · LangGraph + Ollama Log Analytics

Turns syslog data in Parquet (`Timestamp, Hostname, Process, Message`) into incidents, root causes and
recommended actions. You can ask it questions in plain English.

* **Polars** engine: lazy scans, predicate pushdown and streaming. Analyses all 1.46M rows of the bundled
  file in under a second once warm.
* **LangGraph** workflow of specialised agents: planning, loading, classification, error detection,
  anomaly detection, correlation, root-cause analysis, incident summaries and recommendations.
* **Ollama** (Llama 3, Qwen, Mistral, …) for classification, explanations, RCA, summaries and remediation.
  Uses JSON-schema constrained outputs, and every LLM step has a deterministic fallback.
* **RAG** over log *patterns* in **PostgreSQL + pgvector**, with hybrid vector and full-text retrieval.
* **FastAPI** with async background jobs, API-key auth, request-id structured JSON logs and
  readiness probes; **Docker Compose** for the full stack.

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for Mermaid diagrams (system, layers, LangGraph
workflow, RAG, job lifecycle, data model) and the design rationale.

## What it detects

| Capability | How |
|---|---|
| Critical errors | Severity rules: explicit level tokens first, then keyword heuristics (OOM, panic, segfault, disk full, …). |
| Frequent / recurring error patterns | Template mining (IPs, IDs, numbers, timestamps masked) → occurrences and distinct active hours. |
| Service outages | Bursts of connectivity errors (refused, unreachable, 502/503/504); silence gaps on hosts and regular processes. |
| Process failures | OOM kills, crashes, failed systemd units, abnormal exits, PID restarts (with "were there errors right before it?"). |
| Host-specific issues | Per-host error rates, criticals, anomalies and restarts → 0–100 health score and status. |
| Unusual volume spikes | Robust z-score vs. trailing rolling median; daily-recurring spikes are flagged as scheduled jobs. |

## Quick start (local)

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt && pip install -e . --no-deps

# Deterministic analysis, no LLM needed
log-analytics --no-llm analyze -q "What are today's critical issues?" -o report.md

# With Ollama
ollama pull llama3.1:8b && ollama pull nomic-embed-text
log-analytics analyze -q "Analyze issues for hostname ip-127-0-0-1"

# API
log-analytics serve --port 8000          # or: uvicorn log_analytics.api.app:create_app --factory
open http://localhost:8000/docs
```

Convert raw syslog files to Parquet with the streaming ingester. It replaces `parser.py`, also accepts
RFC 3339 timestamps, and processes the 269 MB `logs/syslog` in about 2 s:

```bash
log-analytics ingest logs/syslog logs/syslog.1 -o syslog.parquet
```

## Docker (API + PostgreSQL/pgvector + Ollama)

```bash
cp .env.example .env                    # optional
docker compose up -d --build            # first start pulls the models (several GB)
curl localhost:8000/ready
```

The Parquet file is mounted read-only from `DATA_DIR` (default: this directory) as
`/data/${PARQUET_FILE:-syslog.parquet}`. Choose models with `LLM_MODEL=qwen2.5:7b` / `EMBED_MODEL=…`
(set `EMBEDDING_DIM` to match). GPU support: uncomment the `deploy` block of the `ollama` service.

## Natural-language questions

| Question | Intent | Path |
|---|---|---|
| What are today's critical issues? | `critical_issues` | full pipeline, "today" = last day in the data |
| Summarize errors by host. | `errors_by_host` | skips RCA/summary/recommendation agents |
| Which processes are failing most frequently? | `failing_processes` | error stats + failures/restarts |
| Show recurring error patterns. | `recurring_patterns` | template statistics |
| Analyze issues for hostname XYZ. | `host_analysis` | full pipeline filtered to the host; unknown hosts are reported, not silently ignored |
| Why did errors spike on web-01? | `root_cause` | full pipeline |
| Find logs about SSM credential failures | `semantic_search` | RAG retrieval |

Rules map common phrasings instantly; the LLM planner is used only when no rule matches, and its
entities are validated against hosts and processes that actually exist. Explicit filters in the request
always override what was parsed from the question.

## API

All `/api/v1` routes require `X-API-Key` when `LA_API_KEY` is set. Interactive docs are at `/docs`.

| Method & path | Purpose |
|---|---|
| `GET /health`, `GET /ready` | Liveness; readiness of dataset (required), LLM and vector store (optional) |
| `GET /api/v1/logs` | Filtered, classified, redacted lines: `start`, `end`, `hostname`*, `process`*, `keyword`*, `keyword_mode`, `min_severity`, `limit`, `offset`, `order` (* repeatable) |
| `GET /api/v1/logs/catalog` · `/logs/stats` | Known hosts/processes/time range · volume and distributions |
| `GET /api/v1/insights/top-errors` · `/host-health` · `/anomalies` | Fast deterministic dashboards (no LLM), same filters |
| `POST /api/v1/query` | `{"question": "...", "filter": {...}, "intent": null}` → synchronous `AnalysisResult` |
| `POST /api/v1/analyses` | Background analysis → `202 {run_id, status_url}` |
| `GET /api/v1/analyses/{id}` | Status + result JSON |
| `GET /api/v1/analyses/{id}/reports/{type}?format=json\|markdown` | `executive_summary`, `incident_report`, `root_cause_report`, `top_error_dashboard`, `host_health_report`, `recommended_actions` |
| `GET /api/v1/analyses/{id}/report.md` | Full Markdown report |
| `POST /api/v1/rag/index` · `POST /api/v1/rag/search` · `GET /api/v1/rag/stats` | Index patterns (job) · hybrid search · count |
| `GET /api/v1/workflow/diagram` | Live Mermaid diagram of the compiled LangGraph |

```bash
curl -s localhost:8000/api/v1/query -H 'content-type: application/json' \
     -d '{"question": "Which processes are failing most frequently?"}' | jq -r .answer

curl -s -X POST localhost:8000/api/v1/analyses -H 'content-type: application/json' \
     -d '{"filter": {"start": "2026-09-30T00:00:00Z", "min_severity": "WARNING"}}'
curl -s localhost:8000/api/v1/analyses/<run_id>/report.md
```

## Configuration

Every setting is an `LA_*` environment variable (or `.env`). See [`.env.example`](.env.example).
Notable ones:

| Variable | Default | Notes |
|---|---|---|
| `LA_PARQUET_PATH` | `syslog.parquet` | Path or glob (`/data/*.parquet`) |
| `LA_LOG_YEAR` / `LA_LOG_TIMEZONE` | inferred / `UTC` | Classic syslog has no year; Dec→Jan rollover handled |
| `LA_RELATIVE_TIME_ANCHOR` | `dataset_end` | What "today" means: `dataset_end` for archives, `now` for live feeds |
| `LA_CACHE_DATASET` | `true` | Classify once per data version in memory; `false` = stream every query |
| `LA_LLM_ENABLED` / `LA_OLLAMA_MODEL` | `true` / `llama3.1:8b` | Any Ollama chat model |
| `LA_DATABASE_URL` | unset | Enables RAG and durable run history; without it, runs are kept in memory |
| `LA_REDACT_SECRETS` / `LA_REDACT_EMAILS` | `true` | Applied before analysis, API output, prompts and embeddings |
| `LA_ANOMALY_BUCKET` / `LA_ANOMALY_Z_THRESHOLD` / `LA_SILENCE_GAP_MINUTES` | `5m` / `3.5` / `15` | Detection tuning |

## Development

```bash
pytest -q             # 120+ tests: synthetic scenarios, fake LLM/vector store, API, plus a real-data smoke test
ruff check src tests && ruff format --check src tests
```

The suite covers parsing edge cases (year rollover, timezones, ISO and mixed formats), classification and
redaction, every detector against planted scenarios, the query parser, the Ollama HTTP contract (retries,
schema-repair retry, 404 hint), the circuit breaker, and the whole LangGraph workflow. Workflow tests
include LLM outages, rejection of hallucinated ids and parity between the cached and streaming paths.
API tests cover jobs, reports, RAG and auth.

## Limitations and next steps

* PostgreSQL/pgvector and a live Ollama model are exercised through their adapters' contracts and test
  doubles; validate them in your environment with `docker compose up` and `/ready`.
* One API worker per container: jobs and the classification cache are in-process. Scale with replicas
  plus a shared queue (the `JobRunner` and `RunRepository` ports are the seams).
* Template mining is regex-based; very noisy free-text messages can still split into near-duplicate
  templates. Incidents merge them by known signature, and you can extend `knowledge.py`.
* Not included: a UI, Prometheus metrics, and alert delivery (Slack/PagerDuty). Each would plug into
  the existing reports.
