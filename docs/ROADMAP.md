# Roadmap: remaining optimisation and hardening work

Phases 0 and 1 of the October 2026 audit are done (golden-master test, benchmark harness, CI, fail-closed
auth, capacity limits, single dataset copy, warm-up-aware readiness, run recovery and retention). This file
lists what is left, in priority order, with the evidence behind each item.

Item ids (R = recommendation, S = security, D = database, C = code quality, Q = quick win) match the
audit report.

## Where things stand

Measured on `data/syslog.parquet` (7,819,075 rows, 1 host, 22 processes) with `benchmarks/bench.py`, LLM
off, 12-core Windows laptop. `main` and this branch were A/B-tested on the same day; timings vary ±30–50% between
runs on that machine, so always compare A/B on the same day.

| Metric | Before Phase 1 | After Phase 1 | Target |
|---|---|---|---|
| Full analysis | 9.1–9.9 s | 5.4–7.7 s | ≈ 3–5 s |
| Warm-up (classification) | 46–59 s | 33–45 s | seconds (classify at ingest) |
| Warm-up memory peak | 7.2–7.4 GB | 4.6 GB | < 3 GB |
| Process memory peak | 7.2–7.4 GB | 6.65 GB | **below the container limit** |
| `/insights/top-errors`, `/host-health` | ≈ 7 s each | ≈ 7 s each (untouched) | ms when repeated |

The memory column is private committed bytes on Windows. Working-set numbers drift with free RAM, so the
benchmark reports private bytes instead.

**The API container still cannot hold the full dataset.** `docker-compose.yml` limits the `api` service to
4 GB, but the analyses now peak at about 6.65 GB, which is detector working memory during each run. Until
Phase 2 lands, run the full file with a 6–8 GB limit, or point `LA_PARQUET_PATH` at a smaller time range.

## Phase 2: high-impact performance (≈ 2–3 days)

Every change must keep `tests/test_golden.py` green. Each one also needs an A/B run of `benchmarks/bench.py`
against `main`; use a `git worktree` of `main` with `PYTHONPATH=<worktree>/src`.

| Id | Change | Evidence | Risk |
|---|---|---|---|
| R6 | Vectorise restart detection: compute "errors in the last minute before exit" for every PID with one join, not one full-frame scan per restart event (`ErrorDetector._restarts` / `_errors_before_exit`). | Prototype: 4,475 ms → 57 ms for the 48 restart events. | Low |
| R7 | Compute `failure_kind` (Enum) and `is_outage` (bool) once in `LogEnricher.enrich`. Detectors use the cached columns and fall back to the regexes for frames enriched elsewhere. Move `FAILURE_RULES` and `UNIT_RE` to `patterns.py`. | Saves 1.9–3.3 s (failure regexes) plus 1.0–1.7 s (outage regex) per run and per dashboard. Costs about 3 s once at warm-up and about 35 MB. | Low |
| R8 | Correlator windows: binary search (`search_sorted`) on the time-sorted frame plus a host filter on the slice, not a full filter per anomaly. | Prototype: 310–509 ms → 79 ms for 50 anomalies, identical windows. | Low |
| R9 | Dashboard result cache: a small async LRU that shares one computation between concurrent identical requests, keyed by (data version, filter). Domain models are frozen, so cached results are safe to share. | Repeated dashboard calls drop from about 7 s to ms. | Low |
| R10 | Dictionary-encode low-cardinality columns in the cache: `template` and `hostname`/`process`/`process_name` as Categorical, `severity`/`category` as Enum. | `template` drops from 632 MB to 30 MB (74,923 distinct values); the other columns save 266 MB; the classified frame goes from 1.89 GB to about 1.05 GB. | **Medium.** Cast to `String` wherever ordering or types matter: the `mode().sort().first()` tie-breaks, `sort([... "template"])`, the join and `coalesce` in `classifier._refine`, and `replace_strict` on severity. |
| — | Measure the per-analysis memory peak after R6–R8. If it is still above the limit, find the next allocator (candidates: the `_heartbeat_gaps` unique and sort over the full frame, and the `rows`/`error_rows` copies in the correlator). | The peak moved from warm-up to the analyses in Phase 1. | — |
| — | Size a dedicated thread pool for Polars work (2–4 threads) to avoid oversubscription when error and anomaly detection run in parallel. | Agent timings inflate 1.2–1.7× when the two detectors overlap. | Low |
| R11 | Vector store integrity: filter search by `embedding_model`; upsert `occurrences = GREATEST(...)` and the union of `hostnames`; `SET LOCAL hnsw.iterative_scan = relaxed_order` when pgvector ≥ 0.8. | D3–D5 in the audit. | Medium; re-index after the model filter |
| R12 | Treat LLM output as untrusted: label model-generated remediation steps, drop markdown images from answers, escape `<` and backticks in table cells and evidence. | S3/S4: indirect prompt injection reaches operators. | Low; cosmetic `\<IP>` in raw markdown |

Layout constraint learned in Phase 1: the regex-heavy detectors parallelise **per chunk**. Keep the cached
frame at about one chunk per Polars thread. A single chunk (after `rechunk()`) or the thousands of small
chunks the streaming engine produces both made analyses 2–3× slower.

## Quick wins (each under an hour)

- Q1: Remove `pyarrow` from `requirements.txt` and `pyproject.toml`. The suite passes with pyarrow imports blocked, and it takes 88 MB installed.
- Q2: Set `LANGSMITH_TRACING=false` in the Dockerfile so graph state (log evidence) can never be traced to an external service.
- Q3: Pin Docker images by version and digest (`ollama/ollama`, `pgvector/pgvector`, `python:3.12-slim`).
- Q4: Replace `--forwarded-allow-ips "*"` with a configurable proxy CIDR.
- Q5: Root-cause history: one batched `embed` call for all incidents instead of one per incident.
- Q6: Fix the README's stale performance claims ("1.46M rows … under a second"). Align the compose default dataset path (`./syslog.parquet`) with the real location (`data/syslog.parquet`).
- Q8: Delete dead code: `Severity.from_rank`, `JobRunner.active`, `FrameStore.__len__`.
- Q9: Replace the `assert` in `api/routers/rag.py` (`stats`) with an explicit check.
- Q10: Share one helper for the duplicated `JobAccepted(... url_for("get_analysis"))` blocks.
- Q11: Drop the near-useless `log_patterns_severity` index (5 distinct values).
- Q12: Use `uvicorn` with `uvloop` and `httptools` instead of `uvicorn[standard]`, which pulls in `watchfiles` and `websockets`.

## Phase 3: security, database and code quality (≈ 2–3 days)

- S5: Require a JSON content type, or a key, on body-less `POST /rag/index`. In production this is already mitigated, because R1 requires a key.
- S7: Return generic error messages in production and keep the details in logs. That covers `str(exc)` in API errors, Ollama response bodies in warnings, and the source path on `/ready`.
- S9: Support several API keys (rotation) and log which key accessed what.
- S10: Extend redaction (private-key blocks, GitHub and Slack tokens) and redact user questions before they reach the LLM, the logs or stored runs.
- D6: Move DDL out of app startup into versioned migrations run as a separate step, with an advisory lock.
- D7: Revisit indexes only if profiling shows a need; the `log_patterns` table stays small.
- C1–C4: Remove duplication: the first-match `when/then` builder, pattern ranking, the row-to-model mappers, and knowledge-base regexes that copy detector regexes.
- C7: Add a type checker (pyright or mypy) to CI and remove the `# type: ignore` workarounds, for example in `render_report` and `LLMGateway._call`.
- C8: Turn magic numbers in agents into named constants or settings.
- Tests:
  - Postgres integration tests with testcontainers + pgvector (schema bootstrap, hybrid search, upsert semantics, `fail_unfinished`, `purge`).
  - Security tests: prompt-injection payloads in log lines come out neutralised.
  - Coverage threshold in CI.

## Phase 4: structural (optional, ≈ 1–2 weeks)

1. **Classify at ingest.** `log-analytics ingest --enrich` writes severity, category, template and the
   failure/outage flags into Parquet, together with a rule version. Workers then scan lazily with predicate
   pushdown: no 33–45 s warm-up and no requirement that the dataset fits in RAM.
2. **Separate the API from workers.** Use a PostgreSQL-backed job queue (`SELECT … FOR UPDATE SKIP LOCKED`)
   with run ownership and heartbeats. The API becomes stateless and can scale horizontally, and startup run
   recovery becomes safe with several replicas (today it assumes one).
3. **Rollups.** Keep per (hour, host, process, template) aggregates for dashboards and anomaly grids, so
   most reads touch thousands of rows instead of millions.
