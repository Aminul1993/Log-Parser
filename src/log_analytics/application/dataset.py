"""Dataset access service: filtering, classification (memoised per data version) and pagination.

Rule-based classification costs ~3 µs/row (≈30 vectorised regex passes). With caching on, it runs once
per data version (in the background at startup) and every query becomes a cheap filter over the
classified frame. With caching off, each query streams and classifies only the rows it selects, which
is the right trade-off for datasets larger than memory.
"""

from __future__ import annotations

import asyncio
import threading
import time

import polars as pl
import structlog

from log_analytics.application.analytics.classification import LogEnricher
from log_analytics.application.filters import apply_keyword_filter, apply_row_filter, apply_severity_floor
from log_analytics.domain.models import DatasetCatalog, LogFilter, LogPage, LogRecord
from log_analytics.domain.ports import LogRepository

log = structlog.get_logger(__name__)
CATALOG_LIMIT = 500


class DatasetService:
    def __init__(self, repository: LogRepository, enricher: LogEnricher, *, cache_classified: bool = True) -> None:
        self._repo = repository
        self._enricher = enricher
        self._cache_classified = cache_classified
        self._classified: tuple[str, pl.DataFrame] | None = None
        self._catalog: tuple[str, DatasetCatalog] | None = None
        self._classify_lock = threading.Lock()
        self._catalog_lock = threading.Lock()
        self._warmed = threading.Event()

    def describe_source(self) -> str:
        return self._repo.describe_source()

    @property
    def ready(self) -> bool:
        """False until the classification cache is first built: queries would block behind the build."""
        return not self._cache_classified or self._warmed.is_set()

    # ------------------------------------------------------------------ frames (blocking; run in threads)

    def _classified_dataset(self) -> pl.LazyFrame:
        version = self._repo.version()
        with self._classify_lock:
            if self._classified is None or self._classified[0] != version:
                started = time.perf_counter()
                frame = self._build_classified()
                self._classified = (version, frame)
                self._warmed.set()
                log.info(
                    "dataset_classified",
                    rows=frame.height,
                    size_mb=round(frame.estimated_size("mb"), 1),
                    duration_ms=round((time.perf_counter() - started) * 1000),
                )
            return self._classified[1].lazy()

    def _build_classified(self) -> pl.DataFrame:
        """Classify the whole source, sorted by time, with bounded peak memory.

        Enrichment materialises several full-size string intermediates (secret redaction, template masking).
        Running it over one time-ordered slice per Polars thread keeps those to a fraction of the data, and
        the slices become the frame's chunks: the regex-heavy detectors parallelise per chunk, so one chunk
        per thread is also the fast layout (a single chunk made analyses 2-3x slower on 7.8M rows).
        """
        raw = self._repo.scan().sort("timestamp").collect()
        if raw.is_empty():
            return self._enricher.enrich(raw.lazy()).collect()
        step = -(-raw.height // pl.thread_pool_size())  # ceiling division
        parts = [
            self._enricher.enrich(raw.slice(offset, step).lazy()).collect() for offset in range(0, raw.height, step)
        ]
        del raw  # columns the enricher leaves untouched stay shared with the parts
        # Concatenating sorted, consecutive slices keeps the order; restore the flag so re-sorts stay free.
        return pl.concat(parts, rechunk=False).with_columns(pl.col("timestamp").set_sorted())

    def query(self, flt: LogFilter) -> pl.LazyFrame:
        """Classified rows matching every part of the filter."""
        if self._cache_classified:
            frame = apply_row_filter(self._classified_dataset(), flt)
        else:
            frame = self._enricher.enrich(apply_row_filter(self._repo.scan(), flt))
        return apply_severity_floor(apply_keyword_filter(frame, flt), flt.min_severity)

    def classify(self, frame: pl.DataFrame, flt: LogFilter) -> pl.DataFrame:
        """Finish a working set from :meth:`load_working_set` with the same filter.

        A no-op with the cache on: those rows are already classified and fully filtered.
        """
        if self._cache_classified:
            return frame
        classified = self._enricher.enrich(frame.lazy())
        return apply_severity_floor(apply_keyword_filter(classified, flt), flt.min_severity).collect()

    def warm(self) -> None:
        if self._cache_classified:
            self._classified_dataset()

    # ------------------------------------------------------------------ async API

    async def load_raw(self, flt: LogFilter) -> pl.DataFrame:
        """Normalised rows matching the time/host/process filter (unclassified)."""
        return await asyncio.to_thread(
            lambda: apply_row_filter(self._repo.scan(), flt).sort("timestamp").collect(engine="streaming")
        )

    async def load(self, flt: LogFilter) -> pl.DataFrame:
        """Classified rows matching the full filter."""
        return await asyncio.to_thread(lambda: self.query(flt).sort("timestamp").collect())

    async def load_working_set(self, flt: LogFilter) -> pl.DataFrame:
        """Rows an analysis works on, loaded once: classified and fully filtered when the cache is on (one
        filter over the cached frame), otherwise raw time/host/process matches for :meth:`classify`."""
        return await (self.load(flt) if self._cache_classified else self.load_raw(flt))

    async def page(self, flt: LogFilter, *, limit: int, offset: int, descending: bool = True) -> LogPage:
        def work() -> LogPage:
            frame = self.query(flt)
            total_frame, rows = pl.collect_all(
                [
                    frame.select(pl.len().alias("n")),
                    frame.sort("timestamp", descending=descending).slice(offset, limit),
                ]
            )
            items = [
                LogRecord.model_validate({k: row[k] for k in LogRecord.model_fields})
                for row in rows.iter_rows(named=True)
            ]
            return LogPage(total=int(total_frame["n"][0]), limit=limit, offset=offset, items=items)

        return await asyncio.to_thread(work)

    async def catalog(self) -> DatasetCatalog:
        return await asyncio.to_thread(self._catalog_sync)

    def _catalog_sync(self) -> DatasetCatalog:
        version = self._repo.version()
        with self._catalog_lock:
            if self._catalog and self._catalog[0] == version:
                return self._catalog[1]
        frame = self._repo.scan()
        stats, hosts, processes = pl.collect_all(
            [
                frame.select(
                    pl.len().alias("n"), pl.col("timestamp").min().alias("lo"), pl.col("timestamp").max().alias("hi")
                ),
                frame.group_by("hostname").agg(pl.len().alias("n")).sort("n", descending=True).head(CATALOG_LIMIT),
                frame.group_by("process_name").agg(pl.len().alias("n")).sort("n", descending=True).head(CATALOG_LIMIT),
            ]
        )
        catalog = DatasetCatalog(
            total_rows=int(stats["n"][0]),
            start=stats["lo"][0],
            end=stats["hi"][0],
            hostnames=hosts["hostname"].to_list(),
            process_names=processes["process_name"].to_list(),
        )
        with self._catalog_lock:
            self._catalog = (version, catalog)
        return catalog
