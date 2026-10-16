"""End-to-end benchmark: warm-up, analyses, dashboards and memory on a real dataset (LLM disabled).

    python benchmarks/bench.py data/syslog.parquet [--runs 2] [--json results.json]

Only public entry points are timed, so results stay comparable across refactors of the internals.
Compare a branch against a baseline by running both on the same machine and dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from log_analytics.bootstrap import build_container
from log_analytics.config import Settings
from log_analytics.domain.models import AnalysisRequest, Intent, LogFilter
from log_analytics.infrastructure.logging import configure_logging

GB = 1024**3


def memory_gb() -> tuple[float, float]:
    """(current, peak) memory of this process: resident set on Linux/macOS, private committed bytes on Windows.

    Windows working sets grow and shrink with free system RAM, so they vary between identical runs; private
    bytes track what the process actually allocated, which is closer to what a container limit counts.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters(cb=ctypes.sizeof(Counters))
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PagefileUsage / GB, counters.PeakPagefileUsage / GB
    import resource

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = max_rss / GB if sys.platform == "darwin" else max_rss * 1024 / GB  # bytes on macOS, KiB on Linux
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            current = int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / GB
    except OSError:
        current = peak
    return current, peak


class StepMemory:
    """Highest current memory while one step runs (sampled), so each step reports its own peak."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.max_gb = 0.0

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.max_gb = max(self.max_gb, memory_gb()[0])
            self._stop.wait(self._interval)

    def __enter__(self) -> StepMemory:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.max_gb = max(self.max_gb, memory_gb()[0])


async def _timed(results: dict[str, Any], name: str, work: Callable[[], Awaitable[Any]]) -> Any:
    started = time.perf_counter()
    with StepMemory() as memory:
        value = await work()
    results[name] = round((time.perf_counter() - started) * 1000, 1)
    results[f"{name}_max_memory_gb"] = round(memory.max_gb, 2)
    print(f"{name:<34} {results[name]:>10,.1f} ms   step max mem {memory.max_gb:5.2f} GB", flush=True)
    return value


async def run(path: str, runs: int) -> dict[str, Any]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        parquet_path=path,
        llm_enabled=False,
        database_url=None,
        log_json=False,
        log_level="ERROR",
    )
    configure_logging(settings.log_level, json=False)
    container = await build_container(settings)
    timings: dict[str, Any] = {}
    try:
        catalog = await _timed(timings, "catalog", container.dataset.catalog)
        await _timed(timings, "warm_up", lambda: asyncio.to_thread(container.dataset.warm))
        timings["memory_after_warm_up_gb"] = round(memory_gb()[0], 2)
        for i in range(runs):
            for intent in (Intent.FULL_ANALYSIS, Intent.ERRORS_BY_HOST):
                result = await _timed(
                    timings,
                    f"analysis_{intent.value}_{i}",
                    lambda intent=intent: container.workflow.run(AnalysisRequest(intent=intent)),
                )
                timings[f"agents_{intent.value}_{i}"] = result.timings_ms
        everything = LogFilter()
        await _timed(timings, "insights_top_errors", lambda: container.insights.top_errors(everything))
        await _timed(timings, "insights_host_health", lambda: container.insights.host_health(everything))
        await _timed(timings, "insights_anomalies", lambda: container.insights.anomalies(everything))
        await _timed(timings, "logs_page", lambda: container.dataset.page(everything, limit=100, offset=0))
    finally:
        await container.aclose()
    return {
        "dataset": path,
        "rows": catalog.total_rows,
        "peak_memory_gb": round(memory_gb()[1], 2),
        "timings_ms": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", help="Parquet file or glob")
    parser.add_argument("--runs", type=int, default=2, help="analyses per intent (first run includes any lazy work)")
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args()
    results = asyncio.run(run(args.parquet, args.runs))
    print(
        f"rows={results['rows']:,}  memory after warm-up={results['timings_ms']['memory_after_warm_up_gb']:.2f} GB  "
        f"process peak={results['peak_memory_gb']:.2f} GB"
    )
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
