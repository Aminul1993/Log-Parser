"""Anomaly detection: log-volume spikes, error-rate spikes, silence gaps and service-outage bursts.

Spike scoring uses a robust z-score against a *trailing* rolling median (no look-ahead), with the
scale taken from the median absolute deviation of residuals and floored by a Poisson term so that
low-volume series do not explode on tiny absolute changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import polars as pl

from log_analytics.application.analytics.patterns import (
    ERROR_VALUES,
    OUTAGE_RE,
    coarsest_needed,
    interval_spec,
    parse_duration,
)
from log_analytics.domain.models import Anomaly, AnomalyFindings, AnomalyKind, Severity, TimeBucket

MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True, slots=True)
class AnomalyConfig:
    bucket: str = "5m"
    z_threshold: float = 3.5
    baseline_window: int = 12
    min_count: int = 20
    min_error_count: int = 5
    silence_gap: timedelta = timedelta(minutes=15)
    heartbeat_min_events: int = 30
    heartbeat_gap_factor: float = 10.0
    outage_min_events: int = 5
    outage_critical_events: int = 50
    volume_extreme_ratio: float = 10.0
    volume_extreme_min: int = 1000
    seasonal_min_days: int = 3
    max_buckets: int = 20_000
    max_series_points: int = 500
    max_anomalies: int = 50


class AnomalyDetector:
    def __init__(self, config: AnomalyConfig | None = None) -> None:
        self._cfg = config or AnomalyConfig()
        parse_duration(self._cfg.bucket)  # fail fast on bad configuration

    def detect(self, frame: pl.DataFrame) -> AnomalyFindings:
        if frame.is_empty():
            return AnomalyFindings(bucket=self._cfg.bucket, anomalies=[], volume_series=[], series_bucket="1h")
        span = frame["timestamp"].max() - frame["timestamp"].min()
        bucket = coarsest_needed(span, self._cfg.max_buckets, minimum=self._cfg.bucket)
        grid = self._host_grid(frame, bucket)
        anomalies = [
            *self._seasonal(self._spikes(grid, bucket, "total", AnomalyKind.VOLUME_SPIKE, self._cfg.min_count)),
            *self._seasonal(self._spikes(grid, bucket, "errors", AnomalyKind.ERROR_SPIKE, self._cfg.min_error_count)),
            *self._host_silence(grid, bucket),
            *self._heartbeat_gaps(frame),
            *self._outages(frame, bucket),
        ]
        anomalies.sort(key=lambda a: (a.severity.rank, a.score), reverse=True)
        series, series_bucket = self._series(frame, span)
        return AnomalyFindings(
            bucket=bucket,
            anomalies=anomalies[: self._cfg.max_anomalies],
            volume_series=series,
            series_bucket=series_bucket,
        )

    # ------------------------------------------------------------------ bucketing

    @staticmethod
    def _bucket_counts(frame: pl.DataFrame, bucket: str, keys: list[str]) -> pl.DataFrame:
        return (
            frame.lazy()
            .group_by(*keys, pl.col("timestamp").dt.truncate(bucket).alias("bucket"))
            .agg(
                pl.len().cast(pl.Int64).alias("total"),
                pl.col("severity").is_in(ERROR_VALUES).sum().cast(pl.Int64).alias("errors"),
            )
            .collect()
        )

    def _host_grid(self, frame: pl.DataFrame, bucket: str) -> pl.DataFrame:
        """Dense per-host bucket grid (missing buckets → 0) between each host's first and last bucket."""
        counts = self._bucket_counts(frame, bucket, ["hostname"])
        spans = counts.group_by("hostname").agg(pl.col("bucket").min().alias("lo"), pl.col("bucket").max().alias("hi"))
        axis = pl.DataFrame(
            {"bucket": pl.datetime_range(spans["lo"].min(), spans["hi"].max(), interval=bucket, eager=True)}
        ).with_columns(pl.col("bucket").cast(counts.schema["bucket"]))
        return (
            spans.join(axis, how="cross")
            .filter(pl.col("bucket").is_between(pl.col("lo"), pl.col("hi")))
            .join(counts, on=["hostname", "bucket"], how="left")
            .with_columns(pl.col("total").fill_null(0), pl.col("errors").fill_null(0))
            .drop("lo", "hi")
            .sort("hostname", "bucket")
        )

    def _score(self, grid: pl.DataFrame, value: str) -> pl.DataFrame:
        observed = pl.col(value).cast(pl.Float64)
        scored = grid.with_columns(
            observed.shift(1)
            .rolling_median(window_size=self._cfg.baseline_window, min_samples=3)
            .over("hostname")
            .alias("baseline")
        ).with_columns((observed - pl.col("baseline")).alias("resid"))
        scored = scored.with_columns(
            ((pl.col("resid") - pl.col("resid").median()).abs().median().over("hostname") * MAD_TO_SIGMA).alias("mad")
        )
        scale = pl.max_horizontal(pl.col("mad"), pl.col("baseline").clip(lower_bound=1.0).sqrt(), pl.lit(1.0))
        return scored.with_columns((pl.col("resid") / scale).alias("z"))

    @staticmethod
    def _runs(frame: pl.DataFrame, flag: pl.Expr, keys: list[str], value: str) -> pl.DataFrame:
        """Merge consecutive flagged buckets into runs."""
        flagged = frame.with_columns(flag.fill_null(False).alias("flag"))
        changed = (pl.col("flag") != pl.col("flag").shift(1).over(keys)).fill_null(True)
        aggregations = [
            pl.col("bucket").min().alias("start"),
            pl.col("bucket").max().alias("last_bucket"),
            pl.len().alias("buckets"),
            pl.col(value).max().alias("observed"),
            pl.col(value).sum().alias("observed_sum"),
        ]
        if "baseline" in frame.columns:
            aggregations += [pl.col("baseline").mean().alias("expected"), pl.col("z").max().alias("score")]
        return (
            flagged.with_columns(changed.cast(pl.Int64).cum_sum().over(keys).alias("run_id"))
            .filter(pl.col("flag"))
            .group_by(*keys, "run_id")
            .agg(aggregations)
            .sort("start")
        )

    # ------------------------------------------------------------------ detectors

    def _spikes(self, grid: pl.DataFrame, bucket: str, value: str, kind: AnomalyKind, min_count: int) -> list[Anomaly]:
        threshold = self._cfg.z_threshold
        scored = self._score(grid, value)
        runs = self._runs(scored, (pl.col("z") > threshold) & (pl.col(value) >= min_count), ["hostname"], value)
        step = parse_duration(bucket)
        noun = "log lines" if kind is AnomalyKind.VOLUME_SPIKE else "errors"
        cfg = self._cfg
        anomalies = []
        for row in runs.iter_rows(named=True):
            z, observed, expected = float(row["score"]), float(row["observed"]), float(row["expected"] or 0.0)
            if kind is AnomalyKind.VOLUME_SPIKE:
                # Busy hours are normal for application logs; only an order-of-magnitude jump is an error.
                extreme = observed >= cfg.volume_extreme_min and observed >= cfg.volume_extreme_ratio * max(expected, 1)
                severity = Severity.ERROR if extreme else Severity.WARNING
            else:
                sustained = row["buckets"] >= 3 and z >= 3 * threshold
                severity = Severity.CRITICAL if sustained else Severity.ERROR
            anomalies.append(
                Anomaly(
                    kind=kind,
                    hostname=row["hostname"],
                    start=row["start"],
                    end=row["last_bucket"] + step,
                    observed=observed,
                    expected=round(expected, 2),
                    score=round(z, 2),
                    severity=severity,
                    description=(
                        f"{int(observed)} {noun} per {bucket} on {row['hostname']} vs ~{expected:.0f} expected "
                        f"(robust z={z:.1f}, {row['buckets']} bucket(s), {int(row['observed_sum'])} total)"
                    ),
                )
            )
        return anomalies

    def _seasonal(self, spikes: list[Anomaly]) -> list[Anomaly]:
        """Spikes that start at the same time of day on several days are scheduled work (cron, batch
        jobs, backups), not surprises: keep them visible but downgrade and label them."""
        days: dict[tuple[str, str], set] = {}
        for spike in spikes:
            days.setdefault((spike.hostname, f"{spike.start:%H:%M}"), set()).add(spike.start.date())
        result = []
        for spike in spikes:
            seen_on = days[(spike.hostname, f"{spike.start:%H:%M}")]
            if len(seen_on) < self._cfg.seasonal_min_days:
                result.append(spike)
                continue
            severity = Severity.WARNING if spike.kind is AnomalyKind.VOLUME_SPIKE else spike.severity
            result.append(
                spike.model_copy(
                    update={
                        "severity": severity,
                        "description": f"{spike.description}; recurs daily at {spike.start:%H:%M} UTC on "
                        f"{len(seen_on)} days (likely a scheduled job)",
                    }
                )
            )
        return result

    def _host_silence(self, grid: pl.DataFrame, bucket: str) -> list[Anomaly]:
        step = parse_duration(bucket)
        median = pl.col("total").median().over("hostname")
        runs = self._runs(grid, (pl.col("total") == 0) & (median >= 1), ["hostname"], "total")
        anomalies = []
        for row in runs.iter_rows(named=True):
            duration = row["buckets"] * step
            if duration < self._cfg.silence_gap:
                continue
            minutes = duration.total_seconds() / 60
            anomalies.append(
                Anomaly(
                    kind=AnomalyKind.SILENCE_GAP,
                    hostname=row["hostname"],
                    start=row["start"],
                    end=row["last_bucket"] + step,
                    observed=0.0,
                    expected=round(minutes, 1),
                    score=round(minutes / (self._cfg.silence_gap.total_seconds() / 60), 2),
                    severity=Severity.ERROR if duration >= 4 * self._cfg.silence_gap else Severity.WARNING,
                    description=f"{row['hostname']} emitted no logs for {minutes:.0f} min (possible outage or "
                    "logging pipeline failure)",
                )
            )
        return anomalies

    def _heartbeat_gaps(self, frame: pl.DataFrame) -> list[Anomaly]:
        """Regular (heartbeat-like) processes that went quiet far longer than their usual cadence."""
        keys = ["hostname", "process_name"]
        gap_s = pl.col("timestamp").diff().dt.total_milliseconds().over(keys) / 1000.0
        stats = (
            frame.lazy()
            .select(*keys, "timestamp")
            .unique()
            .sort(*keys, "timestamp")
            .with_columns(gap_s.alias("gap_s"))
            .with_columns(
                pl.len().over(keys).alias("n"),
                pl.col("gap_s").median().over(keys).alias("median_s"),
                pl.col("gap_s").quantile(0.9).over(keys).alias("p90_s"),
            )
            .filter(
                (pl.col("n") >= self._cfg.heartbeat_min_events)
                & (pl.col("median_s") > 0)
                & (pl.col("p90_s") <= 3 * pl.col("median_s"))
            )
            .filter(
                pl.col("gap_s")
                > pl.max_horizontal(
                    pl.lit(self._cfg.silence_gap.total_seconds()),
                    pl.col("median_s") * self._cfg.heartbeat_gap_factor,
                )
            )
            .collect()
        )
        anomalies = []
        for row in stats.iter_rows(named=True):
            gap = timedelta(seconds=row["gap_s"])
            minutes = gap.total_seconds() / 60
            anomalies.append(
                Anomaly(
                    kind=AnomalyKind.SILENCE_GAP,
                    hostname=row["hostname"],
                    process_name=row["process_name"],
                    start=row["timestamp"] - gap,
                    end=row["timestamp"],
                    observed=round(minutes, 1),
                    expected=round(row["median_s"] / 60, 2),
                    score=round(row["gap_s"] / row["median_s"], 1),
                    severity=Severity.ERROR if gap >= 4 * self._cfg.silence_gap else Severity.WARNING,
                    description=(
                        f"{row['process_name']} on {row['hostname']} was silent for {minutes:.0f} min "
                        f"(normally logs every {interval_spec(timedelta(seconds=round(row['median_s'])))})"
                    ),
                )
            )
        return anomalies

    def _outages(self, frame: pl.DataFrame, bucket: str) -> list[Anomaly]:
        hits = frame.filter(pl.col("message").str.contains(OUTAGE_RE))
        if hits.is_empty():
            return []
        keys = ["hostname", "process_name"]
        counts = self._bucket_counts(hits, bucket, keys).sort(*keys, "bucket")
        step = parse_duration(bucket)
        flagged = counts.filter(pl.col("total") >= self._cfg.outage_min_events)
        if flagged.is_empty():
            return []
        # A new run starts whenever the previous flagged bucket is not the adjacent one.
        flagged = flagged.with_columns(
            ((pl.col("bucket").diff().over(keys) != step).fill_null(True))
            .cast(pl.Int64)
            .cum_sum()
            .over(keys)
            .alias("run_id")
        )
        runs = (
            flagged.group_by(*keys, "run_id")
            .agg(
                pl.col("bucket").min().alias("start"),
                pl.col("bucket").max().alias("last_bucket"),
                pl.len().alias("buckets"),
                pl.col("total").sum().alias("events"),
                pl.col("total").max().alias("peak"),
            )
            .sort("start")
        )
        anomalies = []
        for row in runs.iter_rows(named=True):
            anomalies.append(
                Anomaly(
                    kind=AnomalyKind.SERVICE_OUTAGE,
                    hostname=row["hostname"],
                    process_name=row["process_name"],
                    start=row["start"],
                    end=row["last_bucket"] + step,
                    observed=float(row["events"]),
                    expected=0.0,
                    score=round(float(row["peak"]) / self._cfg.outage_min_events, 2),
                    severity=Severity.CRITICAL
                    if row["buckets"] >= 3 and row["events"] >= self._cfg.outage_critical_events
                    else Severity.ERROR,
                    description=(
                        f"{row['events']} connectivity/unavailability errors from {row['process_name']} on "
                        f"{row['hostname']} across {row['buckets']} bucket(s) of {bucket}"
                    ),
                )
            )
        return anomalies

    def _series(self, frame: pl.DataFrame, span: timedelta) -> tuple[list[TimeBucket], str]:
        bucket = coarsest_needed(span, self._cfg.max_series_points)
        counts = self._bucket_counts(frame, bucket, []).sort("bucket")
        return (
            [
                TimeBucket(start=row["bucket"], total=row["total"], errors=row["errors"])
                for row in counts.iter_rows(named=True)
            ],
            bucket,
        )
