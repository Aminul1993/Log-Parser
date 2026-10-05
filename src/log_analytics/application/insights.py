"""Fast, deterministic insights (no LLM) for dashboards and quick API calls."""

from __future__ import annotations

import asyncio

from log_analytics.application.analytics.anomalies import AnomalyDetector
from log_analytics.application.analytics.errors import ErrorDetector
from log_analytics.application.analytics.incidents import IncidentCorrelator
from log_analytics.application.analytics.profile import profile_frame
from log_analytics.application.dataset import DatasetService
from log_analytics.application.reporting.builder import ReportBuilder
from log_analytics.domain.models import AnomalyFindings, DatasetProfile, LogFilter
from log_analytics.domain.reports import HostHealthReport, TopErrorDashboard


class InsightsService:
    def __init__(
        self,
        dataset: DatasetService,
        errors: ErrorDetector,
        anomalies: AnomalyDetector,
        correlator: IncidentCorrelator,
        builder: ReportBuilder,
    ) -> None:
        self._dataset = dataset
        self._errors = errors
        self._anomalies = anomalies
        self._correlator = correlator
        self._builder = builder

    async def profile(self, flt: LogFilter) -> DatasetProfile:
        frame = await self._dataset.load(flt)
        return await asyncio.to_thread(profile_frame, frame)

    async def anomalies(self, flt: LogFilter) -> AnomalyFindings:
        frame = await self._dataset.load(flt)
        return await asyncio.to_thread(self._anomalies.detect, frame)

    async def top_errors(self, flt: LogFilter) -> TopErrorDashboard:
        frame = await self._dataset.load(flt)

        def work() -> TopErrorDashboard:
            return self._builder.top_error_dashboard(
                profile_frame(frame), self._errors.detect(frame), self._anomalies.detect(frame)
            )

        return await asyncio.to_thread(work)

    async def host_health(self, flt: LogFilter) -> HostHealthReport:
        frame = await self._dataset.load(flt)

        def work() -> HostHealthReport:
            errors, anomalies = self._errors.detect(frame), self._anomalies.detect(frame)
            incidents = self._correlator.correlate(frame, errors, anomalies)
            return self._builder.host_health(errors, anomalies, incidents)

        return await asyncio.to_thread(work)
