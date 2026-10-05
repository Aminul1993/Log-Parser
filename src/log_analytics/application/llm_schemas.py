"""Structured-output schemas the LLM must fill (sent to Ollama as JSON Schema → constrained decoding)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from log_analytics.domain.models import Category, Confidence, Intent, Priority


class _LLMOut(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TemplateLabel(_LLMOut):
    fingerprint: str
    category: Category
    label: str = Field(description="2-5 word incident type, e.g. 'IAM credential failure'")


class TemplateLabels(_LLMOut):
    items: list[TemplateLabel]


class PatternExplanation(_LLMOut):
    fingerprint: str
    explanation: str = Field(description="What the error means technically, 1-2 sentences")
    likely_impact: str = Field(description="User/business impact, 1 sentence")


class PatternExplanations(_LLMOut):
    items: list[PatternExplanation]


class RootCauseItem(_LLMOut):
    incident_id: str
    probable_cause: str
    confidence: Confidence
    reasoning: str = Field(description="How the evidence supports the cause, 1-3 sentences")
    contributing_factors: list[str]


class RootCauseBatch(_LLMOut):
    items: list[RootCauseItem]


class IncidentSummaryItem(_LLMOut):
    incident_id: str
    classification: str = Field(description="Incident type, e.g. 'Dependency outage', 'Code defect'")
    impact: str
    summary: str = Field(description="2-3 sentence summary for an on-call engineer")


class IncidentSummaryBatch(_LLMOut):
    headline: str = Field(description="One-line overall status for executives")
    narrative: str = Field(description="3-5 sentence executive summary, plain language")
    items: list[IncidentSummaryItem]


class ActionItem(_LLMOut):
    priority: Priority
    title: str
    steps: list[str] = Field(description="Concrete commands or configuration changes")
    rationale: str
    incident_ids: list[str]
    owner: str = Field(description="Team best placed to act, e.g. 'platform', 'application'")


class ActionPlan(_LLMOut):
    items: list[ActionItem]


class PlannedQuery(_LLMOut):
    intent: Intent
    hostnames: list[str]
    processes: list[str]
    keywords: list[str]
    time_range: str | None = Field(
        default=None, description="Relative or absolute range: 'today', 'yesterday', 'last 6 hours', '2026-09-30'"
    )
    search_text: str | None = None
