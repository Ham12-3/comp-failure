"""Request and response models for the API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "For demonstration only. Not financial or credit advice. The model is trained on a small "
    "sample of public Companies House data and is often wrong; do not use it to make decisions "
    "about a company or the people running it."
)


class PredictRequest(BaseModel):
    company_number: str = Field(min_length=1, max_length=12, examples=["03782379"])


class ReasonOut(BaseModel):
    feature: str
    value: str
    effect: Literal["raises", "lowers"]
    contribution: float = Field(description="Contribution to the log-odds (SHAP)")
    text: str


class LinkOut(BaseModel):
    label: str
    url: str


class PredictResponse(BaseModel):
    company_number: str
    company_name: str | None
    company_status: str | None
    scored: bool
    message: str | None = Field(None, description="Why there is no score, when scored is false")
    score_percent: float | None = Field(
        None, description="Calibrated probability of insolvency within 12 months, in %"
    )
    probability: float | None = None
    percentile: float | None = Field(
        None, description="Share of reference filings with a lower or equal score (0-100)"
    )
    relative_risk: float | None = Field(None, description="Probability / average rate")
    risk_band: Literal["typical", "elevated", "high"] | None = None
    accounts_filed_on: date | None = None
    balance_sheet_date: date | None = None
    accounts_type: str | None = None
    horizon_end: date | None = Field(None, description="End of the 12-month window scored")
    reasons: list[ReasonOut] = []
    links: list[LinkOut] = []
    warnings: list[str] = []
    model_trained_at: str | None = None
    disclaimer: str = DISCLAIMER


class SearchItem(BaseModel):
    company_number: str
    title: str | None = None
    company_status: str | None = None
    date_of_creation: str | None = None
    address_snippet: str | None = None


class SearchResponse(BaseModel):
    items: list[SearchItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_trained_at: str | None
    api_key_configured: bool


class LatencySummary(BaseModel):
    mean: float
    p50: float
    p95: float


class MetricsResponse(BaseModel):
    uptime_seconds: float
    requests_total: int
    client_errors_total: int
    server_errors_total: int
    requests_by_path: dict[str, int]
    latency_ms: LatencySummary
